"""Grouped task loss: action-set NLL, object forward KL and singleton content CE.

One full-vocabulary normalizer per supervised row; no student top-k, dense
vocabulary mask, or duplicate softmax. Correct supports travel as COO edges.
"""
import torch


class _ChunkedNormalizer(torch.autograd.Function):
    """Keep original logits, not an extra full FP32 vocabulary tensor, for backward."""
    @staticmethod
    def forward(ctx, logits):
        result = torch.empty(logits.shape[0], device=logits.device, dtype=torch.float32)
        for start in range(0, logits.shape[0], 1024):
            result[start:start+1024] = logits[start:start+1024].float().logsumexp(-1)
        ctx.save_for_backward(logits, result)
        return result

    @staticmethod
    def backward(ctx, grad):
        logits, norm = ctx.saved_tensors
        result = torch.empty_like(logits)
        for start in range(0, logits.shape[0], 1024):
            probability = (logits[start:start+1024].float() - norm[start:start+1024, None]).exp()
            result[start:start+1024] = (probability * grad[start:start+1024, None]).to(logits.dtype)
        return result


def segment_logsumexp(values, rows, size):
    maximum = values.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, rows, values.detach(), reduce='amax', include_self=True)
    safe = maximum.masked_fill(~maximum.isfinite(), 0.)
    total = values.new_zeros(size).scatter_add(0, rows, (values - safe[rows]).exp())
    # Empty segments must not create NaN gradients (including zero teacher tail).
    return torch.where(total > 0, safe + total.clamp_min(1e-30).log(), -torch.inf)


def selected_token_losses(logits, targets, single, multi, edge_rows, edge_ids, teacher_ids, teacher_logs,
                          preference_rows=None, preference_ids=None):
    """All row indices here are relative to the selected student projection."""
    normalizer = _ChunkedNormalizer.apply(logits)
    ce = normalizer - logits.gather(1, targets[:, None]).squeeze(1).float()
    single_ce = torch.where(single, ce, 0.)
    edge_logs = logits[edge_rows, edge_ids].float() - normalizer[edge_rows]
    logz = segment_logsumexp(edge_logs, edge_rows, logits.shape[0])
    set_loss = torch.where(multi, -logz.clamp_max(0.).expm1(), 0.)
    preference = normalizer * 0.
    teacher_mass = torch.ones_like(normalizer)
    if teacher_ids.shape[0]:
        selected = multi.nonzero(as_tuple=True)[0]
        q = teacher_logs.exp()
        mass = q.sum(-1)
        teacher_mass = teacher_mass.index_copy(0, selected, mass)
        logp = logits[selected[:, None], teacher_ids].float() - normalizer[selected, None]
        # Student-only candidates have known legality but unknown teacher mass.
        # Learn preferences only among observed teacher entries, without assigning
        # artificial zero targets to the other known-legal tokens.
        pref_rows = edge_rows if preference_rows is None else preference_rows
        pref_ids = edge_ids if preference_ids is None else preference_ids
        pref_logs = edge_logs if preference_rows is None else logits[pref_rows, pref_ids].float() - normalizer[pref_rows]
        pref_logz = logz if preference_rows is None else segment_logsumexp(pref_logs, pref_rows, logits.shape[0])
        conditional = logp - pref_logz[selected, None]
        terms = q * torch.where(q > 0, teacher_logs - conditional, 0.)
        # All unreported CORRECT tokens form one coarse tail event. Do not
        # include illegal vocabulary in this tail or renormalize sparse q.
        vocab_size = logits.shape[1]
        keys = (selected[:, None] * vocab_size + teacher_ids)[q > 0].sort().values
        edge_keys = pref_rows * vocab_size + pref_ids
        if keys.numel():
            indices = torch.searchsorted(keys, edge_keys).clamp_max(keys.numel() - 1)
            omitted = keys[indices] != edge_keys
        else:
            omitted = torch.ones_like(pref_ids, dtype=torch.bool)
        logtail = segment_logsumexp(pref_logs[omitted], pref_rows[omitted], logits.shape[0])[selected]
        qtail = (1. - mass).clamp(0., 1.)
        # Rounding can leave a tiny positive qtail even for complete supports.
        positive_tail = (qtail > 1e-6) & logtail.isfinite()
        tail_delta = torch.where(positive_tail,
            qtail.clamp_min(1e-30).log() - (logtail - pref_logz[selected]), 0.)
        pref = terms.sum(-1) + qtail * tail_delta
        preference = preference.index_copy(0, selected, pref)
    mass = torch.where(single, (-ce).exp(), logz.exp())
    return single_ce, set_loss, preference, mass, teacher_mass, normalizer



def task_token_losses(logits, targets, single, multi, objects, edge_rows, edge_ids,
                      teacher_ids, teacher_logs, preference_weight, repair=None):
    """Exact action-set NLL; object top-k+tail forward KL; singleton content CE.

    Object tail is one coarse event over the remaining full vocabulary. No
    object legality enumeration and no normalization of student top-k logits.
    """
    normalizer = _ChunkedNormalizer.apply(logits)
    ce = torch.where(single, normalizer - logits.gather(1, targets[:, None]).squeeze(1).float(), 0.)
    edge_logs = logits[edge_rows, edge_ids].float() - normalizer[edge_rows]
    logz = segment_logsumexp(edge_logs, edge_rows, logits.shape[0])
    repair = torch.zeros_like(multi) if repair is None else repair
    objects = objects & ~repair
    actions = multi & ~objects
    set_loss = torch.where(actions, -logz.clamp_max(0.), 0.)
    preference = normalizer * 0.
    object_kl = normalizer * 0.
    mass = torch.where(single, (-ce).exp(), logz.exp())
    teacher_mass = torch.ones_like(normalizer)
    selected = multi.nonzero(as_tuple=True)[0]
    if selected.numel():
        q = teacher_logs.exp()
        # Only correct floating-point overshoot, never condition on top-k.
        q = q / q.sum(-1, keepdim=True).clamp_min(1.)
        logq = q.clamp_min(1e-30).log()
        logp = logits[selected[:, None], teacher_ids].float() - normalizer[selected, None]
        observed = q > 0
        qmass = q.sum(-1)
        pmass_log = logp.masked_fill(~observed, -torch.inf).logsumexp(-1).clamp_max(0.)
        terms = (q * torch.where(observed, logq - logp, 0.)).sum(-1)
        qtail = (1. - qmass).clamp_min(0.)
        ptail = (-pmass_log.expm1()).clamp_min(1e-30)
        tail = qtail * (qtail.clamp_min(1e-30).log() - ptail.log())
        is_object = objects[selected]
        object_kl = object_kl.index_copy(0, selected, torch.where(is_object, terms + tail, 0.))
        # Action supports are exact and fully returned by the teacher.
        conditional = logp - torch.where(is_object, 0., logz[selected])[:, None]
        pref = (q * torch.where(observed, logq - conditional, 0.)).sum(-1)
        preference = preference.index_copy(0, selected, torch.where(is_object | repair[selected], 0., pref))
        mass = mass.index_copy(0, selected, torch.where(is_object, pmass_log.exp(), logz[selected].exp()))
        teacher_mass = teacher_mass.index_copy(0, selected, qmass)
    composite = ce + set_loss + preference_weight * preference + object_kl
    return ce, set_loss, preference, mass, teacher_mass, normalizer, composite


def correctness_loss_outputs(data, student_logits, token_mask, student_positions=None, preference_weight=0.1):
    """Map packed response supports to causal input rows, preserving empty-rank autograd."""
    selected = token_mask.values().bool()
    causal = selected.nonzero(as_tuple=True)[0]
    flat = student_logits.reshape(-1, student_logits.shape[-1])
    if student_positions is None:
        rows = flat.index_select(0, causal)
    else:
        if not torch.equal(student_positions, causal):
            raise ValueError('Correctness supports do not match selected student projections')
        rows = flat
    input_offsets = data['input_ids'].offsets()
    response_offsets = data['manga_opsd_mask'].offsets()
    shifts = input_offsets[:-1] + input_offsets.diff() - response_offsets.diff() - 1
    response_causal = (torch.arange(data['manga_opsd_mask'].values().numel(), device=rows.device)
        + torch.repeat_interleave(shifts - response_offsets[:-1], response_offsets.diff()))
    single = torch.zeros_like(selected)
    multi = torch.zeros_like(selected)
    single[response_causal] = data['manga_single_mask'].values().bool()
    multi[response_causal] = data['manga_multi_mask'].values().bool()
    if not torch.equal(single | multi, selected) or bool((single & multi).any()):
        raise ValueError('Every supervised token must have exactly one correctness loss class')
    edges = data['manga_correct_edges']
    entries = edges.values().long()
    edge_causal = entries[:, 0] + torch.repeat_interleave(shifts, edges.offsets().diff())
    edge_rows = torch.searchsorted(causal, edge_causal)
    teacher_causal = causal[multi[causal]]
    targets = data['input_ids'].values()[causal + 1].long()
    weights = None
    if 'manga_group_weights' in data:
        weights = rows.new_zeros((selected.numel(), 3), dtype=torch.float32)
        weights[response_causal] = data['manga_group_weights'].values().float()
        object_rows = weights[causal, 1] > 0
        repairs = torch.zeros_like(selected)
        if 'manga_repair_mask' in data:
            repairs[response_causal] = data['manga_repair_mask'].values().bool()

        results = task_token_losses(rows, targets, single[causal], multi[causal], object_rows,
            edge_rows, entries[:, 1], data['teacher_ids'].values()[teacher_causal].long(),
            data['teacher_logprobs'].values()[teacher_causal].float(), preference_weight, repairs[causal])
        ce, set_loss, preference, mass, teacher_mass, normalizer, composite = results
    else:
        pref_rows = pref_ids = None
        if 'manga_preference_edges' in data:
            pref = data['manga_preference_edges']
            pref_entries = pref.values().long()
            pref_causal = pref_entries[:, 0] + torch.repeat_interleave(shifts, pref.offsets().diff())
            pref_rows, pref_ids = torch.searchsorted(causal, pref_causal), pref_entries[:, 1]
        results = selected_token_losses(rows, targets, single[causal], multi[causal], edge_rows, entries[:, 1],
            data['teacher_ids'].values()[teacher_causal].long(),
            data['teacher_logprobs'].values()[teacher_causal].float(), pref_rows, pref_ids)
        ce, set_loss, preference, mass, teacher_mass, normalizer = results
        composite = ce + set_loss + preference_weight * preference
    def scatter(values):
        return values.new_zeros((1, selected.numel())).index_copy(1, causal, values.unsqueeze(0))
    output = dict(distillation_losses=scatter(composite), raw_distillation_losses=scatter(composite.detach()),
        student_mass=scatter(mass.detach()), teacher_mass=scatter(teacher_mass.detach()),
        manga_single_ce=scatter(ce), manga_set_loss=scatter(set_loss),
        manga_preference_kl=scatter(preference))
    if weights is not None:
        output['manga_repair_nll'] = scatter(torch.where(repairs[causal], ce + set_loss, 0.))
        output['manga_repair_mass'] = scatter(torch.where(repairs[causal], mass.detach(), 0.))
        output['manga_repair_rows'] = scatter(repairs[causal].float())
        # Reuse the existing paired first-microbatch replay, no extra forward.
        for name in ('rows', 'mass', 'nll'):
            output['action_repair_' + name] = output['manga_repair_' + name]

        output['manga_action_correctness'] = scatter((ce + set_loss) * weights[causal, 0])
        output['manga_action_preference'] = scatter(preference_weight * preference * weights[causal, 0])
        for index, group in enumerate(('action', 'object', 'content')):
            output['manga_group_' + group] = scatter(composite * weights[causal, index])
    if 'manga_candidate_edges' in data:
        # This measures CURRENT student mass outside the checked union; it remains
        # informative when rollout candidates become stale during minibatch updates.
        checked = data['manga_candidate_edges']
        checked_entries = checked.values().long()
        checked_causal = checked_entries[:, 0] + torch.repeat_interleave(shifts, checked.offsets().diff())
        checked_rows = torch.searchsorted(causal, checked_causal)
        checked_logs = (rows[checked_rows, checked_entries[:, 1]].float().detach()
                        - normalizer.detach()[checked_rows])
        checked_mass = segment_logsumexp(checked_logs, checked_rows, rows.shape[0]).exp()
        approx = torch.zeros_like(selected)
        approx[response_causal] = data['manga_approx_mask'].values().bool()
        approx = approx[causal]
        output['manga_candidate_unchecked'] = scatter(torch.where(approx, (1.-checked_mass).clamp(0.,1.), 0.))
        output['manga_candidate_rows'] = scatter(approx.float())
    # Diagnostics reuse this normalizer instead of reducing the vocabulary again.
    norms = normalizer if student_positions is not None else normalizer.new_zeros(flat.shape[0]).index_copy(0, causal, normalizer)
    return output, norms
