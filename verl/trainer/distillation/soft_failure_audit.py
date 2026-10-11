"""One-update, fixed-prefix audit. Never changes targets, loss, or gradients."""
import torch


def row_snapshot(logits, ids, teacher_logs, viable, sampled, clip):
    """Small CPU record; viable mass covers retained support, not the vocabulary."""
    logq = logits.detach().float().log_softmax(-1)
    q = logq.exp()
    p = teacher_logs.detach().float().exp()
    valid = teacher_logs > -1e8
    viable = viable & valid
    selected_q = q[ids]
    terms = p * (teacher_logs.float() - logq[ids])
    omitted_logs = logq.scatter(0, ids, -torch.inf)
    log_tail = omitted_logs.logsumexp(0)
    teacher_tail = (1 - p.sum()).clamp(0, 1)
    tail_term = (teacher_tail * (teacher_tail.clamp_min(torch.finfo(torch.float32).eps).log() - log_tail)
                 if teacher_tail > 0 and ids.numel() < logits.numel() else teacher_tail.new_zeros(()))
    clipped = terms > clip if clip is not None else torch.zeros_like(viable)
    tail_clipped = bool(clip is not None and tail_term > clip)
    effective = p.masked_fill(clipped, 0)
    effective_tail = teacher_tail * (not tail_clipped)
    # Exact local derivative of coarse KL w.r.t. normalized-temperature logits.
    # Clipped direct terms can still move through softmax normalization.
    grad_raw = selected_q * (p.sum() + teacher_tail) - p
    grad_clip = selected_q * (effective.sum() + effective_tail) - effective
    sampled_matches = (ids == sampled) & valid
    return dict(candidate_ids=ids.tolist(), viable=viable.tolist(), teacher_p=p.tolist(),
                student_p=selected_q.tolist(), sampled_token=int(sampled),
                sampled_p=float(q[sampled]),
                teacher_sampled_p=float(p[sampled_matches].sum()) if sampled_matches.any() else None,
                teacher_tail_p=float(teacher_tail), student_tail_p=float(log_tail.exp()),
                teacher_viable_mass=float(p[viable].sum()), student_viable_mass=float(selected_q[viable].sum()),
                pointwise_kl=terms.tolist(), clipped=clipped.tolist(), tail_clipped=tail_clipped,
                viable_clipped_teacher_mass=float(p[viable & clipped].sum()),
                unclipped_local_logit_grad=grad_raw.tolist(), clipped_local_logit_grad=grad_clip.tolist())


def capture(data, logits, predictors, failure_positions, clip):
    positions = failure_positions.nonzero(as_tuple=True)[0]
    local = torch.searchsorted(predictors, positions)
    if bool((local >= predictors.numel()).any()) or not torch.equal(predictors[local], positions):
        raise ValueError('Failure predictors must be supervised in the real forward')
    ids = data['teacher_ids'].values()[positions]
    logs = data['teacher_logprobs'].values()[positions]
    viable = data['manga_failure_candidates'].values()[positions]
    inputs = data['input_ids'].values()
    offsets = data['input_ids'].offsets().tolist()
    records = []
    for n, position in enumerate(positions.tolist()):
        row = row_snapshot(logits[0, local[n]], ids[n], logs[n], viable[n], inputs[position + 1], clip)
        sample = max(i for i, offset in enumerate(offsets[:-1]) if offset <= position)
        start = offsets[sample]
        row['prefix_ids'] = inputs[start:position + 1].tolist()
        row['predictor_position'] = position
        row['microbatch_sample'] = sample
        if 'index' in data:
            index = data['index'][sample]
            row['dataset_index'] = index.item() if torch.is_tensor(index) else getattr(index, 'data', index)
        records.append(row)
    if logits.requires_grad:
        sampled = inputs[positions + 1]
        def record_gradient(grad):
            # Observe normal backward, rather than run a second backward/VJP.
            selected = grad.detach()[0, local[:, None], ids].float().cpu().tolist()
            failed = grad.detach()[0, local, sampled].float().cpu().tolist()
            for row, values, value in zip(records, selected, failed, strict=True):
                row['actual_backward_candidate_grad'] = values
                row['actual_backward_sampled_grad'] = value
            return None
        logits.register_hook(record_gradient)
    return records


def compare(before, after):
    if len(before) != len(after) or not before:
        raise ValueError('Fixed failure audit needs identical nonempty row sets')
    for a, b in zip(before, after, strict=True):
        if 'actual_backward_candidate_grad' not in a:
            raise RuntimeError('Failure audit did not observe the normal training backward')
        for key in ('prefix_ids', 'candidate_ids', 'teacher_p', 'viable', 'sampled_token', 'predictor_position'):
            if a[key] != b[key]:
                raise ValueError('Fixed failure audit changed ' + key)
    mean = lambda values: sum(values) / len(before)
    return dict(count=len(before),
        sampled_p_before=mean([x['sampled_p'] for x in before]),
        sampled_p_after=mean([x['sampled_p'] for x in after]),
        sampled_p_delta=mean([b['sampled_p']-a['sampled_p'] for a,b in zip(before,after)]),
        sampled_p_decreased_fraction=mean([b['sampled_p'] < a['sampled_p'] for a,b in zip(before,after)]),
        retained_viable_mass_before=mean([x['student_viable_mass'] for x in before]),
        retained_viable_mass_after=mean([x['student_viable_mass'] for x in after]),
        retained_viable_mass_delta=mean([b['student_viable_mass']-a['student_viable_mass'] for a,b in zip(before,after)]),
        teacher_retained_viable_mass=mean([x['teacher_viable_mass'] for x in before]),
        viable_clipped_teacher_mass=mean([x['viable_clipped_teacher_mass'] for x in before]),
        teacher_sampled_in_support_fraction=mean([x['teacher_sampled_p'] is not None for x in before]))
