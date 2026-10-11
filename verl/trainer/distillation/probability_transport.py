"""Redistribute certified illegal mass; retain legal mass and an unknown tail.

Targets come from the first actor forward, never the sampling-filtered policy.
Only sparse edge probabilities and one tail per row are needed. The optional
cache freezes targets across diagnostic replays, not across optimizer steps.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib

import torch

_cache = ContextVar('manga_transport_targets', default=None)


@contextmanager
def fixed_targets():
    token = _cache.set({})
    try:
        yield
    finally:
        _cache.reset(token)


def transport_kl(logits, edge_rows, edge_ids, legal, teacher, exact, target=None):
    from verl.trainer.distillation.correctness_loss import _ChunkedNormalizer
    norm = _ChunkedNormalizer.apply(logits)
    logp = logits[edge_rows, edge_ids].float() - norm[edge_rows]
    p = logp.exp()
    n = logits.shape[0]
    good = norm.new_zeros(n).scatter_add(0, edge_rows, p * legal)
    checked_bad = norm.new_zeros(n).scatter_add(0, edge_rows, p * ~legal)
    bad = torch.where(exact, (1. - good).clamp_min(0.), checked_bad)
    unknown = torch.where(exact, torch.zeros_like(good), (1. - good - checked_bad).clamp_min(0.))
    if target is None:
        q = torch.where(legal, p.detach() + bad.detach()[edge_rows] * teacher, 0.)
        tail = unknown.detach()
        target = (q, tail)
    q, tail = (x.to(device=logits.device, dtype=torch.float32) for x in target)
    terms = q * (q.clamp_min(1e-30).log() - logp)
    loss = norm.new_zeros(n).scatter_add(0, edge_rows, terms)
    # Unknown tokens are a single event, not implicit negative labels.
    loss = loss + tail * (tail.clamp_min(1e-30).log() - unknown.clamp_min(1e-30).log())
    return loss, bad.detach(), unknown.detach(), norm, (q.detach(), tail.detach())


def loss_outputs(data, student_logits, token_mask, student_positions=None):
    selected = token_mask.values().bool()
    causal = selected.nonzero(as_tuple=True)[0]
    flat = student_logits.reshape(-1, student_logits.shape[-1])
    if student_positions is None:
        logits = flat.index_select(0, causal)
    else:
        if not torch.equal(student_positions, causal):
            raise ValueError('Transport predictors differ from selected projection')
        logits = flat
    offsets = data['input_ids'].offsets()
    ro = data['manga_opsd_mask'].offsets()
    shifts = offsets[:-1] + offsets.diff() - ro.diff() - 1
    response_rows = (torch.arange(int(ro[-1]), device=logits.device)
                     + torch.repeat_interleave(shifts - ro[:-1], ro.diff()))

    def field(name):
        # Metadata and fractional command weights stay FP32 even when the
        # actor projects BF16/FP16 logits. Index assignment does not cast.
        full = logits.new_zeros(selected.numel(), dtype=torch.float32)
        full[response_rows] = data[name].values().float()
        return full[causal]

    edges = data['manga_transport_edges']
    e = edges.values()
    edge_causal = e[:, 0].long() + torch.repeat_interleave(shifts, edges.offsets().diff())
    er = torch.searchsorted(causal, edge_causal)
    ei, legal, teacher = e[:, 1].long(), e[:, 2].bool(), e[:, 3].float()
    exact = field('manga_transport_exact').bool()
    cache, key, target = _cache.get(), None, None
    if cache is not None:
        digest = hashlib.sha256()
        for tensor in (data['input_ids'].values(), offsets, e, exact):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        key = digest.hexdigest()
        target = cache.get(key)
    loss, bad, unknown, normalizer, target = transport_kl(logits, er, ei, legal, teacher, exact, target)
    if cache is not None and key not in cache:
        cache[key] = tuple(x.cpu() for x in target)
    weight = field('manga_transport_weights')
    repair = field('manga_repair_mask').bool()
    group = field('manga_transport_group')
    nll = -(1. - bad).clamp_min(1e-30).log()
    zero = loss * 0.

    def scatter(x):
        return x.new_zeros((1, selected.numel())).index_copy(1, causal, x.unsqueeze(0))

    out = {name: scatter(x) for name, x in dict(
        distillation_losses=loss, raw_distillation_losses=loss.detach(),
        student_mass=1.-bad, teacher_mass=torch.ones_like(bad),
        manga_repair_objective=loss * weight,
        manga_repair_rows=repair.float(), manga_repair_raw_nll=nll * repair,
        manga_repair_nll=nll * field('manga_repair_weights'), manga_repair_mass=(1.-bad) * repair,
        manga_single_ce=zero, manga_set_loss=nll, manga_preference_kl=zero,
        manga_transport_bad=bad, manga_transport_unknown=unknown,
        manga_transport_rows=torch.ones_like(bad), manga_transport_group=group,
        manga_transport_repaired=repair.float(), manga_transport_exact=exact.float()).items()}
    for name in ('action', 'object', 'content'):
        out['manga_group_' + name] = scatter(zero)
    for name in ('rows', 'mass', 'nll'):
        out['action_repair_' + name] = out['manga_repair_' + name]
    norms = normalizer if student_positions is not None else normalizer.new_zeros(flat.shape[0]).index_copy(0, causal, normalizer)
    return out, norms
