"""Frozen SFT top-k + tail Forward KL, with bounded vocabulary memory."""
import torch
from types import MethodType
from torch.utils.checkpoint import checkpoint


def _head_forward(self, hidden, *, opsd_indices=None, opsd_preservation=None):
    if opsd_preservation is None:
        return self._opsd_original_forward(hidden)
    losses, mass = opsd_preservation(self._opsd_original_forward, hidden)
    logits = self._opsd_original_forward(hidden.index_select(1, opsd_indices))
    return logits, losses, mass


def preservation_head_forward(head, hidden, indices, preservation):
    """One Module.__call__ per rank, even with differing projection chunk counts.

    Chunk recomputation uses the original forward inside the enclosing head's
    FSDP hooks. Calling the FSDP-wrapped head once per chunk would introduce
    mismatched collectives across ranks with different numbers of normal tokens.
    """
    if not hasattr(head, '_opsd_original_forward'):
        head._opsd_original_forward = head.forward
        head.forward = MethodType(_head_forward, head)
    return head(hidden, opsd_indices=indices, opsd_preservation=preservation)


def coarse_forward_kl(logits, ids, logs):
    """Keep omitted probability as a full-vocabulary tail, never renormalize top-k."""
    x = logits.float()
    logp = x.gather(-1, ids.long()) - x.logsumexp(-1, keepdim=True)
    q = logs.float().exp()
    full = q.shape[-1] == logits.shape[-1]
    q = q / (q.sum(-1, keepdim=True) if full else q.sum(-1, keepdim=True).clamp_min(1.))
    observed = q > 0
    mass = q.sum(-1)
    terms = (q * torch.where(observed, q.clamp_min(1e-30).log() - logp, 0.)).sum(-1)
    qtail = torch.zeros_like(mass) if full else (1. - mass).clamp_min(0.)
    ptail = (-logp.masked_fill(~observed, -torch.inf).logsumexp(-1).clamp_max(0.).expm1()).clamp_min(1e-30)
    kl = terms + qtail * (qtail.clamp_min(1e-30).log() - ptail.log())
    return kl, mass.detach()


def projected_preservation(head, hidden, ids, logs, temperature, chunk_size=128):
    """Checkpoint only projection/loss, not the transformer; retain O(rows*k) targets."""
    losses, masses = [], []
    def evaluate(x, target_ids, target_logs, temp):
        logits = head(x) / temp.clamp_min(1e-8).to(x.dtype)[:, None]
        return coarse_forward_kl(logits, target_ids, target_logs)
    for start in range(0, hidden.shape[0], chunk_size):
        stop = start + chunk_size
        args = (hidden[start:stop], ids[start:stop], logs[start:stop], temperature[start:stop])
        loss, mass = (checkpoint(evaluate, *args, use_reentrant=False)
                      if torch.is_grad_enabled() else evaluate(*args))
        losses.append(loss)
        masses.append(mass)
    if not losses:
        zero = hidden.float().sum(-1) * 0.
        return zero, zero.detach()
    return torch.cat(losses), torch.cat(masses)
