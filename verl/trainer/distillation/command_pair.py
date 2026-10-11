"""Direct command-pair DPO on atomic, differentiable shared-history rows."""
import math
import os
import torch
import torch.nn.functional as F


def direct_loss(logp_rows, id_rows, ref_rows):
    beta = float(os.environ.get('MANGA_COMMAND_PAIR_BETA', '0.1'))
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('Command pair beta must be positive and finite')
    scores, references = [], []
    for logp, ids, ref in zip(logp_rows, id_rows, ref_rows, strict=True):
        if not bool((ids != 0).any()):
            continue
        # Diagnostic callers retain the CPU batch while forward_step moves its
        # local copy to the accelerator. Align labels without detaching scores.
        ids = ids.to(device=logp.device)
        ref = ref.to(device=logp.device)
        active = ids != 0
        labels = ids[active].unique()
        if labels.numel() != 2 or not bool(labels[0] == -labels[1]):
            raise ValueError('Both sides of exactly one pair must occupy the same tree row')
        # model outputs are at causal INPUT positions; IDs label RESPONSE tokens.
        start = logp.numel() - ids.numel() - 1
        if start < 0 or ref.shape != ids.shape:
            raise ValueError('Invalid command pair response alignment')
        causal = logp[start:start + ids.numel()].float()
        scores.append(torch.stack([causal[ids > 0].sum(), causal[ids < 0].sum()]))
        references.append(torch.stack([ref[ids > 0].sum(), ref[ids < 0].sum()]).float())
    if not scores:
        return None
    scores, references = torch.stack(scores), torch.stack(references).detach()
    if not bool(torch.isfinite(scores).all() & torch.isfinite(references).all()):
        raise ValueError('Nonfinite command sequence log probability')
    relative = scores - references
    margin = beta * (relative[:, 0] - relative[:, 1])
    return F.softplus(-margin), margin, scores
