"""Bounded graph-progress rewards; no teacher calls or extra rollouts."""
import math

import torch

BINS = 10
STATS = ('episodes', 'required_units', 'completed_units', 'final_fraction', 'return', 'commands')


def required_units(executor):
    target = executor.target
    # A cluster of size n needs n-1 successful component merges, not n(n-1)/2 edges.
    merges = len(target.characters) - len(set(target.clusters))
    return (len(target.panels) + len(target.characters) + 2 * len(target.texts)
            + len(target.speaker_links) + len(executor.grounding_by_panel) + merges)


def completed_units(executor):
    state = executor.state
    merges = len(state.identities.parent) - sum(k == v for k, v in state.identities.parent.items())
    return (state.entered + len(state.detected_characters) + len(state.detected_texts)
            + len(state.reads) + len(state.speakers) + len(state.groundings) + merges)


def potential(progress, alpha):
    if not 0 < alpha < 1 or not 0 <= progress <= 1:
        raise ValueError('Progress requires 0 < alpha < 1 and 0 <= progress <= 1')
    return math.log1p(-alpha * progress) / math.log1p(-alpha)


def rollout_fields(records, length, total, alpha):
    """Records are (start, end, completed_before, completed_after), including failure.

    With gamma=1 the reward-to-go telescopes. Terminal failure has zero future
    reward; it remains a policy action even where teacher supervision stops.
    """
    returns = torch.zeros(length)
    bins = torch.full((length,), -1, dtype=torch.long)
    starts = torch.zeros(length, dtype=torch.bool)
    weights = torch.zeros(length)
    final = records[-1][3] if records else 0
    denominator = max(total, 1)
    end_value = potential(final / denominator, alpha)
    for start, end, before, after in records:
        if not 0 <= before <= after <= total or not 0 <= start < end <= length:
            raise ValueError('Invalid monotone progress record')
        if before == total:
            continue  # No progress remains: let soft KL teach termination.
        p = before / denominator
        returns[start:end] = end_value - potential(p, alpha)
        bins[start:end] = min(int(p * BINS), BINS-1)
        starts[start] = True
        weights[start:end] = 1. / (end-start)
    return dict(manga_progress_returns=returns, manga_progress_bins=bins,
                manga_progress_starts=starts, manga_progress_weights=weights,
                manga_progress_stats=torch.tensor([1, total, final, final/denominator,
                                                   end_value, int(starts.sum())], dtype=torch.float32))


def baseline_tables(returns, bins, starts):
    """Per-trajectory command sums/counts. Last column is the all-progress fallback."""
    tables = []
    for ret, group, begin in zip(returns.unbind(), bins.unbind(), starts.unbind(), strict=True):
        valid = begin.bool() & (group >= 0)
        ids, values = group[valid].long(), ret[valid].float()
        table = values.new_zeros((2, BINS+1))
        table[0, :BINS].scatter_add_(0, ids, values)
        table[1, :BINS].scatter_add_(0, ids, torch.ones_like(values))
        table[:, BINS] = table[:, :BINS].sum(-1)
        tables.append(table)
    return torch.stack(tables)


def advantages(returns, bins, tables, global_table):
    """Leave the entire trajectory out of its baseline, before microbatch splitting.

    Prefer peers at similar progress. If a bin has no peers, use other
    trajectories' command returns; without any peers use zero. No whitening.
    """
    result = []
    for ret, group, own in zip(returns.unbind(), bins.unbind(), tables, strict=True):
        other = global_table - own
        counts = other[1].clamp_min(0)
        means = other[0] / counts.clamp_min(1)
        baseline = torch.where(counts[:BINS] > 0, means[:BINS], means[BINS])
        result.append(torch.where(group >= 0, ret-baseline[group.clamp_min(0).long()], 0.).detach())
    return torch.nested.as_nested_tensor(result, layout=torch.jagged)
