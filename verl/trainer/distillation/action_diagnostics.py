"""Action legality loss and detached diagnostics sharing the same student logits."""
import torch


def action_diagnostics(data, logits, student_positions=None, train_legality=False, log_normalizers=None):
    inputs = data["input_ids"]
    fields = [data[f"manga_action_ids_{i}"] for i in range(5)]
    if not inputs.is_nested or not all(field.is_nested for field in fields):
        raise ValueError("Action diagnostics require packed jagged inputs")
    supports = torch.stack([field.values() for field in fields], dim=-1).long()
    response_offsets = fields[0].offsets()
    input_offsets = inputs.offsets()
    response_lengths = response_offsets.diff()
    shifts = input_offsets[:-1] + input_offsets.diff() - response_lengths - 1 - response_offsets[:-1]
    causal = torch.arange(supports.shape[0], device=supports.device) + torch.repeat_interleave(shifts, response_lengths)
    selected = supports[:, 0] >= 0
    causal, supports = causal[selected], supports[selected]
    names = ("rows", "legal_mass", "legality_nll", "preference_kl", "total_kl", "teacher_support_mass", "valid_rows")
    result = {"action_" + name: logits.new_zeros((1, int(input_offsets[-1])), dtype=torch.float32) for name in names}
    if not causal.numel():
        if train_legality:
            result["action_illegal_loss"] = result["action_rows"] + logits.reshape(-1)[:0].sum()
        return result
    flat = logits.reshape(-1, logits.shape[-1])
    if student_positions is not None:
        indices = torch.searchsorted(student_positions, causal)
        if bool((indices >= student_positions.numel()).any()) or not torch.equal(student_positions[indices], causal):
            raise ValueError("Action predictors missing from sparse student projections")
    else:
        indices = causal
    valid = supports >= 0
    safe_ids = supports.clamp_min(0)
    if log_normalizers is None:
        rows = flat.index_select(0, indices).float()
        if not train_legality:
            rows = rows.detach()
        logp = rows.gather(-1, safe_ids) - rows.logsumexp(-1, keepdim=True)
    else:
        # Correctness loss already normalized these projections. Gather only
        # five action scores rather than copying/reducing another [actions,V].
        logp = (flat[indices[:, None], safe_ids].float().detach()
                - log_normalizers.index_select(0, indices).detach()[:, None])
    logp = logp.masked_fill(~valid, -torch.inf)
    logz = logp.logsumexp(-1)
    if train_legality:
        illegal = logits.new_zeros((1, int(input_offsets[-1])), dtype=torch.float32)
        illegal[0, causal] = -logz.clamp_max(0.).expm1()
        result["action_illegal_loss"] = illegal
    logz, logp = logz.detach(), logp.detach()
    teacher_ids = data["teacher_ids"].values().index_select(0, causal)
    teacher_logs = data["teacher_logprobs"].values().index_select(0, causal).float()
    match = (teacher_ids[:, :, None] == safe_ids[:, None, :]) & valid[:, None, :]
    q = (teacher_logs.exp()[:, :, None] * match).sum(1)
    mass = q.sum(-1)
    # Only call this an exact decomposition when the sparse teacher contains
    # the complete action distribution. Report coverage for every action row.
    exact = (mass - 1).abs() <= 1e-5
    q = q / mass.clamp_min(1e-30)[:, None]
    logconditional = (logp - logz[:, None]).masked_fill(~valid, 0.)
    preference = (q * (q.clamp_min(1e-30).log() - logconditional)).sum(-1)
    values = dict(rows=torch.ones_like(logz), legal_mass=logz.exp(), legality_nll=-logz,
                  preference_kl=torch.where(exact, preference, 0.),
                  total_kl=torch.where(exact, -logz + preference, 0.),
                  teacher_support_mass=mass, valid_rows=exact.float())
    for name, value in values.items():
        result["action_" + name][0, causal] = value
    return result


def add_action_diagnostic_rates(metrics):
    """Run after worker aggregation; avoid averaging unequal microbatch means."""
    add_paired_action_rates(metrics)
    candidates = 'actor/manga_candidates/'
    count = metrics.get(candidates + 'rows_sum', 0.)
    if candidates + 'rows_sum' in metrics:
        for output, source in (('unchecked_mass', 'unchecked_sum'),
                               ('teacher_observed_legal_mass', 'teacher_observed_legal_mass_sum'),
                               ('checked_tokens_per_position', 'checked_tokens_sum')):
            metrics[candidates + output] = metrics[candidates + source] / count if count else 0.
    prefix = "actor/manga_action_pre_update/"
    if prefix + "rows" not in metrics:
        return
    for name in ("legal_mass", "legality_nll", "preference_kl", "total_kl", "teacher_support_mass"):
        count = metrics[prefix + ("valid_rows" if name in ("preference_kl", "total_kl") else "rows")]
        metrics[prefix + name] = metrics[prefix + name + "_sum"] / count if count else 0.
    metrics[prefix + "illegal_mass"] = 1. - metrics[prefix + "legal_mass"] if metrics[prefix + "rows"] else 0.


@torch.no_grad()
def paired_action_statistics(before, after, tolerance=1e-5):
    """Paired values on identical causal rows; replay does not enter the loss."""
    after = {name: value.values().detach().cpu() if value.is_nested else value.detach().cpu()
             for name, value in after.items() if name.startswith("action_")}
    if not torch.equal(before["action_rows"], after["action_rows"]):
        raise ValueError("Paired action diagnostic positions changed during replay")
    selected = before["action_rows"] > 0
    pre = before["action_legal_mass"][selected]
    post = after["action_legal_mass"][selected]
    delta = post - pre
    up, down = delta > tolerance, delta < -tolerance
    exact = selected & (before["action_valid_rows"] > 0) & (after["action_valid_rows"] > 0)
    stats = dict(rows=selected.float().sum(), increased=up.float().sum(), decreased=down.float().sum(),
                 before_mass_sum=pre.sum(), after_mass_sum=post.sum(),
                 gain_sum=delta[up].sum(), drop_sum=-delta[down].sum(),
                 before_nll_sum=before["action_legality_nll"][selected].sum(),
                 after_nll_sum=after["action_legality_nll"][selected].sum(),
                 valid_rows=exact.float().sum(),
                 before_preference_sum=before["action_preference_kl"][exact].sum(),
                 after_preference_sum=after["action_preference_kl"][exact].sum())
    for label, mask in (("low", pre < .5), ("high", pre >= .9)):
        stats[label + "_rows"] = mask.float().sum()
        stats[label + "_delta_sum"] = delta[mask].sum()
    return stats


def add_paired_action_rates(metrics):
    prefix = "actor/manga_action_paired/"
    if prefix + "rows" not in metrics:
        return
    pairs = [("increase_fraction", "increased", "rows"), ("decrease_fraction", "decreased", "rows"),
             ("mean_gain", "gain_sum", "increased"), ("mean_drop", "drop_sum", "decreased")]
    pairs += [(side + "_" + name, side + "_" + name + "_sum", "rows")
              for side in ("before", "after") for name in ("mass", "nll")]
    pairs += [(side + "_preference", side + "_preference_sum", "valid_rows") for side in ("before", "after")]
    pairs += [(group + "_delta", group + "_delta_sum", group + "_rows") for group in ("low", "high")]
    for output, numerator, denominator in pairs:
        count = metrics[prefix + denominator]
        metrics[prefix + output] = metrics[prefix + numerator] / count if count else 0.
    metrics[prefix + "mass_delta"] = metrics[prefix + "after_mass"] - metrics[prefix + "before_mass"]
