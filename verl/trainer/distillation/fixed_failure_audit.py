"""Fixed original failure cohort; detached, count-weighted diagnostics only."""
import torch


def snapshot(output):
    def flat(name):
        value = output[name]
        return value.values() if value.is_nested else value.reshape(-1)
    mask = flat('manga_repair_rows') > 0
    return flat('manga_repair_raw_nll')[mask].detach().float().cpu().clone()


def compare(baseline, previous, current):
    if baseline.shape != current.shape or previous.shape != current.shape:
        raise RuntimeError('Fixed failure cohort changed shape')
    delta = current - baseline
    return dict(count=float(current.numel()), baseline_nll=float(baseline.sum()),
                current_nll=float(current.sum()), nll_delta=float(delta.sum()),
                step_nll_delta=float((current-previous).sum()),
                baseline_mass=float(baseline.neg().exp().sum()),
                current_mass=float(current.neg().exp().sum()),
                improved_fraction=float((delta < -1e-6).sum()),
                worsened_fraction=float((delta > 1e-6).sum()))
