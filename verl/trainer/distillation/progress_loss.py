"""Direct on-policy progress objective and bounded-cost gradient diagnostics."""
import torch


def add_progress_loss(config, output, data, kl_loss, metrics):
    from verl.trainer.distillation.losses import _response_mask_to_causal_input_mask
    from verl.trainer.manga.progress import STATS
    from verl.utils.metric import Metric, AggregationType
    from verl.utils import tensordict_utils as tu
    groups = data['manga_progress_bins']
    mask = torch.nested.nested_tensor_from_jagged(groups.values() >= 0, groups.offsets())
    causal = _response_mask_to_causal_input_mask(mask, data['input_ids']).values()
    advantage = data['manga_progress_advantages'].values()[mask.values()].detach()
    log_probs = output['manga_progress_log_probs'].values()[causal]
    weights = data['manga_progress_weights'].values()[mask.values()].detach()
    scale = float(data['dp_size']) / max(float(data['manga_progress_command_count']), 1.)
    policy = -(advantage * log_probs * weights).sum() * scale
    weighted = config.manga_progress_weight * policy
    for name, value in (('loss', policy), ('weighted_loss', weighted), ('kl_loss', kl_loss),
                        ('advantage_sum', advantage.sum()), ('advantage_abs_sum', advantage.abs().sum()),
                        ('positive_tokens', (advantage > 0).sum()), ('negative_tokens', (advantage < 0).sum()),
                        ('tokens', mask.values().sum())):
        metrics['manga_progress/' + name] = Metric(value=value.detach(), aggregation=AggregationType.SUM)
    metrics['manga_progress/weight'] = Metric(value=config.manga_progress_weight, aggregation=AggregationType.MEAN)
    stats = data['manga_progress_stats']
    stats = (stats.values() if stats.is_nested else stats).reshape(-1, len(STATS)).sum(0)
    for i, name in enumerate(STATS):
        metrics['manga_progress/' + name] = Metric(value=stats[i], aggregation=AggregationType.SUM)
    if tu.get_non_tensor_data(data, 'manga_progress_probe', False):
        output['_progress_objectives'] = (kl_loss, weighted)
    return kl_loss + weighted


def logit_gradient_probe(objectives, logits):
    """Two loss VJPs ending at shared logits, never two transformer backwards.

    This is a proxy, not the parameter-gradient ratio/cosine. No coefficient
    adaptation is driven by it. Chunk reductions avoid extra full FP32 copies.
    """
    gradients = [torch.autograd.grad(loss, logits, retain_graph=True)[0] for loss in objectives]
    a, b = (g.reshape(-1) for g in gradients)
    moments = logits.new_zeros(3, dtype=torch.float32)
    for start in range(0, a.numel(), 1 << 20):
        x, y = a[start:start+(1 << 20)].float(), b[start:start+(1 << 20)].float()
        moments += torch.stack((x.square().sum(), y.square().sum(), (x*y).sum()))
    ka, pb = moments[:2].sqrt()
    return dict(kl_logit_grad_norm=ka, weighted_progress_logit_grad_norm=pb,
                logit_grad_ratio=pb/ka.clamp_min(1e-20),
                logit_grad_cosine=moments[2]/(ka*pb).clamp_min(1e-20),
                logit_grad_cosine_defined=((ka > 0) & (pb > 0)).float())
