"""Same-token before/after diagnostics; never modify the optimizer or targets."""
import torch
from .command_pair import direct_loss


def gradient_check(module, output, data):
    """Bounded last-norm subspace diagnostic, not full-model attribution."""
    candidates = [(n,p) for n,p in module.named_parameters()
                  if p.requires_grad and 'norm' in n and p.numel() <= 65536]
    if not candidates:
        return {'gradient_subspace_available':0.}
    rest = sum(output['manga_group_'+g].values().sum()
               / max(float(data['manga_'+g+'_group_count']),1.) * data['dp_size']
               for g in ('action','object','content'))
    if not bool(data['manga_group_weights'].values().sum() > 0):
        return {'gradient_subspace_available':0., 'gradient_no_joint_opsd':1.}
    result = direct_loss(output['manga_pair_logp'].unbind(),
                         data['manga_pair_id'].unbind(),data['manga_pair_ref'].unbind())
    values, _, scores = result
    pair = values.sum()/float(data['manga_pair_batch_count'])*data['dp_size']
    parameter = candidates[-1][1]
    def grad(loss):
        g, = torch.autograd.grad(loss, (parameter,), retain_graph=True, allow_unused=True)
        if g is None:
            return None
        if hasattr(g,'to_local'):
            g = g.to_local()
        return g.detach().float().reshape(-1)
    a,b,f = grad(pair),grad(rest),grad(scores[:,1].mean())
    if any(g is None for g in (a,b,f)):
        return {'gradient_subspace_available':0.}
    return {'gradient_subspace_available':1., 'gradient_subspace_numel':a.numel(),
            'gradient_pair_opsd_cosine':float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)),
            'gradient_rejected_dot_pair':float(torch.dot(f,a)),
            'gradient_rejected_dot_opsd':float(torch.dot(f,b))}


def snapshot(output, data):
    result = direct_loss(output['manga_pair_logp'].unbind(),
                         data['manga_pair_id'].unbind(), data['manga_pair_ref'].unbind())
    if result is None:
        raise ValueError('Pair audit requires at least one complete pair')
    losses, margins, scores = result
    return tuple(x.detach().float().cpu().clone() for x in (losses, margins, scores))


def compare(before, after):
    loss0, margin0, score0 = before
    loss1, margin1, score1 = after
    if score0.shape != score1.shape:
        raise ValueError('Pair audit changed the selected commands')
    delta = score1-score0
    better = margin1 > margin0 + 1e-6
    metrics = {'pairs':len(delta), 'loss_before':loss0.mean().item(),
               'loss_after':loss1.mean().item(), 'margin_delta':(margin1-margin0).mean().item(),
               'margin_improved_fraction':better.float().mean().item(),
               'margin_up_rejected_not_down_fraction':
                   (better & (delta[:,1] >= -1e-6)).float().mean().item()}
    for side,name in enumerate(('chosen','rejected')):
        d=delta[:,side]
        metrics.update({name+'_logp_before':score0[:,side].mean().item(),
                        name+'_logp_after':score1[:,side].mean().item(),
                        name+'_logp_delta':d.mean().item(),
                        name+'_increased_fraction':(d>1e-6).float().mean().item(),
                        name+'_decreased_fraction':(d < -1e-6).float().mean().item()})
    return metrics
