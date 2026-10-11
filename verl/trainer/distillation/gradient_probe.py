"""Bounded, opt-in attribution on one jointly eligible microbatch."""
import math
import torch


def values(tensor):
    return tensor.values().reshape(-1) if tensor.is_nested else tensor.reshape(-1)


def cpu(tensor):
    tensor = tensor.detach()
    if hasattr(tensor, 'to_local'):
        tensor = tensor.to_local()
    return tensor.to(device='cpu', dtype=torch.float32)


def probe(module, output, data):
    high = (values(output['action_rows']) > 0) & (values(output['action_legal_mass']) >= .9)
    repair = values(output['manga_repair_rows']) > 0
    metrics = {'high_rows': float(high.sum()), 'repair_rows': float(repair.sum())}
    if not high.any() or not repair.any():
        metrics['skipped_no_joint_support'] = 1.
        return None, metrics
    parameters = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    names, params = zip(*parameters)
    snapshot = [cpu(p).clone() for p in params]
    def gradient(loss):
        return [None if g is None else cpu(g) for g in
                torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)]
    def dot(a, b):
        return sum(float((x*y).sum()) for x, y in zip(a,b) if x is not None and y is not None)
    behavior = values(output['manga_repair_objective']).sum() * data['dp_size'] / max(float(data['manga_behavior_batch_count']), 1.)
    # Match the actual failure-only objective; teaching cohorts remain monitors.
    parts = {'repair': behavior}
    total = behavior
    metrics['loss_split_error'] = float((sum(parts.values()) - total).detach().abs())
    cohorts = {'high': (high, 'manga_action_correctness'), 'repair': (repair, 'manga_repair_nll')}
    references, before = {}, {}
    for label, (mask, field) in cohorts.items():
        objective = values(output[field])[mask].mean()
        references[label] = gradient(objective)
        before[label] = values(output[field]).detach().cpu()[mask.cpu()].clone()
        metrics[label + '_before_nll'] = float(objective.detach())
    # One total VJP checks the decomposition; subtract each component on CPU
    # instead of retaining five full gradient copies.
    residual = gradient(total)
    residual = [None if g is None else g.clone() for g in residual]
    total_norm = math.sqrt(dot(residual, residual))
    for label, loss in parts.items():
        grad = gradient(loss)
        norm = math.sqrt(dot(grad, grad))
        metrics[label + '_norm'] = norm
        metrics[label + '_loss'] = float(loss.detach())
        for cohort, ref in references.items():
            product = dot(ref, grad)
            metrics[cohort + '/' + label + '_dot'] = product
            metrics[cohort + '/' + label + '_cosine'] = product / max(math.sqrt(dot(ref,ref))*norm, 1e-30)
        for i, g in enumerate(grad):
            if g is not None:
                if residual[i] is None:
                    residual[i] = -g.clone()
                else:
                    residual[i].sub_(g)
        del grad
    metrics['gradient_split_relative_error'] = math.sqrt(dot(residual,residual)) / max(total_norm, 1e-30)
    metrics['microbatch_only'] = 1.
    return (module, names, snapshot, references,
            {k:(v[0].cpu(),v[1]) for k,v in cohorts.items()}, before), metrics


def finish(state, output):
    module, names, snapshot, references, cohorts, before = state
    current = dict(module.named_parameters())
    dots = {k:0. for k in references}
    for i, (name, old) in enumerate(zip(names,snapshot)):
        new = cpu(current[name])
        if new.shape != old.shape:
            raise RuntimeError(f'Gradient probe parameter layout changed: {name}')
        delta = new - old
        for label, ref in references.items():
            g = ref[i]
            if g is not None:
                if g.shape != delta.shape:
                    raise RuntimeError(f'Gradient probe parameter layout changed: {name}')
                dots[label] += float((g*delta).sum())
    result = {}
    for label, (mask,field) in cohorts.items():
        after = values(output[field]).detach().cpu()[mask]
        change = after - before[label]
        result.update({label+'/actual_update_dot':dots[label],
                       label+'/after_nll':float(after.mean()), label+'/nll_delta':float(change.mean()),
                       label+'/before_mass':float(before[label].neg().exp().mean()),
                       label+'/after_mass':float(after.neg().exp().mean()),
                       label+'/improved_fraction':float((change < -1e-6).float().mean()),
                       label+'/worsened_fraction':float((change > 1e-6).float().mean())})
    return result
