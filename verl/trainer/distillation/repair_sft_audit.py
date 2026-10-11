"""One-batch repair/SFT attribution through ordinary FSDP backward hooks.

Four diagnostic backwards (repair and three preservation cohorts), one REAL
joint update, then one no-grad paired replay. No counterfactual optimizer step.
"""
import hashlib
import math
import time

import torch

from verl.trainer.distillation.update_counterfactual import adamw_direction, local_float


PARTS = ('repair', 'action', 'object', 'content')
TASKS = ('boundary', 'enter', 'detect', 'read', 'link', 'ground')


def values(x):
    return x.values() if x.is_nested else x


def dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b, strict=True)
               if x is not None and y is not None)


def norm(a):
    return math.sqrt(max(0., dot(a, a)))


def combine(vectors, coefficients):
    result = []
    for tensors in zip(*vectors, strict=True):
        present = [(x, c) for x, c in zip(tensors, coefficients, strict=True) if x is not None]
        result.append(sum((x * c for x, c in present), torch.zeros_like(present[0][0])) if present else None)
    return result


def response_values(full, data):
    """Input-causal scalar outputs -> response positions, retaining autograd."""
    input_offsets = data['input_ids'].offsets()
    response_offsets = data['manga_preserve_mask'].offsets()
    sizes = response_offsets.diff()
    shifts = input_offsets[:-1] + input_offsets.diff() - sizes - 1 - response_offsets[:-1]
    rows = torch.arange(int(response_offsets[-1]), device=values(full).device)
    rows = rows + torch.repeat_interleave(shifts, sizes)
    return values(full).index_select(0, rows)


def observations(output, data):
    preserve = values(data['manga_preserve_mask']).bool()
    groups = values(data['manga_preserve_group']).long()
    if bool((preserve & ((groups < 1) | (groups > 3))).any()):
        raise RuntimeError('Missing preservation diagnostic groups: export MANGA_GRADIENT_PROBE before rollout')
    repair = values(data['manga_repair_mask']).bool()
    if bool((repair & preserve).any()):
        raise RuntimeError('Repair/preservation overlap in diagnostic batch')
    task = values(data['manga_command_types']).long()
    digest = hashlib.sha256(values(data['input_ids']).detach().cpu().numpy().tobytes()).hexdigest()
    record = {'fingerprint': digest}
    for name, tensor in dict(preserve=preserve, repair=repair, group=groups, task=task,
        kl=response_values(output['manga_preserve_kl'], data),
        retained=response_values(output['manga_preserve_mass'], data),
        nll=response_values(output['manga_repair_raw_nll'], data),
        mass=response_values(output['manga_repair_mass'], data)).items():
        record[name] = tensor.detach().cpu().clone()
    if 'manga_transport_bad' in output:
        for name in ('bad', 'unknown', 'rows', 'group', 'repaired', 'exact'):
            record['transport_' + name] = response_values(output['manga_transport_' + name], data).detach().cpu().clone()
    return record


def compare_records(before, after):
    if len(before) != len(after):
        raise RuntimeError('Diagnostic replay microbatch count changed')
    for left, right in zip(before, after, strict=True):
        if left['fingerprint'] != right['fingerprint']:
            raise RuntimeError('Diagnostic replay token prefixes changed')
        for key in ('preserve', 'repair', 'group', 'task'):
            if not torch.equal(left[key], right[key]):
                raise RuntimeError(f'Diagnostic replay changed {key}')
    result = {}
    if not before:
        raise RuntimeError('Diagnostic received no microbatches')
    a = {key: torch.cat([r[key] for r in before]) for key in before[0] if key != 'fingerprint'}
    b = {key: torch.cat([r[key] for r in after]) for key in a}
    cohorts = {'repair': (a['repair'], 'nll'), 'preserve/all': (a['preserve'], 'kl')}
    for i, group in enumerate(PARTS[1:], 1):
        cohorts['preserve/' + group] = (a['preserve'] & (a['group'] == i), 'kl')
        for task, name in enumerate(TASKS):
            cohorts[f'preserve/{group}/{name}'] = (a['preserve'] & (a['group'] == i) & (a['task'] == task), 'kl')
    for task, name in enumerate(TASKS):
        cohorts['repair/' + name] = (a['repair'] & (a['task'] == task), 'nll')
    for name, (mask, field) in cohorts.items():
        count = int(mask.sum())
        result[name + '/count'] = count
        if not count:
            continue
        x, y = a[field][mask].float(), b[field][mask].float()
        delta = y - x
        for key, value in dict(before=x.mean(), after=y.mean(), delta=delta.mean(),
                               before_p95=x.quantile(.95), after_p95=y.quantile(.95),
                               delta_p95=delta.quantile(.95), max_increase=delta.max(),
                               max_abs_change=delta.abs().max(),
                               increased_fraction=(delta > 1e-6).float().mean(),
                               decreased_fraction=(delta < -1e-6).float().mean()).items():
            result[name + '/' + key] = float(value)
        if field == 'kl':
            result[name + '/retained_mass'] = float(a['retained'][mask].mean())
            result[name + '/retained_mass_p05'] = float(a['retained'][mask].quantile(.05))
            result[name + '/before_kl_sum'] = float(x.sum())
        else:
            result[name + '/mass_before'] = float(a['mass'][mask].mean())
            result[name + '/mass_after'] = float(b['mass'][mask].mean())
    if 'transport_bad' in a:
        rows = a['transport_rows'] > 0
        subsets = {'all': rows, 'repair': rows & (a['transport_repaired'] > 0),
                   'unrepaired': rows & (a['transport_repaired'] == 0),
                   'initial_low_risk': rows & (a['transport_bad'] <= .1) & (a['transport_unknown'] <= .01)}
        for i, name in enumerate(('action', 'object', 'other'), 1):
            subsets[name] = rows & (a['transport_group'] == i)
        for name, mask in subsets.items():
            base = 'transport/' + name + '/'
            result[base + 'count'] = int(mask.sum())
            if mask.any():
                x, y = a['transport_bad'][mask], b['transport_bad'][mask]
                result[base + 'illegal_mass_before'] = float(x.mean())
                result[base + 'illegal_mass_after'] = float(y.mean())
                result[base + 'illegal_mass_delta'] = float((y-x).mean())
                result[base + 'worsened_fraction'] = float(((y-x) > 1e-6).float().mean())
                result[base + 'unknown_mass_after'] = float(b['transport_unknown'][mask].mean())
    return result


class RepairSFTAudit:
    def __init__(self, module, optimizer, weight, max_elements=250_000_000):
        self.named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        elements = sum(p.numel() for p in self.params)
        if not self.params or elements > max_elements:
            raise ValueError('Repair/SFT audit requires <=250M trainable parameters; use the existing LoRA configuration')
        if not isinstance(optimizer, torch.optim.AdamW) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Repair/SFT audit requires AdamW and a positive preservation weight')
        self.optimizer, self.weight = optimizer, weight
        self.grads, self.records, self.losses = {}, {}, {}
        self.phase = None
        self.metrics = {'available': 1., 'weight': weight, 'trainable_elements': elements}
        self.before = None
        self.joint = None

    def callback(self, original):
        def wrapped(**kwargs):
            if 'student_logits' in kwargs:
                return original(**kwargs)
            output, data = kwargs['model_output'], kwargs['data']
            full_loss, metrics = original(**kwargs)
            repair = values(output['manga_repair_objective']).sum() * float(data['dp_size']) / max(float(data['manga_behavior_batch_count']), 1.)
            raw_kl = response_values(output['manga_preserve_kl'], data)
            mask, group = values(data['manga_preserve_mask']).bool(), values(data['manga_preserve_group'])
            denominator = max(float(data['manga_preserve_batch_count']), 1.)
            action_weight = float(data['manga_preserve_action_weight']) if 'manga_preserve_action_weight' in data else 0.
            action_denominator = max(float(data['manga_preserve_action_count']), 1.) if 'manga_preserve_action_count' in data else 1.
            effective_weights = [self.weight + action_weight * denominator / action_denominator,
                                 self.weight, self.weight]
            self.effective_weights = effective_weights
            self.metrics['action_extra_weight'] = action_weight
            parts = {'repair': repair}
            for index, name in enumerate(PARTS[1:], 1):
                parts[name] = raw_kl[mask & (group == index)].sum() * effective_weights[index - 1] * float(data['dp_size']) / denominator
            mismatch = float((full_loss - sum(parts.values())).detach().abs())
            self.metrics['loss_split_max_error'] = max(self.metrics.get('loss_split_max_error', 0.), mismatch)
            if mismatch > 1e-5 * max(1., abs(float(full_loss.detach()))):
                raise RuntimeError('Audit split does not match the actual training objective')
            totals = self.losses.setdefault(self.phase, {name: 0. for name in PARTS})
            for name, loss in parts.items():
                totals[name] += float(loss.detach())
            if self.phase in ('repair', 'joint', 'after'):
                self.records.setdefault(self.phase, []).append(observations(output, data))
            return (parts[self.phase] if self.phase in PARTS else full_loss), metrics
        return wrapped

    def capture(self, name, scale=1.):
        self.grads[name] = [None if p.grad is None else local_float(p.grad).clone() / scale for p in self.params]

    def before_step(self, scale, clip_norm):
        self.capture('joint', scale)
        r = self.grads['repair']
        u = combine([self.grads[g] for g in PARTS[1:]], [1., 1., 1.])
        joint = self.grads.pop('joint')
        self.joint = joint
        expected = combine([r, u], [1., 1.])
        error = norm(combine([joint, expected], [1., -1.])) / max(norm(joint), 1e-30)
        self.metrics['gradient_split_relative_error'] = error
        del expected
        self.metrics['repair_norm'] = norm(r)
        self.metrics['preserve_weighted_norm'] = norm(u)
        self.metrics['preserve_raw_norm'] = norm(combine(
            [self.grads[g] for g in PARTS[1:]], [1. / w for w in self.effective_weights]))
        self.metrics['preserve_to_repair_norm'] = norm(u) / max(norm(r), 1e-30)
        self.metrics['joint_norm'] = norm(joint)
        self.metrics['repair_preserve_cosine'] = dot(r, u) / max(norm(r) * norm(u), 1e-30)
        self.metrics['preserve_gradient_nonzero'] = float(norm(u) > 1e-12)
        self.metrics['clip_factor'] = min(1., float(clip_norm) / (norm(joint) + 1e-6))
        # A mismatch means the attribution cannot be interpreted, not that lambda is wrong.
        if not math.isfinite(error) or error > .05:
            raise RuntimeError(f'Repair/SFT gradient decomposition mismatch {error:.4g}; do not interpret attribution')
        references = dict(self.grads, preserve=u)
        for name, g in references.items():
            self.metrics[name + '/norm'] = norm(g)
            self.metrics[name + '/joint_cosine'] = dot(g, joint) / max(norm(g) * norm(joint), 1e-30)
            self.metrics[name + '/sgd_joint_dot'] = dot(g, joint)
            self.metrics[name + '/repair_cosine'] = dot(g, r) / max(norm(g) * norm(r), 1e-30)
            if name != 'repair':
                self.metrics[name + '/norm_balance_lambda'] = norm(r) * self.weight / norm(g) if norm(g) > 1e-12 else -1.
        normal_count = sum(int(x['preserve'].sum()) for x in self.records['joint'])
        for i, name in enumerate(PARTS[1:], 1):
            count = sum(int((x['preserve'] & (x['group'] == i)).sum()) for x in self.records['joint'])
            self.metrics[name + '/token_fraction'] = count / max(normal_count, 1)
            self.metrics[name + '/group_mean_raw_gradient_norm'] = (
                norm(self.grads[name]) * normal_count / (count * self.effective_weights[i - 1]) if count else 0.)
        steps = [self.optimizer.state.get(p, {}).get('step', 0) for p in self.params]
        steps = [float(s) for s in steps]
        self.metrics['adam_step_min'] = min(steps)
        self.metrics['adam_step_max'] = max(steps)
        # Read-only AdamW directions at the real moments/clipping; one candidate at a time.
        factors = [('lambda_0x', 0.), ('lambda_0.1x', .1), ('lambda_1x', 1.), ('lambda_10x', 10.)]
        if norm(u) > 1e-12:
            factors.append(('lambda_norm_balance', norm(r) / norm(u)))
        for label, factor in factors:
            candidate = joint if factor == 1. else combine([r, u], [1., factor])
            direction = adamw_direction(self.optimizer, self.params, candidate, clip_norm)
            self.metrics[label + '/weight'] = factor * self.weight
            self.metrics[label + '/action_extra_weight'] = factor * self.metrics['action_extra_weight']
            self.metrics[label + '/update_norm'] = norm(direction)
            for name, g in references.items():
                self.metrics[label + '/' + name + '_predicted_delta'] = dot(g, direction)
            del direction, candidate
        for name, candidate, fresh in (
                ('fresh_adam', joint, True),
                ('momentum_and_decay_only', [torch.zeros_like(g) if g is not None else None for g in joint], False)):
            direction = adamw_direction(self.optimizer, self.params, candidate, clip_norm, reset_moments=fresh)
            self.metrics[name + '/update_norm'] = norm(direction)
            for label, g in references.items():
                self.metrics[name + '/' + label + '_predicted_delta'] = dot(g, direction)
            del direction
        # Parameter-family attribution can reveal the vision encoder being changed
        # while textual decisions improve. It is descriptive, not a new objective.
        families = {}
        for i, (name, _) in enumerate(self.named):
            family = 'vision' if any(x in name for x in ('visual.', 'vision_')) else ('head' if 'lm_head' in name else 'language')
            families.setdefault(family, []).append(i)
        for family, indices in families.items():
            a, b = [r[i] for i in indices], [u[i] for i in indices]
            self.metrics[family + '/repair_norm'] = norm(a)
            self.metrics[family + '/preserve_norm'] = norm(b)
            self.metrics[family + '/cosine'] = dot(a, b) / max(norm(a) * norm(b), 1e-30)
        self.before = [local_float(p).clone() for p in self.params]

    def after_step(self):
        if self.before is None:
            raise RuntimeError('Diagnostic did not observe an optimizer step')
        dots = dict.fromkeys(PARTS, 0.)
        size = 0.
        for i, (p, before) in enumerate(zip(self.params, self.before, strict=True)):
            current = local_float(p)
            if current.shape != before.shape:
                raise RuntimeError('Diagnostic FSDP parameter layout changed')
            delta = current - before
            size += float(delta.double().square().sum())
            for name in PARTS:
                if self.grads[name][i] is not None:
                    dots[name] += dot([self.grads[name][i]], [delta])
        self.metrics['actual_update_norm'] = math.sqrt(size)
        for name, value in dots.items():
            self.metrics[name + '/actual_update_predicted_delta'] = value
        self.metrics['preserve/actual_update_predicted_delta'] = sum(dots[g] for g in PARTS[1:])
        self.before = None
        self.grads.clear()
        self.joint = None

    def finish(self):
        self.metrics.update(compare_records(self.records['joint'], self.records['after']))
        reproducibility = compare_records(self.records['repair'], self.records['joint'])
        for group in ('repair', 'preserve/all', 'preserve/action', 'preserve/object', 'preserve/content'):
            self.metrics['replay_noise/' + group] = reproducibility.get(group + '/delta', 0.)
            self.metrics['replay_noise/' + group + '_max_abs'] = reproducibility.get(group + '/max_abs_change', 0.)
        for name in PARTS:
            before, after = self.losses['joint'][name], self.losses['after'][name]
            self.metrics[name + '/objective_before'] = before
            self.metrics[name + '/objective_after'] = after
            self.metrics[name + '/objective_delta'] = after - before
            self.metrics[name + '/linearization_error'] = after - before - self.metrics[name + '/actual_update_predicted_delta']
        for phase in ('joint', 'after'):
            self.metrics['total_objective_' + phase] = sum(self.losses[phase].values())
        preserve_delta = sum(self.metrics[g + '/objective_delta'] for g in PARTS[1:])
        self.metrics['preserve/objective_delta'] = preserve_delta
        self.metrics['preserve/linearization_error'] = preserve_delta - self.metrics['preserve/actual_update_predicted_delta']
        if not all(math.isfinite(float(v)) for v in self.metrics.values()):
            raise RuntimeError('Nonfinite repair/SFT diagnostic metric')
        return self.metrics


def run_audit(engine, data, loss_function, weight):
    from verl.trainer.distillation.probability_transport import fixed_targets
    with fixed_targets():
        return _run_audit(engine, data, loss_function, weight)


def _run_audit(engine, data, loss_function, weight):
    """Isolate probes from optimizer/scaler/RNG state; retain one real update."""
    from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
    from verl.utils.metric import Metric, AggregationType
    from verl.utils.fsdp_utils import FSDPModule
    started = time.perf_counter()
    audit = RepairSFTAudit(engine.module, engine.optimizer, weight)
    engine._repair_sft_audit = audit
    callback = audit.callback(loss_function)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(devices[0]) if devices else None
    scaler = getattr(engine, 'scaler', None)
    def reset_rng():
        torch.set_rng_state(cpu_rng)
        if devices:
            torch.cuda.set_rng_state(cuda_rng, devices[0])
    def reshard():
        for module in engine.module.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
    try:
        maybe_fix_3d_position_ids(data)
        engine._capture_action_pair = False
        engine._command_pair_audit_enabled = False
        engine._gradient_probe_active = False
        for name in PARTS:
            engine.optimizer_zero_grad()
            audit.phase = name
            with torch.random.fork_rng(devices=devices):
                reset_rng()
                engine.forward_backward_batch(data, callback, forward_only=False)
            audit.capture(name, scaler.get_scale() if scaler is not None else 1.)
        engine.optimizer_zero_grad()
        audit.phase = 'joint'
        reset_rng()
        result = engine._train_batch_impl(data, callback)
        reshard()
        audit.after_step()
        # forward_only retains full outputs by default; drop them after recording
        # paired scalars to avoid collecting a second copy of every microbatch.
        audit.phase = 'after'
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            reset_rng()
            engine.forward_backward_batch(data, callback, forward_only=True)
        audit.metrics['seconds'] = time.perf_counter() - started
        for name, value in audit.finish().items():
            result['metrics']['manga_repair_sft_audit/' + name] = Metric(value=value, aggregation=AggregationType.MEAN)
        return result
    finally:
        reshard()
        engine.optimizer_zero_grad()
        engine._repair_sft_audit = None
