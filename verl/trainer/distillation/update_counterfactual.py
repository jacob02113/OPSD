"""Full-minibatch action-vs-joint update diagnostic without an extra optimizer step."""
import math
import torch


def local_float(tensor):
    if hasattr(tensor, 'to_local'):
        tensor = tensor.to_local()
    return tensor.detach().float().cpu()


class ActionGradientAudit:
    def __init__(self, module, optimizer, max_elements=250_000_000):
        self.parameters = [p for p in module.parameters() if p.requires_grad]
        size = sum(p.numel() for p in self.parameters)
        if not self.parameters or size > max_elements:
            raise ValueError('Trainable parameter count exceeds the audit memory bound')
        self.optimizer = optimizer
        self.action = [None] * len(self.parameters)
        self.rows = 0
        self.elements = size

    def action_loss(self, **kwargs):
        """Loss callback for an ordinary FSDP backward over the diagnostic pass."""
        output = kwargs['model_output']
        data = kwargs['data']
        count = float(data['manga_action_group_count'])
        if count <= 0:
            raise RuntimeError('Full-batch diagnostic has no action group denominator')
        objective = (output['manga_action_correctness'].values().sum()
                     * float(data['dp_size']) / count)
        if not objective.requires_grad:
            raise RuntimeError('Full-batch action objective is detached from the model')
        return objective, {}

    def capture_action_gradients(self, data, scale=1.):
        """Read the gradients produced by FSDP's normal backward hooks."""
        self.rows = int((data['manga_action_ids_0'].values() >= 0).sum())
        self.action = [None if param.grad is None else local_float(param.grad) / scale
                       for param in self.parameters]
        if not self.rows or not any(g is not None for g in self.action):
            raise RuntimeError('Full-batch diagnostic backward produced no action gradient')

    def finish(self, scale=1., clip_norm=None):
        if not self.rows or not any(g is not None for g in self.action):
            raise RuntimeError('Full-batch diagnostic found no action gradient')
        joint = []
        for param in self.parameters:
            joint.append(None if param.grad is None else local_float(param.grad) / scale)
        action_norm = math.sqrt(sum(float((g*g).sum()) for g in self.action if g is not None))
        joint_norm = math.sqrt(sum(float((g*g).sum()) for g in joint if g is not None))
        dot = sum(float((a*j).sum()) for a,j in zip(self.action,joint)
                  if a is not None and j is not None)
        result = {'available': 1., 'action_rows': float(self.rows),
                  'trainable_elements': float(self.elements),
                  'gradient_cosine': dot / max(action_norm*joint_norm, 1e-30),
                  'sgd_action_only_descent': action_norm*action_norm,
                  'sgd_joint_descent': dot,
                  'action_gradient_norm': action_norm, 'joint_gradient_norm': joint_norm}
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise TypeError('Full-batch diagnostic requires torch AdamW')
        action_step = adamw_direction(self.optimizer, self.parameters, self.action, clip_norm)
        joint_step = adamw_direction(self.optimizer, self.parameters, joint, clip_norm)
        result['adamw_action_only_predicted_loss_delta'] = sum(
            float((g*d).sum()) for g,d in zip(self.action,action_step) if g is not None and d is not None)
        result['adamw_joint_predicted_loss_delta'] = sum(
            float((g*d).sum()) for g,d in zip(self.action,joint_step) if g is not None and d is not None)
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError('Full-batch diagnostic produced nonfinite values')
        return result


def adamw_direction(optimizer, parameters, gradients, clip_norm, *, reset_moments=False):
    """First-order AdamW parameter displacement using existing moments; read-only."""
    norm = math.sqrt(sum(float((g*g).sum()) for g in gradients if g is not None))
    factor = min(1., float(clip_norm)/(norm+1e-6)) if clip_norm is not None and norm else 1.
    groups = {id(param): group for group in optimizer.param_groups for param in group['params']}
    result = []
    for param, raw in zip(parameters, gradients):
        if raw is None:
            result.append(None)
            continue
        group = groups.get(id(param))
        if group is None or group.get('maximize', False) or group.get('amsgrad', False):
            raise ValueError('Unsupported AdamW parameter group')
        lr = float(group['lr']); beta1,beta2 = group['betas']; eps = float(group['eps'])
        state = {} if reset_moments else optimizer.state.get(param, {})
        step = state.get('step', 0)
        step = int(step.item() if torch.is_tensor(step) else step) + 1
        m = local_float(state['exp_avg']) if 'exp_avg' in state else torch.zeros_like(raw)
        v = local_float(state['exp_avg_sq']) if 'exp_avg_sq' in state else torch.zeros_like(raw)
        g = raw * factor
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g.square()
        update = -lr * (m/(1-beta1**step)) / ((v/(1-beta2**step)).sqrt() + eps)
        decay = float(group.get('weight_decay', 0.))
        if decay:
            update -= lr*decay*local_float(param)
        result.append(update)
    return result
