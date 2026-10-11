"""Two KL budgets, with feedback from the ordinary pre-update actor forward."""
import math


class PreservationController:
    def __init__(self, budgets, initial_weight=1., rate=.5, decay=.8):
        if (len(budgets) != 2 or any(not math.isfinite(x) or x <= 0 for x in budgets)
                or not math.isfinite(initial_weight) or not .01 <= initial_weight <= 1e4
                or not math.isfinite(rate) or rate <= 0 or not 0 <= decay < 1):
            raise ValueError('Invalid adaptive preservation settings')
        budgets = [float(x) for x in budgets]
        initial_weight, rate, decay = float(initial_weight), float(rate), float(decay)
        self.budgets = list(budgets)
        self.rate, self.decay = rate, decay
        self.floor = initial_weight
        self.weights = [initial_weight, initial_weight]
        self.ema = [None, None]
        self.updates = 0

    def update(self, measurements, counts):
        # Validate before mutating either constraint. A missing cohort is not KL=0.
        if len(measurements) != 2 or len(counts) != 2:
            raise ValueError('Expected overall and action preservation measurements')
        for value, count in zip(measurements, counts):
            if not math.isfinite(count) or count < 0 or (count and not math.isfinite(value)):
                raise ValueError('Nonfinite preservation controller observation')
        measurements = [float(x) for x in measurements]
        counts = [float(x) for x in counts]
        metrics = {}
        for i, name in enumerate(('all', 'action')):
            value, count, target = measurements[i], counts[i], self.budgets[i]
            used = self.weights[i]
            if count:
                value = max(0., value)  # tolerate tiny numerical negative KL
                self.ema[i] = value if self.ema[i] is None else self.decay*self.ema[i] + (1-self.decay)*value
                error = self.ema[i] / target - 1.
                # Multiplicative dual feedback: bounded change per update, no line search.
                self.weights[i] = min(1e4, max(self.floor, used * math.exp(self.rate * max(-1., min(1., error)))))
            metrics.update({f'{name}/weight_used': used, f'{name}/weight_next': self.weights[i],
                            f'{name}/budget': target, f'{name}/count': count,
                            f'{name}/ema_kl': self.ema[i] if self.ema[i] is not None else 0.,
                            f'{name}/observed': float(count > 0),
                            f'{name}/at_max_weight': float(self.weights[i] == 1e4)})
            if count:
                metrics[f'{name}/kl_over_budget'] = value / target
                metrics[f'{name}/ema_over_budget'] = self.ema[i] / target
        self.updates += 1
        return metrics

    def state_dict(self):
        return dict(version=1, budgets=[float(x) for x in self.budgets], rate=float(self.rate), decay=float(self.decay), floor=float(self.floor),
                    weights=[float(x) for x in self.weights], ema=[None if x is None else float(x) for x in self.ema], updates=int(self.updates))

    def load_state_dict(self, state):
        if (state.get('version') != 1 or state['budgets'] != self.budgets
                or state['rate'] != self.rate or state['decay'] != self.decay or state['floor'] != self.floor):
            raise ValueError('Adaptive preservation checkpoint settings differ from this run')
        if (len(state['weights']) != 2 or len(state['ema']) != 2
                or any(not math.isfinite(x) or not self.floor <= x <= 1e4 for x in state['weights'])
                or any(x is not None and (not math.isfinite(x) or x < 0) for x in state['ema'])
                or not isinstance(state['updates'], int) or state['updates'] < 0):
            raise ValueError('Invalid adaptive preservation checkpoint state')
        self.weights = [float(x) for x in state['weights']]
        self.ema = [None if x is None else float(x) for x in state['ema']]
        self.updates = state['updates']


def action_kl_sum(output, data):
    """Select response action labels at their input-causal predictor positions."""
    import torch
    mask = data['manga_preserve_mask']
    groups = data['manga_preserve_group'].values()
    if bool((mask.values().bool() & ((groups < 1) | (groups > 3))).any()):
        raise ValueError('Adaptive preservation requires labels on every normal token')
    offsets = data['input_ids'].offsets()
    sizes = mask.offsets().diff()
    shifts = offsets[:-1] + offsets.diff() - sizes - 1 - mask.offsets()[:-1]
    selected = mask.values().bool() & (groups == 1)
    rows = torch.arange(groups.numel(), device=groups.device) + torch.repeat_interleave(shifts, sizes)
    return output['manga_preserve_kl'].values()[rows[selected]].sum()


def load_controller_checkpoint(path):
    """Restricted compatibility loader for old NumPy floating-point scalars."""
    import numpy as np
    import torch
    # Metric reduction returned np.float64, which leaked into EMA/weights.
    scalar = np.float64(0).__reduce__()[0]
    allowed = [(scalar, 'numpy.core.multiarray.scalar'),
               (scalar, 'numpy._core.multiarray.scalar'), np.dtype,
               type(np.dtype('float32')), type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)
