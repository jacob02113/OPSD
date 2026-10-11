from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LoRAProbeSnapshot:
    samples: torch.Tensor
    tensor_count: int
    local_numel: int
    parameter_fp32_numel: int
    exp_avg_numel: int
    exp_avg_fp32_numel: int
    exp_avg_sq_numel: int
    exp_avg_sq_fp32_numel: int


def index_lora_optimizer_parameters(module, optimizer) -> dict[str, list[torch.nn.Parameter]]:
    """Bind LoRA names to the exact parameter objects owned by the optimizer.

    FSDP2 may expose temporary mixed-precision compute parameters through
    ``module.named_parameters()`` after a forward. Indexing before the forward and
    retaining optimizer-owned references keeps the probe on FP32 master parameters.
    """
    optimizer_parameter_ids = {
        id(parameter)
        for param_group in optimizer.param_groups
        for parameter in param_group["params"]
    }
    indexed: dict[str, list[torch.nn.Parameter]] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) not in optimizer_parameter_ids:
            continue
        group = _lora_group(name)
        if group is not None:
            indexed.setdefault(group, []).append(parameter)
    if not indexed:
        raise RuntimeError("Could not bind any named LoRA parameter to the optimizer")
    return indexed


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _lora_group(name: str) -> Optional[str]:
    lowered = name.lower()
    if "lora_a" in lowered:
        return "lora_a"
    if "lora_b" in lowered:
        return "lora_b"
    return None


def _sample_tensor(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= limit:
        return flat.float().clone()
    # Evenly cover the local shard instead of repeatedly selecting its prefix.
    indices = torch.linspace(0, flat.numel() - 1, steps=limit, device=flat.device).long()
    return flat.index_select(0, indices).float().clone()


def capture_lora_optimizer_probe(
    indexed_parameters: dict[str, list[torch.nn.Parameter]],
    optimizer,
    *,
    sample_limit_per_group: int = 16384,
    sample_limit_per_tensor: int = 64,
) -> dict[str, LoRAProbeSnapshot]:
    """Capture bounded LoRA samples and exact local dtype counts.

    Sampling is spread across LoRA tensors and bounded per group, so the audit does
    not clone the full adapter. Optimizer-state counts are collected after Adam has
    initialized them and remain zero before the first step.
    """
    accumulators: dict[str, dict] = {}
    for group, parameters in indexed_parameters.items():
        accumulator = accumulators.setdefault(
            group,
            {
                "samples": [],
                "sampled_numel": 0,
                "tensor_count": 0,
                "local_numel": 0,
                "parameter_fp32_numel": 0,
                "exp_avg_numel": 0,
                "exp_avg_fp32_numel": 0,
                "exp_avg_sq_numel": 0,
                "exp_avg_sq_fp32_numel": 0,
            },
        )
        for parameter in parameters:
            local_parameter = _local_tensor(parameter)
            local_numel = local_parameter.numel()
            accumulator["tensor_count"] += 1
            accumulator["local_numel"] += local_numel
            if local_parameter.dtype == torch.float32:
                accumulator["parameter_fp32_numel"] += local_numel

            remaining = sample_limit_per_group - accumulator["sampled_numel"]
            if local_numel and remaining > 0:
                sample = _sample_tensor(local_parameter, min(sample_limit_per_tensor, remaining))
                accumulator["samples"].append(sample)
                accumulator["sampled_numel"] += sample.numel()

            state = optimizer.state.get(parameter, {})
            for state_name in ("exp_avg", "exp_avg_sq"):
                state_tensor = state.get(state_name)
                if not isinstance(state_tensor, torch.Tensor):
                    continue
                local_state = _local_tensor(state_tensor)
                state_numel = local_state.numel()
                accumulator[f"{state_name}_numel"] += state_numel
                if local_state.dtype == torch.float32:
                    accumulator[f"{state_name}_fp32_numel"] += state_numel

    snapshots = {}
    for group, accumulator in accumulators.items():
        samples = accumulator.pop("samples")
        snapshots[group] = LoRAProbeSnapshot(
            samples=torch.cat(samples) if samples else torch.empty(0, dtype=torch.float32),
            **{key: value for key, value in accumulator.items() if key != "sampled_numel"},
        )
    return snapshots


def summarize_lora_optimizer_probe(
    before: dict[str, LoRAProbeSnapshot],
    after: dict[str, LoRAProbeSnapshot],
) -> tuple[dict[str, float], float]:
    """Return scalar metrics per LoRA group and a legacy combined delta L2."""
    metrics: dict[str, float] = {}
    combined_delta_sq = 0.0
    common_groups = sorted(set(before) & set(after))
    if not common_groups:
        return metrics, -1.0
    for group in common_groups:
        old = before[group]
        new = after[group]
        if old.samples.shape != new.samples.shape:
            raise RuntimeError(f"LoRA optimizer probe shape changed for {group}")

        delta = new.samples - old.samples
        sampled_numel = delta.numel()
        delta_sq = float(torch.sum(delta.square()).item()) if sampled_numel else 0.0
        combined_delta_sq += delta_sq
        prefix = f"{group}/"
        metrics[prefix + "tensor_count"] = float(new.tensor_count)
        metrics[prefix + "local_numel"] = float(new.local_numel)
        metrics[prefix + "sampled_numel"] = float(sampled_numel)
        metrics[prefix + "parameter_fp32_fraction"] = (
            new.parameter_fp32_numel / new.local_numel if new.local_numel else -1.0
        )
        metrics[prefix + "sampled_parameter_rms_before"] = (
            float(torch.sqrt(torch.mean(old.samples.square())).item()) if sampled_numel else -1.0
        )
        metrics[prefix + "sampled_parameter_rms_after"] = (
            float(torch.sqrt(torch.mean(new.samples.square())).item()) if sampled_numel else -1.0
        )
        metrics[prefix + "sampled_delta_l2"] = delta_sq**0.5
        metrics[prefix + "sampled_delta_rms"] = (delta_sq / sampled_numel) ** 0.5 if sampled_numel else -1.0
        metrics[prefix + "sampled_changed_fraction"] = (
            float(torch.count_nonzero(delta).item()) / sampled_numel if sampled_numel else -1.0
        )
        metrics[prefix + "exp_avg_initialized_fraction"] = (
            new.exp_avg_numel / new.local_numel if new.local_numel else -1.0
        )
        metrics[prefix + "exp_avg_fp32_fraction"] = (
            new.exp_avg_fp32_numel / new.exp_avg_numel if new.exp_avg_numel else -1.0
        )
        metrics[prefix + "exp_avg_sq_initialized_fraction"] = (
            new.exp_avg_sq_numel / new.local_numel if new.local_numel else -1.0
        )
        metrics[prefix + "exp_avg_sq_fp32_fraction"] = (
            new.exp_avg_sq_fp32_numel / new.exp_avg_sq_numel if new.exp_avg_sq_numel else -1.0
        )

    return metrics, combined_delta_sq**0.5
