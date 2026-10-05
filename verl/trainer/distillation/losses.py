# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}

_MANGA_TEACHER_MASS_CUTOFFS = (16, 32, 64)


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    if distillation_losses_response.numel() == 0:
        # Neutral elements keep metric keys/counts identical across FSDP ranks
        # and micro-batches. A later non-empty micro-batch wins the reduction.
        return {
            "distillation/loss_min": Metric(AggregationType.MIN, float("inf")),
            "distillation/loss_max": Metric(AggregationType.MAX, float("-inf")),
        }
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def _response_mask_to_causal_input_mask(
    response_mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Align a response-token mask with full causal-LM input positions.

    ``response_mask[i, r]`` supervises response token ``r``.  Causal logits
    predicting that token live one position earlier, at
    ``prompt_len[i] + r - 1`` in the full prompt+response sequence.  Teacher
    top-k rows use that full-sequence convention, whereas Manga masks remain
    response-aligned for the final policy loss.
    """
    if not response_mask.is_nested or not input_ids.is_nested:
        raise ValueError("Manga top-k mask alignment requires jagged response masks and input_ids.")

    input_offsets = input_ids.offsets()
    response_offsets = response_mask.offsets()
    if input_offsets.numel() != response_offsets.numel():
        raise ValueError(
            "Manga response mask and input_ids must have the same batch size: "
            f"got {response_offsets.numel() - 1} and {input_offsets.numel() - 1}."
        )

    input_lengths = input_offsets.diff()
    response_lengths = response_offsets.diff()
    prompt_lengths = input_lengths - response_lengths
    if bool((prompt_lengths < 1).any()):
        raise ValueError("Every Manga sequence must contain at least one prompt token.")

    response_values = response_mask.values().bool()
    full_values = torch.zeros(
        int(input_offsets[-1].item()),
        dtype=torch.bool,
        device=response_values.device,
    )
    if response_values.numel() > 0:
        # For flat response index j in sample i:
        # full_index = j + input_start[i] + prompt_len[i] - 1 - response_start[i].
        sample_shift = input_offsets[:-1] + prompt_lengths - 1 - response_offsets[:-1]
        shifts = torch.repeat_interleave(sample_shift, response_lengths)
        target_indices = torch.arange(response_values.numel(), device=response_values.device) + shifts
        full_values[target_indices] = response_values

    return torch.nested.nested_tensor_from_jagged(full_values, offsets=input_offsets)


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
    student_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "fsdp2" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
            token_mask = (
                _response_mask_to_causal_input_mask(data["manga_opsd_mask"], data["input_ids"])
                if distillation_config.manga_opsd_enabled
                else None
            )
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
            token_mask = None
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    normalizers = None
    if distillation_config.manga_opsd_enabled and distillation_config.manga_correctness_loss:
        if config.strategy not in ('fsdp', 'fsdp2', 'veomni'):
            raise ValueError('Manga correctness loss requires the FSDP sparse projection path')
        from verl.trainer.distillation.correctness_loss import correctness_loss_outputs
        outputs, normalizers = correctness_loss_outputs(data, student_logits, token_mask, student_positions,
            distillation_config.manga_preference_weight)
    else:
        outputs = distillation_loss_fn(
            student_logits=student_logits,
            teacher_topk_log_probs=data["teacher_logprobs"],
            teacher_topk_ids=data["teacher_ids"],
            config=distillation_config,
            data_format=data_format,
            token_mask=token_mask,
            **({"student_positions": student_positions} if student_positions is not None else {}),
        )

    if "manga_action_ids_0" in data:
        from verl.trainer.distillation.action_diagnostics import action_diagnostics
        outputs.update(action_diagnostics(data, student_logits, student_positions,
            train_legality=(not distillation_config.manga_correctness_loss
                            and distillation_config.manga_action_illegal_weight > 0),
            log_normalizers=normalizers))

    expected_shape = (1, data["input_ids"].values().numel()) if student_positions is not None else student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
    student_positions: torch.Tensor | None = None,
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format, student_positions)

    # Called as final policy loss
    if distillation_config.manga_opsd_enabled:
        return manga_opsd_loss(config, distillation_config, model_output, data)

    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards and not distillation_loss_config.use_policy_gradient:
        # no need to compute policy loss
        policy_loss = 0.0
        policy_metrics = {}
    else:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
        if not distillation_loss_config.use_task_rewards:
            policy_loss = 0.0

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def _padded(value: torch.Tensor, padding_value: float | bool = 0.0) -> torch.Tensor:
    return value.to_padded_tensor(padding_value) if value.is_nested else value


def _manga_opsd_loss(config, data, distill, topk_metrics):
    """Unified forward KL for command, structural, and boundary tokens."""
    mask = _padded(data["manga_opsd_mask"], False).bool()
    loss = agg_loss(loss_mat=distill, loss_mask=mask,
                    loss_agg_mode=config.loss_agg_mode, **config.global_batch_info)
    metrics = dict(topk_metrics)
    metrics["manga_opsd/loss"] = Metric(value=loss.detach(), aggregation=AggregationType.SUM)
    metrics["manga_opsd/opsd_loss"] = Metric(
        value=(distill.detach() * mask).sum() / mask.sum().clamp_min(1),
        aggregation=AggregationType.MEAN)
    metrics["manga_opsd/opsd_tokens"] = Metric(value=mask.sum(), aggregation=AggregationType.SUM)
    if "manga_delimiter_mask" in data:
        delimiters = _padded(data["manga_delimiter_mask"], False).bool()
        for name, selected in (("delimiter", mask & delimiters), ("non_delimiter", mask & ~delimiters)):
            metrics[f"manga_tokens/{name}_count"] = Metric(value=selected.sum(), aggregation=AggregationType.SUM)
            metrics[f"manga_tokens/{name}_kl_sum"] = Metric(
                value=distill.detach()[selected].sum(), aggregation=AggregationType.SUM)
    if "manga_intent_stats" in data:
        stats = _padded(data["manga_intent_stats"])
        names = ("proposals", "legal_proposals", "exact_proposals", "legal_retained",
                 "direct_target", "dag_prerequisite", "unmatched_target_fallback",
                 "completed_target_fallback", "episodes", "completed_rollouts", "command_limit_reached",
                 "retry_attempts", "retry_successes", "gt_fallbacks", "command_timeouts",
                 "scoring_requests", "teacher_topk_expansions", "teacher_scored_rows",
                 "teacher_missing_mass_sum", "teacher_mixture_missing_mass_sum",
                 "teacher_active_components_sum", "teacher_tail_limit_rows", "teacher_output_tail_mass_sum",
                 "teacher_deterministic_rows",
                 "teacher_skipped_components",
                 "student_generation_requests",
                 "fallback_direct_target", "fallback_dag_prerequisite",
                 "fallback_unmatched_target_fallback", "fallback_completed_target_fallback",
                 "student_forced_tokens", "teacher_image_preprocess_reused",
                 "teacher_submissions_during_preparation", "repair_no_progress", "repair_round_limit",
                 "repair_token_budget", "repair_generated_tokens")
        for i, name in enumerate(names):
            metrics[f"manga_intent/{name}"] = Metric(value=stats[:, i].sum(), aggregation=AggregationType.SUM)
        totals = {name: stats[:, i].sum() for i, name in enumerate(names)}
        for name, numerator, denominator in (
            ("retry_success_rate", "retry_successes", "retry_attempts"),
            ("component_missing_mass", "teacher_missing_mass_sum", "teacher_scored_rows"),
            ("tail_limit_rate", "teacher_tail_limit_rows", "teacher_scored_rows"),
        ):
            metrics[f"manga_intent/{name}"] = Metric(
                value=totals[numerator] / totals[denominator].clamp_min(1), aggregation=AggregationType.MEAN)
        for name in ("teacher_mixture_missing_mass", "teacher_output_tail_mass", "teacher_active_components"):
            metrics[f"manga_intent/{name}"] = Metric(
                value=totals[name + "_sum"] / mask.sum().clamp_min(1), aggregation=AggregationType.MEAN)
    if "manga_command_stats" in data:
        stats = _padded(data["manga_command_stats"])
        types = _padded(data["manga_command_types"]).long()
        starts = _padded(data["manga_command_starts"], False).bool()
        for i, kind in enumerate(("enter", "detect", "read", "link", "ground")):
            for j, name in enumerate(("sampled", "illegal", "changed", "executed")):
                metrics[f"manga_command/{kind}_{name}"] = Metric(
                    value=stats[:, i * 4 + j].sum(), aggregation=AggregationType.SUM)
            for name, positions in (("start_kl", starts), ("content_kl", ~starts)):
                selected = mask & (types == i + 1) & positions
                metrics[f"manga_command/{kind}_{name}"] = Metric(
                    value=distill.detach()[selected].sum() / selected.sum().clamp_min(1),
                    aggregation=AggregationType.MEAN)
        metrics["manga_command/enter_dependency_blocked"] = Metric(
            value=stats[:, 20].sum(), aggregation=AggregationType.SUM)
        from verl.trainer.manga.command_metrics import (
            PROPOSAL_KINDS, PROPOSAL_METRICS, PROPOSAL_STATS_OFFSET,
        )
        if stats.shape[1] >= PROPOSAL_STATS_OFFSET + len(PROPOSAL_KINDS) * len(PROPOSAL_METRICS):
            for i, kind in enumerate(PROPOSAL_KINDS):
                for j, name in enumerate(PROPOSAL_METRICS):
                    index = PROPOSAL_STATS_OFFSET + i * len(PROPOSAL_METRICS) + j
                    metrics[f"manga_command/{kind}_{name}"] = Metric(
                        value=stats[:, index].sum(), aggregation=AggregationType.SUM)
    from verl.trainer.manga.diagnostics import record_opsd_actor
    record_opsd_actor(data, distill)
    return loss, metrics


def manga_opsd_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Correctness objectives or legacy forward KL, sharing command diagnostics."""

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    distill, topk_metrics = compute_forward_kl_topk(config, distillation_config, model_output, data)
    loss, metrics = _manga_opsd_loss(config, data,
        distill.detach() if distillation_config.manga_correctness_loss else distill, topk_metrics)
    if distillation_config.manga_correctness_loss:
        # These values are no longer all-token KL/top-k mass. Keep log names
        # explicit so historical KL curves are not compared to composite loss.
        for key in list(metrics):
            if key.startswith('manga_command/') and key.endswith('_kl'):
                metrics[key[:-3] + '_loss'] = metrics.pop(key)
        for old, new in (
            ('manga_opsd/opsd_loss', 'manga_correctness/token_loss'),
            ('distillation/student_mass', 'manga_correctness/student_target_mass'),
            ('distillation/student_mass_min', 'manga_correctness/student_target_mass_min'),
            ('distillation/teacher_mass', 'manga_correctness/teacher_retained_mass'),
            ('distillation/teacher_mass_min', 'manga_correctness/teacher_retained_mass_min')):
            if old in metrics:
                metrics[new] = metrics.pop(old)
        # Equal group weights; object rows already carry reciprocal per-object
        # token counts. Global denominators keep DP/microbatch splits invariant.
        parts = []
        for index, group in enumerate(('action', 'object', 'content')):
            count = float(data[f'manga_{group}_group_count'])
            part = model_output['manga_group_' + group].values().sum() / max(count, 1.) * data['dp_size']
            parts.append(part)
            metrics['manga_groups/' + group + '_loss'] = Metric(
                value=part.detach(), aggregation=AggregationType.SUM)
            metrics['manga_groups/' + group + '_count'] = Metric(
                value=data['manga_group_weights'].values()[:, index].sum(), aggregation=AggregationType.SUM)
        loss = sum(parts)
        for name in ('correctness', 'preference'):
            key = 'manga_action_' + name
            if key in model_output:
                metrics['manga_groups/action_' + name + '_loss'] = Metric(
                    value=model_output[key].values().detach().sum()
                        / max(float(data['manga_action_group_count']), 1.) * data['dp_size'],
                    aggregation=AggregationType.SUM)
        metrics['manga_correctness/single_tokens'] = Metric(
            value=data['manga_single_mask'].values().sum(), aggregation=AggregationType.SUM)
        metrics['manga_correctness/multi_tokens'] = Metric(
            value=data['manga_multi_mask'].values().sum(), aggregation=AggregationType.SUM)
        metrics['manga_correctness/content_corrected'] = Metric(
            value=data['manga_content_corrections'].values().sum(), aggregation=AggregationType.SUM)
    if 'manga_repair_stats' in data:
        stats = data['manga_repair_stats'].values().reshape(-1, 2)
        for index, name in enumerate(('recorded', 'dropped')):
            metrics['manga_repair/' + name] = Metric(value=stats[:, index].sum(), aggregation=AggregationType.SUM)
    if 'manga_repair_rows' in model_output:
        for name in ('rows', 'nll', 'mass'):
            metrics['manga_repair/' + name + '_sum'] = Metric(
                value=model_output['manga_repair_' + name].values().detach().sum(),
                aggregation=AggregationType.SUM)
    if 'manga_candidate_rows' in model_output:
        # Ratios of globally aggregated SUMs can be read without averaging page means.
        for name in ('rows', 'unchecked'):
            metrics['manga_candidates/' + name + '_sum'] = Metric(
                value=model_output['manga_candidate_' + name].values().detach().sum(),
                aggregation=AggregationType.SUM)
        metrics['manga_candidates/teacher_observed_legal_mass_sum'] = Metric(
            value=data['manga_candidate_teacher_mass'].values().sum(), aggregation=AggregationType.SUM)
        metrics['manga_candidates/checked_tokens_sum'] = Metric(
            value=data['manga_candidate_edges'].values().shape[0], aggregation=AggregationType.SUM)
    weight = distillation_config.manga_action_illegal_weight
    if weight > 0 and not distillation_config.manga_correctness_loss:
        if "action_illegal_loss" not in model_output:
            raise ValueError("Action legality loss requires rollout action supports")
        count = data["manga_action_batch_count"]
        penalty = model_output["action_illegal_loss"].values().sum() / max(float(count), 1.) * data["dp_size"]
        loss = loss + weight * penalty
        metrics["manga_action_loss/contribution"] = Metric(value=weight * penalty.detach(), aggregation=AggregationType.SUM)
    if "action_rows" in model_output:
        # SUMs give exact token-weighted means across microbatches and DP ranks:
        # divide by rows (or valid_rows for preference_kl/total_kl).
        for name in ("rows", "legal_mass", "legality_nll", "preference_kl", "total_kl", "teacher_support_mass", "valid_rows"):
            values = no_padding_2_padding(model_output["action_" + name], data)
            metrics["manga_action_pre_update/" + name + ("" if name.endswith("rows") else "_sum")] = Metric(
                value=values.detach().sum(), aggregation=AggregationType.SUM)

    if "manga_boundary_mask" in data:
        mask = _padded(data["manga_boundary_mask"], False).bool()
        metrics["manga_boundary/kl"] = Metric(
            value=(distill.detach() * mask).sum() / mask.sum().clamp_min(1),
            aggregation=AggregationType.MEAN)
        metrics["manga_boundary/tokens"] = Metric(value=mask.sum(), aggregation=AggregationType.SUM)
        stats = _padded(data["manga_boundary_stats"])
        for i, name in enumerate(("student_requests", "early_eos", "continue_boundaries", "continue_errors",
                                  "terminal_boundaries", "terminal_errors", "mixed_boundary_tokens",
                                  "teacher_budget_skipped", "budget_skipped")):
            metrics[f"manga_boundary/{name}"] = Metric(value=stats[:, i].sum(), aggregation=AggregationType.SUM)
        for name, numerator, denominator in (("early_eos_rate", 1, 0),
                                               ("continue_error_rate", 3, 2),
                                               ("terminal_error_rate", 5, 4)):
            metrics[f"manga_boundary/{name}"] = Metric(
                value=stats[:, numerator].sum() / stats[:, denominator].sum().clamp_min(1),
                aggregation=AggregationType.MEAN)
        metrics["manga_opsd/loss"] = Metric(value=loss.detach(), aggregation=AggregationType.SUM)
    return loss, metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        if loss_config.loss_mode == "frontier_soft_ce":
            response_mask = response_mask.bool()
            command_ids = data["frontier_command_ids"]
            recovery_mask = data["frontier_recovery_mask"]
            endpoint_mask = data["frontier_endpoint_mask"]
            stats = data["frontier_stats"]
            if command_ids.is_nested:
                command_ids = command_ids.to_padded_tensor(0)
            if recovery_mask.is_nested:
                recovery_mask = recovery_mask.to_padded_tensor(False)
            if endpoint_mask.is_nested:
                endpoint_mask = endpoint_mask.to_padded_tensor(False)
            if stats.is_nested:
                stats = stats.to_padded_tensor(0.0)
            command_ids = command_ids.to(device=distillation_losses.device)
            recovery_mask = recovery_mask.bool().to(device=distillation_losses.device) & response_mask
            endpoint_mask = endpoint_mask.bool().to(device=distillation_losses.device) & response_mask
            stats = stats.to(device=distillation_losses.device, dtype=distillation_losses.dtype)
            metric_index = {name: index for index, name in enumerate(FRONTIER_METRIC_KEYS)}

            path_losses = []
            endpoint_losses = []
            trajectory_losses = []
            for batch_index in range(distillation_losses.shape[0]):
                sample_losses = distillation_losses[batch_index]
                sample_commands = command_ids[batch_index]
                sample_mask = response_mask[batch_index]
                sample_recovery = recovery_mask[batch_index]
                command_total = sample_losses.sum() * 0.0
                for command_id in torch.unique(sample_commands[sample_commands > 0]):
                    command_mask = sample_commands.eq(command_id) & sample_mask
                    ordinary_mask = command_mask & ~sample_recovery
                    recovery_tokens = command_mask & sample_recovery
                    command_loss = (
                        sample_losses[ordinary_mask].mean()
                        if ordinary_mask.any()
                        else sample_losses.sum() * 0.0
                    )
                    # Every malformed-suffix token remains supervised, but the
                    # suffix is averaged as one recovery decision.  Summing it
                    # makes one long bad line dominate the entire graph loss
                    # by hundreds of times, even though no token is masked.
                    if recovery_tokens.any():
                        command_loss = command_loss + sample_losses[recovery_tokens].mean()
                    command_total = command_total + command_loss

                initial_nodes = stats[batch_index, metric_index["frontier/initial_nodes"]].clamp_min(1.0)
                path_loss = command_total / initial_nodes
                sample_endpoint = endpoint_mask[batch_index]
                if sample_endpoint.any():
                    remaining_fraction = stats[
                        batch_index, metric_index["frontier/remaining_fraction"]
                    ].clamp(0.0, 1.0)
                    endpoint_loss = sample_losses[sample_endpoint].mean() * (1.0 + remaining_fraction)
                else:
                    endpoint_loss = sample_losses.sum() * 0.0
                path_losses.append(path_loss)
                endpoint_losses.append(endpoint_loss)
                trajectory_losses.append(path_loss + endpoint_loss)

            path_loss_vector = torch.stack(path_losses)
            endpoint_loss_vector = torch.stack(endpoint_losses)
            trajectory_loss_vector = torch.stack(trajectory_losses)
            sequence_mask = torch.ones_like(trajectory_loss_vector, dtype=torch.bool)
            distillation_loss = agg_loss(
                loss_mat=trajectory_loss_vector.unsqueeze(-1),
                loss_mask=sequence_mask.unsqueeze(-1),
                loss_agg_mode="seq-mean-token-mean",
                **config.global_batch_info,
            )
            path_loss_metric = agg_loss(
                loss_mat=path_loss_vector.unsqueeze(-1),
                loss_mask=sequence_mask.unsqueeze(-1),
                loss_agg_mode="seq-mean-token-mean",
                **config.global_batch_info,
            )
            endpoint_loss_metric = agg_loss(
                loss_mat=endpoint_loss_vector.unsqueeze(-1),
                loss_mask=sequence_mask.unsqueeze(-1),
                loss_agg_mode="seq-mean-token-mean",
                **config.global_batch_info,
            )
            distillation_metrics.update(
                {
                    "frontier/path_loss": Metric(AggregationType.SUM, path_loss_metric.detach()),
                    "frontier/endpoint_loss": Metric(AggregationType.SUM, endpoint_loss_metric.detach()),
                    "frontier/total_loss": Metric(AggregationType.SUM, distillation_loss.detach()),
                }
            )
        else:
            distillation_loss = agg_loss(
                loss_mat=distillation_losses,
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                **config.global_batch_info,
            )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    metric_mask = data["manga_opsd_mask"] if distillation_config.manga_opsd_enabled else data["response_mask"]
    if metric_mask.is_nested:
        response_mask_bool = metric_mask.bool().to_padded_tensor(False)
    else:
        response_mask_bool = metric_mask.bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    raw_losses = model_output.get("raw_distillation_losses")
    if raw_losses is not None:
        raw_losses = no_padding_2_padding(raw_losses, data)[response_mask_bool]
        clipped_losses = distillation_losses[response_mask_bool]
        overlap_metrics["distillation/raw_kl_mean"] = Metric(
            value=raw_losses.mean() if raw_losses.numel() else 0.0, aggregation=AggregationType.MEAN)
        overlap_metrics["distillation/clipped_loss_mean"] = Metric(
            value=clipped_losses.mean() if clipped_losses.numel() else 0.0, aggregation=AggregationType.MEAN)
        overlap_metrics["distillation/clipped_position_fraction"] = Metric(
            value=(raw_losses > clipped_losses).float().mean() if raw_losses.numel() else 0.0,
            aggregation=AggregationType.MEAN)
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (
            (valid_overlap_count.float().mean() / k).item() if valid_overlap_count.numel() else 0.0
        )
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    if student_mass.numel() == 0:
        # Auxiliary Manga branches are intentionally GRPO-only. They contain
        # no structural OPSD positions, so top-k mass diagnostics have an empty
        # domain. Keep stable metric keys while using neutral MIN/MAX values;
        # non-empty anchor micro-batches determine the eventual extrema.
        distillation_metrics = {
            "distillation/student_mass": 0.0,
            "distillation/student_mass_min": Metric(AggregationType.MIN, float("inf")),
            "distillation/student_mass_max": Metric(AggregationType.MAX, float("-inf")),
            "distillation/teacher_mass": 0.0,
            "distillation/teacher_mass_min": Metric(AggregationType.MIN, float("inf")),
            "distillation/teacher_mass_max": Metric(AggregationType.MAX, float("-inf")),
            **overlap_metrics,
        }
    else:
        distillation_metrics = {
            "distillation/student_mass": student_mass.mean().item(),
            "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
            "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
            "distillation/teacher_mass": teacher_mass.mean().item(),
            "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
            "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
            **overlap_metrics,
        }

    # Tiny negative values are possible from floating-point cancellation.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


def _legacy_frontier_soft_ce(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return executor soft cross entropy and compact frontier diagnostics."""
    losses = no_padding_2_padding(model_output["distillation_losses"], data)
    legal_mass = no_padding_2_padding(model_output["student_mass"], data)
    fence_losses = no_padding_2_padding(model_output["frontier_fence_losses"], data)
    route_losses = no_padding_2_padding(model_output["frontier_route_losses"], data)
    mask = data["response_mask"].bool()
    if mask.is_nested:
        mask = mask.to_padded_tensor(False)
    valid_losses = losses[mask]
    valid_legal_mass = legal_mass[mask]

    command_start_mask = data["frontier_command_start_mask"]
    token_kinds = data["frontier_token_kinds"]
    recovery_mask = data["frontier_recovery_mask"]
    endpoint_mask = data["frontier_endpoint_mask"]
    if command_start_mask.is_nested:
        command_start_mask = command_start_mask.to_padded_tensor(False)
    if recovery_mask.is_nested:
        recovery_mask = recovery_mask.to_padded_tensor(False)
    if endpoint_mask.is_nested:
        endpoint_mask = endpoint_mask.to_padded_tensor(False)
    if token_kinds.is_nested:
        token_kinds = token_kinds.to_padded_tensor(0)
    command_start_mask = command_start_mask.bool() & mask
    recovery_mask = recovery_mask.bool() & mask
    endpoint_mask = endpoint_mask.bool() & mask

    stats = data["frontier_stats"]
    if stats.is_nested:
        stats = stats.to_padded_tensor(0.0)
    stats = stats.float()
    stat_index = {name: index for index, name in enumerate(FRONTIER_METRIC_KEYS)}

    def stat(name: str) -> torch.Tensor:
        return stats[:, stat_index[name]]

    def progress(kind: str) -> float:
        visited = stat(f"frontier/visited_{kind}").sum()
        total = stat(f"frontier/total_{kind}").sum()
        return (visited / total.clamp_min(1.0)).item()

    metrics = {
        "frontier/soft_ce": valid_losses.mean().item(),
        "frontier/fence_loss": fence_losses[mask].mean().item(),
        "frontier/route_kl": route_losses[mask].mean().item(),
        "frontier/legal_mass": valid_legal_mass.mean().item(),
        "frontier/command_start_legal_mass": (
            legal_mass[command_start_mask].mean().item() if command_start_mask.any() else 0.0
        ),
        "frontier/endpoint_legal_mass": (
            legal_mass[endpoint_mask].mean().item() if endpoint_mask.any() else 0.0
        ),
        "frontier/recovery_legal_mass": (
            legal_mass[recovery_mask].mean().item() if recovery_mask.any() else 0.0
        ),
        "frontier/complete_rate": stat("frontier/complete").mean().item(),
        "frontier/premature_eos_rate": stat("frontier/premature_eos").mean().item(),
        "frontier/synthetic_endpoint_rate": stat("frontier/synthetic_endpoint").mean().item(),
        "frontier/incomplete_without_eos_rate": stat("frontier/incomplete_without_eos").mean().item(),
        "frontier/remaining_fraction": stat("frontier/remaining_fraction").mean().item(),
        "frontier/remaining_nodes": stat("frontier/remaining_nodes").mean().item(),
        "frontier/invalid_command_rate": stat("frontier/invalid_command_rate").mean().item(),
        "frontier/token_coverage": stat("frontier/token_coverage").mean().item(),
        "frontier/recovery_token_fraction": stat("frontier/recovery_token_fraction").mean().item(),
        "frontier/recovery_commands": stat("frontier/recovery_commands").mean().item(),
        "frontier/recovery_loss": losses[recovery_mask].mean().item() if recovery_mask.any() else 0.0,
        "frontier/accepted_commands": stat("frontier/accepted_commands").mean().item(),
        "frontier/mean_legal_actions": stat("frontier/mean_legal_actions").mean().item(),
        "frontier/mean_legal_action_types": stat("frontier/mean_legal_action_types").mean().item(),
        "frontier/detect_success_rate": stat("frontier/detect_success_rate").mean().item(),
        "frontier/mean_detect_iou": stat("frontier/mean_detect_iou").mean().item(),
        "frontier/progress/enter": progress("enter"),
        "frontier/progress/detect_character": progress("detect_character"),
        "frontier/progress/detect_text": progress("detect_text"),
        "frontier/progress/read": progress("read"),
        "frontier/progress/speaker_link": progress("speaker_link"),
        "frontier/progress/identity_link": progress("identity_link"),
        "frontier/progress/ground": progress("ground"),
    }
    for kind in FRONTIER_ACTION_KINDS:
        kind_mask = token_kinds.eq(FRONTIER_ACTION_KIND_IDS[kind]) & mask
        metrics[f"frontier/token_fraction/{kind}"] = kind_mask.sum().div(mask.sum().clamp_min(1)).item()
        metrics[f"frontier/soft_ce/{kind}"] = losses[kind_mask].mean().item() if kind_mask.any() else 0.0
        metrics[f"frontier/legal_mass/{kind}"] = (
            legal_mass[kind_mask].mean().item() if kind_mask.any() else 0.0
        )
    return losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
