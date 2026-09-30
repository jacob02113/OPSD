# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=0 if pad_token_id is None else int(pad_token_id)).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def generate_command(self, prompt_ids, image, mm_processor_kwargs, *,
                               max_tokens, routing_key=None):
        """Generate on the frozen teacher, never on the student rollout server."""
        key = self._resolve_teacher_key(routing_key)
        limit = self.teacher_model_configs[key].inference.max_model_len
        if limit is not None and len(prompt_ids) + max_tokens > limit:
            raise ValueError("Corrective teacher prompt exceeds its context budget")
        return await self.teacher_client[key].generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params={"max_tokens": max_tokens, "temperature": 0.0,
                             "top_p": 1.0, "n": 1, "stop": ["\n"],
                             "include_stop_str_in_output": True, "skip_special_tokens": False},
            image_data=[image], mm_processor_kwargs=mm_processor_kwargs,
        )

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        positions: Optional[list[int]] = None,
        allowed_ids=None,
        topk_override=None,
        return_sampled=False,
        raw_legal_probabilities=False,
        shared_prefill=None,
        action_support=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        max_model_len = teacher_model_config.inference.max_model_len
        if max_model_len is not None and len(sequence_ids) + 1 > max_model_len:
            raise ValueError(
                "Teacher input plus the one-token scoring request exceeds max_model_len: "
                f"sequence={len(sequence_ids)}, required={len(sequence_ids) + 1}, max_model_len={max_model_len}. "
                "For OPSD, filter or shorten privileged solutions, reduce multimodal input size, or increase "
                "the teacher context length."
            )
        sampling_params = _get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config)
        if topk_override is not None:
            sampling_params['prompt_logprobs'] = topk_override
        if allowed_ids is not None:
            if teacher_model_config.inference.name != 'vllm' or positions is None or len(allowed_ids) != len(positions):
                raise ValueError('Local teacher masks require aligned sparse vLLM positions')
            for position, mask in zip(positions, allowed_ids, strict=True):
                if not mask or (isinstance(mask, dict) and len(set(mask['exclude'])) >= mask['vocab_size']):
                    raise ValueError(f'Empty teacher support at position {position}; '
                                     'eliminate the component before submitting to the engine')
            sampling_params['extra_args'] = {'verl_prompt_allowed_ids':
                {str(position): ids for position, ids in zip(positions, allowed_ids, strict=True)}}
        if action_support:
            sampling_params.setdefault('extra_args', {})['verl_action_support'] = action_support
        if shared_prefill is not None:
            sampling_params['_verl_shared_prefill'] = shared_prefill
        server_slices_positions = positions is not None and teacher_model_config.inference.name == "vllm"
        if raw_legal_probabilities:
            if not return_sampled or not teacher_model_config.inference.skip_tokenizer_init:
                raise ValueError('Raw probabilities require metadata return and tokenizer-free teacher')
            sampling_params.setdefault('extra_args', {})['verl_raw_legal_probabilities'] = True
        if server_slices_positions:
            # Internal verl option, removed before vLLM constructs SamplingParams.
            sampling_params["_verl_prompt_logprobs_positions"] = positions
        teacher_output = await client.generate(
            request_id=request_id or uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=sampling_params,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        # torch.gather used by forward_kl_topk requires int64 indices.
        raw_ids = teacher_output.extra_fields["prompt_ids"]
        raw_logprobs = teacher_output.extra_fields["prompt_logprobs"]
        if positions is not None and not server_slices_positions:
            if any(position < 0 or position >= len(sequence_ids) for position in positions):
                raise ValueError(f"Teacher positions outside sequence of length {len(sequence_ids)}: {positions}")
            raw_ids = [raw_ids[position] for position in positions]
            raw_logprobs = [raw_logprobs[position] for position in positions]
        teacher_ids = torch.tensor(raw_ids, dtype=torch.long)
        teacher_logprobs = torch.tensor(raw_logprobs)
        expected_rows = len(sequence_ids) if positions is None else len(positions)
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == expected_rows
        if return_sampled:
            return teacher_ids, teacher_logprobs, teacher_output.extra_fields['prompt_sampled_logprobs']
        return teacher_ids, teacher_logprobs
