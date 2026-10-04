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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
    return_tail: bool = False,
):
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    Returns:
        [B, T, K] tensor in float32 to preserve probability mass near one.
    """
    prefix_shape, V = logits.shape[:-1], logits.shape[-1]
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K).long()  # [N, K], gather requires int64
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        empty = torch.empty((*prefix_shape, K), dtype=torch.float32, device=logits.device)
        return (empty, empty.sum(-1)) if return_tail else empty

    out = torch.empty((N, K), dtype=torch.float32, device=logits.device)
    tail = torch.empty((N,), dtype=torch.float32, device=logits.device) if return_tail else None
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = chunk_topk_logits - log_z
        if return_tail:
            omitted = chunk_logits_fp32.scatter(-1, flat_topk[s:e], -torch.inf)
            tail[s:e] = torch.logsumexp(omitted, dim=-1) - log_z.squeeze(-1)
    selected = out.reshape(*prefix_shape, K)
    return (selected, tail.reshape(prefix_shape)) if return_tail else selected


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor, token_clip: float | None = None) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    positive = p > 0
    delta = torch.where(positive, log_p - log_q, torch.zeros_like(log_p))
    kld = p * delta
    if token_clip is not None:
        # Match official OPSD: cap positive pointwise terms, retain negative terms.
        kld = kld.clamp(max=token_clip)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    token_mask: torch.Tensor | None = None,
    student_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0).float()  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0).long()  # (1, total_nnz, topk)
    if token_mask is not None:
        assert token_mask.is_nested
        token_mask = token_mask.values().unsqueeze(0).bool()

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        if token_mask is not None:
            token_mask = slice_input_tensor(token_mask, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2]
    if student_positions is None:
        assert teacher_topk_ids.shape[:2] == student_logits.shape[:2]
    elif token_mask is None or get_ulysses_sequence_parallel_world_size() != 1:
        raise ValueError("Selected OPSD projection requires a mask and no sequence parallelism")

    # Only supervised command tokens need vocabulary reductions. History and
    # synthetic padding still pass through the model for causal context/DDP.
    output_shape = teacher_topk_ids.shape[:2]
    if token_mask is not None:
        assert token_mask.shape == output_shape
        selected = token_mask
        if student_positions is None:
            student_logits = student_logits[selected]
        else:
            expected_positions = selected.reshape(-1).nonzero(as_tuple=True)[0]
            indices = torch.searchsorted(student_positions, expected_positions)
            if (bool((indices >= student_positions.numel()).any()) or
                    not torch.equal(student_positions[indices], expected_positions)):
                raise ValueError("Selected vocabulary projections do not match the causal loss mask")
            student_logits = student_logits.squeeze(0).index_select(0, indices)
        teacher_topk_log_probs = teacher_topk_log_probs[selected]
        teacher_topk_ids = teacher_topk_ids[selected]

        if student_logits.shape[0] == 0:
            zeros = teacher_topk_log_probs.new_zeros(output_shape)
            # Keep an autograd path through the empty selection to the actor.
            # Distributed padding ranks must participate in backward/collectives;
            # a detached zero (or skipping backward) is not equivalent.
            zero_loss = zeros + student_logits.float().sum()
            return {
                "distillation_losses": zero_loss,
                "raw_distillation_losses": zeros,
                "student_mass": zeros,
                "teacher_mass": zeros,
                "overlap_count": zeros.long(),
                "overlap_token_advantage": zeros,
            }

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs, student_tail_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
            return_tail=True,
        )
    else:
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
        # Compute the complement directly; 1 - topk_mass loses its gradient
        # when rounding makes the selected mass equal to one.
        student_tail_log_probs = student_log_probs.scatter(-1, teacher_topk_ids, -torch.inf).logsumexp(-1)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    token_clip = loss_config.jsd_token_clip
    raw_topk_kl = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)
    topk_kl = (raw_topk_kl if token_clip is None else
               kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs, token_clip=token_clip))
    # The teacher's top-k support is exact, while all omitted vocabulary items
    # form one coarse tail event.  This preserves probability mass and avoids
    # treating a truncated, unnormalised KL sum as a full forward KL.
    eps = torch.finfo(torch.float32).eps
    teacher_tail = (1.0 - teacher_mass.float()).clamp(min=0.0, max=1.0)
    if teacher_topk_ids.shape[-1] == student_logits.shape[-1]:
        tail_kl = torch.zeros_like(teacher_tail)
    else:
        # Exact zero for deterministic targets, including a zero student tail.
        positive_tail = teacher_tail > 0
        tail_delta = torch.where(
            positive_tail, teacher_tail.clamp_min(eps).log() - student_tail_log_probs,
            torch.zeros_like(student_tail_log_probs))
        tail_kl = teacher_tail * tail_delta
    raw_distillation_losses = (raw_topk_kl + tail_kl).detach()
    # The omitted vocabulary is one coarse event in this implementation. Cap
    # that event too so the tail cannot bypass stabilization. This differs from
    # clipping each omitted vocabulary item, whose probabilities are unavailable.
    if token_clip is not None:
        tail_kl = tail_kl.clamp(max=token_clip)
    distillation_losses = topk_kl + tail_kl

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    if token_mask is not None:
        def scatter_selected(values: torch.Tensor) -> torch.Tensor:
            output = values.new_zeros(output_shape)
            output[selected] = values
            return output

        distillation_losses = scatter_selected(distillation_losses)
        raw_distillation_losses = scatter_selected(raw_distillation_losses)
        student_mass = scatter_selected(student_mass)
        teacher_mass = scatter_selected(teacher_mass)
        overlap_count = scatter_selected(overlap_count)
        overlap_token_advantage = scatter_selected(overlap_token_advantage)

    return {
        "distillation_losses": distillation_losses,
        "raw_distillation_losses": raw_distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
