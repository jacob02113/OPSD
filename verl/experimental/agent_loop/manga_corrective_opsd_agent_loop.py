"""Local privileged mixtures, prefix recovery and executor-labelled boundary CE."""
from __future__ import annotations

import asyncio
import copy
import math
import re
import logging
import time
from uuid import uuid4
from typing import Any

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.trainer.manga import MangaExecutor, TargetGraph
from verl.trainer.manga.command_teacher import (
    sg, select_target, signature,
    teacher_chat, dependency_graph, frontier,
    proposal_stats,
)
from verl.utils.tokenizer import build_multimodal_processor_inputs
from verl.utils.tokenizer.chat_template import apply_chat_template

from verl.trainer.manga.local_policy import COMMAND_MAX_TOKENS, COMMAND_STOPS, command_boundary, Support, local_chat, mix_sparse, vocabulary_for, can_match_box


logger = logging.getLogger(__name__)


def _python_value(value: Any) -> Any:
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        return value.item()
    return value.tolist() if hasattr(value, "tolist") else value

def _resolve(sample: dict[str, Any], dotted_key: str) -> Any:
    value: Any = sample
    for part in dotted_key.split("."):
        value = _python_value(value)
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return _python_value(value)

@register("manga_corrective_opsd_agent")
class MangaCorrectiveOPSDLoop(AgentLoopBase):
    def __init__(self, *args, teacher_server_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_server_manager = teacher_server_manager
        cfg = self.config.distillation
        if not cfg.manga_opsd_enabled or teacher_server_manager is None:
            raise ValueError("Manga OPSD requires a frozen teacher server")
        if self.config.actor_rollout_ref.actor.ppo_epochs != 1:
            raise ValueError("Corrective OPSD requires one optimizer epoch per on-policy batch")
        if not self.config.trainer.get("use_v1", False):
            raise ValueError("Corrective command branches require the V1 TransferQueue trainer")
        self.target_key = cfg.manga_target_key
        self.iou_threshold = float(cfg.manga_entity_iou_threshold)
        self.topk = int(cfg.distillation_loss.topk)
        self.response_length = int(self.rollout_config.response_length)
        self.teacher_max_inflight = int(cfg.manga_teacher_max_inflight)
        self.max_commands = int(cfg.manga_max_commands)
        self.teacher_topk_initial = int(cfg.manga_teacher_topk_initial)
        self.teacher_topk_max = int(cfg.manga_teacher_topk_max)
        self.teacher_min_mass = float(cfg.manga_teacher_min_mass)
        self.correctness_loss = bool(cfg.manga_correctness_loss)
        if self.correctness_loss and self.rollout_config.name != "vllm":
            raise ValueError("Sparse student candidate scoring requires the vLLM rollout backend")
        if self.max_commands <= 0:
            raise ValueError("manga_max_commands must be positive")

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _teacher_storage(self, prompt_length: int, response_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros((prompt_length + response_length, self.topk), dtype=torch.long)
        logprobs = torch.full((prompt_length + response_length, self.topk), -1.0e9, dtype=torch.float32)
        return ids, logprobs

    async def _teacher_targets(
        self,
        sequence_ids: list[int],
        image: Image.Image,
        mm_processor_kwargs: dict[str, Any],
        routing_key: str | None,
        request_id: str,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prompt-logprob row p requires token p+1. Later suffix tokens cannot
        # affect that causal distribution and need no teacher prefill.
        if positions:
            if min(positions) < 0 or max(positions) >= len(sequence_ids) - 1:
                raise ValueError("Teacher target requires a following prompt token")
            sequence_ids = sequence_ids[:max(positions) + 2]
        return await self.teacher_server_manager.compute_teacher_logprobs_single(
            sequence_ids=sequence_ids,
            multi_modal_data={"images": [image]},
            mm_processor_kwargs=mm_processor_kwargs,
            routing_key=routing_key,
            request_id=request_id,
            positions=positions,
        )

    async def _generate(
        self,
        prompt_ids: list[int],
        response_prefix: list[int],
        images: list[Any] | None,
        videos: list[Any] | None,
        audios: list[Any] | None,
        mm_processor_kwargs: dict[str, Any],
        sampling_params: dict[str, Any],
        *,
        max_tokens: int,
        stop: list[str],
        priority: int,
        routing_request_id: str,
    ):
        if max_tokens <= 0:
            raise ValueError("Student generation requires a positive remaining token budget.")
        params = dict(sampling_params)
        params.update(
            {
                "max_tokens": max_tokens,
                "stop": stop,
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "logprobs": False,
            }
        )
        output = await self.server_manager.generate(
            # Keep every turn from one trajectory on the same rollout replica.
            # The backend still receives a unique engine request id, while the
            # client routing id preserves prefix-cache locality.
            request_id=routing_request_id,
            prompt_ids=prompt_ids + response_prefix,
            sampling_params=params,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
        )
        if hasattr(self, '_perf_counters'):
            self._perf_counters['student_generation_requests'] += 1
        return output

    async def _privileged_prompt(self, messages, history, image, mm_kwargs, *, hint, key=None):
        # Unlike the student initial-prompt helper, do not cap this full-history
        # teacher conversation at data.max_prompt_length or overwrite actor pixels.
        chat = local_chat(messages, history, hint, key) if key is not None else teacher_chat(messages, history, hint)
        def encode():
            text = apply_chat_template(self.processor, chat, add_generation_prompt=True,
                                       tokenize=False, **self.apply_chat_template_kwargs)
            marker = getattr(self.processor, 'image_token', '<|image_pad|>')
            cache_key = (tuple(im.size for im in image), repr(sorted(mm_kwargs.items())))
            expansion = self._image_expansions.get(cache_key)
            def fast(counts):
                pieces = text.split(marker)
                return self.tokenizer.encode(''.join(piece + (marker * counts[i] if i < len(counts) else '')
                    for i, piece in enumerate(pieces)), add_special_tokens=False)
            if expansion is not None and text.count(marker) == len(expansion):
                return fast(expansion)
            inputs = build_multimodal_processor_inputs(
                self.processor, text=[text], images=image, mm_processor_kwargs=mm_kwargs)
            ids = inputs['input_ids'][0].tolist()
            marker_id = self.tokenizer.convert_tokens_to_ids(marker)
            counts = []
            in_run = False
            for token in ids:
                if token == marker_id:
                    if not counts or not in_run: counts.append(0)
                    counts[-1] += 1
                    in_run = True
                else:
                    in_run = False
            if len(counts) == len(image) and text.count(marker) == len(counts) and fast(counts) == ids:
                self._image_expansions[cache_key] = counts
            return ids
        return await self.loop.run_in_executor(None, encode)

    async def _score_branch(self, *, start, branch, teacher_prompt, image, mm_kwargs,
                            routing_key, history, command, legal, reason, expected,
                            counters, semaphore, student_prompt_ids=None, student_route=None, repair_support=None):
        contexts = teacher_prompt
        # Warm only the original-image history, before the appended thumbnail.
        # Different thumbnail hashes must never contaminate the shared prefix.
        vision_start = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        if not hasattr(self, '_teacher_cache_namespace'):
            self._teacher_cache_namespace = uuid4().hex
        def build_component_schedule():
            # Filter action choices only; object prefixes prune components, not logits.
            text = self._decode(branch)
            action_end = text.find('>') + 1
            boxes = list(sg.BOX_RE.finditer(text.split('<text>', 1)[0]))
            selected_kind = contexts[0]['selected_key'][0]
            selected_source = contexts[0]['selected_source']
            selected_key = contexts[0]['selected_key']
            fixed = [(m.start(), m.end()) for m in re.finditer(
                r'<\|(?:object_ref_start|object_ref_end|box_start|box_end)\|>|</(?:enter|detect|read|ground|text|link_from|link_to)>|<text>|<link_to>', text)]
            # Do not force markup occurring inside the free-text payload.
            payload_start = text.find('<text>')
            payload_end = text.rfind('</text>')
            if payload_start >= 0:
                fixed = [(a,b) for a,b in fixed if b <= payload_start + 6 or a >= payload_end]
            masks, active_rows = [], []
            action_rows = {}
            actions = {c['support'].command.split('>', 1)[0] + '>' for c in contexts}
            encoded_actions = [self.tokenizer.encode(action, add_special_tokens=False) for action in actions]
            previous = 0
            payload_active = None
            payload_boxes = ([ (m.start() + len(sg.BOX_START), m.end() - len(sg.BOX_END))
                               for m in sg.BOX_RE.finditer(text) if m.start() > payload_start ]
                             if selected_kind == 'ground' and payload_start >= 0 else [])
            def payload_mask(begin, end, token):
                return None if any(begin < b and end > a for a, b in payload_boxes) else [token]
            for i, token in enumerate(branch):
                # After the selected head, canonical content and closing syntax
                # are deterministic. Reuse the last head's component set and avoid
                # quadratic prefix decoding / repeated geometry for long captions.
                if payload_active is not None:
                    active_rows.append(payload_active)
                    if payload_boxes:
                        end = len(self._decode(branch[:i+1]))
                        masks.append(payload_mask(previous, end, token))
                        previous = end
                    else:
                        masks.append([token])
                    continue
                end = len(self._decode(branch[:i+1]))
                active = set()
                prefix = text[:previous].split('<text>', 1)[0]
                starts = [m.end() for m in re.finditer(re.escape(sg.BOX_START), prefix)]
                partial_boxes = [prefix[start:].split('<', 1)[0] for start in starts]
                for j, context in enumerate(contexts):
                    key = context['support'].key
                    if previous >= action_end and key[0] != selected_kind:
                        continue
                    if boxes and previous >= boxes[0].end():
                        if context['source'] != selected_source:
                            continue
                    if len(boxes) > 1 and previous >= boxes[1].end() and key != selected_key:
                        continue
                    # Consume only tokens BEFORE the predictor being supervised.
                    targets = context['support'].targets
                    possible = True
                    for box_index, partial in enumerate(partial_boxes):
                        if box_index >= len(targets):
                            possible = False; break
                        threshold = self.iou_threshold if key[0] == 'detect' else 0.95
                        if not can_match_box(partial, targets[box_index], threshold):
                            possible = False; break
                    if possible:
                        active.add(j)
                if not active:
                    raise RuntimeError(f'No teacher at semantic selection boundary: {command!r}')
                active_rows.append(active)
                # Resolve the completed head once, including a token that spans
                # box_end and text markup, before reusing its surviving components.
                if (getattr(self, 'correctness_loss', False) and payload_start >= 0
                        and previous >= payload_start + 6):
                    payload_active = active
                if end > previous and any(a <= previous and end <= b for a,b in fixed):
                    support_mask = [token]
                elif previous < action_end:
                    # Action tokens use the surviving executable choices.
                    # Correctness mode also conditions object/reference tokens below.
                    support_mask = sorted({encoded[i] for encoded in encoded_actions
                        if i < len(encoded) and encoded[:i] == branch[:i]})
                    if not support_mask:
                        raise RuntimeError(f'Empty decision support at {text[:previous]!r}')
                    if (token in support_mask['exclude'] if isinstance(support_mask, dict)
                            else token not in support_mask):
                        raise RuntimeError(f'Accepted command outside decision support at {text[:previous]!r}')
                    action_rows[i] = support_mask
                elif getattr(self, 'correctness_loss', False):
                    # GT text is canonicalized before execution. Once its head
                    # is selected, no teacher guesses are needed for the payload.
                    if payload_start >= 0 and previous >= payload_start + 6:
                        support_mask = payload_mask(previous, end, token)
                    else:
                        # Object candidates are checked after sparse model scoring.
                        # Never walk the entire vocabulary to construct this set.
                        support_mask = None
                else:
                    support_mask = None
                masks.append(support_mask)
                previous = end
            return masks, active_rows, action_rows

        try:
            masks, active_rows, action_rows = await self.loop.run_in_executor(None, build_component_schedule)
        except BaseException:
            ready_jobs = [c['ready'] for c in contexts if 'ready' in c]
            for job in ready_jobs:
                job.cancel()
            await asyncio.gather(*ready_jobs, return_exceptions=True)
            raise

        object_positions = [i for i, mask in enumerate(masks) if mask is None]
        repair_support = (repair_support or {}) if getattr(self, 'correctness_loss', False) else {}
        for position, allowed in repair_support.items():
            masks[position] = allowed
        # Preserve the full readiness barrier and length preflight, but overlap
        # support construction with image/prompt preparation rather than serializing.
        await asyncio.gather(*(c['ready'] for c in contexts))
        for context in contexts:
            starts = [i for i, token in enumerate(context['ids']) if token == vision_start]
            context['prefill'] = (dict(length=starts[1], image_count=1,
                namespace=self._teacher_cache_namespace) if len(starts) == 2 else None)

        async def score(component_index, context):
            # A component participates through its divergence position, then exits.
            # Keep raw legal mass so aggregation conditions the mixture only once.
            count = len(branch)
            if 'ready' in context:
                await context['ready']
            key = self.teacher_server_manager._resolve_teacher_key(routing_key)
            limit = self.teacher_server_manager.teacher_model_configs[key].inference.max_model_len
            if limit is not None and len(context['ids']) + count + 1 > limit:
                return None  # The producer rejects this entire command.
            positions = list(range(len(context['ids'])-1, len(context['ids'])+count-1))
            sequence = context['ids'] + branch[:count]
            result = {}
            action_support = {str(positions[i]): mask for i, mask in action_rows.items()}
            pending = []
            for i, mask in enumerate(masks):
                if i in repair_support:
                    continue
                if component_index not in active_rows[i]:
                    continue
                if isinstance(mask, list) and len(mask) == 1:
                    # Aggregation constructs singleton targets once per token;
                    # no per-component duplicate result is consumed.
                    counters['teacher_deterministic_rows'] += 1
                    counters['teacher_scored_rows'] += 1
                else:
                    pending.append(i)
            if not pending:
                counters['teacher_skipped_components'] += 1
            sparse_mode = getattr(self, 'correctness_loss', False)
            k = min(self.teacher_topk_initial, self.teacher_topk_max)
            while pending:
                async with semaphore:
                    if any('ready' in c and not c['ready'].done() for c in contexts):
                        counters['teacher_submissions_during_preparation'] += 1
                    ids, logs, sampled = await self.teacher_server_manager.compute_teacher_logprobs_single(
                        sequence_ids=sequence[:len(context['ids'])+max(pending)+1],
                        multi_modal_data={'images':[image, context['image']]}, mm_processor_kwargs=mm_kwargs,
                        routing_key=routing_key, request_id=uuid4().hex,
                        positions=[positions[i] for i in pending],
                        allowed_ids=None,
                        topk_override=k, return_sampled=True, raw_legal_probabilities=not sparse_mode,
                        shared_prefill=context['prefill'], action_support=action_support)
                counters['scoring_requests'] += 1
                retry = []
                # Convert whole batches once instead of invoking tiny torch
                # kernels and Python conversions for every scored position.
                id_rows, log_rows = ids.tolist(), logs.tolist()
                legal_masses = torch.as_tensor(sampled, dtype=logs.dtype, device=logs.device)
                masses = [] if sparse_mode else (logs - legal_masses[:, None]).exp().sum(-1).tolist()
                legal_rows = legal_masses.tolist()
                for row, i in enumerate(pending):
                    if sparse_mode:
                        # Keep the accepted token even if it fell outside teacher top-k.
                        values = dict(zip(id_rows[row], log_rows[row]))
                        values[branch[i]] = legal_rows[row]
                        result[i] = (list(values), list(values.values()), 0.)
                        counters['teacher_scored_rows'] += 1
                        continue  # Fixed budget: no adaptive full-prefix rescoring.
                    result[i] = (id_rows[row], log_rows[row], legal_rows[row])
                    mass = masses[row]
                    if mass < self.teacher_min_mass and k < self.teacher_topk_max:
                        retry.append(i)
                    elif mass < self.teacher_min_mass:
                        counters['teacher_tail_limit_rows'] += 1
                    if not retry or i not in retry:
                        counters['teacher_missing_mass_sum'] += max(0.,1-mass)
                        counters['teacher_scored_rows'] += 1
                pending = retry
                if pending:
                    counters['teacher_topk_expansions'] += 1
                k = min(k*4, self.teacher_topk_max)
            return result

        jobs = [asyncio.create_task(score(j, c)) for j, c in enumerate(contexts)]
        try:
            results = await asyncio.gather(*jobs)
        except BaseException:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise
        if any(result is None for result in results):
            return None
        if not hasattr(self, '_diagnostic_action_encodings'):
            self._diagnostic_action_encodings = [
                self.tokenizer.encode(tag, add_special_tokens=False) for tag in sg.ACTION_TAGS]
        # Exclude common syntax such as '<', but retain a single legal option
        # when other DSL actions are possible lexically and illegal in this state.
        decision_support = {i: options for i, options in action_rows.items()
            if len({encoded[i] for encoded in self._diagnostic_action_encodings
                    if i < len(encoded) and encoded[:i] == branch[:i]}) > 1}
        def aggregate():
            from collections import Counter
            deltas = Counter()
            out_ids, out_logs = [], []
            deterministic_cache = {}
            for position in range(len(branch)):
                if position in repair_support:
                    # Placeholder only: executor NLL replaces teacher KL here.
                    ids, logs, _ = mix_sparse([([branch[position]], [0.])], [1.], self.topk)
                    out_ids.append(ids); out_logs.append(logs)
                    continue
                if isinstance(masks[position], list) and len(masks[position]) == 1:
                    token = masks[position][0]
                    if token not in deterministic_cache:
                        deterministic_cache[token] = mix_sparse([([token], [0.])], [1.], self.topk)[:2]
                    ids, logs = deterministic_cache[token]
                    out_ids.append(ids); out_logs.append(logs)
                    deltas['teacher_active_components_sum'] += len(active_rows[position])
                    continue
                rows = [r[position] for r in results if position in r]
                if not rows:
                    raise RuntimeError(f'No active teacher component at position {position}')
                # All components enter with coefficient one. Normalize once by the
                # sum of exact legal probability masses, never by prefix posteriors.
                largest = max(row[2] for row in rows)
                if not math.isfinite(largest):
                    raise RuntimeError('Nonfinite shared teacher legal mass')
                normalizer = largest + math.log(sum(math.exp(row[2]-largest) for row in rows))
                normalized = [(ids, [p-normalizer for p in logs]) for ids, logs, _ in rows]
                ids, logs, mass = mix_sparse(normalized, [1.] * len(rows), self.topk)
                out_ids.append(ids); out_logs.append(logs)
                deltas['teacher_active_components_sum'] += len(rows)
                deltas['teacher_mixture_missing_mass_sum'] += max(0.,1-mass)
                deltas['teacher_output_tail_mass_sum'] += max(0., 1-math.fsum(math.exp(p) for p in logs))
            deltas['supervised_tokens'] += len(branch)
            output = dict(start=start, branch=branch, legal=legal,
                        action_support=decision_support,
                        teacher_ids=torch.tensor(out_ids, dtype=torch.long),
                        teacher_logprobs=torch.tensor(out_logs, dtype=torch.float32))
            if getattr(self, 'correctness_loss', False):
                output['correct_support'] = masks
                output['object_positions'] = object_positions
                output['repair_positions'] = list(repair_support)
            return output, deltas
        output, deltas = await self.loop.run_in_executor(None, aggregate)
        counters.update(deltas)
        return output

    async def _command_context(self, *, executor, command, original, messages, history, mm_kwargs, frozen=False):
        expected = signature(executor, command)
        if expected is not None:
            accepted, reason = command, 'direct_target'
        else:
            accepted, expected, reason = select_target(executor, command)
            if accepted is None:
                return None, None, 'no_target', original, None
        snapshot = executor if frozen else copy.deepcopy(executor)
        contexts = []
        history_text = self._decode(history)
        if not hasattr(self, '_target_image_cache'):
            self._target_image_cache = {}
        preparation_slots = asyncio.Semaphore(min(self.teacher_max_inflight, 16))
        async def prepare(context, candidate):
            async with preparation_slots:
                def thumbnail():
                    from verl.trainer.manga.privileged_thumbnail import render_thumbnail
                    return render_thumbnail(original, context['boxes'], context['support'].key[0])
                image_key = (context['support'].key[0], context['boxes'])
                task = self._target_image_cache.get(image_key)
                if task is None:
                    task = asyncio.ensure_future(self.loop.run_in_executor(None, thumbnail))
                    self._target_image_cache[image_key] = task
                context['image'] = await asyncio.shield(task)
                context['ids'] = await self._privileged_prompt(
                    messages, history_text, [original, context['image']], mm_kwargs,
                    hint=candidate, key=context['support'].key)
        # Publish immutable supports before prompt preparation finishes. Mask
        # construction needs supports only, and each scorer awaits its own prompt.
        candidates = frontier(snapshot)
        def source(command, key):
            if key[0] != 'link':
                return key
            match = sg.LINK_RE.fullmatch(command)
            mapping = (snapshot.state.text_by_student_box if match.group(1) == 'text'
                       else snapshot.state.character_by_student_box)
            return ('link', match.group(1), snapshot.resolve_reference(mapping, sg._match_box(match, 2)))
        selected_source = source(accepted, expected)
        if expected[:2] == ('link', 'identity'):
            oriented = []
            selected_index = selected_source[2]
            for candidate in candidates:
                key = signature(snapshot, candidate)
                if key[:2] == ('link', 'identity') and selected_index in key[2:]:
                    match = sg.LINK_RE.fullmatch(candidate)
                    if source(candidate, key)[2] != selected_index:
                        first = candidate[len('<link_from>'):candidate.index('</link_from>')]
                        second = candidate[candidate.index('<link_to>')+len('<link_to>'):candidate.index('</link_to>')]
                        candidate = '<link_from>' + second + '</link_from><link_to>' + first + '</link_to>'
                oriented.append(candidate)
            candidates = oriented
        for candidate in candidates:
            support = Support(snapshot, candidate, alias=accepted)
            box_text = candidate.split('<text>', 1)[1] if support.key[0] == 'ground' else candidate
            boxes = tuple(dict.fromkeys(sg._match_box(m, 0) for m in sg.BOX_RE.finditer(box_text)))
            context = dict(support=support, image=original, boxes=boxes,
                           source=source(candidate, support.key), selected_source=selected_source, selected_key=expected)
            contexts.append(context)
        for context, candidate in zip(contexts, candidates, strict=True):
            context['ready'] = asyncio.create_task(prepare(context, candidate))
        return accepted, expected, reason, original, contexts

    async def _sample_command(self, prompt_ids, history, images, mm_kwargs, params, priority, route_id,
                              remaining, prefix=None):
        content = list(prefix or [])
        generated = None
        text = self._decode(content)
        if command_boundary(text):
            return generated, content, False
        # One request per command; repair prefixes count toward the same cap.
        budget = min(remaining, COMMAND_MAX_TOKENS) - len(content)
        if budget <= 0:
            return generated, content, True
        generated = await self._generate(
            prompt_ids, history + content, images, None, None, mm_kwargs,
            dict(params, n=1), max_tokens=budget, stop=COMMAND_STOPS,
            priority=int(priority), routing_request_id=route_id)
        ids = list(generated.token_ids)
        content.extend(ids)
        ended = (not ids or self.tokenizer.eos_token_id in ids
                 or command_boundary(self._decode(content)))
        return generated, content, not ended

    async def _repair(self, executor, raw_ids, prompt_ids, history, images, mm_kwargs,
                      sampling_params, priority, route_id, remaining, counters):
        supports = [Support(executor, c) for c in frontier(executor)]
        selected, _, reason = select_target(executor, self._decode(raw_ids).strip())
        if selected is None:
            return None
        # Repair work is not final sequence length: discarded suffixes also cost
        # decoding time. Bound retries separately from the 896-token command cap.
        max_rounds = 16
        token_budget = 2 * COMMAND_MAX_TOKENS
        generated_tokens = 0
        seen = set()
        best_prefix = -1
        stagnant = 0
        rounds = 0
        def bounded_fallback(cause):
            counters['repair_' + cause] += 1
            counters['gt_fallbacks'] += 1
            counters['fallback_' + reason] += 1
            logger.warning("Bounded OPSD repair route=%s cause=%s rounds=%d generated_tokens=%d best_prefix=%d command_tokens=%d",
                           route_id, cause, rounds, generated_tokens, best_prefix, len(raw_ids))
            return selected
        for repair_round in range(min(remaining, max_rounds)):
            if generated_tokens >= token_budget:
                return bounded_fallback('token_budget')
            state = tuple(raw_ids)
            if state in seen:
                return bounded_fallback('no_progress')
            seen.add(state)
            # Locate the first impossible prefix, not the first non-GT coordinate.
            prefix = []
            for token in raw_ids:
                candidate = prefix + [token]
                if not any(s.viable(self._decode(candidate)) for s in supports):
                    break
                prefix = candidate
            else:
                # An incomplete/overlong payload has no impossible token prefix.
                # There is no sound token-local retry point; use the selected target.
                if raw_ids:
                    counters['gt_fallbacks'] += 1
                    counters['fallback_' + reason] += 1
                    return selected
            counters['retry_attempts'] += 1
            # Draw once from the student's conditional legal distribution. No
            # top-k lookup is used to decide whether a legal token exists.
            vocabulary = vocabulary_for(self.tokenizer)
            allowed = await self.loop.run_in_executor(None, vocabulary.union_mask, supports, self._decode(prefix))
            if isinstance(allowed, dict):
                excluded = set(allowed['exclude'])
                allowed = [i for i in range(allowed['vocab_size']) if i not in excluded]
            while not allowed and prefix:
                prefix.pop()
                allowed = await self.loop.run_in_executor(None, vocabulary.union_mask, supports, self._decode(prefix))
                if isinstance(allowed, dict):
                    excluded = set(allowed['exclude'])
                    allowed = [i for i in range(allowed['vocab_size']) if i not in excluded]
            if not allowed:
                counters['gt_fallbacks'] += 1
                counters['fallback_' + reason] += 1
                return selected
            # Use the actual prefix after any rollback, not the optimistic scan.
            if len(prefix) > best_prefix:
                best_prefix, stagnant = len(prefix), 0
            else:
                stagnant += 1
                if stagnant >= 4:
                    return bounded_fallback('no_progress')
            rounds += 1
            # Capture exact raw predictor tokens BEFORE sampling applies its mask.
            # Only actual rejected tokens qualify; truncated payloads do not.
            if len(prefix) < len(raw_ids) and raw_ids[len(prefix)] not in allowed:
                if not hasattr(self, '_repair_records'):
                    self._repair_records = []
                self._repair_records.append((tuple(history + prefix), tuple(allowed)))
            if len(allowed) == 1:
                chosen = allowed
                counters['student_forced_tokens'] += 1
            else:
                sampling = dict(sampling_params, n=1, allowed_token_ids=allowed, top_k=-1, top_p=1.0)
                generated = await self._generate(prompt_ids, history + prefix, images, None, None,
                    mm_kwargs, sampling, max_tokens=1, stop=[], priority=int(priority), routing_request_id=route_id)
                chosen = list(generated.token_ids)
                generated_tokens += len(chosen)
                counters['repair_generated_tokens'] += len(chosen)
                if len(chosen) != 1 or chosen[0] not in allowed:
                    raise RuntimeError('Student legal-token sampling returned an invalid token')
            if generated_tokens >= token_budget:
                return bounded_fallback('token_budget')
            retained = len(prefix) + len(chosen)
            # _sample_command subtracts prefix length from remaining. Limit only
            # newly generated suffix tokens; never charge the retained history.
            repair_remaining = min(remaining, retained + token_budget - generated_tokens)
            _, retried, truncated = await self._sample_command(prompt_ids, history, images, mm_kwargs,
                sampling_params, priority, route_id, repair_remaining, prefix=prefix + chosen)
            produced = max(0, len(retried) - retained)
            generated_tokens += produced
            counters['repair_generated_tokens'] += produced
            text = self._decode(retried).strip()
            if not truncated and signature(executor, text) is not None:
                counters['retry_successes'] += 1
                return text
            # Preserve the repaired prefix and repair any later illegal choice.
            raw_ids = retried
        return bounded_fallback('round_limit')

    async def run(self, sampling_params, priority=0, validate=False, **kwargs):
        # Validation is the original, unshielded student rollout with no teacher.
        from collections import Counter
        started = time.perf_counter()
        messages = list(kwargs["raw_prompt"])
        mm_data = await self.process_multi_modal_info(messages)
        images = mm_data.get("images")
        if not images or len(images) != 1:
            raise ValueError("Corrective OPSD requires exactly one page image")
        original = images[0] if isinstance(images[0], Image.Image) else Image.fromarray(images[0])
        mm_kwargs = self._get_mm_processor_kwargs(mm_data.get("audios"))
        prompt_ids = await self.apply_chat_template(messages, images=images,
                                                    mm_processor_kwargs=mm_kwargs)
        executor = MangaExecutor(TargetGraph.parse(_resolve(kwargs, self.target_key)), self.iou_threshold)
        routing = _python_value(kwargs.get(self.config.distillation.teacher_key))
        routing = str(routing) if routing is not None else None
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Boundary supervision requires an EOS token")
        newline = self.tokenizer.encode("\n", add_special_tokens=False)
        if not newline or self._decode(newline) != "\n":
            raise ValueError("Tokenizer must encode a standalone newline")
        assistant_suffix = ""
        if not validate:
            marker = "MANGA_ASSISTANT_CONTENT_SENTINEL"
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": marker}],
                tokenize=False, add_generation_prompt=False)
            if rendered.count(marker) != 1:
                raise ValueError("Cannot locate assistant content in chat template")
            assistant_suffix = rendered.split(marker, 1)[1]
            if eos not in self.tokenizer.encode(assistant_suffix, add_special_tokens=False):
                raise ValueError("Assistant template must contain the configured EOS")
        dependency_graph(executor)
        history, counters = [], Counter()
        self._perf_counters = counters
        self._target_image_cache = {}
        self._image_expansions = {}
        response_text, command_ranges = "", []
        semaphore = asyncio.Semaphore(self.teacher_max_inflight)
        score_jobs = []
        next_sample = None
        context_job = None
        command_count = 0
        has_eos = False
        min_step = max_step = None
        route_id = uuid4().hex

        def set_response(text):
            nonlocal response_text, history
            candidate_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(candidate_ids) > self.response_length:
                counters["budget_skipped"] += 1
                return False
            response_text, history = text, candidate_ids
            return True

        try:
            while (command_count < self.max_commands or executor.complete()) and len(history) < self.response_length:
                if validate:
                    generated = await self._generate(prompt_ids, history, images, None, None, mm_kwargs,
                        dict(sampling_params,n=1), max_tokens=self.response_length-len(history), stop=COMMAND_STOPS,
                        priority=int(priority), routing_request_id=route_id)
                    sampled_ids, truncated = list(generated.token_ids), False
                elif next_sample is not None:
                    generated, sampled_ids, truncated = await next_sample
                    next_sample = None
                else:
                    generated, sampled_ids, truncated = await self._sample_command(
                        prompt_ids, history, images, mm_kwargs, sampling_params, priority, route_id,
                        self.response_length-len(history))
                if generated is None:
                    break
                step_min = generated.extra_fields.get("min_global_steps", generated.extra_fields.get("global_steps"))
                step_max = generated.extra_fields.get("max_global_steps", generated.extra_fields.get("global_steps"))
                if step_min is not None:
                    min_step = step_min if min_step is None else min(min_step, step_min)
                if step_max is not None:
                    max_step = step_max if max_step is None else max(max_step, step_max)
                raw_ids = sampled_ids
                ended = eos in raw_ids
                has_eos |= ended
                command_ids = raw_ids[:raw_ids.index(eos)] if ended else raw_ids
                command = self._decode(command_ids).strip()
                counters["student_requests"] += 1
                if validate:
                    history.extend(raw_ids)
                    if command:
                        executor.execute(command)
                        command_count += 1
                    if ended or not raw_ids:
                        break
                    continue
                if executor.complete():
                    counters["terminal_boundaries"] += 1
                    counters["terminal_errors"] += int(not ended or bool(command))
                    set_response(response_text + assistant_suffix)
                    break
                self._repair_records = []
                original_command = command
                # Inspect the unmodified proposal, before EOS recovery or repair.
                proposal_kind = executor.action_kind(original_command)
                for name, value in executor.proposal_diagnostics(original_command).items():
                    counters[f"proposal_{proposal_kind}_{name}"] += value
                if not command:
                    if not raw_ids:
                        counters["empty_generation"] += 1
                        break
                    counters['early_eos'] += int(ended)
                    counters['continue_boundaries'] += 1
                    counters['continue_errors'] += int(ended)
                    if not set_response(response_text + '\n'):
                        break
                    # Ban EOS during this recovery; do not repeat unbounded EOS turns.
                    repaired = await self._repair(executor, [], prompt_ids, history, images, mm_kwargs,
                        sampling_params, priority, route_id, self.response_length-len(history), counters)
                    if repaired is None:
                        break
                    command = repaired
                    truncated = False
                    ended = False  # The early EOS was already counted at its boundary.
                legal = signature(executor, original_command) is not None
                kind = executor.action_kind(original_command)
                counters[f"{kind}_sampled"] += 1
                counters[f"{kind}_illegal"] += int(not legal)
                if kind == "enter" and "enter" not in executor.frontier_kinds():
                    counters["enter_dependency_blocked"] += 1
                if truncated:
                    counters['command_timeouts'] += 1
                if signature(executor, command) is None or truncated:
                    command = await self._repair(executor, command_ids, prompt_ids, history, images, mm_kwargs,
                        sampling_params, priority, route_id, self.response_length-len(history), counters)
                    if command is None:
                        counters['no_target'] += 1
                        break
                if getattr(self, 'correctness_loss', False):
                    from verl.trainer.manga.command_teacher import correct_content
                    corrected = correct_content(executor, command)
                    counters['content_corrected'] += int(corrected != command)
                    command = corrected
                pending_jobs = [job for job in score_jobs if not job.done()]
                if len(pending_jobs) >= self.teacher_max_inflight:
                    await asyncio.wait(pending_jobs, return_when=asyncio.FIRST_COMPLETED)
                for job in score_jobs:
                    if job.done():
                        job.result()  # Propagate background failures before more sampling.
                # Freeze the pre-command state before any execution. Prompt
                # preparation and the next student turn can then overlap safely.
                snapshot = copy.deepcopy(executor)
                hint, expected, reason = command, signature(executor, command), 'direct_target'
                if expected is None:
                    raise RuntimeError("Recovery did not produce an executable command")
                image = original
                context_job = asyncio.create_task(self._command_context(
                    executor=snapshot, command=command, original=original, messages=messages,
                    history=list(history), mm_kwargs=mm_kwargs, frozen=True))
                changed = original_command != hint
                counters["proposals"] += 1
                counters["legal_proposals"] += int(legal)
                counters["exact_proposals"] += int(not changed)
                counters["legal_retained"] += int(legal and not changed)
                counters[reason] += 1
                counters[f"{kind}_changed"] += int(changed)
                # Qwen's tokenizer applies NFC (e.g. decomposed Japanese dakuten).
                # Normalize BEFORE recording character offsets; otherwise both
                # round-trip equality and subsequent command ranges can disagree.
                command_text = hint.rstrip("\r\n")
                normalizer = self.tokenizer.backend_tokenizer.normalizer
                if normalizer is not None:
                    command_text = normalizer.normalize_str(command_text)
                candidate = response_text + command_text
                continuation, terminal = candidate + "\n", candidate + assistant_suffix
                command_ranges.append(dict(start_char=len(response_text),
                                           end_char=len(candidate), legal=legal))
                continuation_ids, continuation_commands, _ = self._encode_command_sequence(
                    continuation, command_ranges)
                # Only the accepted suffix needs command-offset alignment. Keep
                # whole-sequence encoding for the alternate budget check, since
                # concatenating separately encoded delimiters changes BPE tokens.
                terminal_ids = self.tokenizer.encode(terminal, add_special_tokens=False)
                if max(len(continuation_ids), len(terminal_ids)) > self.response_length:
                    counters["budget_skipped"] += 1
                    command_ranges.pop()
                    context_job.cancel()
                    await asyncio.gather(context_job, return_exceptions=True)
                    context_job = None
                    break
                if not executor.execute(hint):
                    raise RuntimeError(f"Corrective command was not executable: {hint}")
                finished = executor.complete()
                next_text = terminal if finished else continuation
                if finished:
                    next_ids, next_commands, _ = self._encode_command_sequence(terminal, command_ranges)
                else:
                    next_ids, next_commands = continuation_ids, continuation_commands
                # Speculate only the next unconditioned student command. Do not
                # execute it until this command passes teacher length preflight.
                if not finished and command_count + 1 < self.max_commands and len(next_ids) < self.response_length:
                    next_sample = asyncio.create_task(self._sample_command(
                        prompt_ids, next_ids, images, mm_kwargs, sampling_params, priority, route_id,
                        self.response_length-len(next_ids)))
                _, _, _, _, teacher_prompt = await context_job
                context_job = None
                branch_content = next_commands[-1]["branch"]
                from verl.trainer.manga.repair_supervision import align_repairs
                repair_support, rejected = align_repairs(
                    getattr(self, '_repair_records', ()), next_ids,
                    next_commands[-1]['start'], len(branch_content))
                counters['repair_supervision_dropped'] += rejected
                counters['repair_supervision_recorded'] += len(getattr(self, '_repair_records', ()))
                branch_job = asyncio.create_task(self._score_branch(
                    start=next_commands[-1]["start"], branch=branch_content,
                    teacher_prompt=teacher_prompt, image=image, mm_kwargs=mm_kwargs,
                    routing_key=routing, history=list(next_ids[:next_commands[-1]["start"]]), command=command,
                    legal=legal, reason=reason, expected=expected,
                    counters=counters, semaphore=semaphore,
                    student_prompt_ids=prompt_ids, student_route=route_id, repair_support=repair_support))
                score_jobs.append(branch_job)
                # Length preflight gates acceptance, not ready-component submission.
                await asyncio.gather(*(c['ready'] for c in teacher_prompt))
                key = self.teacher_server_manager._resolve_teacher_key(routing)
                limit = self.teacher_server_manager.teacher_model_configs[key].inference.max_model_len
                if limit is not None and any(len(c['ids']) + len(branch_content) + 1 > limit for c in teacher_prompt):
                    counters["teacher_budget_skipped"] += 1
                    command_ranges.pop()
                    executor = snapshot
                    branch_job.cancel()
                    await asyncio.gather(branch_job, return_exceptions=True)
                    score_jobs.pop()
                    if next_sample is not None:
                        next_sample.cancel()
                        await asyncio.gather(next_sample, return_exceptions=True)
                        next_sample = None
                    break
                counters[f"{executor.action_kind(hint)}_executed"] += 1
                command_count += 1
                counters["early_eos"] += int(ended and not finished)
                response_text, history = next_text, next_ids
                if finished:
                    counters["terminal_boundaries"] += 1
                    counters["terminal_errors"] += int(not ended)
                    break
                counters["continue_boundaries"] += 1
                counters["continue_errors"] += int(ended or not self._decode(command_ids).endswith("\n"))
        except BaseException:
            for job in (next_sample, context_job):
                if job is not None:
                    job.cancel()
            await asyncio.gather(*(job for job in (next_sample, context_job) if job is not None),
                                 return_exceptions=True)
            for job in score_jobs:
                job.cancel()
            await asyncio.gather(*score_jobs, return_exceptions=True)
            raise
        if validate:
            return AgentLoopOutput(
                prompt_ids=prompt_ids, response_ids=history, response_mask=[1] * len(history),
                reward_score=float(executor.complete()), num_turns=2,
                multi_modal_data=mm_data, mm_processor_kwargs=mm_kwargs,
                precomputed_multi_modal_inputs=self.precomputed_multi_modal_inputs,
                metrics=AgentLoopMetrics(generate_sequences=time.perf_counter() - started))
        try:
            supervised = await asyncio.gather(*score_jobs)
        except BaseException:
            for job in score_jobs:
                job.cancel()
            await asyncio.gather(*score_jobs, return_exceptions=True)
            raise
        commands = [item for item in supervised if item is not None]
        history, expected_commands, boundary_positions = self._encode_command_sequence(
            response_text, command_ranges)
        if len(commands) != len(expected_commands):
            raise ValueError("Teacher scoring did not cover every canonical command")
        for scored, expected_command in zip(commands, expected_commands, strict=True):
            if scored["start"] != expected_command["start"] or scored["branch"] != expected_command["branch"]:
                raise ValueError("Teacher targets differ from final canonical tokenization")
        counters["mixed_boundary_tokens"] = sum(
            "\n" in self._decode([history[position]]) and
            self._decode([history[position]]).strip() != ""
            for position in boundary_positions)
        if not commands and not boundary_positions:
            raise RuntimeError(f"No command supervision in page: {dict(counters)}")
        from verl.trainer.manga.diagnostics import enabled, write_record
        if enabled():
            write_record("student_rollout", {
                "sample_index": kwargs.get("index"),
                "response_text": self._decode(history),
                "commands": [
                    {
                        "text": self._decode(item["branch"]),
                        "char_offset": 0,
                    }
                    for item in commands
                ],
                "has_eos": has_eos,
                "command_count": command_count,
                "supervised_command_count": len(commands),
                "illegal_proposals": sum(not item["legal"] for item in commands),
                "legal_proposals": sum(item["legal"] for item in commands),
                "supervised_tokens": sum(len(item["branch"]) for item in commands),
                "counters": dict(counters),
            })
        counters["episodes"] = 1
        counters["completed_rollouts"] = int(executor.complete())
        counters["command_limit_reached"] = int(command_count >= self.max_commands and not executor.complete())
        results = [self._pack_linear_output(prompt_ids, history, commands, mm_data, mm_kwargs, min_step, max_step, boundary_positions, counters)]
        elapsed = time.perf_counter() - started
        for output in results:
            output.metrics = AgentLoopMetrics(generate_sequences=elapsed / len(results))
        actor_tokens = sum(len(output.prompt_ids) + len(output.response_ids) for output in results)
        logger.warning("Intent OPSD sample=%s sequence_tokens=%d supervised_tokens=%d complete=%s counters=%s",
                       kwargs.get("index"), actor_tokens, counters["supervised_tokens"],
                       executor.complete(), dict(counters))
        return results

    def _encode_command_sequence(self, text, ranges):
        """Align teacher content and executor boundaries after whole-response encoding."""
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        commands, boundaries = [], []
        index = 0
        for item in ranges:
            start, end = item["start_char"], item["end_char"]
            while index < len(ids) and offsets[index][0] < start:
                boundaries.append(index)
                index += 1
            first = index
            while index < len(ids) and offsets[index][1] <= end and offsets[index][0] < end:
                index += 1
            if index > first:
                commands.append(dict(start=first, branch=ids[first:index], legal=item["legal"]))
        boundaries.extend(range(index, len(ids)))
        decoded = self._decode(ids)
        if decoded != text:
            mismatch = next((i for i, (a, b) in enumerate(zip(text, decoded)) if a != b),
                            min(len(text), len(decoded)))
            raise ValueError(
                f"Canonical OPSD encoding did not preserve response text at char {mismatch}: "
                f"input={text[max(0, mismatch-24):mismatch+48]!r}, "
                f"decoded={decoded[max(0, mismatch-24):mismatch+48]!r}")
        return ids, commands, boundaries

    def _pack_linear_output(self, prompt_ids, history, commands, mm_data, mm_kwargs, min_step, max_step, boundary_positions, counters):
        response = list(history)
        ids, logprobs = self._teacher_storage(len(prompt_ids), len(response))
        mask = [0] * len(response)
        command_types = [0] * len(response)
        command_starts = [False] * len(response)
        action_ids = [[-1] * len(response) for _ in range(5)]
        single_mask, multi_mask, correct_edges = [False] * len(response), [False] * len(response), []
        repair_mask = [False] * len(response)
        group_weights = [[0., 0., 0.] for _ in response]
        kinds = ("enter", "detect", "read", "link", "ground")
        for item in commands:
            start, size = item["start"], len(item["branch"])
            if any(mask[start:start + size]):
                raise ValueError("Two teacher targets mapped to the same causal predictor")
            mask[start:start + size] = [1] * size
            kind = sg.MangaExecutor.action_kind(self._decode(item["branch"]))
            if kind in kinds:
                command_types[start:start + size] = [kinds.index(kind) + 1] * size
            command_starts[start] = True
            if getattr(self, 'correctness_loss', False):
                for offset in item.get('repair_positions', ()):
                    repair_mask[start + offset] = True
                supports = item['correct_support']
                if len(supports) != size:
                    raise ValueError('Correctness supports must cover the whole command')
                for offset, options in enumerate(supports):
                    position = start + offset
                    single_mask[position] = options is not None and len(options) == 1
                    multi_mask[position] = not single_mask[position]
                    if options is not None and len(options) > 1:
                        correct_edges.extend((position, token) for token in options)
            for offset, options in item.get("action_support", {}).items():
                if len(options) > len(action_ids):
                    raise ValueError("Action diagnostic support exceeds five DSL actions")
                for column, token in enumerate(options):
                    action_ids[column][start + offset] = token
            if getattr(self, 'correctness_loss', False):
                # One object decision comprises its type and coordinate tokens.
                # Fixed markup is content/format, not a decision. Link endpoints
                # are two decisions; each caption reference is another object decision.
                object_groups = {}
                for offset in item.get('object_positions', ()):
                    prefix = self._decode(item['branch'][:offset])
                    object_index = prefix.count(sg.BOX_END)
                    object_groups.setdefault(object_index, []).append(start + offset)
                for positions in object_groups.values():
                    for position in positions:
                        group_weights[position][1] = 1. / len(positions)
                for offset in range(size):
                    position = start + offset
                    if offset in item.get('action_support', {}):
                        group_weights[position][0] = 1.
                    elif not group_weights[position][1]:
                        group_weights[position][2] = 1.
            causal = len(prompt_ids) + start - 1
            ids[causal:causal + size] = item["teacher_ids"]
            logprobs[causal:causal + size] = item["teacher_logprobs"]
        boundary_mask = [False] * len(response)
        for position in boundary_positions:
            boundary_mask[position] = True
        if any(a and b for a, b in zip(mask, boundary_mask)):
            raise ValueError("Command and boundary supervision must not overlap")
        boundary_targets = {}
        for position in boundary_positions:
            causal = len(prompt_ids) + position - 1
            token = response[position]
            if token not in boundary_targets:
                target_ids, target_logs, _ = mix_sparse([([token], [0.])], [1.], self.topk)
                boundary_targets[token] = (
                    torch.tensor(target_ids, dtype=ids.dtype, device=ids.device),
                    torch.tensor(target_logs, dtype=logprobs.dtype, device=logprobs.device))
            ids[causal], logprobs[causal] = boundary_targets[token]
            mask[position] = 1
            single_mask[position] = True
            group_weights[position][2] = 1.
        response_mask = [int(a or b) for a, b in zip(mask, boundary_mask)]
        # Observe delimiter dilution without changing gradient weights. Match
        # exact standalone special tokens; never classify action names as syntax.
        delimiter_ids = set()
        for text in ("<|box_start|>", "<|box_end|>", "<|object_ref_start|>", "<|object_ref_end|>"):
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1 and self._decode(encoded) == text:
                delimiter_ids.add(encoded[0])
        delimiter_mask = [bool(active and token in delimiter_ids) for active, token in zip(mask, response)]
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response, response_mask=response_mask,
            reward_score=0., num_turns=2, metrics=AgentLoopMetrics(),
            multi_modal_data=mm_data, mm_processor_kwargs=mm_kwargs,
            precomputed_multi_modal_inputs=self.precomputed_multi_modal_inputs,
            extra_fields={
                "teacher_ids": ids, "teacher_logprobs": logprobs,
                **({"manga_repair_stats": torch.tensor([counters['repair_supervision_recorded'],
                        counters['repair_supervision_dropped']], dtype=torch.long),
                    "manga_repair_mask": torch.tensor(repair_mask, dtype=torch.bool),
                    "manga_group_weights": torch.tensor(group_weights, dtype=torch.float32),
                    "manga_single_mask": torch.tensor(single_mask, dtype=torch.bool),
                    "manga_multi_mask": torch.tensor(multi_mask, dtype=torch.bool),
                    "manga_content_corrections": torch.tensor([counters['content_corrected']], dtype=torch.long),
                    "manga_correct_edges": torch.tensor(correct_edges, dtype=torch.long).reshape(-1, 2)}
                   if getattr(self, 'correctness_loss', False) else {}),
                "manga_opsd_mask": torch.tensor(mask, dtype=torch.bool),
                "manga_delimiter_mask": torch.tensor(delimiter_mask, dtype=torch.bool),
                "manga_boundary_mask": torch.tensor(boundary_mask, dtype=torch.bool),
                "manga_command_types": torch.tensor(command_types, dtype=torch.long),
                "manga_command_starts": torch.tensor(command_starts, dtype=torch.bool),
                **{f"manga_action_ids_{i}": torch.tensor(values, dtype=torch.long)
                   for i, values in enumerate(action_ids)},
                "manga_command_stats": torch.tensor(
                    [counters[f"{kind}_{metric}"] for kind in kinds
                     for metric in ("sampled", "illegal", "changed", "executed")]
                    + [counters["enter_dependency_blocked"]]
                    + proposal_stats(counters), dtype=torch.float32),
                "manga_boundary_stats": torch.tensor([counters[name] for name in (
                    "student_requests", "early_eos", "continue_boundaries", "continue_errors",
                    "terminal_boundaries", "terminal_errors", "mixed_boundary_tokens",
                    "teacher_budget_skipped", "budget_skipped")], dtype=torch.float32),
                "manga_intent_stats": torch.tensor([counters[name] for name in (
                    "proposals", "legal_proposals", "exact_proposals", "legal_retained",
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
                    "repair_token_budget", "repair_generated_tokens")],
                    dtype=torch.float32),
                "min_global_steps": min_step, "max_global_steps": max_step,
                "manga_opsd_precomputed": True,
            })
