"""Command-boundary corrective OPSD on unchanged student trajectories.

The teacher sees the probe's target hint/overlay and full raw history in a
follow-up conversation. Actor rows use the original student prompt/history.
Only the selected command is supervised; generated corrections never roll in.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from uuid import uuid4
from typing import Any

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from verl.trainer.manga import MangaExecutor, TargetGraph
from verl.trainer.manga.command_teacher import (
    sg, select_target, signature, hint_for, target_boxes, render_target_image,
    teacher_chat, dependency_graph,
)
from verl.utils.tokenizer import build_multimodal_processor_inputs
from verl.utils.tokenizer.chat_template import apply_chat_template

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
        if self.max_commands <= 0:
            raise ValueError("manga_max_commands must be positive")
        self.correction_tokens = int(cfg.manga_correction_max_tokens)

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

    def _split_full_response(self, token_ids):
        """Parse text independently; a boundary-spanning token goes to the next line."""
        eos = self.tokenizer.eos_token_id
        if eos is not None and eos in token_ids:
            token_ids = token_ids[:token_ids.index(eos) + 1]
        text = self._decode(token_ids)
        lines = text.splitlines(keepends=True)
        if not lines:
            return []
        ends, offset = [], 0
        for line in lines:
            offset += len(line)
            ends.append(offset)
        # Prefix lengths preserve the tokenizer's decoding semantics, including
        # UTF-8 fragments. Crossing tokens are not split or duplicated.
        boundaries = {}
        next_end = 0
        lengths = [0]
        for index in range(1, len(token_ids) + 1):
            length = len(self._decode(token_ids[:index]))
            lengths.append(length)
            while next_end < len(ends) and ends[next_end] <= length:
                end = ends[next_end]
                boundaries[end] = index if end == length else index - 1
                next_end += 1
        commands, start, char_start = [], 0, 0
        for line, end in zip(lines, ends, strict=True):
            stop = boundaries.get(end, len(token_ids))
            # The final line owns every remaining sampled token.
            if end == len(text):
                stop = len(token_ids)
            commands.append(dict(ids=token_ids[start:stop], text=line,
                                 char_offset=char_start - lengths[start]))
            start, char_start = stop, end
        return commands

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
        return await self.server_manager.generate(
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

    async def _privileged_prompt(self, messages, history, image, mm_kwargs, hint=None, terminal=False):
        # Unlike the student initial-prompt helper, do not cap this full-history
        # teacher conversation at data.max_prompt_length or overwrite actor pixels.
        chat = teacher_chat(messages, history, hint, terminal=terminal)
        def encode():
            text = apply_chat_template(self.processor, chat, add_generation_prompt=True,
                                       tokenize=False, **self.apply_chat_template_kwargs)
            inputs = build_multimodal_processor_inputs(
                self.processor, text=[text], images=[image], mm_processor_kwargs=mm_kwargs)
            return inputs["input_ids"][0].tolist()
        return await self.loop.run_in_executor(None, encode)

    async def _supervise_command(self, *, executor, command, raw_ids, history,
                                  messages, original, prompt_ids, mm_data, mm_kwargs,
                                  routing_key, min_step, max_step, counters):
        """Read-only on executor/history: return a separate actor branch or None."""
        eos = self.tokenizer.eos_token_id
        is_end = command in sg.CHAT_END_TOKENS
        legal = executor.complete() if is_end else signature(executor, command) is not None
        terminal = executor.complete()
        if is_end and not terminal:
            counters["illegal"] += 1
            branch = []
            teacher_ids_parts = []
            teacher_logprobs_parts = []
            teacher_history = self._decode(history)
            correction_executor = copy.copy(executor)
            correction_executor.state = copy.deepcopy(executor.state)
            selected, expected, reason = select_target(correction_executor, "")
            while not correction_executor.complete():
                if selected is None:
                    counters["no_target"] += 1
                    return None
                hint = hint_for(correction_executor, selected)
                image = await self.loop.run_in_executor(
                    None, lambda: render_target_image(
                        original, target_boxes(correction_executor, selected)))
                teacher_prompt = await self._privileged_prompt(
                    messages, teacher_history, image, mm_kwargs, hint=hint, terminal=False)
                step_branch = self.tokenizer.encode(hint + "\n", add_special_tokens=False)
                if not step_branch or len(history) + len(branch) + len(step_branch) > self.response_length:
                    counters["budget_skipped"] += 1
                    return None
                step_ids, step_logprobs = await self._teacher_targets(
                    teacher_prompt + step_branch, image, mm_kwargs, routing_key, uuid4().hex,
                    list(range(len(teacher_prompt) - 1,
                               len(teacher_prompt) + len(step_branch) - 1)))
                branch.extend(step_branch)
                teacher_ids_parts.append(step_ids)
                teacher_logprobs_parts.append(step_logprobs)
                correction_executor.execute(hint)
                teacher_history += hint + "\n"
                selected, expected, reason = select_target(correction_executor, "")

            if eos is None:
                raise ValueError("Cannot correct an illegal end command without an EOS token")
            terminal_prompt = await self._privileged_prompt(
                messages, teacher_history, original, mm_kwargs, hint=None, terminal=True)
            terminal_branch = [eos]
            if len(history) + len(branch) + len(terminal_branch) > self.response_length:
                counters["budget_skipped"] += 1
                return None
            terminal_ids, terminal_logprobs = await self._teacher_targets(
                terminal_prompt + terminal_branch, original, mm_kwargs, routing_key, uuid4().hex,
                list(range(len(terminal_prompt) - 1,
                           len(terminal_prompt) + len(terminal_branch) - 1)))
            branch.extend(terminal_branch)
            teacher_ids_parts.append(terminal_ids)
            teacher_logprobs_parts.append(terminal_logprobs)
            counters["corrected"] += bool(branch)
            counters["scoring_requests"] += len(teacher_ids_parts)
            counters["supervised_tokens"] += len(branch)
            teacher_ids = torch.cat(teacher_ids_parts, dim=0)
            teacher_logprobs = torch.cat(teacher_logprobs_parts, dim=0)
            output = dict(start=len(history), branch=branch, legal=False,
                          teacher_ids=teacher_ids, teacher_logprobs=teacher_logprobs)
            from verl.trainer.manga.diagnostics import enabled, write_record
            if enabled():
                write_record("command_correction", {
                    "history_ids": list(history), "student_command": command,
                    "branch_ids": branch, "corrected": True, "target_reason": reason,
                    "teacher_prompt_ids": terminal_prompt, "expected": expected,
                })
            return output
        hint = None
        if terminal:
            selected, expected, reason = None, None, "terminal"
            image = original
        else:
            selected, expected, reason = select_target(executor, command)
            if selected is None:
                counters["no_target"] += 1
                return None
            hint = hint_for(executor, selected)
            image = await self.loop.run_in_executor(
                None, lambda: render_target_image(original, target_boxes(executor, selected)))
        teacher_prompt = await self._privileged_prompt(
            messages, self._decode(history), image, mm_kwargs, hint=hint,
            terminal=terminal)
        if legal:
            branch = list(raw_ids)
        else:
            counters["illegal"] += 1
            if terminal:
                if eos is None:
                    raise ValueError("Cannot correct an illegal end command without an EOS token")
                branch = [eos]
            else:
                branch = self.tokenizer.encode(hint, add_special_tokens=False)
            counters["corrected"] += bool(branch)
        if not branch or len(history) + len(branch) > self.response_length:
            counters["budget_skipped"] += 1
            return None
        teacher_ids, teacher_logprobs = await self._teacher_targets(
            teacher_prompt + branch, image, mm_kwargs, routing_key, uuid4().hex,
            list(range(len(teacher_prompt) - 1, len(teacher_prompt) + len(branch) - 1)))
        counters["scoring_requests"] += 1
        counters["supervised_tokens"] += len(branch)
        output = dict(start=len(history), branch=branch, legal=legal,
                      teacher_ids=teacher_ids, teacher_logprobs=teacher_logprobs)
        from verl.trainer.manga.diagnostics import enabled, write_record
        if enabled():
            write_record("command_correction", {
                "history_ids": list(history), "student_command": command,
                "branch_ids": branch, "corrected": not legal, "target_reason": reason,
                "teacher_prompt_ids": teacher_prompt, "expected": expected,
            })
        return output

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
        generated = await self._generate(
            prompt_ids, [], images, None, None, mm_kwargs, dict(sampling_params, n=1),
            max_tokens=self.response_length, stop=[], priority=int(priority), routing_request_id=uuid4().hex)
        min_step = generated.extra_fields.get("min_global_steps", generated.extra_fields.get("global_steps"))
        max_step = generated.extra_fields.get("max_global_steps", generated.extra_fields.get("global_steps"))
        raw = list(generated.token_ids)
        eos = self.tokenizer.eos_token_id
        has_eos = eos is not None and eos in raw
        if has_eos:
            raw = raw[:raw.index(eos)]
        parts = await asyncio.to_thread(self._split_full_response, raw)
        if has_eos:
            parts.append({"ids": [eos], "text": self._decode([eos]), "char_offset": 0})
        # Match the historical full-response command cap: retain an unchanged
        # sampled prefix, without supervising or executing commands beyond it.
        command_count = 0
        for index, part in enumerate(parts):
            command = part["text"].strip()
            if command and command not in sg.CHAT_END_TOKENS:
                if command_count >= self.max_commands:
                    parts = parts[:index]
                    break
                command_count += 1
        retained_ids = [token for part in parts for token in part["ids"]]
        dependency_graph(executor)
        history, jobs, counters = [], [], Counter()
        semaphore = asyncio.Semaphore(self.teacher_max_inflight)

        async def supervise(snapshot, command, ids, prefix):
            async with semaphore:
                return await self._supervise_command(
                    executor=snapshot, command=command, raw_ids=ids, history=prefix,
                    messages=messages, original=original, prompt_ids=prompt_ids, mm_data=mm_data,
                    mm_kwargs=mm_kwargs, routing_key=routing, min_step=min_step, max_step=max_step,
                    counters=counters)

        for part in parts:
            command = part["text"].strip()
            aligned = part["char_offset"] == 0 and self._decode(part["ids"]) == part["text"]
            if command and aligned and not validate:
                # Clone only mutable progress; the immutable target graph is shared.
                snapshot = copy.copy(executor)
                snapshot.state = copy.deepcopy(executor.state)
                jobs.append(asyncio.create_task(supervise(snapshot, command, part["ids"], list(history))))
            elif command and not aligned:
                counters["boundary_skipped"] += 1
            if command not in sg.CHAT_END_TOKENS:
                executor.execute(command)
            history.extend(part["ids"])
        assert history == retained_ids, "Student history was rewritten"
        if validate:
            return AgentLoopOutput(
                prompt_ids=prompt_ids, response_ids=history, response_mask=[1] * len(history),
                reward_score=float(executor.complete()), num_turns=2,
                multi_modal_data=mm_data, mm_processor_kwargs=mm_kwargs,
                precomputed_multi_modal_inputs=self.precomputed_multi_modal_inputs,
                metrics=AgentLoopMetrics(generate_sequences=time.perf_counter() - started))
        try:
            supervised = await asyncio.gather(*jobs)
        except BaseException:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise
        commands = [item for item in supervised if item is not None]
        if not commands:
            raise RuntimeError(f"No command supervision in page: {dict(counters)}")
        from verl.trainer.manga.diagnostics import enabled, write_record
        if enabled():
            write_record("student_rollout", {
                "sample_index": kwargs.get("index"),
                "response_text": self._decode(retained_ids),
                "commands": [
                    {
                        "text": part["text"],
                        "char_offset": part["char_offset"],
                    }
                    for part in parts
                ],
                "has_eos": has_eos,
                "command_count": command_count,
                "supervised_command_count": len(commands),
                "corrected_commands": sum(not item["legal"] for item in commands),
                "original_legal_commands": sum(item["legal"] for item in commands),
                "supervised_tokens": sum(len(item["branch"]) for item in commands),
                "counters": dict(counters),
            })
        results = [self._pack_tree_output(prompt_ids, history, commands, mm_data, mm_kwargs, min_step, max_step)]
        elapsed = time.perf_counter() - started
        for output in results:
            output.metrics = AgentLoopMetrics(generate_sequences=elapsed / len(results))
        actor_tokens = sum(len(output.prompt_ids) + len(output.response_ids) for output in results)
        legal_ends = [item["start"] + len(item["branch"]) for item in commands if item["legal"]]
        independent_tokens = (len(prompt_ids) + max(legal_ends)) if legal_ends else 0
        independent_tokens += sum(len(prompt_ids) + item["start"] + len(item["branch"])
                                  for item in commands if not item["legal"])
        logger.warning("Corrective OPSD sample=%s page_tree_tokens=%d independent_branch_tokens=%d "
                       "shared_input_ratio=%.3f supervised_tokens=%d complete=%s diagnostics=%s",
                       kwargs.get("index"), actor_tokens, independent_tokens,
                       actor_tokens / max(independent_tokens, 1), counters["supervised_tokens"],
                       executor.complete(), dict(counters))
        return results

    def _pack_tree_output(self, prompt_ids, history, commands, mm_data, mm_kwargs, min_step, max_step):
        # One page = one differentiable tree. The original trajectory is the
        # trunk; corrective commands are leaves that never become later history.
        response = list(history)
        layout = [[0, 0, len(prompt_ids) + len(history)]]
        placements = []
        for item in commands:
            start, size = item["start"], len(item["branch"])
            if item["legal"]:
                placements.append((start, item, 0, size))
            else:
                suffix_start = len(response)
                response.extend(item["branch"])
                layout.append([len(prompt_ids) + start, len(prompt_ids) + suffix_start, size])
                # Token one is predicted by the shared branch-point hidden
                # state, not by the preceding physical token of the packed leaf.
                placements.append((start, item, 0, 1))
                if size > 1:
                    placements.append((suffix_start + 1, item, 1, size - 1))
        ids, logprobs = self._teacher_storage(len(prompt_ids), len(response))
        mask = [0] * len(response)
        for start, item, offset, size in placements:
            if any(mask[start:start + size]):
                raise ValueError("Two teacher targets mapped to the same causal predictor")
            mask[start:start + size] = [1] * size
            causal = len(prompt_ids) + start - 1
            ids[causal:causal + size] = item["teacher_ids"][offset:offset + size]
            logprobs[causal:causal + size] = item["teacher_logprobs"][offset:offset + size]
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response, response_mask=mask,
            reward_score=0., num_turns=2, metrics=AgentLoopMetrics(),
            multi_modal_data=mm_data, mm_processor_kwargs=mm_kwargs,
            precomputed_multi_modal_inputs=self.precomputed_multi_modal_inputs,
            extra_fields={
                "teacher_ids": ids, "teacher_logprobs": logprobs,
                "manga_opsd_mask": torch.tensor(mask, dtype=torch.bool),
                "manga_tree_layout": torch.tensor(layout, dtype=torch.long),
                "manga_correction_stats": torch.tensor(
                    [sum(not item["legal"] for item in commands),
                     sum(item["legal"] for item in commands), sum(mask)], dtype=torch.float32),
                "min_global_steps": min_step, "max_global_steps": max_step,
                "manga_opsd_precomputed": True,
            })
