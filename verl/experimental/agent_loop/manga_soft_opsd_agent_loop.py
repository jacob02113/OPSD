"""Raw student rollouts with frozen multi-component soft supervision."""
import asyncio
import copy
import time
from collections import Counter
from uuid import uuid4

import torch
from PIL import Image

from .agent_loop import AgentLoopOutput, AgentLoopMetrics, register
from .manga_corrective_opsd_agent_loop import MangaCorrectiveOPSDLoop, _resolve, _python_value
from verl.trainer.manga import MangaExecutor, TargetGraph
from verl.trainer.manga.command_teacher import frontier as privileged_frontier
from verl.trainer.manga.local_policy import COMMAND_STOPS, COMMAND_MAX_TOKENS, command_boundary
from verl.trainer.manga.raw_mixture import score, STATS


def _student_generation_sampling_params(sampling_params, temperature, validate):
    """Return generation-only params without mutating native actor settings."""
    if validate or temperature is None:
        return sampling_params
    params = dict(sampling_params)
    params['temperature'] = float(temperature)
    if temperature == 0:
        params['top_p'] = 1.0
        params['top_k'] = -1
    return params


@register('manga_soft_opsd_agent')
class MangaSoftOPSDLoop(MangaCorrectiveOPSDLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.distillation.manga_soft_distillation:
            raise ValueError('Soft OPSD loop requires manga_soft_distillation')
        if self.config.distillation.manga_retry_truncated:
            if self.config.trainer.v1.trainer_mode != 'sync' or self.rollout_config.n != 1:
                raise ValueError('Truncated-page retries require synchronous rollout with n=1')
        if self.config.distillation.manga_progress_weight:
            if self.config.trainer.v1.trainer_mode != 'sync':
                raise ValueError('Progress REINFORCE requires synchronous fresh rollouts')
            if self.config.actor_rollout_ref.actor.loss_agg_mode != 'token-mean':
                raise ValueError('The KL component requires token-mean normalization; progress uses command means')
            if (self.rollout_config.temperature != 1 or self.rollout_config.top_p != 1
                    or self.rollout_config.top_k not in (-1, 0)
                    or getattr(self.rollout_config, 'repetition_penalty', 1.) != 1):
                raise ValueError('Progress requires unmodified sampling: temperature=top_p=repetition_penalty=1, top_k=-1')
            generation_temperature = getattr(
                self.config.distillation, 'manga_student_generation_temperature', None)
            if generation_temperature not in (None, 1):
                raise ValueError('Progress policy objective cannot use a generation-only temperature')

    async def _score_raw(self, executor, branch, original, messages, history, mm_kwargs, routing, counters, semaphore, routed):
        try:
            return await self._prepare_and_score(executor, branch, original, messages, history,
                                                 mm_kwargs, routing, counters, semaphore, routed)
        except BaseException as exc:
            if not routed.done():
                routed.set_exception(exc)
            raise

    async def _prepare_and_score(self, executor, branch, original, messages, history, mm_kwargs, routing, counters, semaphore, routed):
        if executor.complete():
            context = dict(terminal=True)
            async def prepare():
                context['ids'] = await self._privileged_prompt(messages, self._decode(history),
                    [original], mm_kwargs, hint=None, terminal=True)
            context['ready'] = asyncio.create_task(prepare())
            contexts = [context]
        else:
            *_, contexts = await self._command_context(executor=executor, command='', original=original,
                messages=messages, history=history, mm_kwargs=mm_kwargs, frozen=True, raw_mixture=True)
        result = await score(self, branch, contexts, original, mm_kwargs, routing, counters, semaphore, routed)
        if hasattr(self, '_soft_failure_labels'):
            record = self._soft_failure_labels.pop(id(branch), None)
            if record is not None:
                self._soft_failure_records.append((len(history)+record[0], record[1]))
        return result

    async def run(self, sampling_params, priority=0, validate=False, **kwargs):
        started = time.perf_counter()
        generation_temperature = (None if validate else getattr(
            self.config.distillation, 'manga_student_generation_temperature', None))
        # Only trajectory collection is tempered. The original params continue
        # through rollout postprocessing, so actor/teacher KL stays at T=1.
        generation_sampling_params = _student_generation_sampling_params(
            sampling_params, generation_temperature, validate)
        messages = list(kwargs['raw_prompt'])
        mm_data = await self.process_multi_modal_info(messages)
        images = mm_data.get('images')
        if not images or len(images) != 1:
            raise ValueError('Manga OPSD requires one page image')
        original = images[0] if isinstance(images[0], Image.Image) else Image.fromarray(images[0])
        mm_kwargs = self._get_mm_processor_kwargs(mm_data.get('audios'))
        prompt = await self.apply_chat_template(messages, images=images, mm_processor_kwargs=mm_kwargs)
        route = uuid4().hex
        if validate:
            # No GT parsing, executor intervention, or teacher requests at evaluation.
            output = await self._generate(prompt, [], images, None, None, mm_kwargs, sampling_params,
                max_tokens=self.response_length, stop=[], priority=int(priority), routing_request_id=route)
            raw = list(output.token_ids)
            return AgentLoopOutput(prompt_ids=prompt, response_ids=raw, response_mask=[1]*len(raw),
                num_turns=1, metrics=AgentLoopMetrics(generate_sequences=time.perf_counter()-started),
                multi_modal_data=mm_data, mm_processor_kwargs=mm_kwargs,
                precomputed_multi_modal_inputs=self.precomputed_multi_modal_inputs)

        executor = MangaExecutor(TargetGraph.parse(_resolve(kwargs, self.target_key)), self.iou_threshold)
        routing = _python_value(kwargs.get(self.config.distillation.teacher_key))
        routing = None if routing is None else str(routing)
        counters = Counter(episodes=1)
        effective_generation_temperature = float(generation_sampling_params.get('temperature', 1.0))
        counters['student_generation_temperature_sum'] = effective_generation_temperature
        counters['student_greedy_episodes'] = int(effective_generation_temperature == 0)
        self._perf_counters = counters
        self._target_image_cache, self._image_expansions = {}, {}
        semaphore = asyncio.Semaphore(self.teacher_max_inflight)
        history, jobs, scored = [], [], []
        import os
        audit_interval = int(os.environ.get('MANGA_SOFT_FAILURE_AUDIT_STEP', '0'))
        if audit_interval < 0:
            raise ValueError('MANGA_SOFT_FAILURE_AUDIT_STEP must be a nonnegative audit interval')
        if audit_interval and self.config.trainer.v1.trainer_mode != 'sync':
            raise ValueError('Fixed failure audit requires synchronous training')
        self.__dict__.pop('_soft_failure_labels', None)
        self.__dict__.pop('_soft_failure_records', None)
        audit_step = kwargs.get('_manga_audit_step', kwargs.get('global_steps', 0))
        audit_active = bool(audit_interval > 0 and not validate and audit_step > 0
                            and audit_step % audit_interval == 0)
        anchor_active = bool(getattr(self.config.distillation, 'manga_failure_anchor_weight', 0.))
        if audit_active or anchor_active:
            self._soft_failure_labels, self._soft_failure_records = {}, []
        progress_weight = float(getattr(self.config.distillation, 'manga_progress_weight', 0.))
        progress_records = []
        if progress_weight:
            from verl.trainer.manga.progress import required_units, completed_units, rollout_fields
            total_units = required_units(executor)
        minimum = maximum = None
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError('Soft OPSD requires an EOS token')

        async def drain():
            for start, task in jobs:
                ids, logs = await task
                scored.append((start, ids, logs))
            jobs.clear()

        try:
            while len(history) < self.response_length:
                if counters['commands'] >= self.max_commands and not executor.complete():
                    counters['command_limit'] = 1
                    break
                output = await self._generate(prompt, history, images, None, None, mm_kwargs,
                    generation_sampling_params,
                    max_tokens=min(COMMAND_MAX_TOKENS, self.response_length-len(history)),
                    stop=COMMAND_STOPS, priority=int(priority), routing_request_id=route)
                raw = list(output.token_ids)
                if not raw:
                    break
                low = output.extra_fields.get('min_global_steps', output.extra_fields.get('global_steps'))
                high = output.extra_fields.get('max_global_steps', output.extra_fields.get('global_steps'))
                if low is not None:
                    minimum = low if minimum is None else min(minimum, low)
                if high is not None:
                    maximum = high if maximum is None else max(maximum, high)
                routed = asyncio.get_running_loop().create_future()
                jobs.append((len(history), asyncio.create_task(self._score_raw(copy.deepcopy(executor),
                    raw, original, messages, list(history), mm_kwargs, routing, counters, semaphore, routed))))
                covered = await routed  # CPU routing only; teacher scoring overlaps the next rollout.
                history.extend(raw)  # No decode/re-encode, GT insertion, forced newline or EOS.
                ended = eos in raw
                decoded = self._decode(raw[:raw.index(eos)] if ended else raw)
                text = decoded.strip()
                # Boundary recognition is text based.  In Qwen tokenization the
                # final ``>\n`` may be one token, so a newline token-id check is
                # neither necessary nor correct.
                delimited = command_boundary(decoded)
                was_complete = executor.complete()
                before = completed_units(executor) if progress_weight else 0
                failed = False
                executed = False
                if text:
                    counters['commands'] += 1
                    if was_complete or not executor.execute(text):
                        counters['illegal_commands'] += 1
                        failed = True
                    else:
                        executed = True
                if progress_weight:
                    progress_records.append((len(history)-len(raw), len(history), before, completed_units(executor)))
                if ended:
                    counters['early_eos'] += int(not executor.complete())
                    break
                if executed and not executor.complete() and not privileged_frontier(executor):
                    # ``complete`` tracks pending target kinds, while the
                    # privileged frontier additionally rejects commands that
                    # cannot execute in the current state.  Stop at this dead
                    # end before launching a teacher job that cannot be routed;
                    # the command that reached it is already being supervised.
                    counters['unrecoverable_commands'] += 1
                    break
                if was_complete or not text:
                    break
                if failed or covered < len(raw):
                    # The failed command suffix remains masked by ``score``.  A
                    # complete delimiter gives the next generation a clean DSL
                    # boundary, so rebuild supervision from the unchanged (on
                    # failure) executor state and the exact raw student history.
                    if not delimited:
                        counters['unrecoverable_commands'] += 1
                        break
                    counters['resynchronized_commands'] += 1
                # Bound outstanding page contexts while overlapping scoring and generation.
                if len(jobs) >= 2:
                    await drain()
            await drain()
        finally:
            tasks = [task for _,task in jobs]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if not history or not scored:
            raise RuntimeError('Student produced no scoreable raw tokens')
        counters['completed'] = int(executor.complete())
        counters['raw_tokens'] = len(history)
        ids, logs = self._teacher_storage(len(prompt), len(history))
        mask = torch.zeros(len(history), dtype=torch.bool)
        for start, ti, tl in scored:
            mask[start:start+len(ti)] = True
            causal = len(prompt)+start-1
            ids[causal:causal+len(ti)] = ti
            logs[causal:causal+len(ti)] = tl
        from verl.trainer.manga.diagnostics import enabled, write_record
        if enabled():
            write_record('raw_soft_rollout', dict(sample_index=kwargs.get('index'),
                response_text=self._decode(history), response_ids=history,
                supervised_mask=mask.tolist(), counters=dict(counters)))
        progress_fields = (rollout_fields(progress_records, len(history), total_units,
                           self.config.distillation.manga_progress_alpha) if progress_weight else {})
        if anchor_active:
            anchor_mask = torch.zeros_like(mask)
            for position, _ in self._soft_failure_records:
                anchor_mask[position] = True
            progress_fields['manga_failure_anchor_mask'] = anchor_mask
        if audit_active:
            failure_mask = torch.zeros_like(mask)
            candidates = torch.zeros_like(ids, dtype=torch.bool)
            for position, legal in self._soft_failure_records:
                failure_mask[position] = True
                candidates[len(prompt)+position-1] = legal
            progress_fields.update(manga_failure_mask=failure_mask, manga_failure_candidates=candidates)
        return AgentLoopOutput(prompt_ids=prompt, response_ids=history,
            response_mask=(mask | (progress_fields['manga_progress_bins'] >= 0)).int().tolist()
                if progress_weight else mask.int().tolist(),
            num_turns=1, reward_score=0., metrics=AgentLoopMetrics(generate_sequences=time.perf_counter()-started),
            multi_modal_data=mm_data, mm_processor_kwargs=mm_kwargs,
            precomputed_multi_modal_inputs=self.precomputed_multi_modal_inputs,
            extra_fields=dict(teacher_ids=ids, teacher_logprobs=logs, manga_opsd_mask=mask,
                **progress_fields,
                manga_soft_stats=torch.tensor([counters[k] for k in STATS], dtype=torch.float32),
                manga_retry_truncated=bool(not executor.complete() or counters['illegal_commands']
                    or counters['masked_suffix_tokens'] or eos not in history),
                manga_retry_valid_commands=int(counters['commands']-counters['illegal_commands']),
                manga_opsd_precomputed=True, min_global_steps=minimum, max_global_steps=maximum))
