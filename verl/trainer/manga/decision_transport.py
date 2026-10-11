"""Command-batched candidate checks and privileged probability transport labels."""
import asyncio
import math
import logging
from uuid import uuid4

import torch

from .local_policy import Support, vocabulary_for, can_match_box, sg
from .command_teacher import frontier
from .preservation_tokens import diagnostic_groups, sampled_positions
from .transport_candidates import supplement_candidates, earlier_coordinate_error

logger = logging.getLogger(__name__)


def decision_supports(executor, command):
    """Cover both spellings of an undirected identity edge.

    The DAG frontier emits each identity pair once. That is sufficient to choose
    a graph operation, but not to certify all token continuations. Keep these
    CPU predicates separate from teacher contexts: no duplicate image/prompt or
    teacher scoring requests are needed for the reverse spelling.
    """
    supports = []
    for candidate in frontier(executor):
        support = Support(executor, candidate, alias=command)
        supports.append(support)
        if support.key[:2] == ('link', 'identity'):
            first = candidate[len('<link_from>'):candidate.index('</link_from>')]
            second = candidate[candidate.index('<link_to>')+len('<link_to>'):candidate.index('</link_to>')]
            reverse = '<link_from>' + second + '</link_from><link_to>' + first + '</link_to>'
            supports.append(Support(executor, reverse, alias=command))
    return supports


def checked_candidates(decode, prefix, candidates, supports, exact=None):
    """Unknown payload semantics are never labelled illegal by syntax alone."""
    if exact is not None:
        return {int(t): True for t in exact}
    if '<text>' in decode(prefix):
        executor = supports[0].executor
        mapping = executor.state.character_by_student_box
        def reference(candidate):
            text = decode(prefix + [int(candidate)]).split('<text>', 1)[1]
            start = text.rfind(sg.BOX_START)
            if start < 0:
                return None  # Not a reference; no semantic quality claim.
            box, marker, suffix = text[start + len(sg.BOX_START):].partition('<')
            if marker and not (sg.BOX_END.startswith(marker + suffix) or (marker + suffix).startswith(sg.BOX_END)):
                return False
            if not any(can_match_box(box, target, .95) for target in mapping):
                return False
            if marker and (marker + suffix).startswith(sg.BOX_END):
                match = sg.BOX_RE.fullmatch(sg.BOX_START + box + sg.BOX_END)
                return match is not None and executor.resolve_reference(mapping, sg._match_box(match, 0)) is not None
            return True
        return {int(t): status for t in candidates if (status := reference(t)) is not None}
    prefix_text = decode(prefix)
    active = [s for s in supports if s.viable(prefix_text)]
    if not active:
        raise ValueError(f'Decision prefix has no executable continuation: prefix={prefix_text!r}, '
                         f'support_count={len(supports)}')
    checked = {}
    for token in candidates:
        text = decode(prefix + [int(token)])
        checked[int(token)] = any(s.viable(text) for s in active)
    return checked


async def score_command(worker, *, executor, command, original, messages, history,
                        branch, raw, gt_start, records, mm_kwargs, routing,
                        student_prompt, student_route, counters, semaphore, legal):
    """One sparse student request and one batch per relevant teacher component.

    The student's request only discovers candidates; actor probabilities define
    the target. Existing command histories/KV locality and thumbnail caches are
    reused. No teacher rescoring loop or vocabulary/coordinate enumeration.
    """
    start, size = len(history), len(branch)
    groups = diagnostic_groups(worker.tokenizer, branch, decision_only=True)
    eligible = set(sampled_positions(0, branch, raw, gt_start))
    repairs = {len(p)-start: list(a) for p, a in records
               if start <= len(p) < start+size and list(p) == history + branch[:len(p)-start]}
    supports = decision_supports(executor, command)
    actions = {s.command.split('>', 1)[0] + '>' for s in supports}
    vocabulary = vocabulary_for(worker.tokenizer)
    prefixes = {}
    exact = {}
    for i in range(size):
        if i in repairs:
            prefixes[i], exact[i] = branch[:i], repairs[i]
        elif i in eligible and groups[i] in (1, 2):
            prefix = branch[:i]
            text = worker._decode(prefix)
            prefixes[i] = prefix
            if groups[i] == 1:
                allowed = vocabulary.action_tokens(actions, text)
                if not allowed or branch[i] not in allowed:
                    raise ValueError('Sampled action is inconsistent with executable support')
                exact[i] = allowed

    teacher_ids = torch.zeros((size, worker.topk), dtype=torch.long)
    teacher_logs = torch.full((size, worker.topk), -1e9)
    if size:
        teacher_ids[:, 0] = torch.tensor(branch)
        teacher_logs[:, 0] = 0.
    result = dict(start=start, branch=branch, legal=legal, teacher_ids=teacher_ids,
                  teacher_logprobs=teacher_logs,
                  correct_support=[repairs.get(i) for i in range(size)],
                  repair_positions=list(repairs), repair_weights={i: 1./max(len(records), 1) for i in repairs},
                  object_positions=[], action_support={}, transport={}, transport_groups=groups)
    if not prefixes:
        return result

    async def student_candidates():
        positions = [i for i in prefixes if i not in exact]
        if not positions:
            return {}
        output = await worker.server_manager.generate(
            request_id=student_route,
            prompt_ids=student_prompt + history + branch[:max(positions)+1],
            sampling_params=dict(max_tokens=1, temperature=1., prompt_logprobs=worker.transport_topk,
                                 _verl_prompt_logprobs_positions=[len(student_prompt)+start+i-1 for i in positions]),
            image_data=[original], mm_processor_kwargs=mm_kwargs)
        ids = output.extra_fields['prompt_ids']
        logs = output.extra_fields['prompt_logprobs']
        if len(ids) != len(positions) or len(logs) != len(positions):
            raise ValueError('Sparse student candidate rows are misaligned')
        counters['transport_student_requests'] += 1
        return {i: [int(t) for t, l in zip(row, lp, strict=True) if math.isfinite(l) and l > -1e8]
                for i, row, lp in zip(positions, ids, logs, strict=True)}

    multi = [i for i in prefixes if len(exact.get(i, ())) != 1]
    student_job = asyncio.create_task(student_candidates())
    contexts, jobs = [], []
    try:
        if multi:
            _, _, _, _, contexts = await worker._command_context(
                executor=executor, command=command, original=original, messages=messages,
                history=history, mm_kwargs=mm_kwargs, frozen=True,
                repair_prefixes=[worker._decode(prefixes[i]) for i in multi])
            if not contexts:
                raise ValueError('Transport has no privileged teacher component')
        if not hasattr(worker, '_teacher_cache_namespace'):
            worker._teacher_cache_namespace = uuid4().hex

        async def score(context, supplement=None):
            active = [i for i in (multi if supplement is None else supplement)
                      if context['support'].viable(worker._decode(prefixes[i]))]
            if not active:
                return {}
            await context['ready']
            key = worker.teacher_server_manager._resolve_teacher_key(routing)
            limit = worker.teacher_server_manager.teacher_model_configs[key].inference.max_model_len
            sequence = context['ids'] + branch[:max(active)+1]
            if limit is not None and len(sequence) + 1 > limit:
                raise ValueError('Transport teacher prefix exceeds max_model_len; increase teacher context budget')
            vision = worker.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            starts = [j for j, token in enumerate(context['ids']) if token == vision]
            prefill = (dict(length=starts[1], image_count=1, namespace=worker._teacher_cache_namespace)
                       if len(starts) == 2 else None)
            async with semaphore:
                ids, logs, sampled = await worker.teacher_server_manager.compute_teacher_logprobs_single(
                    sequence_ids=sequence, positions=[len(context['ids'])+i-1 for i in active],
                    multi_modal_data={'images': [original, context['image']]}, mm_processor_kwargs=mm_kwargs,
                    routing_key=routing, request_id=uuid4().hex,
                    topk_override=worker.topk, return_sampled=True, shared_prefill=prefill,
                    **(dict(allowed_ids=[supplement[i] for i in active], raw_legal_probabilities=True)
                       if supplement is not None else {}))
            counters['scoring_requests'] += 1
            if supplement is not None:
                counters['transport_supplement_requests'] += 1
            rows = {}
            for i, ti, tl, sl in zip(active, ids.tolist(), logs.tolist(), torch.as_tensor(sampled).tolist(), strict=True):
                values = {int(t): l for t, l in zip(ti, tl, strict=True) if math.isfinite(l) and l > -1e8}
                if supplement is None:
                    if not math.isfinite(sl):
                        raise ValueError(f'Missing accepted-token teacher probability: position={i}, '
                                         f'prefix={worker._decode(prefixes[i])!r}')
                    values[branch[i]] = sl
                else:
                    # In raw-legal mode sl is legal-set log mass, not the
                    # observed token's score. Never assign it to branch[i].
                    values = {t:l for t,l in values.items() if t in supplement[i]}
                    if not math.isfinite(sl) or set(values) != set(supplement[i]):
                        raise ValueError(f'Missing requested raw teacher scores: position={i}, '
                                         f'requested={supplement[i]}, returned={list(values)}')
                rows[i] = values
                counters['teacher_scored_rows'] += 1
            return rows

        jobs = [asyncio.create_task(score(c)) for c in contexts]
        parts = await asyncio.gather(*jobs)
        students = await student_job
        # Geometry and candidate decoding run off the event loop, once per union.
        pending, reasons, earlier_errors = {}, {}, {}
        def aggregate(final=False):
            uncovered = []
            for i, prefix in prefixes.items():
                if i in exact and len(exact[i]) == 1:
                    result['transport'][i] = (True, [(exact[i][0], 1., 1.)])
                    continue
                mixtures = {}
                for part in parts:
                    for t, logp in part.get(i, {}).items():
                        mixtures.setdefault(t, []).append(logp)
                # Stable equal-component raw mixture; no candidate-count correction.
                logmix = {}
                for t, values in mixtures.items():
                    m = max(values)
                    logmix[t] = m + math.log(math.fsum(math.exp(x-m) for x in values))
                top = sorted(logmix, key=logmix.get, reverse=True)[:worker.topk]
                candidates = set(students.get(i, ())) | set(top) | {branch[i]} | set(pending.get(i, ()))
                status = checked_candidates(worker._decode, prefix, candidates, supports, exact.get(i))
                if i not in exact and '<text>' not in worker._decode(prefix) and status.get(branch[i]) is not True:
                    # Validate the observed structural choice even when some
                    # other candidate happens to have a usable teacher score.
                    supplement_candidates(worker, prefix, supports, status, branch[i], checked_candidates)
                positives = [t for t, ok in status.items() if ok and t in logmix]
                if not positives:
                    detail = dict(position=i, prefix=worker._decode(prefix),
                                  candidates=len(status), legal=sum(status.values()), scored=len(logmix))
                    if i in exact:
                        raise ValueError(f'No scored legal candidate for exact probability transport: {detail!r}')
                    if not final:
                        proposed, reason = supplement_candidates(worker, prefix, supports, status, branch[i],
                                                                checked_candidates, limit=min(4, worker.topk))
                        reasons[i] = reason
                        if proposed:
                            pending[i] = proposed
                            continue
                    detail['reason'] = reasons.get(i, 'missing_teacher_context')
                    if detail['reason'] == 'dead_coordinate_prefix':
                        box_start = detail['prefix'].rfind(sg.BOX_START)
                        if box_start not in earlier_errors:
                            earlier_errors[box_start] = earlier_coordinate_error(worker, prefix, supports)
                        detail['earlier_position'] = earlier_errors[box_start]
                    # A sparse union is not a complete feasible set. Without a
                    # scored legal destination q cannot be defined: abstain on
                    # this predictor, without inventing a target or treating
                    # the unknown vocabulary tail as illegal.
                    uncovered.append(detail)
                    continue
                maximum = max(logmix[t] for t in positives)
                weights = {t: math.exp(logmix[t]-maximum) for t in positives}
                total = math.fsum(weights.values())
                edges = [(t, float(ok), weights.get(t, 0.) / total) for t, ok in sorted(status.items())]
                result['transport'][i] = (i in exact, edges)
            if uncovered and (final or not pending):
                counters['transport_uncovered_rows'] += len(uncovered)
                for detail in uncovered:
                    counters['transport_skip_' + detail['reason']] += 1
                logger.warning('Transport sparse coverage missing: skipped=%d decisions=%d first=%r',
                               len(uncovered), len(prefixes), uncovered[0])
            return result
        result = await worker.loop.run_in_executor(None, aggregate)
        if pending:
            jobs = [asyncio.create_task(score(c, pending)) for c in contexts]
            extra = await asyncio.gather(*jobs)
            for part, supplement in zip(parts, extra, strict=True):
                for i, values in supplement.items():
                    part.setdefault(i, {}).update(values)
            result = await worker.loop.run_in_executor(None, lambda: aggregate(final=True))
            recovered = len(set(pending) & set(result['transport']))
            counters['transport_supplement_recovered_rows'] += recovered
            logger.info('Transport bounded supplement: requested_rows=%d recovered_rows=%d',len(pending),recovered)
        counters['supervised_tokens'] += len(result['transport'])
        return result
    finally:
        pending = [student_job] + jobs + [c['ready'] for c in contexts]
        for job in pending:
            if not job.done():
                job.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
