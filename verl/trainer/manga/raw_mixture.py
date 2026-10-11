"""Unmasked component mixtures at causal student prefixes.

Select a bounded support from component top-k unions, then fill missing scores
on that support. Selection is approximate; selected probabilities and the
complement mass are exact (up to the teacher backend's numerical precision).
"""
import asyncio
import math
from uuid import uuid4

import torch

from .local_policy import Support

STATS = ('episodes', 'completed', 'commands', 'illegal_commands', 'student_generation_requests',
         'score_requests', 'supplement_requests', 'supervised_tokens', 'masked_suffix_tokens',
         'component_rows', 'tail_mass_sum', 'early_eos', 'command_limit', 'raw_tokens',
         'resynchronized_commands', 'unrecoverable_commands',
         'student_generation_temperature_sum', 'student_greedy_episodes')


def component_supports(contexts):
    supports = []
    for c in contexts:
        if c.get('terminal'):
            supports.append([])
            continue
        support = c['support']
        alternatives = [support]
        if support.key[:2] == ('link', 'identity'):
            command = support.command
            a = command[len('<link_from>'):command.index('</link_from>')]
            b = command[command.index('<link_to>')+len('<link_to>'):command.index('</link_to>')]
            alternatives.append(Support(support.executor,
                '<link_from>'+b+'</link_from><link_to>'+a+'</link_to>'))
        supports.append(alternatives)
    return supports


def component_schedule(decode, branch, contexts):
    """Include the first bad choice; never condition routing on that choice."""
    supports = component_supports(contexts)
    rows = []
    candidates = list(range(len(contexts)))
    payload = False
    for i in range(len(branch)):
        # Once inside unrestricted text, don't repeatedly decode the long caption.
        # The producer ends this command at its closing tag; payload semantics
        # are taught by the teacher, not certified by the routing predicate.
        if not payload:
            # The executor permits leading whitespace. Normalize routing only;
            # teacher scoring still consumes every original student token.
            prefix = decode(branch[:i]).lstrip()
            candidates = [j for j in candidates if (
                not prefix.strip() if contexts[j].get('terminal') else
                any(s.viable(prefix) for s in supports[j]))]
            payload = '<text>' in prefix and bool(candidates)
        if not candidates:
            break
        rows.append(tuple(candidates))
    return rows


def failure_labels(decode, branch, contexts, schedule, ids, eos):
    """Routing failures only; free text semantics and omitted candidates are unknown."""
    position = len(schedule) - 1
    active = schedule[position]
    supports = component_supports(contexts)
    prefix = branch[:position]
    # A payload is intentionally unrestricted by the routing policy.
    if '<text>' in decode(prefix).lstrip():
        return None

    def viable(token):
        if token == eos:
            return any(contexts[j].get('terminal') or (
                contexts[j].get('finishes_graph') and
                any(s.matches_completed(decode(prefix).strip()) for s in supports[j])) for j in active)
        text = decode(prefix + [token]).lstrip()
        return any((not text.strip()) if contexts[j].get('terminal') else
                   any(s.viable(text) for s in supports[j]) for j in active)

    if viable(branch[position]):
        return None
    return position, torch.tensor([viable(token) for token in ids[position].tolist()], dtype=torch.bool)


def select_support(parts, schedule, width):
    selected = []
    for i, active in enumerate(schedule):
        mass = {}
        for j in active:
            for token, lp in parts[j][i].items():
                mass[token] = mass.get(token, 0.) + math.exp(lp) / len(active)
        selected.append(sorted(mass, key=lambda t: (-mass[t], t))[:width])
    return selected


def assemble(parts, schedule, selected, width):
    ids, logs = [], []
    for i, active in enumerate(schedule):
        row = []
        for token in selected[i]:
            values = [parts[j][i][token] for j in active]  # missing scores are errors
            maximum = max(values)
            lp = maximum + math.log(math.fsum(math.exp(v-maximum) for v in values)) - math.log(len(active))
            row.append(lp)
        mass = math.fsum(math.exp(v) for v in row)
        if not row or not math.isfinite(mass) or not 0 < mass <= 1.001:
            raise ValueError('Invalid raw teacher mixture mass')
        padding = []
        token = 0
        while len(padding) + len(row) < width:
            if token not in selected[i]:
                padding.append(token)
            token += 1
        ids.append(selected[i] + padding)
        logs.append(row + [-1e9] * len(padding))
    return torch.tensor(ids, dtype=torch.long).reshape(-1, width), torch.tensor(logs, dtype=torch.float32).reshape(-1, width)


async def score(worker, branch, contexts, original, mm_kwargs, routing, counters, semaphore, routed=None):
    jobs = []
    try:
        schedule = await worker.loop.run_in_executor(None, component_schedule, worker._decode, branch, contexts)
        if not schedule:
            raise ValueError('No teacher component at initial command prefix')
        if routed is not None:
            routed.set_result(len(schedule))
        if not hasattr(worker, '_teacher_cache_namespace'):
            worker._teacher_cache_namespace = uuid4().hex

        component_positions = [[] for _ in contexts]
        for i, active in enumerate(schedule):
            for j in active:
                component_positions[j].append(i)

        async def request(j, wanted=None):
            positions = component_positions[j] if wanted is None else list(wanted)
            if not positions:
                return {}
            c = contexts[j]
            await c['ready']
            sequence = c['ids'] + branch[:max(positions)+1]
            key = worker.teacher_server_manager._resolve_teacher_key(routing)
            limit = worker.teacher_server_manager.teacher_model_configs[key].inference.max_model_len
            if limit is not None and len(sequence)+1 > limit:
                raise ValueError('Raw mixture teacher context exceeds max_model_len')
            vision = worker.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            starts = [i for i, t in enumerate(c['ids']) if t == vision]
            prefill = (dict(length=starts[1], image_count=1, namespace=worker._teacher_cache_namespace)
                       if len(starts) == 2 else None)
            async with semaphore:
                ids, logs, _ = await worker.teacher_server_manager.compute_teacher_logprobs_single(
                    sequence_ids=sequence, positions=[len(c['ids'])+i-1 for i in positions],
                    multi_modal_data={'images': [original] if c.get('terminal') else [original, c['image']]},
                    mm_processor_kwargs=mm_kwargs, routing_key=routing, request_id=uuid4().hex,
                    topk_override=worker.topk, shared_prefill=prefill, return_sampled=True,
                    **(dict(allowed_ids=[wanted[i] for i in positions], raw_legal_probabilities=True)
                       if wanted is not None else {}))
            counters['score_requests'] += 1
            counters['supplement_requests'] += int(wanted is not None)
            rows = {}
            for i, ti, lp in zip(positions, ids.tolist(), logs.tolist(), strict=True):
                values = {int(t): float(p) for t,p in zip(ti,lp,strict=True) if math.isfinite(p) and p > -1e8}
                if wanted is not None:
                    if not set(wanted[i]).issubset(values):
                        raise ValueError('Teacher omitted requested raw mixture scores')
                    values = {t: values[t] for t in wanted[i]}
                rows[i] = values
            return rows

        jobs = [asyncio.create_task(request(j)) for j in range(len(contexts))]
        parts = await asyncio.gather(*jobs)
        selected = await worker.loop.run_in_executor(None, select_support, parts, schedule, worker.topk)
        missing = [{i: [t for t in selected[i] if t not in part[i]]
                    for i in part if any(t not in part[i] for t in selected[i])} for part in parts]
        jobs = [asyncio.create_task(request(j, wanted)) for j,wanted in enumerate(missing)]
        supplements = await asyncio.gather(*jobs)
        for part, extra in zip(parts, supplements, strict=True):
            for i, values in extra.items():
                part[i].update(values)
        ids, logs = await worker.loop.run_in_executor(None, assemble, parts, schedule, selected, worker.topk)
        if hasattr(worker, '_soft_failure_labels'):
            record = await worker.loop.run_in_executor(None, failure_labels, worker._decode, branch,
                contexts, schedule, ids, worker.tokenizer.eos_token_id)
            if record is not None:
                worker._soft_failure_labels[id(branch)] = record
        counters['supervised_tokens'] += len(schedule)
        counters['masked_suffix_tokens'] += len(branch)-len(schedule)
        counters['component_rows'] += sum(map(len, schedule))
        counters['tail_mass_sum'] += float((1-logs.exp().sum(-1)).clamp_min(0).sum())
        return ids, logs
    except BaseException as exc:
        if routed is not None and not routed.done():
            routed.set_exception(exc)
        raise
    finally:
        pending = jobs + [c['ready'] for c in contexts]
        for job in pending:
            if not job.done():
                job.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
