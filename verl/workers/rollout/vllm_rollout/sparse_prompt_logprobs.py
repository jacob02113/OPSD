"""Selected-position prompt scoring for the vLLM 0.19 GPU runner.

Position p means the hidden state AFTER input token p, predicting token p+1.
Only the internal verl client may request sparse rows. Unselected CPU top-k
rows are placeholders and must never be used as supervision.
"""
import hashlib
import inspect
import logging
from types import MethodType

import torch

POSITIONS_KEY = "verl_prompt_logprobs_positions"
PROMPT_HASH_KEY = "verl_prompt_logprobs_prompt_hash"
PROJECTION_CHUNK_SIZE = 1024
MASKS_KEY = 'verl_prompt_allowed_ids'


def _prompt_hash(token_ids):
    return hashlib.sha256(",".join(map(str, token_ids)).encode("ascii")).hexdigest()


def attach_positions(sampling_params, positions, prompt_length, prompt_ids=None):
    """Pass the selection through vLLM's serialized SamplingParams.extra_args."""
    if positions is None:
        return
    if sampling_params.get("prompt_logprobs") is None:
        raise ValueError("Selected prompt positions require prompt_logprobs")
    if any(type(p) is not int or p < 0 or p >= prompt_length - 1 for p in positions):
        raise ValueError("Selected predictor positions must have a following prompt token")
    extra_args = dict(sampling_params.get("extra_args") or {})
    extra_args[POSITIONS_KEY] = sorted(set(positions))
    if prompt_ids is not None:
        if len(prompt_ids) != prompt_length:
            raise ValueError("Sparse prompt metadata length mismatch")
        extra_args[PROMPT_HASH_KEY] = _prompt_hash(prompt_ids)
    sampling_params["extra_args"] = extra_args


def _positions(request):
    params = request.sampling_params
    return (getattr(params, "extra_args", None) or {}).get(POSITIONS_KEY)


def selected_prompt_logprobs(runner, hidden_states, num_scheduled_tokens):
    """Preserve vLLM's full CPU row indexing without dense vocabulary projection."""
    from vllm.v1.outputs import LogprobsTensors

    original = runner._verl_dense_prompt_logprobs
    counts = runner.num_prompt_logprobs
    sparse_ids = {rid for rid in counts if _positions(runner.requests[rid]) is not None}
    if not sparse_ids:
        return original(hidden_states, num_scheduled_tokens)

    # Ordinary generation/scoring still uses the backend's unmodified method.
    dense_counts = {rid: k for rid, k in counts.items() if rid not in sparse_ids}
    runner.num_prompt_logprobs = dense_counts
    try:
        result = original(hidden_states, num_scheduled_tokens) if dense_counts else {}
    finally:
        runner.num_prompt_logprobs = counts
        for rid in list(counts):
            if rid not in sparse_ids and rid not in dense_counts:
                del counts[rid]

    progress = runner.input_batch.in_progress_prompt_logprobs_cpu
    groups = {}
    completed = []
    for rid in list(counts):
        if rid not in sparse_ids or rid not in num_scheduled_tokens:
            continue
        request = runner.requests[rid]
        prompt = request.prompt_token_ids
        if prompt is None:
            raise ValueError("Sparse prompt scoring requires token IDs")
        k = counts[rid]
        if k < 0:
            raise ValueError("Sparse prompt scoring requires finite top-k")
        positions = _positions(request)
        if any(p < 0 or p >= len(prompt) - 1 for p in positions):
            raise ValueError("Sparse positions changed relative to the worker prompt")
        if not getattr(runner, "_verl_sparse_projection_logged", False):
            logging.getLogger(__name__).warning(
                "Sparse teacher prompt scoring active: prompt_tokens=%d, selected_rows=%d, projection_cap=%d",
                len(prompt), len(positions), PROJECTION_CHUNK_SIZE,
            )
            runner._verl_sparse_projection_logged = True
        rows = progress.get(rid)
        if rows is None:
            # The server compresses visual placeholders; vLLM expands them again
            # before creating this worker request. Compare only at this stage.
            expected_hash = (request.sampling_params.extra_args or {}).get(PROMPT_HASH_KEY)
            if expected_hash is not None and _prompt_hash(prompt) != expected_hash:
                raise ValueError(
                    "Sparse teacher expanded prompt tokenization mismatch: "
                    "vLLM worker tokens differ from the original processor tokens"
                )
            rows = LogprobsTensors.empty_cpu(len(positions), k + 1)
            rows.logprob_token_ids.zero_()
            rows.logprobs.fill_(-torch.inf)
            rows.selected_token_ranks.fill_(1)
            progress[rid] = rows

        start = request.num_computed_tokens
        scheduled = num_scheduled_tokens[rid]
        remaining = len(prompt) - start - 1
        end = start + min(scheduled, max(remaining, 0))
        selected = [p for p in positions if start <= p < end]
        offset = int(runner.query_start_loc.np[runner.input_batch.req_id_to_index[rid]])
        extra = request.sampling_params.extra_args or {}
        raw_mode = bool(extra.get('verl_raw_legal_probabilities', False))
        masks = extra.get(MASKS_KEY)
        # Group compatible requests, preserving each row's owner, target and mask.
        group = groups.setdefault((k, raw_mode), [])
        compact_indices = {p: i for i, p in enumerate(positions)}
        group.extend((rows, compact_indices[p], offset + p - start, prompt[p + 1],
                      masks[str(p)] if masks is not None else None,
                      extra.get('verl_action_support', {}).get(str(p))) for p in selected)
        if scheduled > remaining:
            completed.append((rid, rows, positions, len(prompt)-1, k))

    for (k, raw_mode), entries in groups.items():
        if not entries:
            continue
        outputs = []
        for begin in range(0, len(entries), PROJECTION_CHUNK_SIZE):
            chunk = entries[begin:begin + PROJECTION_CHUNK_SIZE]
            local = torch.tensor([entry[2] for entry in chunk], device=hidden_states.device)
            logits = runner.model.compute_logits(hidden_states.index_select(0, local))
            raw_normalizer = logits.float().logsumexp(-1) if raw_mode else None
            allow_rows, allow_cols, restricted_rows = [], [], []
            deny_rows, deny_cols = [], []
            for row_index, entry in enumerate(chunk):
                allowed = entry[4]
                if allowed is None:
                    continue
                if isinstance(allowed, dict):
                    excluded = list(allowed['exclude'])
                    excluded.extend(range(allowed['vocab_size'], logits.shape[-1]))
                    deny_rows.extend([row_index] * len(excluded))
                    deny_cols.extend(excluded)
                else:
                    if not allowed:
                        raise ValueError('Empty command support at teacher predictor')
                    restricted_rows.append(row_index)
                    allow_rows.extend([row_index] * len(allowed))
                    allow_cols.extend(allowed)
            if restricted_rows:
                indices = torch.tensor([allow_rows, allow_cols], device=logits.device, dtype=torch.long)
                keep = logits[indices[0], indices[1]]
                rows_to_clear = torch.tensor(restricted_rows, device=logits.device, dtype=torch.long)
                logits.index_fill_(0, rows_to_clear, -torch.inf)
                logits[indices[0], indices[1]] = keep
            if deny_rows:
                indices = torch.tensor([deny_rows, deny_cols], device=logits.device, dtype=torch.long)
                logits[indices[0], indices[1]] = -torch.inf
            if raw_mode:
                # Carry the legal mass in the sampled-token slot, which the
                # raw-mixture client does not use for posterior conditioning.
                probabilities = logits.float() - raw_normalizer[:, None]
                legal_mass = probabilities.logsumexp(-1)
            else:
                probabilities = runner.sampler.compute_logprobs(logits)
            # Calibrate full-vocabulary action probabilities BEFORE top-k.
            # Applying the same floor to each component guarantees it for their
            # equal-weight mixture, while preserving within-set preferences.
            for row_index, entry in enumerate(chunk):
                support = entry[5]
                if not support:
                    continue
                legal = torch.zeros(probabilities.shape[-1], dtype=torch.bool, device=probabilities.device)
                legal[support] = True
                row = probabilities[row_index]
                log_yes = row[legal].logsumexp(0)
                log_no = row[~legal].logsumexp(0)
                floor = row.new_tensor(0.9).log()
                adjust = log_yes < floor
                yes_delta = torch.where(adjust, floor-log_yes, row.new_zeros(()))
                no_delta = torch.where(adjust, row.new_tensor(0.1).log()-log_no, row.new_zeros(()))
                probabilities[row_index] = row + torch.where(legal, yes_delta, no_delta)
            target = torch.tensor([entry[3] for entry in chunk], device=hidden_states.device)
            token_ids, logprobs, ranks = runner.sampler.gather_logprobs(probabilities, k, target)[:3]
            if raw_mode:
                # Reserved out-of-vocabulary ID avoids overwriting an actual
                # top-k token when vLLM builds its logprob dictionary.
                token_ids[:, 0] = logits.shape[-1]
                logprobs[:, 0] = legal_mass
            outputs.append((token_ids, logprobs, ranks))
            del logits, probabilities
        # Transfer compact results together after all projection chunks; never
        # synchronize once per request. Only top-k tensors survive each chunk.
        cpu_ids = torch.cat([item[0] for item in outputs]).cpu()
        cpu_logs = torch.cat([item[1] for item in outputs]).cpu()
        cpu_ranks = torch.cat([item[2] for item in outputs]).cpu()
        del outputs, token_ids, logprobs, ranks
        begin = 0
        while begin < len(entries):
            rows = entries[begin][0]
            end = begin + 1
            while end < len(entries) and entries[end][0] is rows:
                end += 1
            positions_cpu = torch.tensor([entry[1] for entry in entries[begin:end]])
            rows.logprob_token_ids.index_copy_(0, positions_cpu, cpu_ids[begin:end].to(rows.logprob_token_ids.dtype))
            rows.logprobs.index_copy_(0, positions_cpu, cpu_logs[begin:end].to(rows.logprobs.dtype))
            rows.selected_token_ranks.index_copy_(0, positions_cpu, cpu_ranks[begin:end].to(rows.selected_token_ranks.dtype))
            begin = end

    for rid, rows, positions, prompt_rows, k in completed:
        # Match upstream: at equality the final prompt token is still pending.
        # Cached/skipped prefixes must not silently leave requested rows as
        # placeholders. Check a top-k entry (the target itself may be banned).
        if positions:
            selected_rows = rows.logprobs
            valid = torch.isfinite(selected_rows).any(dim=-1)
            if not valid.all():
                missing = [p for p, ok in zip(positions, valid.tolist()) if not ok]
                raise RuntimeError(f"Sparse teacher request {rid} has unscored predictor positions: {missing[:8]}")
        # vLLM's public prompt-logprobs path still expects full prompt indexing.
        # Expand only at completion, not for every live chunked-prefill request.
        dense = LogprobsTensors.empty_cpu(prompt_rows, k + 1)
        dense.logprob_token_ids.zero_()
        dense.logprobs.fill_(-torch.inf)
        dense.selected_token_ranks.fill_(-1)
        indices = torch.tensor(positions, dtype=torch.long)
        dense.logprob_token_ids.index_copy_(0, indices, rows.logprob_token_ids)
        dense.logprobs.index_copy_(0, indices, rows.logprobs)
        dense.selected_token_ranks.index_copy_(0, indices, rows.selected_token_ranks)
        result[rid] = dense
        del counts[rid]
        del progress[rid]
    return result


def install_sparse_prompt_logprobs(runner):
    if hasattr(runner, "_verl_dense_prompt_logprobs"):
        return
    import vllm
    if not vllm.__version__.split("+")[0].startswith("0.19."):
        # Normal rollout on other backends is unaffected; sparse requests are
        # rejected by the server before submission instead of silently going dense.
        return
    method = runner._get_prompt_logprobs_dict
    if list(inspect.signature(method).parameters) != ["hidden_states", "num_scheduled_tokens"]:
        raise RuntimeError("Unsupported vLLM prompt scoring signature; sparse projection was not installed")
    for name in ("num_prompt_logprobs", "input_batch", "requests", "query_start_loc", "sampler", "model"):
        if not hasattr(runner, name):
            raise RuntimeError(f"Unsupported vLLM runner: missing {name}")
    runner._verl_dense_prompt_logprobs = method
    runner._get_prompt_logprobs_dict = MethodType(selected_prompt_logprobs, runner)
    logging.getLogger(__name__).warning(
        "Selected prompt-logprob projection installed: vLLM=%s, max_projection_rows=%d, cross_request_batching=True",
        vllm.__version__, PROJECTION_CHUNK_SIZE,
    )


def install_sparse_prompt_output():
    """Skip placeholder rows before vLLM's tensor-to-Python conversion.

    Rank -1 is private wire metadata for unscored rows. Real vocabulary ranks
    are nonnegative. Ordinary unmarked requests retain the upstream path.
    """
    import vllm
    if not vllm.__version__.split('+')[0].startswith('0.19.'):
        return
    from vllm.v1.engine.logprobs import LogprobsProcessor
    from vllm.v1.outputs import LogprobsTensors
    from vllm.logprobs import FlatLogprobs
    if hasattr(LogprobsProcessor, '_verl_original_prompt_update'):
        return
    original = LogprobsProcessor._update_prompt_logprobs
    if list(inspect.signature(original).parameters) != ['self', 'prompt_logprobs_tensors']:
        raise RuntimeError('Unsupported vLLM prompt output signature')

    def update(self, tensors):
        ranks = tensors.selected_token_ranks
        if not bool((ranks == -1).any()):
            return original(self, tensors)
        if ranks.device.type != 'cpu':
            raise RuntimeError('Sparse prompt output must already be on CPU')
        if bool((ranks < -1).any()):
            raise RuntimeError('Invalid sparse prompt rank metadata')
        selected = torch.nonzero(ranks != -1, as_tuple=False).flatten()
        positions = selected.tolist()
        destination = self.prompt_logprobs
        if destination is None:
            raise RuntimeError('Missing prompt logprob container')
        compact = LogprobsTensors(
            tensors.logprob_token_ids.index_select(0, selected),
            tensors.logprobs.index_select(0, selected),
            ranks.index_select(0, selected))
        # Preserve upstream conversion for real rows, including rank metadata
        # and optional token decoding. Never convert full placeholder tensors.
        temporary = FlatLogprobs() if isinstance(destination, FlatLogprobs) else []
        self.prompt_logprobs = temporary
        try:
            if positions:
                original(self, compact)
        finally:
            self.prompt_logprobs = destination
        if len(temporary) != len(positions):
            raise RuntimeError('Sparse prompt output position count mismatch')

        def append_empty(count):
            if isinstance(destination, FlatLogprobs):
                offset = len(destination.logprobs)
                destination.start_indices.extend([offset] * count)
                destination.end_indices.extend([offset] * count)
            else:
                destination.extend([None] * count)

        cursor = 0
        for index, position in enumerate(positions):
            append_empty(position-cursor)
            destination.append(temporary[index])
            cursor = position + 1
        append_empty(len(ranks)-cursor)

    LogprobsProcessor._verl_original_prompt_update = original
    LogprobsProcessor._update_prompt_logprobs = update
    print('Sparse prompt output conversion installed: unscored rows bypass Python logprob construction', flush=True)
