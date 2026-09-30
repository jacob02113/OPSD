"""vLLM 0.19 scheduler: reuse only KV strictly before sparse predictor rows."""
from types import MethodType
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class _PrefixRequest:
    def __init__(self, request, boundary):
        self._request = request
        self.num_tokens = min(request.num_tokens, boundary + 1)
        self.skip_reading_prefix_cache = False

    def __getattr__(self, name):
        return getattr(self._request, name)


class _SparsePrefixMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manager = self.kv_cache_manager
        original = manager.get_computed_blocks

        def get_computed_blocks(manager, request):
            params = request.sampling_params
            extra = (getattr(params, 'extra_args', None) or {}) if params else {}
            positions = extra.get('verl_prompt_logprobs_positions')
            if positions:
                # Upstream limits hits to request.num_tokens - 1 and block-aligns.
                # The earliest predictor hidden state must NOT be a cache hit.
                result = original(_PrefixRequest(request, min(positions)))
                return result
            return original(request)

        manager.get_computed_blocks = MethodType(get_computed_blocks, manager)


class SparseTeacherScheduler(_SparsePrefixMixin, Scheduler):
    pass


class SparseTeacherAsyncScheduler(_SparsePrefixMixin, AsyncScheduler):
    pass
