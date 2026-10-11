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
import uuid

from verl.utils import tensordict_utils as tu

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def _add_batch_to_generate(self):
        if not self.config.distillation.get('manga_retry_truncated', False):
            return super()._add_batch_to_generate()
        if self.parameter_sync_step != 1 or self.config.actor_rollout_ref.rollout.n != 1:
            raise ValueError('Page retries require one synchronous update and n=1')
        if self._resolve_filter_groups_metric() is not None:
            raise ValueError('Page retries cannot discard/filter rollout groups')
        pending = getattr(self, '_manga_retry_pages', {})
        size = self.config.data.train_batch_size
        if len(pending) > size:
            raise ValueError('Retry queue exceeds batch size; do not shrink batch size on resume')
        batches, active = [], {}
        for page, attempt, previous in pending.values():
            # Keep input data only, never reuse student tokens or teacher scores.
            page = page.clone()
            uid = str(uuid.uuid4())
            tu.assign_non_tensor_stack(page, 'uid', [uid])
            tu.assign_non_tensor_data(page, 'global_steps', self.global_steps)
            batches.append(page)
            active[uid] = (page, attempt, previous)
        fresh = size-len(pending)
        if fresh:
            batch = self._next_train_batch(fresh)
            batches.append(batch)
            for i, uid in enumerate(batch['uid']):
                active[str(uid)] = (batch[i:i+1].clone(), 0, None)
        batch = batches[0] if len(batches) == 1 else tu.concat_tensordict(batches)
        tu.assign_non_tensor_data(batch, 'global_steps', self.global_steps)
        self._manga_retry_active = active
        self._manga_retry_metrics = {'manga_retry/retried_pages': len(pending), 'manga_retry/new_pages': fresh}
        self._submit_batch_to_rollout(batch)

    def step(self, metrics, timing_raw):
        batch = super().step(metrics, timing_raw)
        if not self.config.distillation.get('manga_retry_truncated', False):
            return batch
        active = self._manga_retry_active
        pending, seen = {}, set()
        report = self._manga_retry_metrics
        report.update({'manga_retry/improved_pages': 0, 'manga_retry/valid_command_delta': 0,
                       'manga_retry/retry_attempt_max': 0})
        for key, tag in zip(batch.keys, batch.tags, strict=True):
            if tag.get('is_padding', False):
                continue
            uid = key.split('_')[0]
            if uid not in active or uid in seen or 'manga_retry_truncated' not in tag:
                raise RuntimeError('Retry batch must contain exactly one fresh tagged rollout per input')
            seen.add(uid)
            page, attempt, previous = active[uid]
            current = int(tag['manga_retry_valid_commands'])
            if previous is not None:
                report['manga_retry/improved_pages'] += int(current > previous)
                report['manga_retry/valid_command_delta'] += current-previous
            report['manga_retry/retry_attempt_max'] = max(report['manga_retry/retry_attempt_max'], attempt)
            if tag['manga_retry_truncated']:
                pending[uid] = (page, attempt+1, current)
        if seen != set(active):
            raise RuntimeError('A retained page disappeared from the rollout batch')
        self._manga_retry_pages = pending
        self._manga_retry_active = {}
        report['manga_retry/retained_pages'] = len(pending)
        metrics.update(report)
        # Submission occurs in the NEXT step, after on_step_end syncs new weights.
        return batch

    def on_step_end(self):
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
