"""Opt-in frozen-SFT audit using the unchanged synchronous rollout pipeline."""
import json
from pathlib import Path


def install_frozen_audit(trainer):
    specification = trainer.config.trainer.get('frozen_audit_batches')
    if not specification:
        return
    batches = []
    for part in str(specification).split(','):
        ends = list(map(int, part.split('-')))
        batches.extend(range(ends[0], ends[-1] + 1))
    if not batches or batches != sorted(set(batches)) or batches[0] < 1:
        raise ValueError('Use increasing, non-overlapping 1-based batch ranges')
    cfg = trainer.config
    if trainer.global_steps != 0 or cfg.trainer.resume_mode != 'disable':
        raise ValueError('Frozen audit requires resume_mode=disable and the SFT model path')
    if (trainer.use_critic or trainer.trainer_mode != 'sync' or trainer.parameter_sync_step != 1
            or cfg.actor_rollout_ref.rollout.n != 1
            or cfg.data.get('gen_batch_size') not in (None, cfg.data.train_batch_size)
            or cfg.algorithm.get('filter_groups', {}).get('enable', False)
            or cfg.trainer.v1.sampler.get('sync_refill_failed_groups', False)):
        raise ValueError('Audit requires synchronous, unfiltered one-batch-per-step sampling')
    if batches[-1] > len(trainer.train_dataloader):
        raise ValueError('Requested batch exceeds the first epoch')
    cfg.trainer.save_freq = cfg.trainer.test_freq = -1
    cfg.trainer.val_before_train = False
    cfg.trainer.critic_warmup = 0
    trainer.total_training_steps = len(batches)
    original_fetch = trainer._fetch_one_gen_batch
    ordinal = 0
    selected = iter(batches)
    output = Path(str(cfg.trainer.get('frozen_audit_output', 'logs/frozen_sft_audit.jsonl')))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Choose a fresh audit output: {output}')

    def fetch():
        nonlocal ordinal
        target = next(selected)
        while ordinal < target:
            batch = original_fetch()
            ordinal += 1
        identifiers = {}
        for key in ('extra_info', 'data_source'):
            if key in batch:
                value = batch[key]
                identifiers[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
        print('FROZEN_AUDIT_BATCH ' + json.dumps(
            dict(original_batch=target, samples=identifiers), ensure_ascii=False, default=str), flush=True)
        return batch

    def no_update(batch, metrics):
        import transfer_queue as tq
        from verl.trainer.manga.command_metrics import (
            PROPOSAL_KINDS, PROPOSAL_METRICS, PROPOSAL_STATS_OFFSET, add_proposal_rates,
        )
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=['manga_command_stats', 'manga_intent_stats'])
        def totals(key):
            value = data[key]
            rows = value.unbind() if value.is_nested else value
            return sum(row for row, tag in zip(rows, batch.tags) if not tag.get('is_padding', False))
        command, intent = totals('manga_command_stats'), totals('manga_intent_stats')
        for i, kind in enumerate(PROPOSAL_KINDS):
            for j, name in enumerate(PROPOSAL_METRICS):
                metrics[f'actor/manga_command/{kind}_{name}'] = float(
                    command[PROPOSAL_STATS_OFFSET + i * len(PROPOSAL_METRICS) + j])
        for index, name in ((0, 'proposals'), (1, 'legal_proposals'), (8, 'episodes'),
                            (9, 'completed_rollouts'), (11, 'retry_attempts'), (13, 'gt_fallbacks')):
            metrics['actor/manga_intent/' + name] = float(intent[index])
        add_proposal_rates(metrics)
        record = dict(original_batch=ordinal, **metrics)
        with output.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record) + '\n')
        return batch  # No actor forward, backward or optimizer step.

    trainer._fetch_one_gen_batch = fetch
    trainer._update_actor = no_update
