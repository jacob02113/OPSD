#!/usr/bin/env bash
# v4.9: local privileged teacher mixture; retain legal student commands
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CUDA_HOME="$CONDA_PREFIX"

STUDENT_MODEL=${STUDENT_MODEL:-/hdd/u202212063031/magi/Qwen3-VL/models/Qwen3-VL-4B-DSL-SFT-v3.1-e1}
TRAIN_FILE=${TRAIN_FILE:-data/train.parquet}
VAL_FILE=${VAL_FILE:-data/val.parquet}
PROMPT_KEY=${PROMPT_KEY:-prompt}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_MODEL=${TEACHER_MODEL:-$STUDENT_MODEL}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.85}
MANGA_TEACHER_MAX_INFLIGHT=${MANGA_TEACHER_MAX_INFLIGHT:-32}
TEACHER_MAX_CONCURRENT_REQUESTS=${TEACHER_MAX_CONCURRENT_REQUESTS:-64}
TEACHER_MM_PROCESSOR_CACHE_GB=${TEACHER_MM_PROCESSOR_CACHE_GB:-4}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-24576}
TEACHER_TOPK_MAX=${TEACHER_TOPK_MAX:-128}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-32}
ROLLOUT_TP=${ROLLOUT_TP:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-True}
FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-4}
FILTER_OVERLONG_PROMPTS_CACHE_DIR=${FILTER_OVERLONG_PROMPTS_CACHE_DIR:-cache/opsd}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ENTITY_IOU_THRESHOLD=${ENTITY_IOU_THRESHOLD:-0.5}
MANGA_MAX_COMMANDS=${MANGA_MAX_COMMANDS:-256}
OLD_LOG_PROB_ENTROPY=${OLD_LOG_PROB_ENTROPY:-False}
ACTOR_LR=${ACTOR_LR:-1e-6}
MANGA_ACTION_ILLEGAL_WEIGHT=${MANGA_ACTION_ILLEGAL_WEIGHT:-1.0}
MANGA_CORRECTNESS_LOSS=${MANGA_CORRECTNESS_LOSS:-True}
MANGA_PREFERENCE_WEIGHT=${MANGA_PREFERENCE_WEIGHT:-0.2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-16}
TEST_FREQ=${TEST_FREQ:--1}
PROJECT_NAME=${PROJECT_NAME:-mangatrace}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mangatrace_a_opsd_v4.9.9}
SAVE_STUDENT_ROLLOUTS=${SAVE_STUDENT_ROLLOUTS:-False}

rollout_max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
export VERL_FILE_LOGGER_PATH="logs/${EXPERIMENT_NAME}-${start_time}.metrics.jsonl"
mkdir -p logs
if [[ "$SAVE_STUDENT_ROLLOUTS" == "True" ]]; then
    export MANGA_DIAG_DIR="${MANGA_DIAG_DIR:-logs/${EXPERIMENT_NAME}-${start_time}/diagnostics}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files="['$TRAIN_FILE']" \
    data.val_files="['$VAL_FILE']" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    data.prompt_key=$PROMPT_KEY \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS \
    data.filter_overlong_prompts_workers=$FILTER_OVERLONG_PROMPTS_WORKERS \
    data.filter_overlong_prompts_cache=True \
    data.filter_overlong_prompts_cache_dir="$FILTER_OVERLONG_PROMPTS_CACHE_DIR" \
    data.truncation=error \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.calculate_old_log_prob_entropy=$OLD_LOG_PROB_ENTROPY \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=$rollout_max_model_len \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.agent.default_agent_loop=manga_corrective_opsd_agent \
    distillation.enabled=True \
    distillation.manga_opsd_enabled=True \
    distillation.manga_action_illegal_weight=$MANGA_ACTION_ILLEGAL_WEIGHT \
    distillation.manga_correctness_loss=$MANGA_CORRECTNESS_LOSS \
    distillation.manga_preference_weight=$MANGA_PREFERENCE_WEIGHT \
    distillation.n_gpus_per_node=$TEACHER_NGPUS_PER_NODE \
    distillation.nnodes=$TEACHER_NNODES \
    distillation.manga_teacher_max_inflight=$MANGA_TEACHER_MAX_INFLIGHT \
    distillation.manga_teacher_topk_max=$TEACHER_TOPK_MAX \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEM_UTIL \
    distillation.teacher_models.teacher_model.inference.max_model_len=$((rollout_max_model_len + 2048)) \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_MAX_NUM_BATCHED_TOKENS \
    distillation.teacher_models.teacher_model.inference.max_concurrent_requests=$TEACHER_MAX_CONCURRENT_REQUESTS \
    distillation.teacher_models.teacher_model.inference.free_cache_engine=False \
    +distillation.teacher_models.teacher_model.inference.enable_sleep_mode=False \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.mm_processor_cache_gb=$TEACHER_MM_PROCESSOR_CACHE_GB \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs=$TEACHER_TOPK_MAX \
    distillation.manga_target_key=target_json \
    distillation.manga_entity_iou_threshold=$ENTITY_IOU_THRESHOLD \
    distillation.manga_max_commands=$MANGA_MAX_COMMANDS \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk=$DISTILLATION_TOPK \
    distillation.distillation_loss.use_chunked_topk=True \
    distillation.distillation_loss.chunked_topk_chunk_size=1024 \
    distillation.distillation_loss.jsd_token_clip=null \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    trainer.use_v1=True \
    trainer.balance_batch=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger='["console","file"]' \
    trainer.val_before_train=False \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@" 2>&1 | tee "logs/${EXPERIMENT_NAME}-${start_time}.log"
