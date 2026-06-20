#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29501}

MODEL_PATH=${MODEL_PATH:-downloads/Lance_3B_Video}
VIT_PATH=${VIT_PATH:-downloads/Qwen2.5-VL-ViT}
DATASET_CONFIG=${DATASET_CONFIG:-data/configs/lance_finetune_example.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-results/lance_finetune}
MAX_NUM_FRAMES=${MAX_NUM_FRAMES:-121}

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  train/finetune_lance.py \
  --model_path "${MODEL_PATH}" \
  --llm_path "${MODEL_PATH}" \
  --vit_path "${VIT_PATH}" \
  --vit_type qwen_2_5_vl_original \
  --dataset_config_file "${DATASET_CONFIG}" \
  --results_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${OUTPUT_DIR}/checkpoints" \
  --visual_gen True \
  --visual_und True \
  --max_latent_size 64 \
  --max_num_frames "${MAX_NUM_FRAMES}" \
  --latent_patch_size 1 1 1 \
  --sharding_strategy FULL_SHARD \
  --num_shard "${NUM_GPUS}" \
  --lr 2e-5 \
  --warmup_steps 100 \
  --total_steps 1000 \
  --save_every 500 \
  --log_every 10 \
  --num_workers 2 \
  --wandb_offline True
