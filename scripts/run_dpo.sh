#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-7B-Instruct-bnb-4bit}"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[run_dpo.sh] 未找到本地模型 $MODEL_DIR，将尝试从 HuggingFace 下载"
  MODEL_DIR="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
fi

SFT_ADAPTER="${SFT_ADAPTER_PATH:-outputs/sft_qwen25_7b_hasumi_lora_v2}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-outputs/dpo_qwen25_7b_hasumi_lora_v2}"
BETA="${DPO_BETA:-0.05}"
LR="${DPO_LR:-5e-5}"
BATCH_SIZE="${DPO_BATCH_SIZE:-2}"
GRAD_ACC="${DPO_GRAD_ACC:-4}"
MAX_SEQ_LEN="${DPO_MAX_SEQ_LEN:-2048}"
MAX_PROMPT_LEN="${DPO_MAX_PROMPT_LEN:-1024}"

echo "[run_dpo.sh] sft_adapter=$SFT_ADAPTER, output_dir=$DPO_OUTPUT_DIR, beta=$BETA"

uv run python -m src.training.dpo_unsloth \
  --sft_adapter_path "$SFT_ADAPTER" \
  --base_model_name "$MODEL_DIR" \
  --dpo_data_path data/dpo_train.jsonl \
  --output_dir "$DPO_OUTPUT_DIR" \
  --beta "$BETA" \
  --learning_rate "$LR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --num_train_epochs 1 \
  --max_seq_length "$MAX_SEQ_LEN" \
  --max_prompt_length "$MAX_PROMPT_LEN" \
  --report_to none \
  "$@"
