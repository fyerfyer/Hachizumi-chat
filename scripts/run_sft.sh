#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-models/Qwen2.5-7B-Instruct-bnb-4bit}"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[run_sft.sh] 未找到本地模型 $MODEL_DIR，将尝试从 HuggingFace 下载"
  MODEL_DIR="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
fi

OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/sft_qwen25_7b_hasumi_lora_v2}"
EPOCHS="${SFT_EPOCHS:-3}"
LR="${SFT_LR:-1.5e-4}"
BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
GRAD_ACC="${SFT_GRAD_ACC:-2}"
MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-2048}"

echo "[run_sft.sh] output_dir=$OUTPUT_DIR, epochs=$EPOCHS, lr=$LR"

uv run python -m src.training.sft_unsloth \
  --model_name "$MODEL_DIR" \
  --data_path data/sft_train.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --max_seq_length "$MAX_SEQ_LEN" \
  --report_to none \
  "$@"
