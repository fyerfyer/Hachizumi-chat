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

uv run python -m src.training.dpo_unsloth \
  --sft_adapter_path outputs/sft_qwen25_7b_hasumi_lora \
  --base_model_name "$MODEL_DIR" \
  --dpo_data_path data/dpo_train.jsonl \
  --output_dir outputs/dpo_qwen25_7b_hasumi_lora \
  --beta 0.1 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --max_seq_length 2048 \
  --max_prompt_length 1024 \
  --report_to none \
  "$@"
