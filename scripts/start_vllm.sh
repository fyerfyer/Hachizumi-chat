#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# 默认使用 GPU 1，避免和 GPU 0 上可能运行的任务冲突
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# vLLM 0.24.0 V1 engine 默认启用 FlashInfer sampler，会触发 JIT 编译。
# 系统 /usr/bin/nvcc 为 CUDA 11.5，不支持 Ada Lovelace (sm_89)，
# 因此禁用 FlashInfer sampler，回退到原生 PyTorch top-k/top-p 采样。
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1

MODEL_DIR="${MERGED_MODEL_DIR:-outputs/hasumi_qwen25_7b_merged_v2}"
PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
MODEL_NAME="${VLLM_MODEL_NAME:-hasumi}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[start_vllm.sh] 未找到合并模型 $MODEL_DIR，请先完成 merge_and_export"
  exit 1
fi

echo "[start_vllm.sh] 启动 vLLM 服务"
echo "  model: $MODEL_DIR"
echo "  port:  $PORT"
echo "  gpu:   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  util:  $GPU_UTIL"

echo "$PORT" > .vllm_port

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" \
  "$@"
