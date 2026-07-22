#!/usr/bin/env bash
# Upload Hachizumi model artifacts to ModelScope.
# Run with: MODELSCOPE_API_TOKEN=ms-xxxx bash scripts/upload_models_to_modelscope.sh

set -euo pipefail

TOKEN="${MODELSCOPE_API_TOKEN:?MODELSCOPE_API_TOKEN is required}"
export MODELSCOPE_API_TOKEN="$TOKEN"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

upload() {
    local repo_id="$1"
    local local_path="$2"
    local log_file="$LOG_DIR/modelscope_upload_${repo_id//\//_}.log"

    echo "[$(date -Iseconds)] Uploading $repo_id from $local_path ..."
    set -f
    uv run modelscope upload \
        --repo-type model \
        --commit-message "Initial upload of Hachizumi model artifacts" \
        --use-cache \
        --max-workers 4 \
        "$repo_id" \
        "$local_path" \
        > "$log_file" 2>&1
    set +f
    echo "[$(date -Iseconds)] Done: $repo_id"
}

upload "fyerfyer/hachizumi-qwen25-7b-sft-lora-v3" "$ROOT_DIR/outputs/sft_qwen25_7b_hasumi_lora_v3"
upload "fyerfyer/hachizumi-qwen25-7b-dpo-lora-v3" "$ROOT_DIR/outputs/dpo_qwen25_7b_hasumi_lora_v3"
upload "fyerfyer/hachizumi-qwen25-7b-merged-v3" "$ROOT_DIR/outputs/hasumi_qwen25_7b_merged_v3"
upload "fyerfyer/hachizumi-qwen25-7b-gguf-v3" "$ROOT_DIR/outputs/hasumi_qwen25_7b_gguf_v3"

echo "[$(date -Iseconds)] All model uploads completed."
