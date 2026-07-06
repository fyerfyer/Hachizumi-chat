#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

uv run python -m src.data_process.run_all --skip-api
