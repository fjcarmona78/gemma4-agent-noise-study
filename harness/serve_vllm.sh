#!/usr/bin/env bash
# Serve the competition model for `swegemma eval` on a single RTX 5090.
# Mirrors the Kaggle scorer (32k context, gemma4 parsers, thinking on) except TP=1.
# Extra vLLM flags can be appended, e.g. ./serve_vllm.sh --kv-cache-dtype fp8
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-31b-it-qat-w4a16-ct}"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "${SERVED_NAME:-gemma-4-31b-it-qat-w4a16-ct}" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --max-num-seqs 2 \
  --no-scheduler-reserve-full-isl \
  "$@"
