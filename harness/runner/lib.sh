#!/usr/bin/env bash
# Shared helpers for queue jobs. Job files in queue/pending/ are sourced by queue_runner.sh
# after this file, so they can call run_experiment directly.

PROJECT=~/projects/gemma-agent
PY=~/miniforge3/envs/gemma-agent/bin/python

# Kaggle-equivalent budget: 4 local minutes ~= 5 Kaggle minutes (see eval_config.yaml).
DEFAULT_MINUTES=4
DEFAULT_TOOL_CALLS=50
DEFAULT_CMD_TIMEOUT=120

# run_experiment NAME SUBMISSION_DIR TASK_LIST_FILE [MINUTES]
#   Validates the submission, then evaluates it; results go to results/NAME/.
#   Env overrides: TASKS_FILE, SNAPSHOTS_DIR, EVAL_SCRIPT (run_rebench.py for SWE-rebench task images),
#   TOOL_CALLS, CMD_TIMEOUT, MODELS_YAML.
run_experiment() {
  local name=$1 submission=$2 tasks=$3 minutes=${4:-$DEFAULT_MINUTES}
  cd "$PROJECT" || return 1
  if [ -e "results/$name/task_results.jsonl" ]; then
    echo "results/$name already exists; refusing to overwrite" >&2
    return 1
  fi
  "$PY" -I runner/validate_submission.py "$submission" || return 1
  python "${EVAL_SCRIPT:-run_eval.py}" eval --tasks "${TASKS_FILE:-data/tasks.jsonl}" \
    --snapshots-dir "${SNAPSHOTS_DIR:-data/snapshots}" \
    --sandbox docker --concurrency 1 --display quiet --submission-dir "$submission" \
    --results-dir "results/$name" --max-tool-calls "${TOOL_CALLS:-$DEFAULT_TOOL_CALLS}" \
    --max-time-minutes "$minutes" --timeout-seconds "${CMD_TIMEOUT:-$DEFAULT_CMD_TIMEOUT}" \
    ${MODELS_YAML:+--models-yaml "$MODELS_YAML"} \
    --task-ids $(cat "$tasks") > "results/$name.log" 2>&1
}

# run_recorded NAME SUBMISSION_DIR TASK_LIST_FILE [MINUTES]
#   run_experiment through finetune/proxy.py, which records every chat request/response to
#   finetune/requests/NAME.jsonl (fine-tuning data in the exact format the model was served).
run_recorded() {
  local name=$1 rec=finetune/requests/$1.jsonl proxy_pid status
  cd "$PROJECT" || return 1
  mkdir -p finetune/requests
  if [ -e "$rec" ]; then
    echo "$rec already exists; refusing to overwrite" >&2
    return 1
  fi
  setsid nohup "$PY" finetune/proxy.py --log "$rec" --port 8001 \
    > "finetune/requests/proxy_$name.log" 2>&1 < /dev/null &
  proxy_pid=$!
  for _ in $(seq 1 30); do curl -sf -m 5 localhost:8001/v1/models > /dev/null && break; sleep 1; done
  if ! curl -sf -m 5 localhost:8001/v1/models > /dev/null; then
    echo "recording proxy did not start (see finetune/requests/proxy_$name.log)" >&2
    kill "$proxy_pid" 2> /dev/null
    return 1
  fi
  export MODEL_PROXY_URL=http://127.0.0.1:8001/v1
  run_experiment "$@"
  status=$?
  unset MODEL_PROXY_URL
  kill "$proxy_pid" 2> /dev/null
  return $status
}

# stop_vllm: stop the vLLM server and wait until its GPU memory is released.
stop_vllm() {
  local used
  pkill -f 'bin/vllm serve' 2> /dev/null
  for _ in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 3000 ] && return 0
    sleep 5
  done
  return 1
}

# serve_lora NAME ADAPTER_DIR LOG
#   Restart vLLM as Gemma4ForCausalLM (vLLM 0.19.1 supports LoRA only on that class) with
#   ADAPTER_DIR served as model NAME (the name the harness requests for `adapter: NAME`).
#   Pause the watchdog first (kill -STOP "$WATCHDOG_PID") or it restarts plain vLLM; restore
#   afterwards with stop_vllm, start_vllm and kill -CONT "$WATCHDOG_PID".
serve_lora() {
  local name=$1 adapter=$2 vlog=$3
  cd "$PROJECT" || return 1
  stop_vllm || return 1
  setsid nohup bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate gemma-agent && exec ./serve_vllm.sh --hf-overrides '{\"architectures\": [\"Gemma4ForCausalLM\"]}' --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules $name=$adapter" \
    > "$vlog" 2>&1 < /dev/null &
  for _ in $(seq 1 144); do
    vllm_up && return 0
    grep -q "Engine core initialization failed" "$vlog" 2> /dev/null && return 1
    sleep 5
  done
  return 1
}
