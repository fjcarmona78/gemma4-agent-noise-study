#!/usr/bin/env bash
# Runs queue/pending/*.job one at a time, forever, so the GPU never sits idle.
#   - Jobs run in filename order (prefix them 10_, 20_, ... to control order).
#   - A job is a bash snippet; runner/lib.sh is sourced first, so it can call run_experiment.
#   - Finished jobs move to queue/done/ with a .exit file holding their exit code.
#   - A watchdog restarts vLLM if it dies or stalls (request waiting, nothing running).
#   - Between jobs, leftover sandbox containers are removed and results/SUMMARY.md is rebuilt.
# Start detached:  setsid nohup runner/queue_runner.sh > logs/queue_runner.log 2>&1 < /dev/null & disown
# Stop:            kill the queue_runner.sh process (the current job keeps running unless killed too).

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gemma-agent
cd ~/projects/gemma-agent || exit 1
source runner/lib.sh

log() { echo "$(date '+%F %T') $*"; }

vllm_up() { curl -sf -m 10 localhost:8000/v1/models >/dev/null; }

start_vllm() {
  log "starting vLLM"
  pkill -f 'bin/vllm serve' 2>/dev/null; sleep 10
  setsid nohup bash -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate gemma-agent && exec ./serve_vllm.sh' \
    >> logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do vllm_up && { log "vLLM ready"; return 0; }; sleep 5; done
  log "vLLM failed to start within 10 min"; return 1
}

watchdog() {
  local stalled=0
  while true; do
    sleep 120
    if ! vllm_up; then
      log "watchdog: vLLM not responding"; start_vllm; stalled=0; continue
    fi
    if tail -1 logs/vllm.log | grep -q 'Running: 0 reqs, Waiting: [1-9]'; then
      stalled=$((stalled + 1))
      if [ $stalled -ge 5 ]; then log "watchdog: vLLM stalled ~10 min, restarting"; start_vllm; stalled=0; fi
    else
      stalled=0
    fi
  done
}

# An evaluation is running, or an older queue script (logs/step1*.sh) is about to start one.
eval_running() { ps -eo args | grep -qE '^python run_eval.py eval|^bash logs/step1c?\.sh'; }

clean_containers() {
  local ids
  ids=$(docker ps -aq --filter ancestor=swebench-sandbox:latest)
  [ -n "$ids" ] && { log "removing leftover sandbox containers: $(echo $ids)"; docker rm -f $ids >/dev/null; }
}

watchdog & WATCHDOG_PID=$!
trap 'kill $WATCHDOG_PID 2>/dev/null' EXIT
log "queue runner started (watchdog pid $WATCHDOG_PID)"

while true; do
  job=$(ls queue/pending/*.job 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 300; continue; fi
  # Wait for any evaluation started outside the queue (e.g. logs/step1c.sh) to finish first.
  while eval_running; do sleep 60; done
  vllm_up || start_vllm || { sleep 300; continue; }
  clean_containers

  name=$(basename "$job" .job)
  mv "$job" queue/running/
  log "START $name"
  ( source runner/lib.sh; source "queue/running/$name.job" ) > "logs/job_$name.log" 2>&1
  code=$?
  mv "queue/running/$name.job" queue/done/
  echo "$code" > "queue/done/$name.exit"
  log "END $name (exit $code)"

  clean_containers
  "$PY" -I runner/summarize.py > /dev/null 2>> logs/queue_runner.log
done
