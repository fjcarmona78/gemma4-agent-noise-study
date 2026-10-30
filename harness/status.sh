#!/usr/bin/env bash
# One-screen status of the evaluation pipeline. Usage: ./status.sh
cd ~/projects/gemma-agent || exit 1

echo "== Services"
curl -sf -m 5 localhost:8000/v1/models >/dev/null && echo "vLLM: up" || echo "vLLM: DOWN"
pgrep -f 'runner/queue_runner.sh' >/dev/null && echo "queue runner: running" || echo "queue runner: NOT running"
pgrep -f 'logs/auto_submit.py' >/dev/null && echo "auto-submit: waiting ($(tail -1 logs/auto_submit.log))" \
  || echo "auto-submit: not running ($(tail -1 logs/auto_submit.log 2>/dev/null))"

echo; echo "== Current evaluation"
current=$(ps -eo args | grep '^python run_eval.py eval' | grep -oE 'results-dir [^ ]+' | cut -d' ' -f2)
if [ -n "$current" ]; then
  total=$(ps -eo args | grep '^python run_eval.py eval' | sed 's/.*--task-ids//' | wc -w)
  done=$(wc -l < "$current/task_results.jsonl" 2>/dev/null || echo 0)
  solved=$(grep -c '"resolved": true' "$current/task_results.jsonl" 2>/dev/null || echo 0)
  echo "$current: $done/$total tasks done, $solved solved"
else
  echo "none"
fi

echo; echo "== Queue"
echo "pending: $(ls queue/pending 2>/dev/null | tr '\n' ' ')"
echo "running: $(ls queue/running 2>/dev/null | tr '\n' ' ')"
echo "done:    $(ls queue/done/*.job 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

echo; echo "== Results (results/SUMMARY.md)"
~/miniforge3/envs/gemma-agent/bin/python -I runner/summarize.py | sed -n '5,$p'
