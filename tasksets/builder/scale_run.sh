#!/bin/bash
# SWE-rebench scale-up (user-approved 2026-10-13 noon): pull 100 task images (paced under Docker Hub's
# 100/hour anonymous limit, stop below 150 GB free), then gold + empty checks through run_rebench.py,
# keep tasks where the reference patch passes and an empty patch fails, remove invalid tasks' images.
set -u
cd ~/projects/gemma-agent
PY=~/miniforge3/envs/gemma-agent/bin/python
DIR=data/rebench_scale
LOG=logs/rebench_scale.log
export REBENCH_IMAGES=$DIR/images.json REBENCH_TASKS=$DIR/tasks.jsonl
echo "$(date '+%F %T') start" >> $LOG
: > $DIR/pulled_ids.txt
$PY -I -c "import json; [print(k, v['image']) for k, v in json.load(open('$DIR/images.json')).items()]" | while read id img; do
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 150 ]; then echo "$(date '+%F %T') stop pulling: ${free}G free" >> $LOG; break; fi
  start=$(date +%s)
  if timeout 1800 docker pull -q "$img" > /dev/null 2>&1; then echo "$id" >> $DIR/pulled_ids.txt; else echo "$(date '+%F %T') pull failed $id" >> $LOG; fi
  wait=$(( 37 - ($(date +%s) - start) )); [ $wait -gt 0 ] && sleep $wait
done
echo "$(date '+%F %T') pulled $(wc -l < $DIR/pulled_ids.txt)" >> $LOG
common="--tasks $DIR/tasks.jsonl --snapshots-dir $DIR/snapshots --sandbox docker --concurrency 1 --display quiet --submission-dir submissions/v4b --max-tool-calls 50 --max-time-minutes 4 --timeout-seconds 300"
REBENCH_GOLD=1 $PY run_rebench.py eval $common --results-dir results/rebench_scale_gold --task-ids $(cat $DIR/pulled_ids.txt) > results/rebench_scale_gold.log 2>&1
$PY run_rebench.py eval $common --results-dir results/rebench_scale_empty --skip-agent-patch --task-ids $(cat $DIR/pulled_ids.txt) > results/rebench_scale_empty.log 2>&1
$PY -I - <<'PYEOF' >> $LOG
import json
gold = {json.loads(l)['instance_id']: json.loads(l)['resolved'] for l in open('results/rebench_scale_gold/task_results.jsonl')}
empty = {json.loads(l)['instance_id']: json.loads(l)['resolved'] for l in open('results/rebench_scale_empty/task_results.jsonl')}
valid = [t for t in gold if gold[t] and t in empty and not empty[t]]
open('data/rebench_scale/valid_ids.txt', 'w').write('\n'.join(valid) + '\n')
images = json.load(open('data/rebench_scale/images.json'))
open('data/rebench_scale/invalid_images.txt', 'w').write('\n'.join(images[t]['image'] for t in gold if t not in valid) + '\n')
print(f'gold pass {sum(gold.values())}/{len(gold)}, empty pass {sum(empty.values())}/{len(empty)}, VALID {len(valid)}')
PYEOF
while read img; do [ -n "$img" ] && docker rmi -f "$img" > /dev/null 2>&1; done < $DIR/invalid_images.txt
echo "$(date '+%F %T') done; $(df -h / | tail -1 | awk '{print $4}') free" >> $LOG
