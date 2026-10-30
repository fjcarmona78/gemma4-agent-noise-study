#!/bin/bash
# Pull + validate one SWE-rebench task batch for teacher runs (free sources only, user rule):
#   batch_run.sh DIR      (DIR made by make_v2_batch.py: tasks.jsonl, images.json, snapshots/, task_ids.txt)
# Pulls each image (paced under Docker Hub's 100/hour limit, stops below 150 GB free), records each V2 image's
# layout (WorkingDir = repo folder, python3 path, image PYTHONPATH merged into run_rebench's V2_ENV), then runs
# the gold + empty checks through run_rebench.py and keeps tasks whose reference patch passes and empty patch
# fails; images of invalid tasks are removed.
set -u
cd ~/projects/gemma-agent
PY=~/miniforge3/envs/gemma-agent/bin/python
DIR=$1; NAME=$(basename "$DIR"); LOG=logs/$NAME.log
export REBENCH_IMAGES=$DIR/images.json REBENCH_TASKS=$DIR/tasks.jsonl
echo "$(date '+%F %T') start" >> $LOG
: > $DIR/pulled_ids.txt; : > $DIR/layout.tsv
$PY -I -c "import json; [print(k, v['image']) for k, v in json.load(open('$DIR/images.json')).items()]" | while read id img; do
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 150 ]; then echo "$(date '+%F %T') stop pulling: ${free}G free" >> $LOG; break; fi
  start=$(date +%s)
  if timeout 1800 docker pull -q "$img" > /dev/null 2>&1; then
    wd=$(docker image inspect -f '{{.Config.WorkingDir}}' "$img")
    py=$(docker run --rm --network none --entrypoint sh "$img" -c 'command -v python3 || command -v python' 2> /dev/null | tail -1)
    pp=$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$img" | grep '^PYTHONPATH=' | cut -d= -f2-)
    printf '%s\t%s\t%s\t%s\n' "$id" "$wd" "$py" "$pp" >> $DIR/layout.tsv
    echo "$id" >> $DIR/pulled_ids.txt
  else
    echo "$(date '+%F %T') pull failed $id" >> $LOG
  fi
  wait=$(( 37 - ($(date +%s) - start) )); [ $wait -gt 0 ] && sleep $wait
done
echo "$(date '+%F %T') pulled $(wc -l < $DIR/pulled_ids.txt)" >> $LOG
$PY - "$DIR" <<'PYEOF' >> $LOG
import json, sys
sys.path.insert(0, '.')
import run_rebench
d = sys.argv[1]
images = json.load(open(f'{d}/images.json'))
for line in open(f'{d}/layout.tsv'):
    tid, wd, py, pp = (line.rstrip('\n').split('\t') + ['', '', '', ''])[:4]
    env = dict(run_rebench.V2_ENV)
    if pp:
        env['PYTHONPATH'] = env['PYTHONPATH'] + ':' + pp
    images[tid].update({'repo_dir': wd or '/testbed', 'python': py or '/usr/local/bin/python3', 'env': env})
json.dump(images, open(f'{d}/images.json', 'w'), indent=1)
print('layouts recorded:', sum(1 for _ in open(f'{d}/layout.tsv')))
PYEOF
common="--tasks $DIR/tasks.jsonl --snapshots-dir $DIR/snapshots --sandbox docker --concurrency 1 --display quiet --submission-dir submissions/v4b --max-tool-calls 50 --max-time-minutes 4 --timeout-seconds 300"
REBENCH_GOLD=1 $PY run_rebench.py eval $common --results-dir results/${NAME}_gold --task-ids $(cat $DIR/pulled_ids.txt) > results/${NAME}_gold.log 2>&1
$PY run_rebench.py eval $common --results-dir results/${NAME}_empty --skip-agent-patch --task-ids $(cat $DIR/pulled_ids.txt) > results/${NAME}_empty.log 2>&1
$PY -I - "$DIR" "$NAME" <<'PYEOF' >> $LOG
import json, sys
d, name = sys.argv[1], sys.argv[2]
gold = {json.loads(l)['instance_id']: json.loads(l)['resolved'] for l in open(f'results/{name}_gold/task_results.jsonl')}
empty = {json.loads(l)['instance_id']: json.loads(l)['resolved'] for l in open(f'results/{name}_empty/task_results.jsonl')}
valid = [t for t in gold if gold[t] and t in empty and not empty[t]]
open(f'{d}/valid_ids.txt', 'w').write('\n'.join(valid) + '\n')
images = json.load(open(f'{d}/images.json'))
open(f'{d}/invalid_images.txt', 'w').write('\n'.join(images[t]['image'] for t in gold if t not in valid) + '\n')
print(f'gold pass {sum(gold.values())}/{len(gold)}, empty pass {sum(empty.values())}/{len(empty)}, VALID {len(valid)}')
PYEOF
while read img; do [ -n "$img" ] && docker rmi -f "$img" > /dev/null 2>&1; done < $DIR/invalid_images.txt
echo "$(date '+%F %T') done; $(df -h / | tail -1 | awk '{print $4}') free" >> $LOG
