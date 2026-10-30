#!/bin/bash
# After the V2 teacher run (job 85b): free the images of V2 tasks the teacher did not solve (no training value;
# the solved tasks' images stay for possible repeats), then pull and validate the recent-set tasks that the
# 150 GB floor left unpulled (data/rebench_recent/unpulled_ids.txt) as batch data/rebench_recent_b, and merge
# the valid ones, with their recorded layouts, into data/rebench_recent (valid_ids, images.json, eval_ids =
# tasks created Feb 2025 or later, after Gemma 4's January 2025 pretraining cutoff).
#   bash tools/rebench/extend_recent.sh      (CPU and disk only; log: logs/extend_recent.log)
set -u
cd ~/projects/gemma-agent || exit 1
PY=~/miniforge3/envs/gemma-agent/bin/python
LOG=logs/extend_recent.log
echo "$(date '+%F %T') start" >> $LOG

$PY -I - <<'EOF' >> $LOG 2>&1
import json, subprocess
im = json.load(open('data/rebench_v2/images.json'))
res = {json.loads(l)['instance_id']: json.loads(l)['resolved'] for l in open('results/v2_teacher_qwen36/task_results.jsonl')}
failed = [t for t, ok in res.items() if not ok]
for t in failed:
    subprocess.run(['docker', 'rmi', '-f', im[t]['image']], capture_output=True)
print(f'removed the images of {len(failed)} V2 tasks the teacher failed; kept {sum(res.values())} solved')
EOF
df -h / | tail -1 | awk '{print "free after cleanup:", $4}' >> $LOG

B=data/rebench_recent_b
rm -rf $B; mkdir -p $B/snapshots
$PY -I - <<'EOF' >> $LOG 2>&1
import json, shutil
from pathlib import Path
R, B = Path('data/rebench_recent'), Path('data/rebench_recent_b')
ids = (R / 'unpulled_ids.txt').read_text().split()
im = json.load(open(R / 'images.json'))
tasks = [json.loads(l) for l in open(R / 'tasks.jsonl')]
(B / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks if t['instance_id'] in ids))
json.dump({t: im[t] for t in ids}, open(B / 'images.json', 'w'), indent=1)
(B / 'task_ids.txt').write_text('\n'.join(ids) + '\n')
for t in ids:
    shutil.copy(R / 'snapshots' / f'{t}.tgz', B / 'snapshots' / f'{t}.tgz')
print(len(ids), 'tasks in batch b')
EOF
bash tools/rebench/batch_run.sh $B

$PY -I - <<'EOF' >> $LOG 2>&1
import json
from pathlib import Path
R, B = Path('data/rebench_recent'), Path('data/rebench_recent_b')
valid_b = (B / 'valid_ids.txt').read_text().split()
im, imb = json.load(open(R / 'images.json')), json.load(open(B / 'images.json'))
for t in valid_b:
    im[t] = imb[t]
json.dump(im, open(R / 'images.json', 'w'), indent=1)
valid = [t for t in (R / 'valid_ids.txt').read_text().split() if t not in valid_b] + valid_b
(R / 'valid_ids.txt').write_text('\n'.join(valid) + '\n')
tasks = {json.loads(l)['instance_id']: json.loads(l) for l in open(R / 'tasks.jsonl')}
ev = [t for t in valid if tasks[t]['created_at'][:7] >= '2025-02']
(R / 'eval_ids.txt').write_text('\n'.join(ev) + '\n')
print(f'recent set: {len(valid)} valid, eval list {len(ev)} tasks (Feb 2025 or later)')
EOF
echo "$(date '+%F %T') done" >> $LOG
