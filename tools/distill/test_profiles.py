import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / 'projects' / 'gemma-agent' / 'finetune'))
import build_dataset as bd  # noqa: E402

out = Path(sys.argv[1])
# 1) v3 profile regression: must reproduce adapter v2's dataset exactly
bd.build(argparse.Namespace(max_tokens=20480, max_per_task=1, out=str(out / 'v3_regress.jsonl'), profile='v3'))
old = [json.loads(l) for l in open(bd.PROJECT / 'finetune' / 'data' / 'v2.jsonl')]
new = [json.loads(l) for l in open(out / 'v3_regress.jsonl')]
key = lambda e: (e['task'], e['run'], e['chain'])
same = {key(e): (e['input_ids'], e['spans']) for e in old} == {key(e): (e['input_ids'], e['spans']) for e in new}
print('V3 REGRESSION identical to finetune/data/v2.jsonl:', same, len(old), len(new))
# 2) v4 profile on the partial dev91_v4 recording, verified vs as-teacher
bd.build(argparse.Namespace(max_tokens=20480, max_per_task=1, out=str(out / 'v4_verified.jsonl'), profile='v4'))
bd.V4_RECORDED_RUNS, bd.TEACHER_RUNS = [], ['dev91_v4']
bd.build(argparse.Namespace(max_tokens=20480, max_per_task=1, out=str(out / 'v4_as_teacher.jsonl'), profile='v4'))
a = {key(e): (e['input_ids'], e['spans'], e['note']) for e in map(json.loads, open(out / 'v4_verified.jsonl'))}
b = {key(e): (e['input_ids'], e['spans'], e['note']) for e in map(json.loads, open(out / 'v4_as_teacher.jsonl'))}
print('V4 verified examples:', len(a), '| as-teacher:', len(b), '| identical:', a == b)
for k in a:
    print('  ', k, 'tokens', len(a[k][0]), 'spans', len(a[k][1]), a[k][2])
