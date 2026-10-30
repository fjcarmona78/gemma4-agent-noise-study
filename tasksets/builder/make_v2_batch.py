"""Select a batch of SWE-rebench-V2 verified Python tasks (permissive license, repos not in the test set or earlier
batches, one task per repo) and write them in swegemma's format plus images.json with V2 layout entries
(repo_dir/python/env are filled in after pulling, by inspecting each image).

    python make_v2_batch.py V2_PARQUET EXCLUDE_REPOS_JSON N SEED OUT_DIR [MIN_YEAR]

MIN_YEAR keeps only tasks created in that year or later (e.g. 2025 for a recent-tasks dev set, where
memorized fixes are less likely).
"""
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

src, exclude_path, n, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
min_year = sys.argv[6] if len(sys.argv) > 6 else '0000'
# Hard rule (user, 2026-10-13): never import a repository of the rebench test set, whatever list is passed.
BENCH_REPOS = Path(__file__).resolve().parents[2] / 'data' / 'rebench_scale' / 'bench_repos.json'
exclude = set(json.load(open(exclude_path))) | set(json.load(open(BENCH_REPOS)))
PERMISSIVE = ('MIT', 'Apache', 'BSD', 'ISC', 'PSF', 'Unlicense')
# python_base_37 images ship Python 3.7, too old for the harness's helpers (Path.is_relative_to needs 3.9):
# in batch 1 this base predicted all 17 such rejects and none of the valid tasks.
rows = [r for r in pq.read_table(src).to_pylist()
        if (r['language'] or '').lower() == 'python' and any(p in (r['license'] or '') for p in PERMISSIVE)
        and r['repo'] not in exclude and (r['install_config'] or {}).get('base_image_name') != 'python_base_37'
        and str(r.get('created_at') or '')[:4] >= min_year]
random.seed(seed)
random.shuffle(rows)
picked, repos = [], set()
for r in rows:
    if r['repo'] in repos:
        continue
    repos.add(r['repo'])
    picked.append(r)
    if len(picked) == n:
        break
(out / 'snapshots').mkdir(parents=True, exist_ok=True)
images, tasks = {}, []
for r in picked:
    tasks.append({k: r.get(k) or '' for k in ('instance_id', 'repo', 'base_commit', 'problem_statement', 'patch', 'test_patch')}
                 | {'hints_text': '', 'created_at': str(r.get('created_at') or '')})
    name = r['image_name']
    if name.startswith('prime/primeintellect/'):
        name = 'swerebenchv2/' + name[len('prime/primeintellect/'):]  # the Docker Hub original (see the dataset card)
    images[r['instance_id']] = {'image': name, 'license': r['license'], 'layout': 'v2',
                                'FAIL_TO_PASS': list(r['FAIL_TO_PASS'] or []), 'PASS_TO_PASS': list(r['PASS_TO_PASS'] or [])}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        data = b'placeholder: the repo comes from the task image\n'
        info = tarfile.TarInfo('REBENCH_PLACEHOLDER')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (out / 'snapshots' / f"{r['instance_id']}.tgz").write_bytes(buf.getvalue())
(out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
(out / 'images.json').write_text(json.dumps(images, indent=1))
(out / 'task_ids.txt').write_text('\n'.join(images) + '\n')
print(f'{len(picked)} tasks from {len(repos)} repos written to {out} (pool after filters: {len(rows)})')
