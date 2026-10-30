"""Pick SWE-rebench pilot tasks (select_pilot.py PARQUET[,PARQUET] N OUT_IDS_JSON [EXCLUDE_REPOS_JSON [MIN_CREATED]]): permissive license, prebuilt image, clear self-contained issue, valid tests,
easy/moderate difficulty, one task per repo, small-to-medium gold patch."""
import json
import random
import sys

import pyarrow.parquet as pq

PERMISSIVE = ('MIT', 'Apache', 'BSD')
FAMOUS = {'django/django', 'sympy/sympy', 'scikit-learn/scikit-learn', 'matplotlib/matplotlib', 'pandas-dev/pandas',
          'pytest-dev/pytest', 'sphinx-doc/sphinx', 'astropy/astropy', 'psf/requests', 'pallets/flask', 'numpy/numpy',
          'pydata/xarray', 'pylint-dev/pylint', 'Textualize/rich', 'fastapi/fastapi', 'encode/httpx', 'tiangolo/fastapi'}
import pyarrow as pa
rows = pa.concat_tables([pq.read_table(f) for f in sys.argv[1].split(',')]).to_pylist()  # one path or a comma-separated list
MIN_CREATED = sys.argv[5] if len(sys.argv) > 5 else ''  # 'YYYY-MM': keep tasks created then or later (recent sets)
n_out = int(sys.argv[2]) if len(sys.argv) > 2 else 10
cands = []
for r in rows:
    lic = r.get('license_name') or ''
    if not r.get('docker_image') or not any(p in lic for p in PERMISSIVE) or r['repo'] in FAMOUS:
        continue
    if str(r.get('created_at') or '')[:7] < MIN_CREATED:
        continue
    try:
        meta = json.loads(r['meta']) if isinstance(r['meta'], str) else (r['meta'] or {})
        f2p = json.loads(r['FAIL_TO_PASS']) if isinstance(r['FAIL_TO_PASS'], str) else list(r['FAIL_TO_PASS'] or [])
    except ValueError:
        continue
    ic = json.loads(r['install_config']) if isinstance(r['install_config'], str) else (r['install_config'] or {})
    try:
        py = tuple(int(x) for x in str(ic.get('python') or '0').split('.')[:2])
    except ValueError:
        py = (0, 0)
    if py < (3, 9):  # the harness's own helper scripts need Python >= 3.9 (pilot: a 3.6 env broke patch apply)
        continue
    score = meta.get('llm_score') or {}
    bad = set(meta.get('failed_lite_validators') or [])
    if score.get('issue_text_score', 9) > 1 or score.get('test_score', 9) > 0 or score.get('difficulty_score', 9) > 1:
        continue
    if bad & {'has_hyperlinks', 'has_media'} or not f2p or len(f2p) > 20:
        continue
    patch_lines = sum(1 for l in r['patch'].splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---')))
    if not 3 <= patch_lines <= 60 or len(r['problem_statement']) < 200:
        continue
    cands.append((r, patch_lines, score, meta.get('num_modified_files')))
print('candidates after filters:', len(cands), 'repos:', len({c[0]['repo'] for c in cands}))
EXCLUDE = set(json.load(open(sys.argv[4]))) if len(sys.argv) > 4 else set()
cands = [c for c in cands if c[0]['repo'] not in EXCLUDE]
random.seed(13)
random.shuffle(cands)
picked, repos = [], set()
for r, lines, score, nfiles in cands:
    if r['repo'] in repos:
        continue
    repos.add(r['repo'])
    picked.append(r)
    print(f"{r['instance_id']:45s} {r['license_name'][:12]:12s} diff={score.get('difficulty_score')} lines={lines:3d} files={nfiles} f2p={len(json.loads(r['FAIL_TO_PASS']) if isinstance(r['FAIL_TO_PASS'], str) else r['FAIL_TO_PASS'])} image={r['docker_image']}")
    if len(picked) == n_out:
        break
json.dump([r['instance_id'] for r in picked], open(sys.argv[3], 'w')) if len(sys.argv) > 3 else None
