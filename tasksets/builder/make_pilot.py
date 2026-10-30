"""Write SWE-rebench pilot tasks in swegemma's tasks.jsonl format, an instance->image map with the
fields our runner needs (test command, FAIL_TO_PASS/PASS_TO_PASS), and placeholder snapshots (the
rebench wrapper takes the repo from the task image's /testbed instead)."""
import io
import json
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

src, ids_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ids = json.loads(ids_path.read_text())
import pyarrow as pa
rows = {r['instance_id']: r for r in pa.concat_tables([pq.read_table(f) for f in str(src).split(',')]).to_pylist()
        if r['instance_id'] in set(ids)}  # one path or a comma-separated list
(out / 'snapshots').mkdir(parents=True, exist_ok=True)
tasks, images = [], {}
for i in ids:
    r = rows[i]
    tasks.append({k: r.get(k) or '' for k in ('instance_id', 'repo', 'base_commit', 'problem_statement', 'hints_text',
                                              'patch', 'test_patch', 'created_at')})
    ic = json.loads(r['install_config']) if isinstance(r['install_config'], str) else r['install_config']
    images[i] = {'image': r['docker_image'], 'test_cmd': ic.get('test_cmd'), 'python': ic.get('python'),
                 'FAIL_TO_PASS': json.loads(r['FAIL_TO_PASS']) if isinstance(r['FAIL_TO_PASS'], str) else r['FAIL_TO_PASS'],
                 'PASS_TO_PASS': json.loads(r['PASS_TO_PASS']) if isinstance(r['PASS_TO_PASS'], str) else r['PASS_TO_PASS'],
                 'license': r['license_name']}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        data = b'placeholder: the repo comes from the task image (/testbed)\n'
        info = tarfile.TarInfo('REBENCH_PLACEHOLDER')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (out / 'snapshots' / f'{i}.tgz').write_bytes(buf.getvalue())
(out / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
(out / 'images.json').write_text(json.dumps(images, indent=1))
(out / 'task_ids.txt').write_text('\n'.join(ids) + '\n')
print(len(tasks), 'tasks written to', out)
for i in ids:
    print(' ', i, images[i]['python'], len(images[i]['FAIL_TO_PASS']), 'F2P', len(images[i]['PASS_TO_PASS']), 'P2P |', (images[i]['test_cmd'] or '')[:60])
