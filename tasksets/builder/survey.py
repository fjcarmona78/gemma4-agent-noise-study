"""Survey SWE-rebench task metadata: splits, licenses, meta labels, install_config keys, patch sizes."""
import collections
import json
import sys

import pyarrow.parquet as pq

path = sys.argv[1]
t = pq.read_table(path)
print(path.split('/')[-1], 'rows', t.num_rows, 'columns', t.column_names)
rows = t.slice(0, min(t.num_rows, 4000)).to_pylist()
lic = collections.Counter(r.get('license_name') for r in rows)
print('licenses (first 4000):', lic.most_common(10))
ex = rows[0]
for k in ('instance_id', 'repo', 'version', 'created_at'):
    print(k, '=', ex.get(k))
meta = ex.get('meta')
print('meta sample:', (meta if isinstance(meta, str) else json.dumps(meta))[:600])
ic = ex.get('install_config')
print('install_config sample:', (ic if isinstance(ic, str) else json.dumps(ic))[:900])
env = ex.get('environment')
print('environment sample:', str(env)[:300])
print('FAIL_TO_PASS sample:', str(ex.get('FAIL_TO_PASS'))[:300])
