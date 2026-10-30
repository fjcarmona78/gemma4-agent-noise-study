"""Oracle context-packet diagnostic (2026-10-20). For each task, build a compact packet of PRE-FIX code chosen with the
reference patch (symbol selection only; no corrected lines, no hidden assertions) and append it to the problem statement.

  python tools/context_packet.py TASK_ID [TASK_ID ...]   -> data/rebench_recent/packets/<id>.md, tasks_ctx.jsonl, ctx_ids.txt

Packet = for every source file the patch touches: its import block, the full pre-fix definition enclosing each hunk
(def/class from the hunk header, cut by ast), up to 12 direct references repo-wide, and the pre-fix test functions
that mention those symbols (from the files the test patch touches, else a grep of tests/). Hard cap per packet.
"""
import ast, json, re, subprocess, sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(P / 'data/rebench_recent/tasks.jsonl')}
IMAGES = json.load(open(P / 'data/rebench_recent/images.json'))
OUT = P / 'data/rebench_recent/packets'; OUT.mkdir(exist_ok=True)
CAP = 9000

def sh(img, repo, cmd):
    r = subprocess.run(['docker', 'run', '--rm', '--entrypoint', 'sh', img, '-c', f'cd {repo} && {cmd}'], capture_output=True, text=True, timeout=300)
    return r.stdout

def enclosing(src, name, old_line):
    """Source of the def/class `name` that contains old_line (ast), else the 60 lines around old_line."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    lines = src.splitlines()
    best = None
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                if node.lineno <= old_line <= getattr(node, 'end_lineno', node.lineno) or best is None:
                    best = node
    if best:
        a, b = best.lineno, min(getattr(best, 'end_lineno', best.lineno), best.lineno + 119)
        return f'{name} (lines {a}-{b})', '\n'.join(f'{i:5d}  {l}' for i, l in enumerate(lines[a-1:b], a))
    a, b = max(1, old_line - 30), min(len(lines), old_line + 30)
    return f'around line {old_line}', '\n'.join(f'{i:5d}  {l}' for i, l in enumerate(lines[a-1:b], a))

def build(tid):
    t = TASKS[tid]; meta = IMAGES[tid]; img = meta if isinstance(meta, str) else meta.get('image')
    repo = (meta.get('repo_dir') if isinstance(meta, dict) else None) or subprocess.run(['docker', 'inspect', '--format', '{{.Config.WorkingDir}}', img], capture_output=True, text=True).stdout.strip() or '/testbed'
    patch, tpatch = t['patch'], t['test_patch']
    files = {}  # file -> [(name, old_line)]
    cur = None
    for m in re.finditer(r'^(?:\+\+\+ b/(\S+)|@@ -(\d+),?\d* \+\d+,?\d* @@(.*))$', patch, re.M):
        if m.group(1):
            cur = m.group(1)
            if re.search(r'(^|/)(tests?|docs?|changes|changelog|newsfragments|CHANGES)(/|$)|\.(rst|md|txt|yml|yaml|cfg|toml)$', cur): cur = None
            else: files.setdefault(cur, [])
        elif cur:
            hdr = m.group(3).strip(); nm = re.search(r'(?:def|class)\s+(\w+)', hdr)
            files[cur].append((nm.group(1) if nm else None, int(m.group(2))))
    test_files = [f for f in re.findall(r'^\+\+\+ b/(\S+)', tpatch, re.M)]
    symbols = sorted({n for v in files.values() for n, _ in v if n})
    out = [f'# Code context for this issue (pre-fix repository state; selected regions only)\n']
    prefix_sources = []
    for f, hunks in files.items():
        src = sh(img, repo, f'cat {f}')
        if not src: continue
        prefix_sources.append(src)
        imports = [l for l in src.splitlines()[:80] if re.match(r'\s*(from|import)\s', l)][:20]
        out.append(f'\n## {f}\nImports:\n' + '\n'.join('    ' + l for l in imports))
        seen = set()
        for name, old in hunks:
            key = name or old
            if key in seen: continue
            seen.add(key)
            label, body = enclosing(src, name, old) if name else enclosing(src, '', old)
            out.append(f'\n### {f}: {label}\n```python\n{body}\n```')
    if symbols:
        pat = '\\|'.join(f'\\b{s}\\b' for s in symbols[:8])
        refs = sh(img, repo, f"git grep -n '{pat}' -- '*.py' | grep -v '^\\(tests\\?/\\|.*test_\\)' | grep -v 'def \\|class ' | head -12")
        if refs.strip(): out.append('\n## Direct references (file:line)\n' + refs.strip())
        tsrc = ''
        for tf in test_files:
            tsrc += sh(img, repo, f'test -f {tf} && cat {tf}')
        prefix_sources.append(tsrc)
        if not tsrc.strip():
            hits = sh(img, repo, f"git grep -l '{pat}' -- 'tests/*.py' 'test/*.py' '*/tests/*.py' | head -2")
            for tf in hits.split():
                tsrc += sh(img, repo, f'cat {tf}')
        tests = []
        for m in re.finditer(r'^(def test_\w+\(.*?)(?=^def |\Z)', tsrc, re.M | re.S):
            if any(re.search(rf'\b{s}\b', m.group(1)) for s in symbols): tests.append(m.group(1).strip()[:1200])
        if tests: out.append('\n## Existing tests that exercise these symbols (pre-fix)\n```python\n' + '\n\n'.join(tests[:3]) + '\n```')
    packet = '\n'.join(out)
    if len(packet) > CAP: packet = packet[:CAP] + '\n... (packet truncated)'
    # leak check: an added patch line that does NOT exist anywhere in the pre-fix sources we read must not be in the packet
    added = [l[1:].strip() for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++') and len(l.strip()) > 12]
    novel = [a for a in added if not any(a in src for src in prefix_sources)]
    leaks = [a for a in novel if a in packet]
    assert not leaks, f'corrected code leaked into the packet: {leaks[:3]}'
    (OUT / f'{tid}.md').write_text(packet)
    return packet

if __name__ == '__main__':
    ids = sys.argv[1:]
    rows = []
    for tid in ids:
        pk = build(tid); t = dict(TASKS[tid]); t['problem_statement'] = t['problem_statement'].rstrip() + '\n\n' + pk
        rows.append(t); print(f'{tid}: packet {len(pk)} chars')
    (P / 'data/rebench_recent/tasks_ctx.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    (P / 'data/rebench_recent/ctx_ids.txt').write_text('\n'.join(ids) + '\n')
