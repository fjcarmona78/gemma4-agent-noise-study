"""Route C (hinted runs): write a copy of the task file whose problem statements end with a hint
listing WHERE the reference fix changes the code (file + enclosing function/class + number of
places), never WHAT to write. Successful hinted runs become training data with the hint removed
(finetune/build_dataset.py), teaching the model to finish multi-part changes.

    python runner/make_hints.py TASK_LIST OUT_TASKS_JSONL

The hint block always starts with HINT_MARK so it can be removed exactly.
"""
import ast
import collections
import json
import re
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
HINT_MARK = '\n\nHint (from the task authors): '


def is_test_file(path: str) -> bool:
    parts = path.split('/')
    return any(p in ('tests', 'test') for p in parts[:-1]) or parts[-1].startswith('test_')


def hunks(patch: str) -> list[tuple[str, int, bool]]:
    """(file, first original line of the hunk, file is new) for non-test files."""
    out, current, new = [], None, False
    for line in patch.splitlines():
        if line.startswith('--- '):
            new = line.startswith('--- /dev/null')
            current = None if new else line[6:].strip()
        elif line.startswith('+++ b/') and current is None:
            current = line[6:].strip()
        elif line.startswith('@@') and current and not is_test_file(current):
            m = re.match(r'@@ -(\d+)(?:,(\d+))? ', line)
            start, n = int(m.group(1)), int(m.group(2) or 1)
            # first changed line: skip unchanged context lines at the hunk start
            out.append((current, start + (3 if n > 6 else 0), new))
    return out


def enclosing(source: str, line: int) -> str | None:
    """Innermost class/function (dotted) whose body spans `line` in the original file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    best = None

    def walk(node, prefix):
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{prefix}.{child.name}' if prefix else child.name
                if child.lineno <= line <= (child.end_lineno or child.lineno):
                    best = name
                    walk(child, name)
            else:
                walk(child, prefix)
    walk(tree, '')
    return best


def read_snapshot_file(task_id: str, path: str) -> str | None:
    tgz = PROJECT / 'data' / 'snapshots' / f'{task_id}.tgz'
    with tarfile.open(tgz) as tar:
        for name in (path, f'./{path}'):
            try:
                f = tar.extractfile(name)
            except KeyError:
                continue
            if f:
                return f.read().decode('utf-8', 'ignore')
    return None


def make_hint(task: dict) -> str | None:
    places = collections.Counter()
    for path, line, new in hunks(task['patch']):
        if new:
            places[(path, None, True)] += 1
            continue
        source = read_snapshot_file(task['instance_id'], path)
        where = enclosing(source, line) if source else None
        places[(path, where, False)] += 1
    if not places:
        return None
    total = sum(places.values())
    items = []
    for (path, where, new), n in places.items():
        if new:
            items.append(f'a new file `{path}`')
        elif where:
            items.append(f'`{path}` in `{where}`' + (f' ({n} places)' if n > 1 else ''))
        else:
            items.append(f'`{path}` at module level' + (f' ({n} places)' if n > 1 else ''))
    return (f'{HINT_MARK}the complete fix changes {total} place{"s" if total > 1 else ""}: '
            + '; '.join(items) + '. Make all of these changes.')


def main() -> None:
    wanted = set(Path(sys.argv[1]).read_text().split())
    out = Path(sys.argv[2])
    rows, made = [], 0
    for line in (PROJECT / 'data' / 'tasks.jsonl').read_text().splitlines():
        task = json.loads(line)
        if task['instance_id'] in wanted:
            hint = make_hint(task)
            if hint:
                task['problem_statement'] = task['problem_statement'].rstrip('\n') + hint
                made += 1
        rows.append(json.dumps(task))
    out.write_text('\n'.join(rows) + '\n')
    print(f'{out}: {made} of {len(wanted)} listed tasks got a hint')


if __name__ == '__main__':
    main()
