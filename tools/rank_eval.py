"""Offline check of a file-ranking command for the localization experiment.

For each public task: query = problem statement, documents = the snapshot's .py files (tests
excluded), gold = files the reference patch edits. Reports how often a gold file is in the
top 1/3/5/10 for TF-IDF and BM25 scoring. Snapshots are read in memory (nothing extracted).

Usage: rank_eval.py TASKS_JSONL SNAPSHOT_DIR
"""
import collections
import json
import math
import re
import sys
import tarfile
from pathlib import Path

WORD = re.compile(r'[a-z_][a-z0-9_]{2,}')


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def gold_files(patch: str) -> set[str]:
    out = set()
    for line in patch.splitlines():
        if line.startswith('+++ b/'):
            out.add(line[6:].strip())
        elif line.startswith('--- a/'):
            out.add(line[6:].strip())
    return out


def load_docs(tgz: Path) -> dict[str, collections.Counter]:
    docs = {}
    with tarfile.open(tgz) as tar:
        names = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.py')]
        prefix = ''
        for m in names:
            path = m.name
            parts = path.split('/')
            # snapshot archives may wrap the repo in one top-level directory
            if prefix == '' and len(parts) > 1 and all(n.name.split('/')[0] == parts[0] for n in names[:20]):
                prefix = parts[0] + '/'
            rel = path[len(prefix):] if path.startswith(prefix) else path
            if any(p in ('tests', 'test') for p in rel.split('/')[:-1]) or rel.split('/')[0].startswith('.'):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            docs[rel] = collections.Counter(tokens(f.read().decode('utf-8', 'ignore')))
    return docs


def rank_tfidf(q: list[str], docs: dict) -> list[str]:
    n = len(docs)
    df = collections.Counter(w for c in docs.values() for w in set(c))
    terms = set(q)
    score = {p: sum((1 + math.log(c[w])) * math.log(n / df[w]) for w in terms if c[w]) for p, c in docs.items()}
    return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])]


def rank_bm25(q: list[str], docs: dict, k1=1.2, b=0.75) -> list[str]:
    n = len(docs)
    df = collections.Counter(w for c in docs.values() for w in set(c))
    lens = {p: sum(c.values()) for p, c in docs.items()}
    avg = sum(lens.values()) / max(n, 1)
    terms = set(q)
    score = {}
    for p, c in docs.items():
        s = 0.0
        for w in terms:
            tf = c[w]
            if tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[p] / avg))
        score[p] = s
    return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])]


def main() -> None:
    tasks = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    snaps = Path(sys.argv[2])
    hits = {name: collections.Counter() for name in ('tfidf', 'bm25')}
    counted = 0
    for t in tasks:
        tgz = snaps / f"{t['instance_id']}.tgz"
        gold = {g for g in gold_files(t['patch']) if g.endswith('.py') and not any(p in ('tests', 'test') for p in g.split('/')[:-1])}
        if not tgz.exists() or not gold:
            continue
        docs = load_docs(tgz)
        if not gold & set(docs):
            continue  # gold file is new (not in the snapshot): ranking cannot find it
        counted += 1
        q = tokens(t['problem_statement'])
        for name, fn in (('tfidf', rank_tfidf), ('bm25', rank_bm25)):
            order = fn(q, docs)
            first = next((i for i, p in enumerate(order) if p in gold), 10 ** 6)
            for k in (1, 3, 5, 10):
                hits[name][k] += first < k
    print(f'tasks with an existing gold .py source file: {counted}')
    for name, h in hits.items():
        print(f'{name}: ' + ', '.join(f'top{k} {100 * h[k] / counted:.0f}%' for k in (1, 3, 5, 10)))


if __name__ == '__main__':
    main()
