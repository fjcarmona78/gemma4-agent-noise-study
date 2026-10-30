"""Drop-in replacement for run_eval.py that runs the OOD-45 split (data/ood45: 45 post-release bug fixes in 17
Python projects, swegemma format, from the Kaggle dataset rodrigospalma/ood45-bugfix-split) in the standard Docker
sandbox with the dataset's per-repository offline wheel sets.

Keeps run_eval.py's Kaggle-fidelity settings (events compaction, context caching, dependency pinning) and changes
only where the task dependencies come from:

- the wheel set for the sandbox interpreter (wheels/x86_64-cp313/manifest.json) is materialized once as one
  directory per repository under data/ood45/wheel_dirs/, each a set of symlinks into wheels/house/;
- while a task runs, resolve_wheels_dir() returns that task's repository directory instead of the single
  wheels directory the harness expects (the harness passes config only, so the current repo is kept in a global;
  the runner uses --concurrency 1);
- the harness's unpacked-wheel cache (sp_base.tar, keyed by name only) is keyed by the exact wheel set, as the
  dataset's own tools/ood45_harness.py does, so repositories do not inherit each other's site-packages.

    python run_ood45.py eval --tasks data/ood45/tasks.jsonl --snapshots-dir data/ood45/snapshots ... (swegemma flags)

OOD_GOLD=1 grades each task's reference patch instead of running the agent (environment check; the negative
check is --skip-agent-patch).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import run_eval  # noqa: F401  (compaction/caching config and dependency pinning)
from swegemma import cli, evaluate
from swegemma.harness import container_setup

OOD = Path(os.environ.get('OOD_DIR', 'data/ood45'))
WHEEL_SET = os.environ.get('OOD_WHEEL_SET', 'x86_64-cp313')
WHEEL_DIRS = OOD / 'wheel_dirs' / WHEEL_SET
_CURRENT = {'repo': ''}


def materialize_wheels() -> dict[str, Path]:
    """One directory of symlinks per repository (manifest key = repo name without owner)."""
    manifest = json.loads((OOD / 'wheels' / WHEEL_SET / 'manifest.json').read_text())
    house = (OOD / 'wheels' / 'house').resolve()
    dirs = {}
    for repo_dir, names in manifest.items():
        target = WHEEL_DIRS / repo_dir
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            link = target / name
            if not link.exists():
                os.symlink(house / name, link)
        dirs[repo_dir] = target
    return dirs


REPO_DIRS = materialize_wheels()


def _wheels_for(repo: str) -> Path | None:
    return REPO_DIRS.get(repo.split('/', 1)[-1]) if repo else None


_original_resolve = container_setup.resolve_wheels_dir


def _resolve_per_repo(config=None):
    found = _wheels_for(_CURRENT['repo'])
    return found if found is not None else _original_resolve(config)


container_setup.resolve_wheels_dir = _resolve_per_repo
evaluate.resolve_wheels_dir = _resolve_per_repo

_original_tar = container_setup._build_unpacked_wheels_tar


def _keyed_tar(selected_wheels, cache_name, marker_name):
    sig = sorted(f'{p}:{Path(p).stat().st_size}' for p in selected_wheels)
    digest = hashlib.sha1('\n'.join(sig).encode()).hexdigest()[:12]
    return _original_tar(selected_wheels, f'{cache_name}_{digest}', marker_name)


container_setup._build_unpacked_wheels_tar = _keyed_tar

_original_evaluate_task = evaluate.Evaluator.evaluate_task


async def _evaluate_task_with_repo(self, task, *args, **kwargs):
    _CURRENT['repo'] = task.repo
    try:
        return await _original_evaluate_task(self, task, *args, **kwargs)
    finally:
        _CURRENT['repo'] = ''


evaluate.Evaluator.evaluate_task = _evaluate_task_with_repo

if os.environ.get('OOD_GOLD'):
    GOLD = {row['instance_id']: row['patch'] for row in
            map(json.loads, filter(str.strip, (OOD / 'tasks.jsonl').read_text().splitlines()))}

    async def _gold_agent(self, task, *args, **kwargs):
        return GOLD[task.instance_id], None, evaluate.SessionTrace()

    evaluate.Evaluator._run_agent_sandbox = _gold_agent

if __name__ == '__main__':
    sys.exit(cli.main(sys.argv[1:]))
