"""Drop-in replacement for run_eval.py that runs SWE-rebench tasks, each in its own prebuilt task image.

nebius/SWE-rebench publishes a Docker image per task (Docker Hub `swerebench/sweb.eval.x86_64.*`) with
the repository at /testbed and its environment installed in the `testbed` conda env. This wrapper
keeps run_eval.py's Kaggle-fidelity settings (events compaction, context caching) and changes only the
sandbox, for both the agent's container and the verification container:

- each task's containers start from the task's image, with the testbed env first on PATH;
- no wheels, editable install, dependency install or hermetic pytest.ini/conftest.py (the image
  already has the repo's own environment and test configuration);
- the repo moves from /testbed to /workspace and /testbed becomes a symlink to it, so the image's
  editable install keeps importing the agent's edits; then the usual git exclude + baseline commit.

    python run_rebench.py eval --tasks data/rebench_pilot/tasks.jsonl \
        --snapshots-dir data/rebench_pilot/snapshots ... (same flags as swegemma)

Images come from data/rebench_pilot/images.json (or REBENCH_IMAGES=path). Snapshots are placeholders.
REBENCH_GOLD=1 grades each task's reference patch instead of running the agent (environment check).
"""

import json
import os
import shlex
import sys
from pathlib import Path

import run_eval  # noqa: F401  (applies the compaction/caching config; its dependency pinning is replaced below)
from swegemma import cli, evaluate
from swegemma.harness import agent_runner, container_setup, verification

IMAGES = json.loads(Path(os.environ.get('REBENCH_IMAGES', 'data/rebench_pilot/images.json')).read_text())
# Per-image layout. V1 images (nebius/SWE-rebench): repo at /testbed, conda env `testbed`. V2 images
# (docker.io swerebenchv2/*): repo at the image's WorkingDir (e.g. /actualpy), system Python, and pip/XDG
# caches pointed at /workspace by the image's ENV (moved to /tmp so they never land in the agent's diff).
# An images.json entry may override repo_dir, python (absolute path only; V1 metadata stores the Python version
# there) and env; missing keys take the V1 values.
V1_DEFAULTS = {
    'repo_dir': '/testbed',
    'python': '/opt/miniconda3/envs/testbed/bin/python',
    'env': {'PATH': '/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:'
                    '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'},
}
# V2 images do not pip-install the package (it imports only from the repo folder), so it goes on PYTHONPATH
# (src layouts included; a missing entry is harmless).
V2_ENV = {'PYTHONPATH': '/workspace:/workspace/src', 'XDG_CACHE_HOME': '/tmp/.cache', 'PIP_CACHE_DIR': '/tmp/.cache/pip', 'PYTHONPYCACHEPREFIX': '/tmp/.cache/pycache',
          'XDG_CONFIG_HOME': '/tmp/.config', 'XDG_DATA_HOME': '/tmp/.local/share'}
CURRENT = dict(V1_DEFAULTS)  # layout of the task being evaluated (the harness runs one task at a time)
# The images' environments were built with the repo at /<name> (e.g. /pendulum, /textual/src) and the repo
# was then copied elsewhere, so the editable install can point at a path that does not exist and the
# package imports only from inside the repo (a script in /tmp gets ModuleNotFoundError). Every missing
# absolute path in the env's .pth files and setuptools __editable__ finders gets its top-level folder
# symlinked to /workspace. SWE-rebench also leaves the issue text as an untracked issue.md in some repos;
# Kaggle's workspace has no such file, so it is removed.
MOVE_REPO = r"""set -e; shopt -s dotglob
if [ "$REPO" != /workspace ]; then
  mkdir -p /workspace; find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  mv "$REPO"/* /workspace/; rmdir "$REPO"; ln -s /workspace "$REPO"
fi
cd /workspace; git ls-files --error-unmatch issue.md > /dev/null 2>&1 || rm -f issue.md
SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
for p in $( (grep -h "^/" "$SP"/*.pth 2> /dev/null; grep -oh "'/[^']*'" "$SP"/__editable__*.py 2> /dev/null | tr -d "'") | sort -u); do
  top=/$(echo "$p" | cut -d/ -f2)
  [ -e "$p" ] || [ -e "$top" ] || ln -s /workspace "$top"
done"""


def _skip(*args, **kwargs):
    return None


def _workspace_from_repo(docker, container_id, snapshot_path=None, *, patch_path=None, base_snapshot_path=None):
    script = f"REPO={shlex.quote(CURRENT['repo_dir'])}; PY={shlex.quote(CURRENT['python'])}; " + MOVE_REPO
    res = docker.exec(container_id, script)
    if res.exit_code != 0:
        raise container_setup.ContainerSetupError(f"moving {CURRENT['repo_dir']} to /workspace failed: {res.stderr or res.stdout}")


for module in (agent_runner, verification, container_setup, evaluate):
    for name, fn in (('setup_container_wheels', _skip), ('install_editable_package', _skip),
                     ('install_test_dependencies', _skip), ('setup_workspace_test_config', _skip),
                     ('extract_snapshot', _workspace_from_repo),
                     ('extract_deduplicated_snapshot', _workspace_from_repo)):
        if hasattr(module, name):
            setattr(module, name, fn)

_evaluate_task = evaluate.Evaluator.evaluate_task
_BASE_ENV: dict | None = None


async def _evaluate_task_in_its_image(self, task, *args, **kwargs):
    global _BASE_ENV
    manager = self.sandbox
    spec = {**V1_DEFAULTS, **IMAGES[task.instance_id]}
    if not str(spec['python']).startswith('/'):  # V1 metadata keeps the Python version ("3.9") under this key
        spec['python'] = V1_DEFAULTS['python']
    CURRENT.clear()
    CURRENT.update(spec)
    if _BASE_ENV is None:
        _BASE_ENV = dict(manager.config.environment or {})
    manager.config.image = spec['image']
    manager._image_resolved = True
    manager.config.environment = {**_BASE_ENV, **spec['env']}  # rebuilt per task: nothing leaks between tasks
    manager._idle_containers.clear()  # never reuse a container started from another task's image
    return await _evaluate_task(self, task, *args, **kwargs)


evaluate.Evaluator.evaluate_task = _evaluate_task_in_its_image

if os.environ.get('REBENCH_GOLD'):
    # Gold check (no model needed): the "agent" returns the reference patch, so the harness's own
    # verify_task grades it exactly as it grades an agent's patch (expect resolved). The negative
    # check is --skip-agent-patch, which grades an empty patch (expect unresolved).
    GOLD = {row['instance_id']: row['patch'] for row in map(json.loads, Path(
        os.environ.get('REBENCH_TASKS', 'data/rebench_pilot/tasks.jsonl')).read_text().splitlines())}

    async def _gold_agent(self, task, *args, **kwargs):
        return GOLD[task.instance_id], None, evaluate.SessionTrace()

    evaluate.Evaluator._run_agent_sandbox = _gold_agent

SNAPSHOT_AFTER = int(os.environ.get('REBENCH_SNAPSHOT_AFTER') or 0)
if SNAPSHOT_AFTER:
    # Diagnostic: save the working tree's diff after tool call N (taken at the start of call N+1, so only tasks
    # that go past N get one) under results/<run>/patches_atN/, so the patch an agent had at the public
    # bundle's 28-call cap can be graded against its final patch (REBENCH_PATCH_DIR below). Reads only: no
    # `git add -N`, so the agent's own `git status`/`git diff` output is unchanged.
    from swegemma import context as _context
    _CTX = next(c for c in vars(_context).values() if isinstance(c, type) and hasattr(c, 'check_budget'))
    _check_budget = _CTX.check_budget
    _RESULTS = Path(sys.argv[sys.argv.index('--results-dir') + 1]) if '--results-dir' in sys.argv else Path('results/_snapshots')
    _SNAPSHOT_CMD = ('cd /workspace && (git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD); '
                     'for f in $(git ls-files --others --exclude-standard); do git diff --no-index --binary /dev/null "$f"; done; true')

    def _check_budget_and_snapshot(self, *args, **kwargs):
        res = _check_budget(self, *args, **kwargs)
        try:
            if res is None and kwargs.get('count_tool_call', True) and self.tool_calls_used == SNAPSHOT_AFTER + 1:
                out = self.exec(_SNAPSHOT_CMD, timeout=60)
                d = _RESULTS / f'patches_at{SNAPSHOT_AFTER}'
                d.mkdir(parents=True, exist_ok=True)
                (d / f'{self.task_id}.patch').write_text(getattr(out, 'stdout', '') or '')
        except Exception as e:  # a diagnostic must never change the run
            print(f'snapshot after call {SNAPSHOT_AFTER} failed for {getattr(self, "task_id", "?")}: {e}', file=sys.stderr)
        return res

    _CTX.check_budget = _check_budget_and_snapshot

if os.environ.get('REBENCH_PATCH_DIR'):
    # Grade saved patches (e.g. the call-28 snapshots) instead of running the agent: the "agent" returns the
    # task's patch file, or an empty patch when there is none (the task ended before the snapshot point).
    PATCH_DIR = Path(os.environ['REBENCH_PATCH_DIR'])

    async def _saved_patch_agent(self, task, *args, **kwargs):
        f = PATCH_DIR / f'{task.instance_id}.patch'
        return (f.read_text() if f.exists() else ''), None, evaluate.SessionTrace()

    evaluate.Evaluator._run_agent_sandbox = _saved_patch_agent

if __name__ == '__main__':
    sys.exit(cli.main(sys.argv[1:]))
