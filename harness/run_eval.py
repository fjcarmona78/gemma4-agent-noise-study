"""Drop-in replacement for `swegemma eval` with two local-fidelity fixes.

1. The stock CLI never sets events compaction or context caching, but the Kaggle
   scorer (scripts/inference.py, HARNESS_README section 7.2) does.
2. The public harness's dependency injection (a) puts the published release of the
   package under test (rich 15, fastapi 0.141, ...) in site-packages, where it can
   shadow the agent's /workspace edits, and (b) installs the newest wheel of every
   package, breaking snapshots that pin older ones (57/129 need starlette<1.0).
   After each container's dependency install, sandbox_pin_deps.py fixes both.

Usage is identical to swegemma:

    python run_eval.py eval --tasks data/tasks.jsonl ... (same flags as swegemma)
"""

import sys
from pathlib import Path

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from swegemma import cli, evaluate
from swegemma.evaluate import EvalConfig
from swegemma.harness import agent_runner, container_setup, verification

PIN_SCRIPT = Path(__file__).resolve().parent / 'sandbox_pin_deps.py'


def _kaggle_eval_config(**kwargs) -> EvalConfig:
    kwargs.setdefault(
        'events_compaction_config',
        EventsCompactionConfig(
            compaction_interval=5,
            overlap_size=2,
            token_threshold=14_336,
            event_retention_size=5,
        ),
    )
    kwargs.setdefault(
        'context_cache_config',
        ContextCacheConfig(min_tokens=2_048, ttl_seconds=1_800, cache_intervals=10),
    )
    return EvalConfig(**kwargs)


_original_install = container_setup.install_test_dependencies


def _install_and_pin(docker, container_id, repo='', **kwargs):
    _original_install(docker, container_id, repo, **kwargs)
    if type(docker).__name__ in {'SubprocessManager', 'SubprocessSandbox'}:
        return
    docker.copy_to(container_id, PIN_SCRIPT, '/tmp/_pin_deps.py')
    docker.exec(container_id, f'python3 /tmp/_pin_deps.py {repo}; rm -f /tmp/_pin_deps.py')


cli.EvalConfig = _kaggle_eval_config
for module in (container_setup, agent_runner, verification, evaluate):
    module.install_test_dependencies = _install_and_pin

if __name__ == '__main__':
    sys.exit(cli.main(sys.argv[1:]))
