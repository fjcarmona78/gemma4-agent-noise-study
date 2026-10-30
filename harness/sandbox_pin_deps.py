"""Runs INSIDE a sandbox container after swegemma installs test dependencies.

Fixes two problems with the public harness's dependency injection, offline:

1. It unpacks every small wheel in /wheels into site-packages, including the
   published release of the task's own package (rich 15.0.0, fastapi 0.141.1, ...).
   That copy can shadow the editable /workspace install, so tests silently run
   against the release instead of the agent's edits. We delete the shadowing copy.
2. It installs the newest wheel of each package (e.g. starlette 1.6.0), which breaks
   older snapshots whose pyproject.toml requires earlier versions (most fastapi tasks
   pin starlette<1.0). We re-pin each violated core dependency to the newest
   compatible wheel in /wheels.

Usage: python3 sandbox_pin_deps.py [owner/repo]
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import InvalidVersion, Version

WHEELS_DIR = '/wheels'
PYPROJECT = '/workspace/pyproject.toml'
COMPATIBLE_TAGS = ('py3-none-any', 'py2.py3-none-any', 'cp313-cp313', 'abi3')


def norm(name: str) -> str:
    return re.sub(r'[-_.]+', '_', name).lower()


def available_wheels() -> dict[str, list[tuple[Version, str]]]:
    found: dict[str, list[tuple[Version, str]]] = {}
    for fname in os.listdir(WHEELS_DIR):
        if not fname.endswith('.whl') or not any(t in fname for t in COMPATIBLE_TAGS):
            continue
        parts = fname[:-4].split('-')
        try:
            found.setdefault(norm(parts[0]), []).append((Version(parts[1]), fname))
        except InvalidVersion:
            pass
    return found


def is_editable(dist: metadata.Distribution) -> bool:
    direct_url = dist.read_text('direct_url.json') or '{}'
    return bool(json.loads(direct_url).get('dir_info', {}).get('editable'))


def remove_installed(dist: metadata.Distribution) -> None:
    """Deletes a distribution by hand: harness-unpacked wheels have no RECORD for pip."""
    dist_info = Path(str(dist._path))
    site_packages = dist_info.parent
    top_level = dist.read_text('top_level.txt') or norm(dist.metadata['Name'])
    for module in top_level.split():
        for target in (site_packages / module, site_packages / f'{module}.py'):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
    shutil.rmtree(dist_info, ignore_errors=True)


def unshadow_workspace_package(project_names: set[str]) -> None:
    """Removes non-editable installs of the package under test so /workspace wins."""
    for dist in list(metadata.distributions()):
        name = dist.metadata['Name'] or ''
        if norm(name) in project_names and not is_editable(dist):
            version = dist.version
            remove_installed(dist)
            print(f'pin_deps: removed shadowing {name} {version} from site-packages')


def main() -> None:
    try:
        with open(PYPROJECT, 'rb') as f:
            project = tomllib.load(f).get('project', {})
    except (OSError, tomllib.TOMLDecodeError):
        project = {}

    project_names = {norm(sys.argv[1].rsplit('/', 1)[-1])} if len(sys.argv) > 1 else set()
    if project.get('name'):
        project_names.add(norm(project['name']))
    unshadow_workspace_package(project_names)

    deps = project.get('dependencies', [])
    wheels = available_wheels()
    for dep in deps:
        try:
            req = Requirement(dep)
        except Exception:
            continue
        if not req.specifier or (req.marker and not req.marker.evaluate()):
            continue
        try:
            current = Version(metadata.version(req.name))
        except (metadata.PackageNotFoundError, InvalidVersion):
            continue
        if current in req.specifier:
            continue
        candidates = sorted(w for w in wheels.get(norm(req.name), []) if w[0] in req.specifier)
        if not candidates:
            print(f'pin_deps: no wheel satisfies {req}', file=sys.stderr)
            continue
        version, fname = candidates[-1]
        remove_installed(metadata.distribution(req.name))
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '--root-user-action=ignore', '--no-deps', '--no-index',
             os.path.join(WHEELS_DIR, fname)],
            check=False,
        )
        print(f'pin_deps: {req.name} {current} -> {version}')


if __name__ == '__main__':
    main()
