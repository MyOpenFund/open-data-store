"""The rename to open-data-store is complete: no old name outside the migration note."""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OLD = re.compile(rb"my[-_ ]?open[-_ ]?data", re.IGNORECASE)
ALLOWED = {"docs/MIGRATION.md", "tests/test_no_old_name.py"}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.stdout.split("\0") if f]


def test_no_old_name_left():
    hits = []
    for rel in _tracked_files():
        if rel in ALLOWED:
            continue
        if OLD.search(rel.encode()):
            hits.append(rel)
        try:
            data = (REPO / rel).read_bytes()
        except OSError:  # deleted in the work tree, a directory (submodule), ...
            continue
        hits += [f"{rel}:{i}" for i, line in enumerate(data.splitlines(), 1) if OLD.search(line)]
    assert not hits, "old name left:\n" + "\n".join(hits)


def test_distribution_cli_and_entry_point_group_renamed():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "myopenfund-open-data-store"
    assert project["scripts"] == {"open-data-store": "open_data_store.cli:main"}
    assert list(project["entry-points"]) == ["open_data_store.contracts"]
