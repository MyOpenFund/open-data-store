"""The rename to open-data-store is complete: no old name outside the migration note."""

import re
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = re.compile(r"\bmyopendata\b|MYOPENDATA|my-open-data|my_open_data|MyOpenData|My Open Data")
ALLOWED = {"docs/MIGRATION.md", "tests/test_no_old_name.py"}


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, check=True, capture_output=True, text=True)
    return out.stdout.split("\n")


def test_no_old_name_left():
    hits = []
    for rel in filter(None, _tracked_files()):
        if rel in ALLOWED:
            continue
        if OLD.search(rel):
            hits.append(rel)
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{rel}:{i}" for i, line in enumerate(text.splitlines(), 1) if OLD.search(line)]
    assert not hits, "old name left:\n" + "\n".join(hits)


def test_distribution_cli_and_entry_point_group_renamed():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "open-data-store"
    assert project["scripts"] == {"open-data-store": "open_data_store.cli:main"}
    assert list(project["entry-points"]) == ["open_data_store.contracts"]
