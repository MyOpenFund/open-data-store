"""Publication metadata: the distribution is published on PyPI as `myopenfund-open-data-store`.

The import name (`open_data_store`), the command and the entry-point group never change.
"""

import importlib.metadata
import re
import tomllib
from pathlib import Path

import yaml

import open_data_store

ROOT = Path(__file__).resolve().parent.parent
DIST = "myopenfund-open-data-store"
CREDIT = (
    "Created and maintained by [Marc Jeulin](https://github.com/jeulinmarc), "
    "part of [MyOpenFund](https://github.com/MyOpenFund)."
)
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
BLOB = "https://github.com/MyOpenFund/open-data-store/blob/main/"


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _project():
    return _pyproject()["project"]


def _workflow(path):
    # PyYAML reads the bare key `on` as the boolean True.
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))
    return data


def _upload_index(steps):
    return [
        i for i, s in enumerate(steps) if s.get("uses", "").startswith("actions/upload-artifact@")
    ]


def test_name_and_dynamic_version():
    project = _project()
    assert project["name"] == DIST
    assert "version" in project["dynamic"]
    assert "version" not in project
    assert _pyproject()["tool"]["hatch"]["version"]["path"] == "src/open_data_store/__init__.py"


def test_installed_version_matches_init():
    assert importlib.metadata.version(DIST) == open_data_store.__version__ == "0.1.0"


def test_credit():
    project = _project()
    assert project["authors"] == [{"name": "Marc Jeulin"}]
    assert project["maintainers"] == [{"name": "Marc Jeulin"}]
    assert "myopenfund" in project["keywords"]
    urls = project["urls"]
    assert urls["Organization"] == "https://github.com/MyOpenFund"
    assert urls["Homepage"] == "https://github.com/MyOpenFund/open-data-store"
    assert urls["Source"] == "https://github.com/MyOpenFund/open-data-store"
    assert urls["Issues"] == "https://github.com/MyOpenFund/open-data-store/issues"
    assert "Changelog" not in urls  # no CHANGELOG in this repo
    assert project["readme"] == "README.md"


def test_license_spdx_without_classifier_and_python_classifiers():
    project = _project()
    assert project["license"] == "MIT"
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    for version in ("3.11", "3.12", "3.13"):
        assert f"Programming Language :: Python :: {version}" in project["classifiers"]


def test_license_copyright():
    assert "Copyright (c) 2026 Marc Jeulin" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_readme_credit_install_and_migration():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# open-data-store"
    assert [line for line in lines[1:] if line.strip()][0] == CREDIT
    for extra in ("writer", "publisher", "reader"):
        assert f'pip install "{DIST}[{extra}]"' in text
    assert 'pip install "open-data-store[' not in text
    assert "pip uninstall -y open-data-store" in text
    assert "git+https://github.com/MyOpenFund/open-data-store" in text
    assert "Working across repos" in text


def test_readme_links_are_absolute():
    # README.md is the PyPI description: relative links would 404 there.
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "](#" not in text
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    assert targets
    relative = [t for t in targets if not t.startswith("https://")]
    assert not relative, relative
    assert f"{BLOB}docs/MIGRATION.md" in text


def test_dev_extra_has_pyyaml():
    assert any(d.startswith("pyyaml") for d in _project()["optional-dependencies"]["dev"])


def test_no_git_urls_in_metadata():
    project = _project()
    deps = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        deps += extra
    assert not [d for d in deps if "git+" in d or "@" in d]


def test_release_workflow_publishes_through_environment():
    workflow = _workflow(RELEASE)
    assert workflow["permissions"] == {}
    assert workflow["on"]["push"]["tags"] == ["v*"]
    assert list(workflow["on"]["workflow_dispatch"]["inputs"]) == ["target"]
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == (
        "${{ github.event_name == 'workflow_dispatch' && 'testpypi' || 'pypi' }}"
    )
    assert publish["permissions"] == {"id-token": "write"}
    with_id_token = [
        name for name, job in workflow["jobs"].items() if "id-token" in job.get("permissions", {})
    ]
    assert with_id_token == ["publish"]
    assert publish["needs"] in ("build", ["build"])


def test_actions_pinned_by_sha():
    ci = yaml.safe_load(CI.read_text(encoding="utf-8"))
    texts = [RELEASE.read_text(encoding="utf-8"), yaml.safe_dump(ci["jobs"]["package"])]
    for text in texts:
        uses = re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)", text, re.M)
        assert uses
        for ref in uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_tag_check_present_with_prefix_v():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    checks = [
        i for i, s in enumerate(steps) if "tag_version" in s.get("run", "") and "exit 1" in s["run"]
    ]
    assert len(checks) == 1
    assert steps[checks[0]]["if"] == "github.event_name == 'push'"
    assert "${GITHUB_REF_NAME#v}" in steps[checks[0]]["run"]
    assert DIST in steps[checks[0]]["run"]
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_build_checks_tag_is_on_default_branch():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    checks = [i for i, s in enumerate(steps) if "merge-base --is-ancestor" in s.get("run", "")]
    assert len(checks) == 1
    assert "origin/main" in steps[checks[0]]["run"]
    assert steps[checks[0]]["if"] == "github.event_name == 'push'"
    assert steps[0]["with"]["fetch-depth"] == 0
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_testpypi_publish_skips_existing_but_pypi_does_not():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert steps["Publish to TestPyPI"]["with"]["skip-existing"] is True
    assert "skip-existing" not in steps["Publish to PyPI"].get("with", {})


def test_concurrency_group():
    assert _workflow(RELEASE)["concurrency"]["group"] == "release-${{ github.ref }}"


def test_github_release_is_push_only_and_verifies_tag():
    job = _workflow(RELEASE)["jobs"]["github-release"]
    assert job["if"] == "github.event_name == 'push'"
    runs = " ".join(s.get("run", "") for s in job["steps"])
    assert "--verify-tag" in runs
    assert "--generate-notes" in runs


def test_smoke_uses_cli_help_and_reader_extra():
    for path in (RELEASE, CI):
        text = path.read_text(encoding="utf-8")
        assert '/bin/open-data-store" --help' in text, path
        assert "[reader]" in text, path


def test_dependabot_covers_github_actions_only():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    assert [(u["package-ecosystem"], u["schedule"]["interval"]) for u in config["updates"]] == [
        ("github-actions", "weekly")
    ]


def test_ci_keeps_its_jobs_and_adds_package():
    jobs = _workflow(CI)["jobs"]
    assert {"test", "stack", "package"} <= set(jobs)
    runs = " ".join(s.get("run", "") for s in jobs["package"]["steps"])
    assert "build==1.6.1 twine==6.2.0" in runs
    assert "twine check --strict" in runs


def test_nas_compose_installs_the_published_distribution_name():
    text = (ROOT / "deploy" / "nas" / "compose.yaml").read_text(encoding="utf-8")
    assert '"myopenfund-open-data-store[publisher] @ git+' in text
    assert '"open-data-store[' not in text
