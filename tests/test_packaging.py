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
    assert importlib.metadata.version(DIST) == open_data_store.__version__


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
        "${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi')"
        " && 'testpypi' || 'pypi' }}"
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
    checkout = [s for s in steps if s.get("uses", "").startswith("actions/checkout@")]
    assert checkout[0]["with"]["fetch-depth"] == 0
    upload = _upload_index(steps)
    assert upload and checks[0] < upload[0]


def test_testpypi_publish_skips_existing_but_pypi_does_not():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert steps["Publish to TestPyPI"]["with"]["skip-existing"] is True
    assert "skip-existing" not in steps["Publish to PyPI"].get("with", {})


def test_concurrency_group():
    assert _workflow(RELEASE)["concurrency"]["group"] == "release-${{ github.ref }}"


def test_dispatch_inputs():
    inputs = _workflow(RELEASE)["on"]["workflow_dispatch"]["inputs"]
    assert list(inputs) == ["target"]
    assert inputs["target"]["type"] == "choice"
    assert inputs["target"]["options"] == ["pypi", "testpypi"]
    assert inputs["target"]["default"] == "pypi"
    assert inputs["target"]["required"] is True


def test_build_dispatch_runs_on_default_branch_only():
    steps = _workflow(RELEASE)["jobs"]["build"]["steps"]
    guard = [s for s in steps if "refs/heads/main" in s.get("run", "")]
    assert len(guard) == 1
    assert guard[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert "exit 1" in guard[0]["run"]
    assert steps.index(guard[0]) == 0


def test_build_computes_tag_and_refuses_an_existing_pypi_tag():
    job = _workflow(RELEASE)["jobs"]["build"]
    step = [s for s in job["steps"] if s.get("id") == "tag"]
    assert len(step) == 1
    run = step[0]["run"]
    assert step[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert "importlib.metadata" in run and DIST in run
    assert 'tag="v$version"' in run
    assert 'git ls-remote --tags origin "refs/tags/$tag"' in run
    assert '"$INPUT_TARGET" = "pypi"' in run
    assert "already exists" in run and "exit 1" in run
    upload = _upload_index(job["steps"])
    assert job["steps"].index(step[0]) < upload[0]
    assert job["outputs"]["tag"] == "${{ steps.tag.outputs.tag || github.ref_name }}"
    assert job["outputs"]["sha"] == "${{ github.sha }}"


def test_testpypi_only_for_testpypi_target():
    steps = {s["name"]: s for s in _workflow(RELEASE)["jobs"]["publish"]["steps"] if "name" in s}
    assert "inputs.target == 'testpypi'" in steps["Publish to TestPyPI"]["if"]
    assert "inputs.target == 'testpypi'" in steps["Publish to PyPI"]["if"]
    assert steps["Publish to PyPI"]["if"].startswith("${{ !(")


def _release_job():
    job = _workflow(RELEASE)["jobs"]["github-release"]
    assert job["if"] == (
        "github.event_name == 'push' || "
        "(github.event_name == 'workflow_dispatch' && inputs.target == 'pypi')"
    )
    assert job["permissions"] == {"contents": "write"}
    assert set(job["needs"]) == {"build", "publish"}
    return job


def _assert_tag_created_and_pushed(run):
    # no personal address in a public workflow: the tagger email is the actor's noreply one
    assert 'user.name="Marc Jeulin"' in run
    assert 'user.email="${GITHUB_ACTOR_ID}+${GITHUB_ACTOR}@users.noreply.github.com"' in run
    assert 'tag -a "$TAG"' in run and '"$SHA"' in run
    assert 'git push origin "refs/tags/$TAG"' in run


def test_github_release_creates_and_pushes_the_tag_on_dispatch_not_testpypi():
    job = _release_job()
    checkout = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkout) == 1
    assert checkout[0]["if"] == "github.event_name == 'workflow_dispatch'"
    assert checkout[0]["with"]["persist-credentials"] is True
    assert checkout[0]["with"]["ref"] == "${{ needs.build.outputs.sha }}"
    tag = [s for s in job["steps"] if "git push" in s.get("run", "")]
    assert len(tag) == 1
    assert tag[0]["if"] == "github.event_name == 'workflow_dispatch'"
    _assert_tag_created_and_pushed(tag[0]["run"])


def test_dispatch_tag_step_is_idempotent():
    run = next(s for s in _release_job()["steps"] if "git push" in s.get("run", ""))["run"]
    lookup = run.index('git ls-remote origin "refs/tags/$TAG" "refs/tags/$TAG^{}"')
    assert lookup < run.index("git push")
    assert '"$remote_sha" = "$SHA"' in run  # already there at the built commit: skip
    assert "already exists" in run and "exit 1" in run  # there, but elsewhere: fail
    assert '[ -z "$remote_sha" ]' in run  # absent: create


def test_github_release_is_created_or_completed_on_both_paths():
    job = _release_job()
    release = [s for s in job["steps"] if "gh release create" in s.get("run", "")]
    assert len(release) == 1
    assert "if" not in release[0]
    run = release[0]["run"]
    assert 'gh release view "$TAG"' in run
    assert 'gh release upload "$TAG" dist/* --repo "$GITHUB_REPOSITORY" --clobber' in run
    assert '--target "$SHA"' in run and '--title "$TAG"' in run
    assert "--verify-tag" in run
    assert "--generate-notes" in run


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
    assert "build==1.6.1 twine==7.0.0" in runs
    assert "twine check --strict" in runs


def test_nas_compose_installs_the_published_distribution_name():
    text = (ROOT / "deploy" / "nas" / "compose.yaml").read_text(encoding="utf-8")
    assert '"myopenfund-open-data-store[publisher] @ git+' in text
    assert '"open-data-store[' not in text


def test_recipes_extra_pins_its_runtime():
    extras = _project()["optional-dependencies"]
    for name in ("recipes", "dev"):
        deps = extras[name]
        assert "exchange_calendars==4.13.2" in deps
        assert any(re.fullmatch(r"numpy(>=[\d.]+)?", d) for d in deps), deps
        assert any(d.startswith("pandas") and "<4" in d for d in deps), deps


# ---- fail-closed approval gate (release hardening) -------------------------


def _gate_job():
    job = _workflow(RELEASE)["jobs"]["build"]
    gate = [i for i, s in enumerate(job["steps"]) if s.get("name") == "Check the approval gate"]
    assert len(gate) == 1
    return job, gate[0]


def test_release_checks_the_approval_gate_before_building():
    job, gate = _gate_job()
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    step = job["steps"][gate]
    run = step["run"]
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "inputs.target == 'testpypi'" in step["env"]["ENV_NAME"]
    assert "'testpypi'" in step["env"]["ENV_NAME"] and "'pypi'" in step["env"]["ENV_NAME"]
    assert 'gh api "repos/$GITHUB_REPOSITORY/environments/$ENV_NAME"' in run
    assert "if ! " in run  # an API failure (non-200) is tested, not ignored
    assert "required_reviewers" in run and "exit 1" in run
    for i, other in enumerate(job["steps"]):
        uses = other.get("uses", "")
        building = uses.startswith(
            ("actions/checkout@", "actions/setup-python@", "actions/upload-artifact@")
        ) or "build" in other.get("run", "")
        if i != gate and (building or other.get("name", "").startswith("Build")):
            assert gate < i, other


def _run_gate(tmp_path, response, status):
    """Run the gate step against a stub `gh` that prints `response` and exits `status`."""
    import os
    import shutil
    import subprocess

    import pytest

    if shutil.which("jq") is None or shutil.which("bash") is None:
        pytest.skip("needs jq and bash")
    job, gate = _gate_job()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "gh"
    stub.write_text(f"#!/bin/sh\ncat <<'JSON'\n{response}\nJSON\nexit {status}\n")
    stub.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "MyOpenFund/example",
        "ENV_NAME": "pypi",
    }
    return subprocess.run(
        ["bash", "-e", "-c", job["steps"][gate]["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_gate_passes_with_a_required_reviewer(tmp_path):
    rules = (
        '{"protection_rules": [{"type": "wait_timer"}, {"type": "required_reviewers",'
        ' "reviewers": [{"type": "User", "reviewer": {"login": "release-approver"}}]}]}'
    )
    assert _run_gate(tmp_path, rules, 0).returncode == 0


def test_gate_fails_without_a_required_reviewer(tmp_path):
    for rules in (
        '{"protection_rules": []}',
        '{"protection_rules": [{"type": "required_reviewers", "reviewers": []}]}',
        '{"protection_rules": [{"type": "branch_policy"}]}',
        "{}",
    ):
        result = _run_gate(tmp_path, rules, 0)
        assert result.returncode == 1, rules
        assert "required reviewer" in result.stdout + result.stderr


def test_gate_fails_closed_when_the_api_call_fails(tmp_path):
    result = _run_gate(tmp_path, "gh: Not Found (HTTP 404)", 1)
    assert result.returncode == 1
    assert "HTTP 404" in result.stdout + result.stderr


def test_how_to_release_names_the_version_source():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("**How to release**", 1)[1].split("\n\n", 3)[1]
    assert "`__version__` in `src/open_data_store/__init__.py`" in section
    assert "and `version`" not in section
    assert "pip install -e" in section


def test_dev_extra_has_twine_7_for_metadata_2_5():
    # hatchling 1.33 writes Metadata-Version 2.5, which twine 6.2 rejects
    dev = _project()["optional-dependencies"]["dev"]
    assert "twine>=7" in dev
    assert not [d for d in dev if d.startswith("twine") and d != "twine>=7"]
