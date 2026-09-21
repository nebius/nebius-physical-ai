"""Guard the CI controls that prevent duplicate and superseded PR work.

Also pins the CI gates to the `make` targets that claim to reproduce them. A gate
a contributor cannot run locally, or can run and get a different answer from, is
how `make lint` came to be red on main for three unused imports that the narrower
CI scope never saw.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
MAKEFILE = REPO_ROOT / "Makefile"
AUTOMATIC_PR_WORKFLOWS = ("security-regression.yml",)


def _load_workflow(name: str) -> dict:
    # BaseLoader preserves the key `on`; PyYAML's YAML 1.1 SafeLoader treats it
    # as a boolean, while GitHub Actions follows YAML 1.2 semantics here.
    with (WORKFLOW_DIR / name).open(encoding="utf-8") as workflow_file:
        return yaml.load(workflow_file, Loader=yaml.BaseLoader)


def test_automatic_pr_workflows_cancel_superseded_commits() -> None:
    discovered = {
        path.name
        for path in WORKFLOW_DIR.glob("*.y*ml")
        if "pull_request" in _load_workflow(path.name)["on"]
    }
    assert discovered == set(AUTOMATIC_PR_WORKFLOWS)

    for name in AUTOMATIC_PR_WORKFLOWS:
        workflow = _load_workflow(name)
        assert "pull_request" in workflow["on"], name
        assert workflow["concurrency"] == {
            "group": "pr-gate-${{ github.event.pull_request.number || github.event.merge_group.head_sha || github.ref }}",
            "cancel-in-progress": "${{ github.event_name == 'pull_request' || github.event_name == 'push' }}",
        }, name


def test_one_pr_workflow_owns_every_merge_gate() -> None:
    """Keep PR updates atomic instead of spawning six independent workflows."""

    workflow = _load_workflow("security-regression.yml")
    assert set(workflow["on"]) == {"pull_request", "merge_group", "push"}
    jobs = workflow["jobs"]
    assert jobs["test-gate"]["uses"] == "./.github/workflows/test.yml"
    assert jobs["lint-gate"]["uses"] == "./.github/workflows/lint.yml"
    assert jobs["guardrails-gate"]["uses"] == (
        "./.github/workflows/harness-guardrails.yml"
    )
    assert jobs["gitleaks"]["name"] == "gitleaks"
    assert jobs["scan"]["name"] == "scan"
    required = set(jobs["security-regression"]["needs"])
    assert required == set(jobs) - {"security-regression", "ci-timing-report"}
    assert jobs["ci-timing-report"]["needs"] == "security-regression"


def test_test_and_lint_do_not_duplicate_feature_branch_pushes() -> None:
    for name in (
        "confidentiality-scan.yml",
        "gitleaks.yml",
        "harness-guardrails.yml",
        "lint.yml",
    ):
        workflow = _load_workflow(name)
        assert workflow["on"]["push"] == {"branches": ["main"]}, name
        assert "pull_request" not in workflow["on"], name

    test = _load_workflow("test.yml")["on"]
    assert set(test) == {"workflow_call", "schedule", "workflow_dispatch"}
    assert test["schedule"] == [{"cron": "23 5 * * *"}]


def test_main_validation_cancels_superseded_commits() -> None:
    for name in (
        "confidentiality-scan.yml",
        "gitleaks.yml",
        "harness-guardrails.yml",
        "lint.yml",
    ):
        cancellation = _load_workflow(name)["concurrency"]["cancel-in-progress"]
        assert cancellation in (
            "true",
            "${{ github.event_name == 'pull_request' || github.event_name == 'push' }}",
        ), name


def test_merge_queue_suite_is_sharded_and_scheduled_audit_keeps_compatibility() -> None:
    workflow = _load_workflow("test.yml")
    job = workflow["jobs"]["test"]
    python_matrix = job["strategy"]["matrix"]["python-version"]
    assert '["pull_request", "merge_group"]' in python_matrix
    assert '["3.12"]' in python_matrix
    assert '["3.10", "3.12", "3.14"]' in python_matrix
    shard_matrix = job["strategy"]["matrix"]["shard"]
    assert '["pull_request", "merge_group"]' in shard_matrix
    assert "[1, 2, 3, 4]" in shard_matrix
    assert "[1, 2, 3, 4, 5]" in shard_matrix
    assert job["strategy"]["fail-fast"] == (
        "${{ github.event_name == 'merge_group' || github.event_name == 'pull_request' }}"
    )
    assert job["needs"] == "scope"
    assert job["if"] == "needs.scope.outputs.full_suite != 'false'"
    assert "continue-on-error" not in job
    assert workflow["on"]["workflow_call"] == ""

    smoke = workflow["jobs"]["pr-smoke"]
    assert smoke["if"] == (
        "needs.scope.outputs.full_suite == 'false' && "
        "(github.event_name == 'pull_request' || needs.scope.outputs.prose_only == 'true')"
    )
    commands = "\n".join(step.get("run", "") for step in smoke["steps"])
    assert "npa/tests/smoke" in commands
    assert "test_ci_workflows.py" in commands

    assert workflow["jobs"]["browser-mocked"]["if"] == (
        "needs.scope.outputs.browser != 'false'"
    )
    browser_steps = workflow["jobs"]["browser-mocked"]["steps"]
    browser_step_names = {step["name"] for step in browser_steps}
    for version in ("3.10", "3.14"):
        assert f"Set up Python {version} compatibility" in browser_step_names
        install_step = f"Install Python {version} compatibility environment"
        assert install_step in browser_step_names
        assert f"Run Python {version} compatibility regressions" in browser_step_names


def test_compatibility_regressions_run_before_heavy_dependencies() -> None:
    job = _load_workflow("test.yml")["jobs"]["test"]
    steps = job["steps"]
    regression = _step("test.yml", "test", "compatibility and image scan")
    install = _step("test.yml", "test", "CPU checkpoint")
    assert steps.index(regression) < steps.index(install)
    assert regression["if"] == "matrix.shard == 1"
    assert regression["env"] == {
        "NPA_CI_SHARD_INDEX": "1",
        "NPA_CI_TOTAL_SHARDS": "1",
    }
    assert "continue-on-error" not in regression
    for path in (
        "npa/tests/guardrails/test_ci_workflows.py",
        "npa/tests/docker/test_base_image_scan.py",
        "npa/tests/orchestration/skypilot/test_workflow_logs.py",
        "npa/tests/workbench/test_cosmos3_nano_video_server.py",
    ):
        assert path in regression["run"]


def test_coverage_shards_are_parallel_and_merged_before_enforcement() -> None:
    workflow = _load_workflow("test.yml")
    test_job = workflow["jobs"]["test"]
    pytest_step = _step("test.yml", "test", "pytest coverage shard")
    assert "-n auto" in pytest_step["run"]
    assert "--dist worksteal" in pytest_step["run"]
    assert "--cov=src/npa" in pytest_step["run"]
    assert "--cov=npa" not in pytest_step["run"]
    assert "--cov-fail-under" not in pytest_step["run"]
    assert test_job["env"] == {
        "NPA_PROJECT_ID": "project-test-00000000",
        "NPA_S3_BUCKET": "test-bucket-00000000",
        "NPA_E2E_PROJECT_ID": "project-test-00000000",
        "NPA_E2E_GROOT_BUCKET": "test-bucket-00000000",
        "NPA_CI_SHARD_INDEX": "${{ matrix.shard }}",
        "NPA_CI_TOTAL_SHARDS": '${{ contains(fromJSON(\'["pull_request", "merge_group"]\'), github.event_name) && 5 || 4 }}',
        "COVERAGE_FILE": ".coverage.${{ matrix.python-version }}.${{ matrix.shard }}",
        "NPA_CI_TIMING_OUTPUT": "ci-timings-${{ matrix.python-version }}-${{ matrix.shard }}.json",
        "NPA_REQUIRE_FFMPEG": "1",
    }

    coverage = workflow["jobs"]["coverage"]
    assert coverage["needs"] == ["scope", "test"]
    assert coverage["if"] == (
        "${{ !cancelled() && needs.scope.outputs.full_suite != 'false' "
        "&& needs.test.result == 'success' }}"
    )
    report = _step("test.yml", "coverage", "merged coverage floor")["run"]
    assert "coverage combine" in report
    assert "--fail-under=60" in report
    profile = _step("test.yml", "coverage", "Merge Python 3.12 duration")["run"]
    assert "merge_ci_test_timings.py" in profile


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", ""])
def test_required_gate_rejects_unsuccessful_children(result: str) -> None:
    """Execute the aggregate check with each unsuccessful child result.

    Args:
        result: Unsuccessful or missing dependency conclusion.
    Returns:
        None.
    Raises:
        AssertionError: A required dependency can fail without rejecting the merge.
    """
    gate = _load_workflow("security-regression.yml")["jobs"]["security-regression"]
    assert gate["if"] == "${{ always() }}"
    step = gate["steps"][0]
    for event in ("pull_request", "merge_group"):
        environment = {name: "success" for name in step["env"]}
        environment["EVENT_NAME"] = event
        for name in environment.keys() - {"EVENT_NAME"}:
            rejected = subprocess.run(
                ["bash", "-c", step["run"]],
                env={**os.environ, **environment, name: result},
                capture_output=True,
            )
            assert rejected.returncode != 0, (event, name, result)
        accepted = subprocess.run(
            ["bash", "-c", step["run"]],
            env={**os.environ, **environment},
            capture_output=True,
        )
        assert accepted.returncode == 0


def test_ci_installers_pin_versions_cache_packages_and_keep_cpu_runtime() -> None:
    """Preserve reproducibility and real CPU coverage when speeding up setup.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: A test environment loses dependency pins or CPU routing.
    """
    jobs = _load_workflow("test.yml")["jobs"]
    for name in ("test", "affected-tests", "pr-smoke", "browser-mocked"):
        setup = next(
            step for step in jobs[name]["steps"] if "setup-uv@" in step.get("uses", "")
        )
        assert setup["with"]["version"] == "0.12.5"
        assert setup["with"]["enable-cache"] == "true"
        assert "npa/ci/requirements.txt" in setup["with"]["cache-dependency-glob"]
        installs = [
            step["run"]
            for step in jobs[name]["steps"]
            if "uv pip install" in step.get("run", "")
        ]
        assert installs and all(
            "-c npa/ci/requirements.txt" in command for command in installs
        )
    for name in ("test", "affected-tests"):
        command = _step(
            "test.yml", name, "CPU checkpoint" if name == "test" else "Install affected"
        )["run"]
        assert "--torch-backend cpu" in command
        assert "assert torch.version.cuda is None" in command
    assert (
        "ci_requirements.py --check"
        in _step("test.yml", "scope", "dependency pins")["run"]
    )


def test_timing_report_is_read_only_and_runs_after_the_required_gate() -> None:
    """Keep timing metadata collection outside the required validation path.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Reporting gains write access or delays a required check.
    """
    workflow = _load_workflow("security-regression.yml")
    job = workflow["jobs"]["ci-timing-report"]
    assert job["needs"] == "security-regression"
    assert job["if"] == "${{ !cancelled() }}"
    assert job["permissions"] == {"actions": "read", "contents": "read"}
    steps = job["steps"]
    assert steps[0]["with"] == {"persist-credentials": "false"}
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "--paginate --slurp" in commands
    assert "/attempts/${GITHUB_RUN_ATTEMPT}" in commands
    assert all("download-artifact" not in step.get("uses", "") for step in steps)
    assert "ci-timing-report" not in workflow["jobs"]["security-regression"]["needs"]


def test_every_python_312_candidate_publishes_module_timings() -> None:
    """Retain failed-shard measurements and merge complete successful profiles.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Timing evidence is confined to scheduled audits.
    """
    upload = _step("test.yml", "test", "Upload Python 3.12 timing")
    assert "always()" in upload["if"]
    assert "matrix.python-version == '3.12'" in upload["if"]
    assert "schedule" not in upload["if"]
    merge = _step("test.yml", "coverage", "Merge Python 3.12 duration")
    assert "if" not in merge


def test_browser_execution_has_one_owner_and_remains_blocking() -> None:
    """Keep real browser coverage without nesting a second run inside pytest.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Browser coverage is duplicated or made optional.
    """
    browser = _load_workflow("test.yml")["jobs"]["browser-mocked"]
    steps = [step for step in browser["steps"] if "cy:mock" in step.get("run", "")]
    assert len(steps) == 1
    assert "if" not in steps[0] and "continue-on-error" not in steps[0]
    assert "continue-on-error" not in browser
    assert browser["needs"] == "scope"
    source = (REPO_ROOT / "npa/tests/cli/test_agent_foxglove.py").read_text()
    assert "def test_ci_executes_mocked_agent_cypress" not in source


def test_test_scope_is_trusted_and_does_not_filter_required_security_jobs() -> None:
    """Keep test shortcuts independent from always-required security and docs gates.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Selection trusts candidate policy or removes required gates.
    """
    scope = _load_workflow("test.yml")["jobs"]["scope"]
    assert scope["steps"][0]["with"]["fetch-depth"] == "0"
    command = scope["steps"][-1]["run"]
    assert 'git show "${BASE_SHA}:${policy}"' in command
    assert '"${RUNNER_TEMP}/ci_test_scope.py"' in command
    assert "echo 'full_suite=true'" in command
    assert "echo 'browser=true'" in command
    parent = _load_workflow("security-regression.yml")
    for event in ("pull_request", "merge_group"):
        assert parent["on"][event] == ""
    for job in ("test-gate", "lint-gate", "guardrails-gate", "gitleaks", "scan"):
        assert parent["jobs"][job]["if"] == "github.event_name != 'push'"
    for job in ("security-scanners", "security-runtime", "image-security"):
        assert "if" not in parent["jobs"][job]


def test_affected_tests_fail_the_candidate_and_use_json_arguments() -> None:
    """Require selected PR tests without treating repository paths as shell code.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Affected tests become optional or bypass subprocess checking.
    """
    job = _load_workflow("test.yml")["jobs"]["affected-tests"]
    assert job["needs"] == "scope"
    assert job["if"] == (
        "github.event_name == 'pull_request' && needs.scope.outputs.full_suite == 'false' "
        "&& needs.scope.outputs.test_paths != '[]'"
    )
    assert "continue-on-error" not in job
    step = job["steps"][-1]
    assert step["env"]["TEST_PATHS"] == "${{ needs.scope.outputs.test_paths }}"
    assert "${{" not in step["run"]
    assert "check=True" in step["run"]
    assert '"-n", "auto"' in step["run"]


def _make_recipe(target: str) -> list[str]:
    """The command lines of one Makefile target, with variables left unexpanded."""

    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if re.match(rf"^{re.escape(target)}\s*:", line)
    )
    recipe = []
    for line in lines[start + 1 :]:
        if line.startswith("\t"):
            recipe.append(line.lstrip("\t").strip())
        elif line.strip() and not line.startswith("#"):
            break
    return recipe


def _make_prereqs(target: str) -> list[str]:
    """The prerequisite targets on a Makefile target's own line."""

    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        match = re.match(rf"^{re.escape(target)}\s*:([^=].*)?$", line)
        if match:
            return (match.group(1) or "").split()
    raise AssertionError(f"no Makefile target named {target!r}")


def _step(workflow: str, job: str, name_fragment: str) -> dict:
    steps = _load_workflow(workflow)["jobs"][job]["steps"]
    matches = [step for step in steps if name_fragment in step.get("name", "")]
    # A renamed step should read as a renamed step, not as StopIteration from a
    # generator, in a file whose whole job is to explain what drifted.
    assert matches, (
        f"no step matching {name_fragment!r} in {workflow} job {job!r}; "
        f"steps are {[step.get('name') for step in steps]}"
    )
    assert len(matches) == 1, (
        f"{name_fragment!r} matches {len(matches)} steps in {workflow} job {job!r}: "
        f"{[step.get('name') for step in matches]}"
    )
    return matches[0]


def test_ci_lints_the_same_tree_as_make_lint() -> None:
    """The blocking ruff scope and `make lint` must not drift apart.

    They already did once: CI checked `src tests` while `make lint` checked the
    whole package, so unused imports collected in npa/docker, npa/examples and
    npa/scripts and only the documented contributor command failed.
    """

    ci_ruff = _step("lint.yml", "ruff", "Ruff check")["run"]
    make_lint = " ".join(_make_recipe("lint"))

    # Both forms run from npa/. Compare the argument that decides the scope.
    assert "cd npa" in ci_ruff and "cd npa" in make_lint
    ci_paths = ci_ruff.split("ruff check", 1)[1].split()
    make_paths = make_lint.split("ruff check", 1)[1].split()
    assert ci_paths == make_paths, (
        f"lint.yml lints {ci_paths} but `make lint` lints {make_paths}; "
        "widen or narrow both together"
    )


def test_docs_drift_gate_matches_the_make_target() -> None:
    ci_docs = _step("lint.yml", "docs-drift", "CLI reference drift")["run"]
    make_docs_check = " ".join(_make_recipe("docs-check"))

    assert "scripts/build_docs.sh --check" in ci_docs
    assert "scripts/build_docs.sh --check" in make_docs_check


def test_guardrail_gate_matches_the_make_target() -> None:
    ci_guardrails = _step("harness-guardrails.yml", "guardrails", "guardrail tests")[
        "run"
    ]
    make_guardrails = " ".join(_make_recipe("test-guardrails"))

    assert "tests/guardrails" in ci_guardrails
    assert "tests/guardrails" in make_guardrails
    assert "-n auto --dist worksteal" in ci_guardrails


def _npa_bin_chosen_by_docs_target(python: str) -> str:
    """What `make docs-check PYTHON=<python>` would hand build_docs.sh as NPA_BIN."""

    recipe = subprocess.run(
        ["make", "-n", "docs-check", f"PYTHON={python}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [line for line in recipe.splitlines() if "scripts/build_docs.sh" in line]
    assert lines, f"docs-check recipe no longer invokes build_docs.sh:\n{recipe}"
    line = lines[0]
    assignment = line.split("bash scripts/build_docs.sh", 1)[0]
    # `printenv` rather than `printf "$NPA_BIN"`: the recipe sets NPA_BIN as a
    # command prefix, which reaches the child's environment but not an argument the
    # calling shell has already expanded.
    return subprocess.run(
        ["sh", "-c", f"{assignment} printenv NPA_BIN || true"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={k: v for k, v in os.environ.items() if k != "NPA_BIN"},
    ).stdout.strip()


def test_docs_target_leaves_npa_bin_unset_when_it_cannot_derive_one(tmp_path) -> None:
    """The docs targets must not shadow build_docs.sh's own npa resolution.

    Deriving NPA_BIN as `dirname $(command -v $(PYTHON))/npa` unconditionally made
    every docs target fail in the setup CONTRIBUTING prescribes: `command -v` is
    empty when the interpreter is not on PATH, so NPA_BIN became `./npa` and
    suppressed both the npa/.venv fallback and any npa that was on PATH.
    """

    assert _npa_bin_chosen_by_docs_target("/nonexistent/python") == ""

    # `python` frequently is not on PATH at all -- only `python3` is -- which is the
    # exact shape that produced "./npa".
    assert _npa_bin_chosen_by_docs_target("definitely-not-a-real-python") == ""


def test_docs_target_uses_the_console_script_beside_an_explicit_python(
    tmp_path,
) -> None:
    """A PYTHON override still selects that interpreter's own npa when one exists."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "npa"):
        script = bin_dir / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

    assert _npa_bin_chosen_by_docs_target(str(bin_dir / "python")) == str(
        bin_dir / "npa"
    )


def test_check_target_does_not_claim_the_coverage_floor() -> None:
    """`make check` is the reproducible subset, not a full stand-in for test.yml.

    test.yml runs pytest with --cov-fail-under, plus the CLI-install and
    source-drift steps. `make test` runs no coverage, so `check` can pass while
    test.yml fails the floor. Whenever that stays true, CONTRIBUTING has to say so
    rather than presenting `make check` as the whole gate.
    """

    coverage_report = _step("test.yml", "coverage", "merged coverage floor")["run"]
    floor = re.search(r"--fail-under=(\d+)", coverage_report)
    assert floor, "test.yml no longer enforces a coverage floor; update CONTRIBUTING"

    # `check` is prerequisites only, so the recipes that matter are its children's.
    check_path = [
        line
        for target in ["check", *_make_prereqs("check")]
        for line in _make_recipe(target)
    ]
    assert check_path, "expected `make check` to reach some recipe"
    assert not any("cov" in line for line in check_path), (
        "`make check` now runs coverage; drop the caveat from CONTRIBUTING instead"
    )

    # Anchor on what the caveat names: the floor plus the two steps test.yml runs
    # that no make target does. Deleting the caveat, or bumping the floor in CI
    # without updating it, fails here instead of quietly overstating `make check`.
    unreproduced_steps = ("Run CLI install test", "Warn on source drift")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for name in unreproduced_steps:
        script = re.search(r"[\w./-]+\.sh", _step("test.yml", "test", name)["run"])
        assert script, f"expected step {name!r} to run a script"
        assert Path(script.group()).name in contributing, (
            f"test.yml step {name!r} runs {script.group()}, which `make check` does "
            "not; CONTRIBUTING has to say so"
        )
    assert f"--cov-fail-under={floor.group(1)}" in contributing, (
        "CONTRIBUTING must name the coverage floor `make check` does not enforce"
    )


def test_advisory_mypy_is_manual_only() -> None:
    lint = _load_workflow("lint.yml")
    typecheck = _load_workflow("typecheck.yml")

    assert "mypy" not in lint["jobs"]
    assert typecheck["on"] == {"workflow_dispatch": ""}
    assert set(typecheck["jobs"]) == {"mypy"}
    assert typecheck["permissions"] == {"contents": "read"}
