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

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
MAKEFILE = REPO_ROOT / "Makefile"
AUTOMATIC_PR_WORKFLOWS = (
    "confidentiality-scan.yml",
    "gitleaks.yml",
    "harness-guardrails.yml",
    "image-security-scan.yml",
    "lint.yml",
    "test.yml",
)


def _load_workflow(name: str) -> dict:
    # BaseLoader preserves the key `on`; PyYAML's YAML 1.1 SafeLoader treats it
    # as a boolean, while GitHub Actions follows YAML 1.2 semantics here.
    with (WORKFLOW_DIR / name).open(encoding="utf-8") as workflow_file:
        return yaml.load(workflow_file, Loader=yaml.BaseLoader)


def test_automatic_pr_workflows_cancel_superseded_commits() -> None:
    expected_group = (
        "${{ github.workflow }}-"
        "${{ github.event.pull_request.number || github.run_id }}"
    )
    expected_cancel = "${{ github.event_name == 'pull_request' }}"

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
            "group": expected_group,
            "cancel-in-progress": expected_cancel,
        }, name


def test_test_and_lint_do_not_duplicate_feature_branch_pushes() -> None:
    for name in ("test.yml", "lint.yml"):
        workflow = _load_workflow(name)
        assert workflow["on"]["push"] == {"branches": ["main"]}, name


def test_pr_test_matrix_uses_one_version_and_main_keeps_compatibility() -> None:
    workflow = _load_workflow("test.yml")
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]

    assert "github.event_name == 'pull_request'" in versions
    assert "[\"3.12\"]" in versions
    assert "[\"3.10\",\"3.12\",\"3.14\"]" in versions


def test_test_workflow_uses_the_pinned_development_ffmpeg() -> None:
    """Media validation must not silently skip or wait on runner OS mutation."""

    workflow = _load_workflow("test.yml")
    job = workflow["jobs"]["test"]
    steps = job["steps"]
    install_index = next(
        index for index, step in enumerate(steps) if step.get("name") == "Install npa"
    )
    ffmpeg_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Configure test ffmpeg"
    )
    ffmpeg_step = steps[ffmpeg_index]["run"]

    assert job["env"]["NPA_REQUIRE_FFMPEG"] == "1"
    assert install_index < ffmpeg_index
    assert "imageio_ffmpeg.get_ffmpeg_exe()" in ffmpeg_step
    assert 'test -x "$ffmpeg_bin"' in ffmpeg_step
    assert 'ln -s "$ffmpeg_bin" "$test_bin/ffmpeg"' in ffmpeg_step
    assert re.search(
        r"raw\.githubusercontent\.com/imageio/imageio-binaries/"
        r"[0-9a-f]{40}/ffmpeg/ffprobe-linux64-v4\.1",
        ffmpeg_step,
    )
    assert re.search(r"[0-9a-f]{64}  \$test_bin/ffprobe", ffmpeg_step)
    assert "sha256sum --check --strict" in ffmpeg_step
    assert 'echo "$test_bin" >> "$GITHUB_PATH"' in ffmpeg_step
    assert '"$test_bin/ffprobe" -version' in ffmpeg_step
    assert "apt-get" not in ffmpeg_step


def test_test_workflow_fetches_history_for_pinned_source_provenance() -> None:
    """Reviewed hosted-source commits must be resolvable in the CI checkout."""

    checkout = _step("test.yml", "test", "Check out repository")

    assert checkout["uses"] == "actions/checkout@v6"
    assert checkout["with"]["fetch-depth"] == "0"


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

    ci_pytest = _step("test.yml", "test", "pytest")["run"]
    floor = re.search(r"--cov-fail-under=(\d+)", ci_pytest)
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
