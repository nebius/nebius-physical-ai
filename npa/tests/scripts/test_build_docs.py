"""Guardrail: the CLI-reference generator refuses to run without a working npa.

`docs/cli/` is generated from live `npa --help` and drift-gated by `lint.yml`, and
the generator starts from a clean slate so removed commands do not linger as
orphan pages. Every individual help fetch tolerates a non-zero exit, because a
leaf command may legitimately fail `--help`.

Those two facts combined were a data-loss bug. `npa` is only on PATH while a venv
is activated, so running the script without one documented nothing: the in-place
mode deleted every page under `docs/cli/` and exited 0, and `--check` reported the
entire reference as drifted. Both failure modes looked like a docs problem rather
than a missing interpreter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "build_docs.sh"
DOCS_CLI = REPO_ROOT / "docs" / "cli"


def _run(args: list[str], npa_bin: str, cwd: Path) -> subprocess.CompletedProcess:
    # An empty PATH is not enough on its own: the script resolves npa/.venv/bin/npa
    # relative to its own location, so the copied tree must not contain one.
    env = {**os.environ, "NPA_BIN": npa_bin, "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A minimal copy of the script plus the pages it would overwrite."""

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    for helper in ("_generate_docs_index.py", "_help_to_markdown.py"):
        shutil.copy2(SCRIPT.parent / helper, root / "scripts" / helper)
    shutil.copytree(DOCS_CLI, root / "docs" / "cli")
    return root


@pytest.mark.parametrize("args", [[], ["--check"]])
def test_missing_npa_fails_loudly(checkout: Path, args: list[str]) -> None:
    before = sorted(p.name for p in (checkout / "docs" / "cli").glob("*.md"))
    assert before, "fixture should copy real pages"

    result = _run(args, npa_bin="npa-does-not-exist", cwd=checkout)

    # Assert the harm before the exit status: the pre-fix script reported
    # "Docs generated for groups:" with an empty list and exited 0, having already
    # cleared the directory.
    after = sorted(p.name for p in (checkout / "docs" / "cli").glob("*.md"))
    assert after == before, "regeneration without a working npa emptied docs/cli"
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot run" in result.stderr


def test_unreadable_npa_is_not_treated_as_a_working_cli(checkout: Path) -> None:
    """An executable that exists but cannot serve --help must fail the same way."""

    broken = checkout / "broken-npa"
    broken.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    broken.chmod(0o755)

    result = _run(["--check"], npa_bin=str(broken), cwd=checkout)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot run" in result.stderr
