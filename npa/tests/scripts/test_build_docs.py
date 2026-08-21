"""Guardrail: the CLI-reference generator never publishes an empty reference.

`docs/cli/` is generated from live `npa --help` and drift-gated by `lint.yml`.
Every individual help fetch tolerates a non-zero exit, because a leaf command may
legitimately fail `--help`, and the generator used to clear the directory up front
so removed commands could not linger as orphan pages.

Together those made an empty walk destructive: an in-place run deleted all pages,
printed ``Docs generated for groups:`` with an empty list and exited 0, while
``--check`` reported the whole reference as drifted. Two inputs reach that state,
and neither means the CLI is broken:

* no runnable ``npa`` at all -- it is only on PATH while a venv is activated; and
* an ``npa`` that answers ``--help`` fine but whose output no longer matches the
  Rich box-drawing regex in ``discover_commands`` (a Typer/Rich upgrade, or a
  ``COLUMNS`` value that rewraps the table).

Every case here asserts the pages survived, because that is the actual harm.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "build_docs.sh"
DOCS_CLI = REPO_ROOT / "docs" / "cli"
SCRIPT_HELPERS = ("_generate_docs_index.py", "_help_to_markdown.py")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A minimal copy of the generator plus the pages it would overwrite.

    The copied script is the one executed, not the repo's. build_docs.sh resolves
    npa/.venv relative to its own location, so running the original would consult
    the real checkout and make these tests pass or fail depending on whether the
    developer happens to have a venv there.
    """

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts" / SCRIPT.name)
    for helper in SCRIPT_HELPERS:
        shutil.copy2(SCRIPT.parent / helper, root / "scripts" / helper)
    shutil.copytree(DOCS_CLI, root / "docs" / "cli")
    return root


def _pages(checkout: Path) -> list[str]:
    return sorted(p.name for p in (checkout / "docs" / "cli").glob("*.md"))


def _fake_npa(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run(
    checkout: Path, args: list[str], npa_bin: str | None = None
) -> subprocess.CompletedProcess:
    # A pruned PATH is not enough on its own: the script falls back to
    # npa/.venv/bin/npa under its own directory, so the fixture must not have one
    # unless a test puts it there deliberately.
    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    if npa_bin is None:
        env.pop("NPA_BIN", None)
    else:
        env["NPA_BIN"] = npa_bin
    return subprocess.run(
        ["bash", str(checkout / "scripts" / SCRIPT.name), *args],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("args", [[], ["--check"]])
def test_missing_npa_fails_without_touching_docs(
    checkout: Path, args: list[str]
) -> None:
    before = _pages(checkout)
    assert before, "fixture should copy real pages"

    result = _run(checkout, args, npa_bin="npa-does-not-exist")

    assert _pages(checkout) == before, "a failed run emptied docs/cli"
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot run" in result.stderr


def test_npa_that_cannot_serve_help_is_not_treated_as_working(checkout: Path) -> None:
    """An executable that exists but exits non-zero on --help must fail the same way."""

    broken = _fake_npa(checkout / "broken-npa", "exit 3")

    result = _run(checkout, ["--check"], npa_bin=str(broken))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot run" in result.stderr


@pytest.mark.parametrize("args", [[], ["--check"]])
def test_help_without_a_command_table_fails_without_touching_docs(
    checkout: Path, args: list[str]
) -> None:
    """The realistic regression: a healthy npa whose help no longer parses.

    This npa succeeds, prints a plausible banner, and offers nothing the
    box-drawing regex in `discover_commands` can match -- which is what a Typer or
    Rich rendering change looks like from here.
    """

    healthy_but_unparsable = _fake_npa(
        checkout / "quiet-npa",
        'echo "Usage: npa [OPTIONS] COMMAND [ARGS]..."\necho "Nebius Physical AI"',
    )
    before = _pages(checkout)

    result = _run(checkout, args, npa_bin=str(healthy_but_unparsable))

    assert _pages(checkout) == before, "an unparsable command table emptied docs/cli"
    assert result.returncode == 1, result.stdout + result.stderr
    assert "found no commands" in result.stderr
    # The message has to point at the parser, or the next reader re-debugs the CLI.
    assert "discover_commands" in result.stderr


def test_venv_npa_is_used_when_nothing_is_on_path(checkout: Path) -> None:
    """With NPA_BIN unset the generator falls back to the repo venv, not PATH.

    Proven through the failure path: the fallback npa is deliberately broken, so
    the error naming its path is evidence that it was the binary selected.
    """

    venv_npa = _fake_npa(checkout / "npa" / ".venv" / "bin" / "npa", "exit 3")

    result = _run(checkout, ["--check"], npa_bin=None)

    assert result.returncode == 1, result.stdout + result.stderr
    assert str(venv_npa) in result.stderr
