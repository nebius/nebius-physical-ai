"""Guard: the package must compile without any ``SyntaxWarning``.

A first-time user's very first command is ``npa --version`` (the README verify
step). It once printed ``SyntaxWarning: invalid escape sequence '\\s'`` from an
embedded f-string in ``npa/src/npa/cli/agent.py`` — a single-backslash regex
class written into a non-raw f-string. Worse, ``\\b`` in that position silently
collapses to a backspace byte, corrupting the regex without any warning.

This guard compiles every shipped module's source with ``SyntaxWarning``
promoted to an error, so any invalid escape sequence anywhere in the package
fails here instead of greeting a new user on install.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest


PACKAGE_SRC = Path(__file__).resolve().parents[2] / "src" / "npa"


def _source_files() -> list[Path]:
    return sorted(PACKAGE_SRC.rglob("*.py"))


def test_package_has_source_files() -> None:
    # Sanity: the glob must actually find modules, or the guard is vacuous.
    files = _source_files()
    assert files, f"no python sources found under {PACKAGE_SRC}"
    assert (PACKAGE_SRC / "cli" / "agent.py") in files


def test_no_module_emits_syntax_warning() -> None:
    offenders: list[str] = []
    for path in _source_files():
        source = path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            try:
                compile(source, str(path), "exec")
            except SyntaxWarning as exc:  # promoted invalid escape / etc.
                rel = path.relative_to(PACKAGE_SRC.parents[1])
                offenders.append(f"{rel}: {exc}")
            except SyntaxError as exc:  # genuinely un-parseable module
                rel = path.relative_to(PACKAGE_SRC.parents[1])
                offenders.append(f"{rel}: SyntaxError: {exc}")
    assert not offenders, "modules must compile without SyntaxWarning:\n" + "\n".join(offenders)


def test_guard_would_catch_a_bad_escape() -> None:
    """Self-test: an invalid escape must trip the same filter this guard uses.

    With ``SyntaxWarning`` promoted to an error, CPython surfaces the invalid
    escape as a ``SyntaxError`` at ``compile()`` time, so the guard catches both.
    """
    bad = 'x = "\\s"\n'  # a single-backslash-s in a non-raw string literal
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        with pytest.raises((SyntaxWarning, SyntaxError)):
            compile(bad, "<bad>", "exec")
