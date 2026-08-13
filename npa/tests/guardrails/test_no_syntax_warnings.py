"""Guard: the package must compile without any ``SyntaxWarning``.

A first-time user's very first command is ``npa --version`` (the README verify
step). It once printed ``SyntaxWarning: invalid escape sequence '\\s'`` from an
embedded f-string in ``npa/src/npa/cli/agent.py`` — a single-backslash regex
class written into a non-raw f-string. Worse, ``\\b`` in that position silently
collapses to a backspace byte, corrupting the regex without any warning.

This guard compiles every shipped module's source with the invalid-escape
warning promoted to an error, so any invalid escape sequence anywhere in the
package fails here instead of greeting a new user on install.

Version nuance that matters: CPython only made invalid escape sequences a
``SyntaxWarning`` in 3.12; on 3.10/3.11 they are a ``DeprecationWarning``. Since
``requires-python >= 3.10`` (and Ubuntu 22.04 LTS ships 3.10), the guard must
promote **both** categories, or it silently misses the exact regression it
exists to catch on the versions where new users most often hit it.
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


# Invalid escapes surface as SyntaxWarning (>=3.12) or DeprecationWarning
# (3.10/3.11); promoting both to errors keeps the guard effective across the
# whole supported matrix. When promoted, ``compile()`` raises either the warning
# itself or a SyntaxError depending on the CPython version.
_ESCAPE_WARNING_CATEGORIES = (SyntaxWarning, DeprecationWarning)
_ESCAPE_FAILURES = (SyntaxWarning, DeprecationWarning, SyntaxError)


def test_no_module_emits_syntax_warning() -> None:
    offenders: list[str] = []
    for path in _source_files():
        source = path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            for category in _ESCAPE_WARNING_CATEGORIES:
                warnings.simplefilter("error", category)
            try:
                compile(source, str(path), "exec")
            except _ESCAPE_FAILURES as exc:  # promoted invalid escape / unparseable
                rel = path.relative_to(PACKAGE_SRC.parents[1])
                label = "SyntaxError: " if isinstance(exc, SyntaxError) else ""
                offenders.append(f"{rel}: {label}{exc}")
    assert not offenders, (
        "modules must compile without an invalid-escape warning:\n"
        + "\n".join(offenders)
    )


def test_guard_would_catch_a_bad_escape() -> None:
    """Self-test: an invalid escape must trip the same filter this guard uses.

    The warning category differs by CPython version (``SyntaxWarning`` on >=3.12,
    ``DeprecationWarning`` on 3.10/3.11), and once promoted to an error
    ``compile()`` may raise the warning or a ``SyntaxError`` — so the guard (and
    this self-test) must accept all of them.
    """
    bad = 'x = "\\s"\n'  # a single-backslash-s in a non-raw string literal
    with warnings.catch_warnings():
        for category in _ESCAPE_WARNING_CATEGORIES:
            warnings.simplefilter("error", category)
        with pytest.raises(_ESCAPE_FAILURES):
            compile(bad, "<bad>", "exec")
