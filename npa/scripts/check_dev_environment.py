"""Fail fast when the invoking interpreter's `npa` import does not resolve to this checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


def _repo_root() -> Path:
    # This file lives at <repo_root>/npa/scripts/check_dev_environment.py.
    return Path(__file__).resolve().parent.parent.parent


def _new_venv_recipe(npa_dir: Path) -> str:
    """One shell-safe, self-aborting recipe: never touches a path that already exists.

    Args:
        npa_dir: The checkout's `npa/` directory.
    Returns:
        A single `&&`-chained command line. `test ! -e` and `test ! -L` both
        have to pass before anything is created, so a name collision (a real
        directory, a file, or a symlink already at that path) aborts the
        whole line with no side effect, rather than reinitializing or
        following whatever is already there.
    Raises:
        None.
    """
    venv = shlex.quote(str(npa_dir / ".venv-local"))
    pip = shlex.quote(str(npa_dir / ".venv-local" / "bin" / "pip"))
    target = shlex.quote(f"{npa_dir}[dev,adapter]")
    return f"test ! -e {venv} && test ! -L {venv} && python3 -m venv {venv} && {pip} install -e {target}"


def _missing_install_message(expected_src: Path, npa_dir: Path, error: BaseException) -> str:
    src = shlex.quote(str(expected_src))
    return (
        f"error: `import npa` failed under {sys.executable}: {error}\n"
        "This interpreter cannot import this checkout's package. Point PYTHONPATH at\n"
        "this checkout's source instead of installing into an interpreter whose\n"
        "provenance this check cannot confirm:\n"
        f"  export PYTHONPATH={src}\n"
        "or create a new virtualenv owned by this checkout (aborts safely instead of\n"
        "touching anything that already exists at that path):\n"
        f"  {_new_venv_recipe(npa_dir)}"
    )


def _drift_message(expected_src: Path, resolved_src: Path, npa_dir: Path) -> str:
    src = shlex.quote(str(expected_src))
    return (
        f"error: {sys.executable} imports `npa` from:\n"
        f"  {resolved_src}\n"
        "not this checkout's source:\n"
        f"  {expected_src}\n"
        "\n"
        "This happens when npa/.venv is shared (for example symlinked) across git\n"
        "worktrees or clones: the venv's editable install still points at whichever\n"
        "checkout last ran `pip install -e`, so pytest collects THIS checkout's test\n"
        "files but exercises a DIFFERENT checkout's production code, with no error\n"
        "or non-zero exit to flag it. Fix one of:\n"
        "\n"
        "  1. Point PYTHONPATH at this checkout's source (safe, no side effects):\n"
        f"       export PYTHONPATH={src}\n"
        "  2. Give this checkout a new virtualenv, without touching npa/.venv itself:\n"
        f"       {_new_venv_recipe(npa_dir)}"
    )


def check(python_executable: str, repo_root: Path) -> str | None:
    """Return an error message if `npa` does not resolve to `repo_root`, else None.

    Args:
        python_executable: Path reported by the interpreter running this check
            (normally `sys.executable`); only used in messages.
        repo_root: Root of the checkout this process should be validating.

    Returns:
        A human-readable error message, or `None` when the import is correctly
        scoped to this checkout.

    Raises:
        None.
    """
    expected_src = (repo_root / "npa" / "src").resolve()
    npa_dir = repo_root / "npa"

    try:
        import npa
    except ImportError as error:
        return _missing_install_message(expected_src, npa_dir, error)

    resolved_src = Path(npa.__file__).resolve().parent.parent
    if resolved_src != expected_src:
        return _drift_message(expected_src, resolved_src, npa_dir)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="Checkout root to validate against (default: this script's own checkout).",
    )
    args = parser.parse_args(argv)

    error = check(sys.executable, args.repo_root)
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
