"""Resolve installed Codex executables after extension updates remove old paths."""

import os
from pathlib import Path
import platform
import shutil
import sys


def resolve_executable(preferred=None):
    """Keep an available executable, or discover its installed replacement.

    Args:
        preferred: Previously configured executable, when available.
    Returns:
        An executable path for the current operating system and architecture.
    Raises:
        RuntimeError: No installed Codex executable is available.
    """
    if preferred and _executable(Path(preferred)):
        return str(preferred)
    system = "macos" if platform.system() == "Darwin" else "linux"
    machine = "aarch64" if platform.machine() in {"arm64", "aarch64"} else "x86_64"
    candidates = []
    for directory in (
        ".vscode",
        ".vscode-insiders",
        ".vscode-server",
        ".vscode-server-insiders",
    ):
        pattern = f"extensions/openai.chatgpt-*/bin/{system}-{machine}/codex"
        candidates.extend(
            path
            for path in (Path.home() / directory).glob(pattern)
            if _executable(path)
        )
    if candidates:
        return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))
    binary = shutil.which("codex")
    if binary:
        return binary
    raise RuntimeError("Install Codex and sign in on this host before opening chat.")


def _executable(path):
    return path.is_file() and os.access(path, os.X_OK)


def main(arguments=None):
    """Replace this launcher with Codex while preserving arguments and signals.

    Args:
        arguments: Preferred executable followed by unchanged Codex arguments.
    Returns:
        None; successful execution replaces the launcher process.
    Raises:
        RuntimeError: Codex is unavailable or the preferred path was omitted.
        OSError: The selected executable cannot start.
    """
    arguments = sys.argv[1:] if arguments is None else arguments
    if not arguments:
        raise RuntimeError("Supply the configured Codex executable.")
    binary = resolve_executable(arguments[0])
    os.execv(binary, [binary, *arguments[1:]])


if __name__ == "__main__":
    main()
