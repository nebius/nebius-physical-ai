"""Prevent a mobile resume from creating a second writer for a legacy Codex session."""

from pathlib import Path


def owned_elsewhere(thread, socket, proc_root=Path("/proc")):
    """Detect rollout files still held by a different local Codex process.

    Args:
        thread: Persisted Codex thread metadata.
        socket: Socket of the shared runtime that owns this UI.
        proc_root: Linux process filesystem, replaceable by test fixtures.
    Returns:
        Whether another Codex runtime currently owns this rollout.
    Raises:
        OSError: Process ownership cannot be determined.
    """
    if thread.get("status", {}).get("type") != "notLoaded" or not thread.get("path"):
        return False
    rollout = Path(thread["path"]).resolve()
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if (process / "comm").read_text().strip() != "codex":
                continue
            args = (process / "cmdline").read_bytes().split(b"\0")
            if ("unix://" + socket).encode() in args:
                continue
            if any(fd.resolve() == rollout for fd in (process / "fd").iterdir()):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False
