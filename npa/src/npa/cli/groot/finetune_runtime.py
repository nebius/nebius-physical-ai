"""Execution helpers for GR00T fine-tuning commands."""

from __future__ import annotations

import shlex
import subprocess


def run_local_finetune(command: str, *, stream: bool) -> tuple[int, str, str]:
    """Run the pinned GR00T trainer in the current GPU container."""

    completed = subprocess.run(
        shlex.split(command),
        text=True,
        stdout=None if stream else subprocess.PIPE,
        stderr=None if stream else subprocess.PIPE,
        check=False,
    )
    return completed.returncode, completed.stdout or "", completed.stderr or ""
