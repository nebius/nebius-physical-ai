"""Resolve a harness's output path, and refuse early if it cannot be written.

Eight harnesses here write to a hardcoded `/data/<name>.json`. That is correct inside the
container they document, which mounts the evidence directory at `/data`. Outside it, the
directory does not exist, and the failure arrives at the *last* line: the harness runs the
whole measurement and then loses it to a `FileNotFoundError`. One of them takes about 75
seconds to reach that point.

So this does two things, both narrow:

* lets the destination be redirected with `OPEN3D_OUT_DIR`, keeping each harness's own
  filename, so a reviewer outside the container can keep the result;
* resolves and checks the destination at import time, so an unwritable one is reported
  before the measurement runs rather than after.

It deliberately does not invent the directory. A harness that silently creates somewhere
to write is harder to reason about than one that says where it was pointed and stops.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Redirects every harness output in this directory, keeping the individual filenames.
OUT_DIR_ENV = "OPEN3D_OUT_DIR"


def output_path(default: str) -> Path:
    """Where this harness should write, verified writable now rather than at the end."""
    override = os.environ.get(OUT_DIR_ENV)
    resolved = Path(override) / Path(default).name if override else Path(default)

    parent = resolved.parent
    if not parent.is_dir():
        raise SystemExit(
            f"Cannot write {resolved}: {parent} does not exist. That default is the "
            f"container mount point, where the evidence directory appears at /data. "
            f"Outside the container, set {OUT_DIR_ENV} to a directory that exists and "
            f"the report will be written there under the same filename."
        )
    if not os.access(parent, os.W_OK):
        raise SystemExit(f"Cannot write {resolved}: {parent} is not writable.")
    return resolved
