"""Shared harness output, input, reading-label, and sampling controls.

Eight harnesses here write to a hardcoded `/data/<name>.json`. That is correct inside the
container they document, which mounts the evidence directory at `/data`. Outside it, the
directory does not exist, and the failure arrives at the *last* line: the harness runs the
whole measurement and then loses it to a `FileNotFoundError`. One of them takes about 75
seconds to reach that point.

So this does two things, both narrow:

* lets the destination be redirected with `OPEN3D_OUT` (an exact file) or
  `OPEN3D_OUT_DIR` (a directory that keeps each harness's filename), so a reviewer
  outside the container can keep the result;
* resolves and checks the destination at import time, so an unwritable one is reported
  before the measurement runs rather than after.

It deliberately does not invent the directory. A harness that silently creates somewhere
to write is harder to reason about than one that says where it was pointed and stops.

`seed_sampling` controls the Open3D-global sampler used by the harnesses that call it.
The fixed default makes repeat runs comparable, while `OPEN3D_SEED` exposes sensitivity.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Redirects every harness output in this directory, keeping the individual filenames.
OUT_DIR_ENV = "OPEN3D_OUT_DIR"
#: Redirects one harness output to an exact path. It wins when both overrides are set.
OUT_FILE_ENV = "OPEN3D_OUT"

#: Reading labels shared with the runtime so a renamed contract cannot silently turn a
#: detector into an always-false comparison.
FAR_READING = "far from any observation"
NEAR_READING = "near-threshold surface"

SEED_ENV = "OPEN3D_SEED"
DEFAULT_SEED = 20261101


def output_path(default: str) -> Path:
    """Where this harness should write, verified writable now rather than at the end."""
    file_override = os.environ.get(OUT_FILE_ENV)
    directory_override = os.environ.get(OUT_DIR_ENV)
    resolved = (
        Path(file_override)
        if file_override
        else Path(directory_override) / Path(default).name
        if directory_override
        else Path(default)
    )

    parent = resolved.parent
    if not parent.is_dir():
        raise SystemExit(
            f"Cannot write {resolved}: {parent} does not exist. That default is the "
            f"container mount point, where the evidence directory appears at /data. "
            f"Outside the container, set {OUT_FILE_ENV} to an exact file or "
            f"{OUT_DIR_ENV} to a directory that exists; the directory form keeps "
            f"the report's default filename."
        )
    if not os.access(parent, os.W_OK):
        raise SystemExit(f"Cannot write {resolved}: {parent} is not writable.")
    return resolved


def seed_sampling() -> int:
    """Seed Open3D sampling and return the seed that a report must record."""
    import open3d as o3d

    raw = os.environ.get(SEED_ENV)
    if raw is None:
        seed = DEFAULT_SEED
    else:
        try:
            seed = int(raw)
        except ValueError:
            raise SystemExit(f"{SEED_ENV}={raw!r} is not an integer.") from None
    o3d.utility.random.seed(seed)
    return seed


def _read(kind: str, path: str | Path, count) -> object:
    """Read a geometry and refuse an empty one, because Open3D will not.

    `o3d.io.read_point_cloud` and `read_triangle_mesh` do not raise on a missing or
    unreadable file. They log a warning to stderr and return an *empty* geometry, so the
    failure surfaces much later and somewhere else: `band_profile_across_scenes` died in a
    numpy reduction with `zero-size array to reduction operation maximum`, and
    `consumer_occupancy` inside Open3D's C++ with `IndexError: _Map_base::at`. Neither names
    a file.

    Emptiness is checked rather than existence alone, because a file that exists and does not
    parse reads the same way.
    """
    resolved = Path(path)

    # Before importing Open3D, so a missing input is still named on a machine without it.
    if not resolved.exists():
        staged = (
            " That path is under /data, a volume mounted into the container rather than part "
            "of the repository, so outside the container it has to be staged there."
            if resolved.is_absolute() and resolved.parts[:2] == ("/", "data")
            else ""
        )
        raise SystemExit(
            f"No {kind} at {resolved}.{staged} Open3D would have returned an empty {kind} "
            f"here and failed later somewhere unrelated."
        )

    import open3d as o3d

    reader = {
        "point cloud": o3d.io.read_point_cloud,
        "mesh": o3d.io.read_triangle_mesh,
    }[kind]

    geometry = reader(str(resolved))
    if count(geometry) == 0:
        raise SystemExit(
            f"The {kind} at {resolved} is empty. The file exists, so this is a parse failure "
            f"or a truncated write rather than a missing input; Open3D reports both by "
            f"returning nothing and logging a warning."
        )
    return geometry


def read_cloud(path: str | Path) -> object:
    """A point cloud that is known to have points."""
    return _read("point cloud", path, lambda c: len(c.points))


def read_mesh(path: str | Path) -> object:
    """A mesh that is known to have vertices."""
    return _read("mesh", path, lambda m: len(m.vertices))
