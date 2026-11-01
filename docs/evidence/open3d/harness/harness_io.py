"""Shared plumbing for the harnesses here: where the report goes, and how sampling is seeded.

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

`seed_sampling` exists because these harnesses sample point clouds from meshes, and an
unseeded sample makes every run a different measurement. Open3D 0.20.0 exposes no `seed`
parameter on either sampler, so the only control is the library-global one.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Redirects every harness output in this directory, keeping the individual filenames.
OUT_DIR_ENV = "OPEN3D_OUT_DIR"

#: The reading four harnesses here test for, named once instead of spelled out in each.
#:
#: It used to be `"extrapolated shell"`. Commit `a828e58fd3` renamed it on the same branch of
#: the same condition, because the old name claimed the reading could detect fabrication and
#: it cannot. Four harnesses kept comparing against the old literal, so their detectors
#: silently stopped firing: `floor-across-geometry-and-sampling` had 27 of 32 readings true
#: when its evidence was committed, and 0 afterwards. Nothing failed, because a detector that
#: cannot fire reports no defects.
#:
#: `test_open3d_reading_contract.py` holds this against the `reads_as` assignments in the
#: runner, so the next rename breaks a test instead of a measurement.
FAR_READING = "far from any observation"

#: Overrides the sampling seed, so a reviewer can ask whether a result is seed-specific.
SEED_ENV = "OPEN3D_SEED"

#: Arbitrary and fixed. Nothing about the measurements depends on this value being special;
#: it is pinned only so that two runs of the same harness are the same measurement. Whether
#: a given conclusion survives a *different* seed is a separate question, which
#: `seed_sensitivity.py` measures rather than assumes.
DEFAULT_SEED = 20261101


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


def seed_sampling() -> int:
    """Make this harness's point sampling repeatable, and report the seed it used.

    Open3D is imported here rather than at module scope so that `output_path` stays usable
    on a machine without the package.
    """
    import open3d as o3d

    raw = os.environ.get(SEED_ENV)
    if raw is None:
        seed = DEFAULT_SEED
    else:
        try:
            seed = int(raw)
        except ValueError:
            raise SystemExit(
                f"{SEED_ENV}={raw!r} is not an integer."
            ) from None

    o3d.utility.random.seed(seed)
    return seed
