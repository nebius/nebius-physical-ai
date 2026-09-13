"""Keep Arena viewport capture separate from embodiment-camera observations.

Arena uses one ``enable_cameras`` CLI flag for two distinct concerns: enabling
Kit's render path and adding every embodiment-mounted camera sensor.  Viewport
recording needs the former, while adding the latter caused an unnecessary and
unstable camera workload for NPA's GR1 replay.  The private NPA marker therefore
leaves the upstream policy-runner flag enabled and masks only the embodiment
sensor construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path


POLICY_RUNNER_CONTEXT = """\
        # Re-apply enable_cameras: the full parse resets it to default False.
        if args_cli.record_camera_video or args_cli.record_viewport_video:
            args_cli.enable_cameras = True
"""

EMBODIMENT_IMPORT = """\
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
"""

EMBODIMENT_IMPORT_PATCHED = """\
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any
"""

EMBODIMENT_ASSIGNMENT = """\
        self.enable_cameras = enable_cameras
"""

EMBODIMENT_ASSIGNMENT_PATCHED = """\
        # NPA's viewport recorder still needs the upstream render flag, but it
        # does not consume robot-mounted sensor observations.  Mask only those
        # sensors so ``env.render()`` continues to advance the Kit viewport.
        viewport_only = os.environ.get("NPA_ISAAC_ARENA_VIEWPORT_ONLY") == "1"
        self.enable_cameras = enable_cameras and not viewport_only
"""


def patch_sources(policy_runner: Path, embodiment_base: Path) -> None:
    policy_text = policy_runner.read_text(encoding="utf-8")
    if policy_text.count(POLICY_RUNNER_CONTEXT) != 1:
        raise RuntimeError("pinned Arena viewport-camera patch context changed")

    embodiment_text = embodiment_base.read_text(encoding="utf-8")
    if (
        embodiment_text.count(EMBODIMENT_IMPORT) != 1
        or embodiment_text.count(EMBODIMENT_ASSIGNMENT) != 1
    ):
        raise RuntimeError("pinned Arena embodiment-camera patch context changed")
    embodiment_base.write_text(
        embodiment_text.replace(EMBODIMENT_IMPORT, EMBODIMENT_IMPORT_PATCHED).replace(
            EMBODIMENT_ASSIGNMENT, EMBODIMENT_ASSIGNMENT_PATCHED
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_runner", type=Path)
    parser.add_argument("embodiment_base", type=Path)
    args = parser.parse_args()
    patch_sources(args.policy_runner, args.embodiment_base)


if __name__ == "__main__":
    main()
