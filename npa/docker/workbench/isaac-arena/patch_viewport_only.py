"""Keep Arena viewport capture separate from embodiment-camera observations.

Arena's policy runner correctly enables Kit camera support before launching the
application, but later reuses the same flag to add every embodiment-mounted
camera whenever either video recorder is selected.  NPA exposes only viewport
recording.  Its private marker keeps Kit rendering enabled while preventing
unused robot cameras from entering the environment graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """\
        # Re-apply enable_cameras: the full parse resets it to default False.
        if args_cli.record_camera_video or args_cli.record_viewport_video:
            args_cli.enable_cameras = True
"""

NEW = """\
        # Re-apply environment-mounted cameras only when their recorder needs
        # them.  AppLauncher camera support was already enabled above for both
        # recorders; the NPA viewport-only path must not instantiate unused
        # embodiment cameras as a side effect of recording the Kit viewport.
        viewport_only = os.environ.get("NPA_ISAAC_ARENA_VIEWPORT_ONLY") == "1"
        if args_cli.record_camera_video or (
            args_cli.record_viewport_video and not viewport_only
        ):
            args_cli.enable_cameras = True
        elif args_cli.record_viewport_video and viewport_only:
            args_cli.enable_cameras = False
"""


def patch_policy_runner(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError("pinned Arena viewport-camera patch context changed")
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_runner", type=Path)
    args = parser.parse_args()
    patch_policy_runner(args.policy_runner)


if __name__ == "__main__":
    main()
