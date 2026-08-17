#!/usr/bin/env python3
"""In-image entry point for decode validation of generated LTX-2.5 video.

The checks live in ``npa/src/npa/workbench/ltx2/video_check.py``, copied into the
image verbatim by the Dockerfile, for the same reason the licensing gate is: the
module the repo's tests exercise and the module that decides whether a GPU run
passed must be one module.

Runs on the image's own Python before upstream's venv exists, so it may use only
the standard library and ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import video_check  # noqa: E402  (path is set above so the copied module resolves)

EX_SOFTWARE = 70


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Generated clip to validate.")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=24,
        help="Reject a clip that decodes to fewer frames than this.",
    )
    parser.add_argument(
        "--artifact", default="", help="Write the evidence JSON to this path."
    )
    parser.add_argument(
        "--capability", default="", help="Capability name recorded in the evidence."
    )
    args = parser.parse_args(argv[1:])

    try:
        result = video_check.validate_video(
            args.video, min_frames=args.min_frames, capability=args.capability
        )
    except video_check.VideoCheckError as error:
        print(f"npa-ltx2: {error}", file=sys.stderr)
        return EX_SOFTWARE

    payload = result.as_dict()
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.artifact:
        with open(args.artifact, "w", encoding="utf-8") as handle:
            handle.write(body)
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
