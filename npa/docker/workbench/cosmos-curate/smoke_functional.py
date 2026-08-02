"""Golden eval for the npa-cosmos-curate image: a real Cosmos Curator run.

Synthesizes a short clip with the image's own ffmpeg, drives the real upstream
curator stages over it, and asserts that upstream produced its canonical output —
transcoded clips under ``clips/`` and per-clip metadata under ``metas/v0/`` with a
motion score. Anything less (an import probe, a manifest) would not prove the
curator actually curated.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Big enough that upstream's motion-vector stage does not reject the clip for
# "too small resolution", and long enough to split into several fixed-stride clips.
CLIP_WIDTH = 1280
CLIP_HEIGHT = 704
CLIP_SECONDS = 8
CLIP_LEN_S = 3.0
MIN_CLIP_LEN_S = 1.0


def _synthesize(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate=24:duration={CLIP_SECONDS}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libopenh264",
            str(path),
        ],
        check=True,
        timeout=300,
    )


def main() -> int:
    from npa.workbench.cosmos_curate import curate_videos, ingest_output, probe_availability

    availability = probe_availability()
    if not availability.can_run_in_process:
        print(f"COSMOS_CURATE_SMOKE_FAIL {availability.reason()}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="cosmos-curate-smoke-") as tmp:
        root = Path(tmp)
        source = root / "input"
        source.mkdir()
        _synthesize(source / "smoke.mp4")
        output = root / "output"

        run = curate_videos(
            input_dir=source,
            output_dir=output,
            clip_len_s=CLIP_LEN_S,
            min_clip_length_s=MIN_CLIP_LEN_S,
            motion_filter="score-only",
        )
        ingested = ingest_output(output)

        clips = ingested["clips"]
        if not clips:
            print("COSMOS_CURATE_SMOKE_FAIL curator wrote no clip metadata", file=sys.stderr)
            return 1
        if len(ingested["clip_files"]) != len(clips):
            print(
                f"COSMOS_CURATE_SMOKE_FAIL {len(clips)} metas but "
                f"{len(ingested['clip_files'])} clip files",
                file=sys.stderr,
            )
            return 1
        scored = [clip for clip in clips if clip.motion_score_global_mean is not None]
        if not scored:
            print("COSMOS_CURATE_SMOKE_FAIL no clip carries a motion score", file=sys.stderr)
            return 1
        if run.errors:
            print(f"COSMOS_CURATE_SMOKE_FAIL stage errors: {run.errors}", file=sys.stderr)
            return 1

        print(
            "COSMOS_CURATE_SMOKE_OK "
            + json.dumps(
                {
                    "engine": run.engine,
                    "encoder": run.encoder,
                    "source": run.source,
                    "clips": len(clips),
                    "motion_scored": len(scored),
                    "first_clip": clips[0].to_dict(),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
