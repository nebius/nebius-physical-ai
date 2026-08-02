"""Golden eval for the npa-cosmos-evaluator image: a real hallucination check.

Synthesizes a source clip and two "augmented" clips with the image's own ffmpeg —
one that preserves the source's motion and one showing a different scene — then
scores both with upstream's own ``HallucinationProcessor`` and with the in-repo
port of the same algorithm. It asserts the check actually discriminates (the
preserving clip passes, the different one fails) and that the two engines agree,
which is the property that makes the port a stand-in for upstream rather than a
different metric wearing its name.

Deliberately needs no model weights and no network: the hallucination check is
classical computer vision. Attribute verification is exercised separately, since it
calls a hosted VLM and so needs a credential this smoke should not require.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Large enough that the check has real motion to compare, small enough to stay fast.
WIDTH = 640
HEIGHT = 480
SECONDS = 3
# The two engines decode through different paths (upstream's ffmpeg wrapper vs a raw
# grayscale pipe) and use different distance transforms, so exact equality is not
# the property under test; agreeing on the verdict and to within this margin is.
MAX_SCORE_DELTA = 0.05


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        timeout=300,
    )


def _synthesize(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source.mp4"
    preserving = root / "augmented_preserving.mp4"
    different = root / "augmented_different.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size={WIDTH}x{HEIGHT}:rate=10:duration={SECONDS}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(source),
    )
    # Appearance-only change: same motion, new brightness/saturation.
    _ffmpeg(
        "-i", str(source), "-vf", "eq=brightness=0.12:saturation=1.4",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(preserving),
    )
    # A different scene entirely, so its motion is hallucinated relative to source.
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=10:duration={SECONDS}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(different),
    )
    return source, preserving, different


def main() -> int:
    from npa.workbench.cosmos_evaluator import check_hallucination
    from npa.workbench.cosmos_evaluator.evaluate import evaluator_engine_summary

    summary = evaluator_engine_summary()
    if summary["engine"] != "cosmos-evaluator-upstream":
        print(f"COSMOS_EVALUATOR_SMOKE_FAIL upstream checkout not resolved: {summary}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="cosmos-evaluator-smoke-") as tmp:
        source, preserving, different = _synthesize(Path(tmp))

        results: dict[str, dict[str, dict[str, object]]] = {}
        for label, augmented in (("preserving", preserving), ("different", different)):
            upstream = check_hallucination(
                clip_id=label, original_video=source, augmented_video=augmented
            )
            port = check_hallucination(
                clip_id=label,
                original_video=source,
                augmented_video=augmented,
                prefer_upstream=False,
            )
            results[label] = {"upstream": upstream.to_dict(), "port": port.to_dict()}

            if upstream.engine != "cosmos-evaluator-upstream":
                print(
                    f"COSMOS_EVALUATOR_SMOKE_FAIL {label} did not run upstream: {upstream.engine}",
                    file=sys.stderr,
                )
                return 1
            if port.engine != "cosmos-evaluator-npa-port":
                print(
                    f"COSMOS_EVALUATOR_SMOKE_FAIL {label} did not run the port: {port.engine}",
                    file=sys.stderr,
                )
                return 1
            if upstream.total_frames < 2:
                print(
                    f"COSMOS_EVALUATOR_SMOKE_FAIL {label} decoded {upstream.total_frames} frames",
                    file=sys.stderr,
                )
                return 1
            if upstream.passed != port.passed:
                print(
                    f"COSMOS_EVALUATOR_SMOKE_FAIL {label}: engines disagree on the verdict "
                    f"(upstream={upstream.passed} port={port.passed})",
                    file=sys.stderr,
                )
                return 1
            delta = abs(upstream.score - port.score)
            if delta > MAX_SCORE_DELTA:
                print(
                    f"COSMOS_EVALUATOR_SMOKE_FAIL {label}: engines disagree by {delta:.4f} "
                    f"(upstream={upstream.score:.6f} port={port.score:.6f})",
                    file=sys.stderr,
                )
                return 1

        if not results["preserving"]["upstream"]["passed"]:
            print(
                "COSMOS_EVALUATOR_SMOKE_FAIL an appearance-only augmentation should pass: "
                f"{results['preserving']['upstream']}",
                file=sys.stderr,
            )
            return 1
        if results["different"]["upstream"]["passed"]:
            print(
                "COSMOS_EVALUATOR_SMOKE_FAIL a different scene should fail: "
                f"{results['different']['upstream']}",
                file=sys.stderr,
            )
            return 1

        print("COSMOS_EVALUATOR_SMOKE_OK " + json.dumps({"source": summary["upstream_source"], "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
