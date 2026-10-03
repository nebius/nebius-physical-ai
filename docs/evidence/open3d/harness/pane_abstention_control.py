"""Controls for the pane audit's no-divider abstention, after it was reordered.

`audit()` declared that a frame with no locatable divider returns
`pane_detection: failed`. It did not: the refusal sat *below* `_pane_body(frame, split)`,
which computes `split - 40`, so an abstention raised TypeError before reaching the branch
that describes it. No committed frame in this set is flat, so nothing here exercised it.

Two controls, because either alone proves little:

- Negative: a flat frame with no divider at all. It must reach the declared abstention.
  The pre-fix call order is reproduced directly against the same frame, so the control
  shows the TypeError it used to raise rather than only asserting the fixed behaviour.
- Positive: the run's actual capture, which must still return the divider and the
  unclipped verdict it returned before the reordering. Moving a refusal earlier can
  silence real answers, and this is the answer that must not move.

Run from the repository root:
    python3 docs/evidence/open3d/harness/pane_abstention_control.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from viewer_pane_audit import _pane_body, _pane_bounds, audit

EVIDENCE = Path(__file__).resolve().parent.parent
CAPTURE = EVIDENCE / "viewer-ui" / "ordinary-native-replay.png"
#: The window the capture was taken at, so the control is the same shape as the real input.
FLAT_SIZE = (1024, 768)
#: Mid grey. Any uniform fill works; the point is that no column carries a colour step.
FLAT_VALUE = 128


def _flat_frame(directory: Path) -> Path:
    path = directory / "flat-no-divider.png"
    Image.fromarray(
        np.full((FLAT_SIZE[1], FLAT_SIZE[0], 3), FLAT_VALUE, dtype=np.uint8)
    ).save(path)
    return path


def _pre_fix_order(frame: np.ndarray) -> str:
    """What the old call order did with an abstention: use the split before checking it."""

    try:
        _pane_body(frame, _pane_bounds(frame))
    except TypeError as error:
        return f"TypeError: {error}"
    return "no error"


def main() -> int:
    with tempfile.TemporaryDirectory() as workspace:
        flat = _flat_frame(Path(workspace))
        frame = np.asarray(Image.open(flat).convert("RGB")).astype(int)
        negative = {
            "control": "flat frame, no divider",
            "size": list(FLAT_SIZE),
            "fill": FLAT_VALUE,
            "divider_found": _pane_bounds(frame),
            "pre_fix_call_order": _pre_fix_order(frame),
            "audit": audit(flat),
        }

    positive = {
        "control": "the run's actual native capture",
        "frame": str(CAPTURE.relative_to(EVIDENCE)),
        "frame_sha256": hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
        "audit": audit(CAPTURE),
    }

    report = {
        "what_this_checks": (
            "that the declared no-divider abstention is reachable, and that reaching it "
            "earlier did not silence the verdict on a frame that has a divider"
        ),
        "generator": "harness/pane_abstention_control.py",
        "audited": "harness/viewer_pane_audit.py",
        "negative_control": negative,
        "positive_control": positive,
    }
    print(json.dumps(report, indent=2))

    if negative["divider_found"] is not None:
        print("flat control found a divider; it is not a no-divider control", file=sys.stderr)
        return 1
    if negative["audit"].get("pane_detection") != "failed":
        print("flat control did not reach the declared abstention", file=sys.stderr)
        return 1
    if not negative["pre_fix_call_order"].startswith("TypeError"):
        print("the pre-fix call order no longer reproduces the bug", file=sys.stderr)
        return 1
    if positive["audit"].get("scene_pane", {}).get("clipped") is not False:
        print("the actual capture's verdict moved", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
