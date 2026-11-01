"""Bind the delivered frames' measurements to the frames' actual bytes.

A review lane found that the nine frames under `frames-refit/` carry this evidence set's entire
visual argument while not one of their digests was recorded anywhere: nine frames named in two
records, eighteen places where a frame is named and its digest is absent. The consequence is
quiet rather than loud. Replace a frame and every recorded measurement keeps describing the old
bytes, and nothing says so -- which is the same shape as the stale-manifest concern this set has
been corrected for twice before, except that the digests were missing rather than out of date.

So each row here puts a frame's path, its digest, and the measurements taken *from those bytes*
in one object. The digest sits beside a `path` key on purpose: that is the pairing the evidence
guard already understands, so these hashes are checked by the existing gate without teaching it a
new convention.

The measurements are copied from `frame-edge-audit-after.json`, which computed them, rather than
recomputed here. Recomputing would let this file and that one drift apart while both looked
healthy; copying means a mismatch is a failure in `--verify` instead of a difference nobody reads.

On `content_fraction`: it is recorded as a **regression ratchet, not a quality gate**. The review
lane is right that every delivered frame sits at or below the occupancy that prompted the original
complaint, and four are under 13%. A minimum-occupancy gate would be the wrong remedy, because the
cheapest way to pass one is to crop in -- and what gets cropped is the sparse, incomplete geometry
these views exist to show. Pinning the measured value instead catches a silent reframing without
rewarding a tighter crop. The honest state of that complaint is: measured, published, and
deliberately ungated, with the reason written down.

Run from the repository root:
    python3 docs/evidence/open3d/harness/bind_delivered_frames.py            # write
    python3 docs/evidence/open3d/harness/bind_delivered_frames.py --verify   # gate

`--verify` exits non-zero if any frame's bytes, or any copied measurement, no longer matches.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "frames-refit"
AUDIT = ROOT / "frame-edge-audit-after.json"
MANIFEST = ROOT / "delivered-frames-manifest.json"

#: Measurements copied from the audit row, by the name the audit gives them. Kept to the ones a
#: reader would use to decide whether a frame still shows what it showed: how much of the canvas
#: the content covers, how far it spans, and whether it touches an edge.
COPIED = ("content_fraction", "content_pixels", "content_span_fraction", "is_clipped")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def audit_rows() -> dict[str, dict]:
    rows = json.loads(AUDIT.read_text())["frames"]
    return {row["frame"]: row for row in rows}


def build() -> dict:
    measured = audit_rows()
    frames = []
    for png in sorted(FRAMES.glob("*.png")):
        row = measured.get(png.name)
        if row is None:
            raise SystemExit(
                f"{png.name} is delivered but {AUDIT.name} never measured it; a frame with no "
                "measurement cannot be bound to one"
            )
        entry = {
            "path": f"{FRAMES.name}/{png.name}",
            "sha256": digest(png),
            "bytes": png.stat().st_size,
        }
        entry.update({key: row[key] for key in COPIED if key in row})
        frames.append(entry)

    unmeasured = set(measured) - {Path(f["path"]).name for f in frames}
    return {
        "what": "Every delivered frame, its exact bytes, and the measurements taken from those "
        "bytes, in one row each, so that replacing a frame cannot leave a recorded measurement "
        "describing bytes that are gone.",
        "why_this_exists": "A review lane hashed all nine committed frames and searched every "
        "evidence record: no record contained the digest of any delivered frame. Eighteen places "
        "named a frame without its digest. The measurements were tied to the frames by filename "
        "alone.",
        "measurements_copied_from": AUDIT.name,
        "measurements_are_not_recomputed_here": "Copying rather than recomputing means this file "
        "and the record that measured these frames cannot drift apart while both look healthy. A "
        "mismatch fails --verify instead of sitting unread in two places.",
        "content_fraction_is_a_ratchet_not_a_gate": "Occupancy is pinned so a silent reframing is "
        "caught, and deliberately not floored. The cheapest way to pass a minimum-occupancy gate "
        "is to crop in, and what gets cropped is the sparse incomplete geometry these views exist "
        "to show. Every delivered frame sits at or below the occupancy that prompted the original "
        "complaint and four are under 13%; that is measured, published and ungated on purpose.",
        "what_this_does_not_establish": [
            "Nothing about whether the frames are good. This binds measurements to bytes; it is "
            "not a judgement about framing, legibility or reconstruction quality.",
            "Nothing about the geometry. A frame can be byte-stable and still show a bad mesh.",
            "No claim that these are the only frames that matter, only that these are the nine "
            "delivered here.",
        ],
        "measured_but_not_delivered": sorted(unmeasured),
        "generator": "harness/bind_delivered_frames.py",
        "frames": frames,
    }


def verify() -> int:
    if not MANIFEST.exists():
        print(f"{MANIFEST.name} does not exist; run without --verify to write it")
        return 1
    recorded = json.loads(MANIFEST.read_text())
    findings: list[str] = []

    measured = audit_rows()
    by_path = {row["path"]: row for row in recorded["frames"]}

    delivered = {f"{FRAMES.name}/{p.name}" for p in FRAMES.glob("*.png")}
    for missing in sorted(delivered - set(by_path)):
        findings.append(f"{missing} is delivered but the manifest does not bind it")
    for stale in sorted(set(by_path) - delivered):
        findings.append(f"{stale} is bound by the manifest but is no longer delivered")

    for path, row in sorted(by_path.items()):
        png = ROOT / path
        if not png.is_file():
            continue  # already reported as no longer delivered
        actual = digest(png)
        if actual != row["sha256"]:
            findings.append(
                f"{path} is bound to {row['sha256'][:12]}.. and its bytes hash to {actual[:12]}.."
            )
            continue
        audit_row = measured.get(png.name)
        if audit_row is None:
            findings.append(f"{path} is bound but {AUDIT.name} no longer measures it")
            continue
        for key in COPIED:
            if key in row and key in audit_row and row[key] != audit_row[key]:
                findings.append(
                    f"{path}: manifest records {key}={row[key]!r} but {AUDIT.name} says "
                    f"{audit_row[key]!r}"
                )

    if findings:
        print(f"{len(findings)} finding(s):")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(
        f"All {len(by_path)} delivered frames hash as bound, and every copied measurement still "
        f"matches {AUDIT.name}."
    )
    return 0


def main() -> int:
    if "--verify" in sys.argv:
        return verify()
    MANIFEST.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
