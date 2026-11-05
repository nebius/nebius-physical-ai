"""Every evidence record that names its producer must still match its bytes.

This exists because a `ruff format` pass over the evidence harnesses re-wrapped one
argument list in `preview_from_retained_plys.py`, changing nothing it does and changing
its sha256. `capability-record-provenance-bound-preview.json` went on claiming the old
digest, so a record asserting "these bytes produced this artifact" was false, and
nothing in CI noticed: `audit_evidence_references.py` finds it, and no gate runs that
script.

The lane's own instruction is not to upgrade accepted proof to new bytes, so the
restored state is the original file rather than a restamped record. This guard is the
part that makes the next one loud instead of silent.

Needs no Open3D and reads no large artifacts -- it hashes `.py` files only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"

#: Keys a record may use to name a producing script, and to pin its bytes. Both shapes
#: are in use across this directory; the pair has to be found in the same object for
#: the claim to be unambiguous.
PATH_KEYS = ("script", "harness", "path", "file")
DIGEST_KEYS = ("script_sha256", "harness_sha256", "sha256")


def _claims(node: Any, found: list[tuple[str, str, str | None]]) -> None:
    if isinstance(node, dict):
        path = next(
            (node[key] for key in PATH_KEYS if isinstance(node.get(key), str)), None
        )
        digest_key = next(
            (key for key in DIGEST_KEYS if isinstance(node.get(key), str)), None
        )
        digest = node[digest_key] if digest_key else None
        if path and digest and path.endswith(".py"):
            if digest_key and digest_key.endswith("_sha256"):
                at_commit = node.get(
                    f"{digest_key[: -len('_sha256')]}_bytes_committed_at"
                )
            else:
                at_commit = node.get("bytes_committed_at")
            found.append(
                (path, digest, at_commit if isinstance(at_commit, str) else None)
            )
        for value in node.values():
            _claims(value, found)
    elif isinstance(node, list):
        for value in node:
            _claims(value, found)


def test_every_record_that_pins_its_producer_still_matches_it() -> None:
    checked = []
    for record in sorted(EVIDENCE.glob("*.json")):
        claims: list[tuple[str, str, str | None]] = []
        _claims(json.loads(record.read_text()), claims)
        for path, digest, at_commit in claims:
            target = EVIDENCE / path
            if not target.exists():
                # Expected: most hashes in this directory name a run's own outputs,
                # which are recorded by hash and not committed.
                continue
            if at_commit:
                relative = target.relative_to(EVIDENCE.parents[2])
                result = subprocess.run(
                    ["git", "show", f"{at_commit}:{relative.as_posix()}"],
                    cwd=EVIDENCE.parents[2],
                    capture_output=True,
                    check=False,
                )
                if result.returncode:
                    continue
                actual = hashlib.sha256(result.stdout).hexdigest()
            else:
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
            assert actual.startswith(digest[:12]), (
                f"{record.name} claims {digest[:12]} for {path}, which is actually "
                f"{actual[:12]}. Either the producer changed without the record being "
                "regenerated, or a formatting pass touched it -- the second is what "
                "happened last time, and the fix was to restore the bytes rather than "
                "restamp the record."
            )
            checked.append((record.name, path))

    assert checked, (
        "no record in this directory pins a producing script any more, so this guard "
        "is passing because it has nothing to check"
    )
