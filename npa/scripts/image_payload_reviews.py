"""Content-bound dispositions for reviewed public, non-operational matches.

The credential detector still runs unchanged. A disposition requires the entire
member to match reviewed bytes, size, and finding kind. A filename, prefix, or
changed library cannot establish that identity. The source records explain the
matches; this does not replace any other publication scanner.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import IO

from image_payload_credentials import CONTENT_CHUNK, content_credential

CATALOG = Path(__file__).with_name("image-payload-content-reviews.json")
KINDS = {"aws_access_key_id", "private_key_content", "credential_assignment"}


def load_reviews(path: Path = CATALOG) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "npa.image-payload-content-reviews.v1":
        raise ValueError("unsupported payload content review schema")
    reviews = {}
    for row in document["content"]:
        digest = row["sha256"]
        counts = row["match_counts"]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not counts
            or set(counts) - KINDS
            or any(type(count) is not int or count <= 0 for count in counts.values())
            or not row["reason"]
            or not row["source"]["url"].startswith("https://")
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", row["source"]["revision"])
            or digest in reviews
        ):
            raise ValueError("invalid or duplicate payload content review")
        reviews[digest] = row
    return reviews


class _HashingReader:
    def __init__(self, stream: IO[bytes]):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        payload = self.stream.read(size)
        self.digest.update(payload)
        self.size += len(payload)
        return payload


def reviewed_content_credential(
    stream: IO[bytes], reviews: dict[str, dict]
) -> tuple[str | None, dict | None]:
    reader = _HashingReader(stream)
    kind = content_credential(reader)
    if kind is None:
        return None, None
    # The detector can return at the first match. Hash every remaining byte,
    # including an appended credential, before attempting a disposition.
    while reader.read(CONTENT_CHUNK):
        pass
    digest = reader.digest.hexdigest()
    row = reviews.get(digest)
    if row is None or row["bytes"] != reader.size or kind not in row["match_counts"]:
        return kind, None
    return None, {
        "sha256": digest,
        "bytes": reader.size,
        "match_counts": row["match_counts"],
        "source": row["source"],
    }
