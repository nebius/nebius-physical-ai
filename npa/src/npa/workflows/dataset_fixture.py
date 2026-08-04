"""Build a raw sensor record set for the dataset-of-record live specs.

Why this exists
---------------
``dataset-of-record-smoke.yaml`` and ``dataset-ingest-curate.yaml`` read
``config.raw_sensor_uri`` — a JSON document of raw sensor records — and had no live
coverage because nothing seeded one. ``workbench.dataset.ingest`` then fails with
``raw sensor data not found`` / ``raw sensor data has no records``.

The contract is small and enforced by ``dataset/ingestion.py``:

* the document is ``{"records": [...]}`` (or a bare list);
* every record needs ``record_id`` (unique), ``modality`` and ``uri``;
* ``event`` / ``location`` / ``timestamp`` / ``quality`` / ``embedding`` each add
  0.2 to a record's ``completeness``;
* a record counts as **corrupt** when ``uri`` is empty or ``quality.corruption > 0.5``.

Both specs then gate on ``completeness_min`` (0.3 / 0.5), ``max_corruption_rate``
(0.5 / 0.1) and query for an ``event_of_interest`` (``cut_in``), with the stricter spec
also filtering on ``location_of_interest`` (``san_francisco``) and ``min_quality`` 0.5.
The generated set is built to satisfy the **stricter** of the two, so one fixture serves
both: every record is fully populated (completeness 1.0), none is corrupt, and a known
share carries the queried event/location.

Standard library only, so it can be generated inside the live harness.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

DEFAULT_RECORD_COUNT = 12
DEFAULT_MODALITIES: tuple[str, ...] = ("camera", "lidar", "radar")
DEFAULT_EVENT = "cut_in"
DEFAULT_LOCATION = "san_francisco"
#: Share of records carrying the queried event/location. Kept above zero so a
#: ``dataset query`` stage returns rows rather than an empty (but "successful") result.
DEFAULT_EVENT_SHARE = 0.5
FIXTURE_SCHEMA = "npa.dataset.raw_sensor.v1"


class DatasetFixtureError(RuntimeError):
    """Raised when a raw sensor fixture cannot be built or published."""


def build_records(
    *,
    count: int = DEFAULT_RECORD_COUNT,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    event: str = DEFAULT_EVENT,
    location: str = DEFAULT_LOCATION,
    event_share: float = DEFAULT_EVENT_SHARE,
    bucket_uri: str = "s3://example-bucket/raw-sensor",
) -> list[dict[str, Any]]:
    """Return fully populated, non-corrupt raw sensor records."""

    if count < 1:
        raise DatasetFixtureError(f"count must be >= 1, got {count}")
    if not modalities:
        raise DatasetFixtureError("at least one modality is required")
    if not 0.0 <= event_share <= 1.0:
        raise DatasetFixtureError(f"event_share must be within [0, 1], got {event_share}")

    tagged = max(1, round(count * event_share)) if event_share else 0
    records: list[dict[str, Any]] = []
    for index in range(count):
        modality = modalities[index % len(modalities)]
        is_tagged = index < tagged
        records.append(
            {
                "record_id": f"rec-{index:04d}",
                "modality": modality,
                "uri": f"{bucket_uri.rstrip('/')}/{modality}/frame_{index:04d}.bin",
                # Every completeness field is present, so completeness == 1.0 and the
                # stricter spec's completeness_min of 0.5 is satisfied.
                "event": event if is_tagged else "lane_keep",
                "location": location if is_tagged else "phoenix",
                "timestamp": f"2026-07-31T00:{index % 60:02d}:00Z",
                # corruption well under the 0.5 threshold => zero corrupt records, so
                # max_corruption_rate 0.1 holds too.
                "quality": {"completeness": 1.0, "corruption": 0.0, "sharpness": 0.9},
                "embedding": [round(0.01 * (index + offset), 4) for offset in range(4)],
            }
        )
    return records


def build_document(**kwargs: Any) -> dict[str, Any]:
    """Return the ``{"records": [...]}`` document the ingester reads."""

    records = build_records(**kwargs)
    return {
        "schema": FIXTURE_SCHEMA,
        "record_count": len(records),
        "records": records,
    }


def write_document(output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write the fixture document locally and return it."""

    document = build_document(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise DatasetFixtureError(f"expected an s3:// URI, got {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise DatasetFixtureError(f"s3 URI needs a key: {uri!r}")
    return parsed.netloc, key


def publish(uri: str, *, client: Any | None = None, **kwargs: Any) -> dict[str, Any]:
    """Build the document and put it at an ``s3://`` object URI."""

    bucket, key = split_s3_uri(uri)
    document = build_document(**kwargs)
    body = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if client is None:  # pragma: no cover - real S3 only
        import boto3
        from botocore.client import Config

        boto_kwargs: dict[str, Any] = {"config": Config(signature_version="s3v4")}
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            boto_kwargs["endpoint_url"] = endpoint
        client = boto3.client("s3", **boto_kwargs)
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return {**document, "uri": uri}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for staging a shared fixture."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="", help="s3:// object URI to publish to.")
    parser.add_argument("--output-path", default="", help="Local path to write instead.")
    parser.add_argument("--count", type=int, default=DEFAULT_RECORD_COUNT)
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    args = parser.parse_args(argv)

    if not args.uri and not args.output_path:
        parser.error("pass --uri or --output-path")
    try:
        if args.uri:
            result = publish(
                args.uri, count=args.count, event=args.event, location=args.location
            )
        else:
            result = write_document(
                args.output_path, count=args.count, event=args.event, location=args.location
            )
    except DatasetFixtureError as exc:
        print(f"Error: {exc}")
        return 2
    summary = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
