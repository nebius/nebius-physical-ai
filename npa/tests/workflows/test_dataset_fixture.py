"""Unit coverage for the raw sensor fixture (no infrastructure).

The assertions encode the contract ``npa/src/npa/workbench/dataset/ingestion.py``
enforces, and the gate thresholds of the two specs that consume it — the *stricter* of
which (``dataset-ingest-curate``) is what the fixture must satisfy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workbench.dataset.schemas import (
    CANONICAL_REQUIRED_FIELDS,
    COMPLETENESS_FIELDS,
    SensorSchema,
)
from npa.workflows.dataset_fixture import (
    DEFAULT_EVENT,
    DEFAULT_LOCATION,
    FIXTURE_SCHEMA,
    DatasetFixtureError,
    build_document,
    build_records,
    publish,
    split_s3_uri,
    write_document,
)


def test_records_carry_every_required_field() -> None:
    records = build_records(count=6)

    assert len(records) == 6
    for record in records:
        for field in CANONICAL_REQUIRED_FIELDS:
            assert record.get(field), f"{field} must be present and non-empty"


def test_records_are_fully_complete_and_not_corrupt() -> None:
    """completeness == 1.0 and zero corruption satisfy the stricter spec's gates."""

    records = build_records(count=8)

    for record in records:
        for field in COMPLETENESS_FIELDS:
            assert record.get(field), f"{field} contributes to completeness"
        # `_is_corrupt` flags an empty uri or quality.corruption > 0.5.
        assert record["uri"]
        assert record["quality"]["corruption"] <= 0.5


def test_record_ids_are_unique() -> None:
    """A duplicate record_id makes the ingester raise."""

    records = build_records(count=20)

    assert len({record["record_id"] for record in records}) == 20


def test_the_real_normalizer_accepts_the_fixture() -> None:
    """End-to-end through the shipped ingester, no mocks."""

    from npa.workbench.dataset.ingestion import compute_quality_stats, normalize_records

    records = build_records(count=10)
    # An empty declared schema means "canonical required fields, any modality" —
    # the default the ingest CLI uses when a spec declares no sensor schema.
    normalized, corrupt = normalize_records(records, SensorSchema())
    stats = compute_quality_stats(normalized, corrupt)

    assert len(normalized) == 10
    assert corrupt == 0
    assert stats.record_count == 10
    assert all(record.completeness == 1.0 for record in normalized)


def test_a_known_share_carries_the_queried_event_and_location() -> None:
    """A `dataset query` stage must return rows, not an empty success."""

    records = build_records(count=10, event_share=0.5)

    tagged = [r for r in records if r["event"] == DEFAULT_EVENT]
    assert len(tagged) == 5
    assert all(record["location"] == DEFAULT_LOCATION for record in tagged)
    # ...and the rest are genuinely different, so a filter is actually exercised.
    assert any(record["event"] != DEFAULT_EVENT for record in records)


def test_modalities_are_spread_across_records() -> None:
    records = build_records(count=9, modalities=("camera", "lidar", "radar"))

    assert {record["modality"] for record in records} == {"camera", "lidar", "radar"}


def test_document_shape_is_what_the_loader_reads() -> None:
    document = build_document(count=3)

    assert document["schema"] == FIXTURE_SCHEMA
    assert document["record_count"] == 3
    assert isinstance(document["records"], list) and len(document["records"]) == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"count": 0}, "count must be >= 1"),
        ({"modalities": ()}, "at least one modality"),
        ({"event_share": 1.5}, "event_share must be within"),
    ],
)
def test_degenerate_input_is_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(DatasetFixtureError, match=match):
        build_records(**kwargs)


def test_write_document_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "records.json"

    write_document(path, count=4)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 4


@pytest.mark.parametrize("uri", ["", "bucket/key", "s3://bucket", "https://x/y"])
def test_split_s3_uri_rejects_bad_input(uri: str) -> None:
    with pytest.raises(DatasetFixtureError):
        split_s3_uri(uri)


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[tuple[str, str, bytes]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self.puts.append((Bucket, Key, Body))


def test_publish_uses_the_injected_client() -> None:
    client = _FakeS3()

    result = publish("s3://bucket/raw-sensor/records.json", client=client, count=2)

    assert result["uri"] == "s3://bucket/raw-sensor/records.json"
    assert client.puts and client.puts[0][:2] == ("bucket", "raw-sensor/records.json")
    payload = json.loads(client.puts[0][2].decode())
    assert payload["record_count"] == 2


def test_fixture_satisfies_both_shipped_specs_gates() -> None:
    """Pin the gate thresholds the fixture is built against."""

    from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec
    from npa.orchestration.npa_workflow.spec import load_spec

    for name in ("dataset-of-record-smoke.yaml", "dataset-ingest-curate.yaml"):
        path = resolve_npa_workflow_spec(name)
        assert path is not None, name
        config = load_spec(path).config
        # completeness 1.0 clears any completeness_min <= 1.0.
        assert float(config["completeness_min"]) <= 1.0, name
        # zero corrupt records clears any max_corruption_rate >= 0.
        assert float(config["max_corruption_rate"]) >= 0.0, name
        assert config["event_of_interest"] == DEFAULT_EVENT, (
            f"{name} queries {config['event_of_interest']!r}; the fixture tags "
            f"{DEFAULT_EVENT!r}"
        )
        wanted_location = str(config.get("location_of_interest") or "")
        assert wanted_location in ("", DEFAULT_LOCATION), (
            f"{name} filters location {wanted_location!r}; the fixture tags "
            f"{DEFAULT_LOCATION!r}"
        )


def test_the_real_normalizer_honours_a_declared_modality_list() -> None:
    """A declared schema must also accept the fixture, not just the permissive default."""

    from npa.workbench.dataset.ingestion import normalize_records

    records = build_records(count=6, modalities=("camera", "lidar"))
    schema = SensorSchema(modalities=["camera", "lidar"])

    normalized, corrupt = normalize_records(records, schema)

    assert len(normalized) == 6 and corrupt == 0
