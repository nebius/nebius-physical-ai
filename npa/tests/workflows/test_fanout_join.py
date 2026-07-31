"""Unit coverage for the fan-out barrier/join stage (no S3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows import fanout_join


@pytest.fixture()
def local_storage(monkeypatch, tmp_path: Path):
    root = tmp_path / "s3"

    def _local(uri: str) -> Path:
        return root / uri[len("s3://") :] if uri.startswith("s3://") else Path(uri)

    def fake_download_json(uri: str):
        path = _local(uri)
        if not path.is_file():
            raise FileNotFoundError(uri)
        return json.loads(path.read_text(encoding="utf-8"))

    def fake_upload_json(payload, uri):
        path = _local(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return uri

    def fake_list_keys(uri: str):
        base = _local(uri)
        bucket = uri[len("s3://") :].split("/", 1)[0]
        if not base.exists():
            return []
        return [str(p.relative_to(root / bucket)) for p in sorted(base.rglob("*")) if p.is_file()]

    monkeypatch.setattr(fanout_join, "_download_json", fake_download_json)
    monkeypatch.setattr(fanout_join, "_upload_json", fake_upload_json)
    monkeypatch.setattr(fanout_join, "_list_keys", fake_list_keys)
    return root


def _write_shard(root: Path, shard: str, count: int) -> None:
    path = root / "bucket/captions" / shard / "captions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-VL-72B-Instruct",
                "captions": [{"image": f"{shard}-{i}.png", "caption": "x"} for i in range(count)],
            }
        ),
        encoding="utf-8",
    )


def test_join_shards_merges_every_shard(local_storage) -> None:
    for shard, count in (("shard-a", 2), ("shard-b", 3), ("shard-c", 1)):
        _write_shard(local_storage, shard, count)

    report = fanout_join.join_shards(
        shards_uri="s3://bucket/captions/",
        report_uri="s3://bucket/reports/join_report.json",
        shards="shard-a,shard-b,shard-c",
        run_id="run-1",
    )

    assert report["schema"] == fanout_join.JOIN_REPORT_SCHEMA
    assert report["shard_count"] == 3
    assert report["joined_shards"] == 3
    assert report["total_items"] == 6
    assert report["missing_shards"] == []
    written = json.loads((local_storage / "bucket/reports/join_report.json").read_text())
    assert {entry["shard"] for entry in written["shards"]} == {"shard-a", "shard-b", "shard-c"}


def test_join_shards_fails_when_a_shard_is_missing(local_storage) -> None:
    _write_shard(local_storage, "shard-a", 2)
    _write_shard(local_storage, "shard-b", 2)

    with pytest.raises(RuntimeError, match="1 shard\\(s\\) missing"):
        fanout_join.join_shards(
            shards_uri="s3://bucket/captions/",
            report_uri="s3://bucket/reports/join_report.json",
            shards="shard-a,shard-b,shard-c",
        )
    # The partial report is still published so the failure is debuggable.
    written = json.loads((local_storage / "bucket/reports/join_report.json").read_text())
    assert written["missing_shards"] == ["shard-c"]


def test_join_shards_discovers_shards_when_not_listed(local_storage) -> None:
    _write_shard(local_storage, "shard-a", 1)
    _write_shard(local_storage, "shard-b", 1)

    report = fanout_join.join_shards(
        shards_uri="s3://bucket/captions/",
        report_uri="s3://bucket/reports/",
    )

    assert sorted(entry["shard"] for entry in report["shards"]) == ["shard-a", "shard-b"]
    assert report["report_uri"].endswith(fanout_join.JOIN_REPORT_FILENAME)


def test_count_items_handles_shapes() -> None:
    assert fanout_join._count_items({"captions": [1, 2]}, "captions") == 2
    assert fanout_join._count_items({"records": [1]}, "captions") == 1
    assert fanout_join._count_items([1, 2, 3], "captions") == 3
    assert fanout_join._count_items({"nothing": 1}, "captions") == 0
