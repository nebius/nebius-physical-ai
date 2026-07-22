from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from npa.workflows.artifacts import (
    Artifact,
    ArtifactDiscoveryError,
    artifact_media_type,
    download_s3_uri,
    find_run_artifacts,
    list_all_runs,
    list_artifacts,
    list_run_categories,
    list_runs,
    render_hint_for_object,
    select_preferred_artifact,
)


class _FakePaginator:
    def __init__(self, pages: list[dict]):
        self._pages = pages

    def paginate(self, **_kwargs):
        for page in self._pages:
            yield page


class _FakeS3:
    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.download_calls: list[tuple[str, str, str]] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self._pages)

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        self.download_calls.append((bucket, key, dest))
        Path(dest).write_text("ok", encoding="utf-8")


def _obj(key: str, size: int = 1, ts: str = "2026-06-30T00:00:00+00:00") -> dict:
    return {
        "Key": key,
        "Size": size,
        "LastModified": datetime.fromisoformat(ts).astimezone(timezone.utc),
    }


def test_list_artifacts_returns_all_objects_including_unknown_extension() -> None:
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    _obj("run-a/reports/sim2real.rrd", 128),
                    _obj("run-a/metrics/report.json", 22),
                    _obj("run-a/raw/new-format.fooz", 19),
                ]
            }
        ]
    )
    artifacts = list_artifacts("bucket", "run-a", s3=s3)
    keys = [item.key for item in artifacts]
    assert "run-a/reports/sim2real.rrd" in keys
    assert "run-a/metrics/report.json" in keys
    assert "run-a/raw/new-format.fooz" in keys
    unknown = next(item for item in artifacts if item.key.endswith(".fooz"))
    assert unknown.render == "download"
    assert unknown.inline is False


@pytest.mark.parametrize("suffix", [".newkind", ".novelblob", ".artifactx"])
def test_new_artifact_type_is_discoverable_without_code_changes(suffix: str) -> None:
    s3 = _FakeS3([{"Contents": [_obj(f"run-b/data/object{suffix}", 7)]}])
    artifacts = list_artifacts("bucket", "run-b", s3=s3)
    assert len(artifacts) == 1
    assert artifacts[0].key.endswith(suffix)
    assert artifacts[0].render == "download"


def test_select_preferred_artifact_ranks_rerun_highest() -> None:
    artifacts = [
        Artifact("run", "run/frame.png", "s3://bucket/run/frame.png", 1, "2026-01-01T00:00:00+00:00", "image", True),
        Artifact("run", "run/trace.rrd", "s3://bucket/run/trace.rrd", 1, "2026-01-01T00:00:00+00:00", "rerun", True),
        Artifact("run", "run/out.mp4", "s3://bucket/run/out.mp4", 1, "2026-01-01T00:00:00+00:00", "video", True),
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.render == "rerun"


def test_select_preferred_artifact_chooses_run_report_rrd_before_component_images() -> None:
    artifacts = [
        Artifact(
            "run",
            "run/component-io/vlm/input/rollout/camera-001.ppm",
            "s3://bucket/run/component-io/vlm/input/rollout/camera-001.ppm",
            1,
            "2026-01-02T00:00:00+00:00",
            "image",
            True,
        ),
        Artifact(
            "run",
            "run/reports/sim2real.rrd",
            "s3://bucket/run/reports/sim2real.rrd",
            1,
            "2026-01-01T00:00:00+00:00",
            "rerun",
            True,
        ),
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.key.endswith("reports/sim2real.rrd")


def test_select_preferred_artifact_keeps_unknown_download_selectable() -> None:
    artifacts = [
        Artifact("run", "run/raw.foo", "s3://bucket/run/raw.foo", 1, "2026-01-01T00:00:00+00:00", "download", False)
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.key.endswith("raw.foo")


def test_list_runs_reports_truncation_explicitly() -> None:
    s3 = _FakeS3(
        [
            {"Contents": [_obj("run-1/a.txt"), _obj("run-2/b.txt"), _obj("run-3/c.txt")]},
        ]
    )
    page = list_runs("bucket", limit=2, s3=s3)
    assert page.total_runs == 3
    assert page.truncated is True
    assert len(page.runs) == 2


def test_download_s3_uri_fetches_explicit_object(tmp_path: Path) -> None:
    s3 = _FakeS3([])
    dest = tmp_path / "artifact.bin"
    output = download_s3_uri("s3://bucket-a/path/to/object.bin", dest, s3=s3)
    assert output == dest
    assert s3.download_calls == [("bucket-a", "path/to/object.bin", str(dest))]


def test_render_hint_detects_text_csv_and_unknown_fallback() -> None:
    assert render_hint_for_object(key="x/table.csv") == "text"
    assert render_hint_for_object(key="x/video.bin", content_type="video/mp4") == "video"
    assert render_hint_for_object(key="x/opaque.new") == "download"


def test_artifact_media_type_prefers_explicit_browser_types() -> None:
    assert artifact_media_type("demo.mp4") == "video/mp4"
    assert artifact_media_type("demo.webm") == "video/webm"
    assert artifact_media_type("shot.png") == "image/png"
    assert artifact_media_type("notes.md").startswith("text/plain")
    assert artifact_media_type("blob.bin") == "application/octet-stream"


def test_list_runs_requires_positive_limit() -> None:
    with pytest.raises(ArtifactDiscoveryError):
        list_runs("bucket", limit=0, s3=_FakeS3([]))


class _PrefixAwareS3:
    """Fake S3 that honors Prefix + Delimiter over an in-memory key store."""

    def __init__(self, keys: list[tuple[str, str]]):
        # keys: list of (key, iso_ts)
        self._keys = keys

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        store = self._keys

        class _P:
            def paginate(self, Bucket=None, Prefix="", Delimiter=None):  # noqa: N803
                if Delimiter:
                    seen: set[str] = set()
                    cps: list[dict] = []
                    for key, _ts in store:
                        if not key.startswith(Prefix):
                            continue
                        rest = key[len(Prefix):]
                        if "/" not in rest:
                            continue
                        seg = rest.split("/", 1)[0]
                        cp = Prefix + seg + "/"
                        if seg and cp not in seen:
                            seen.add(cp)
                            cps.append({"Prefix": cp})
                    yield {"CommonPrefixes": cps}
                else:
                    contents = [
                        _obj(key, ts=ts) for key, ts in store if key.startswith(Prefix)
                    ]
                    yield {"Contents": contents}

        return _P()


_LAYOUT = [
    ("checkpoints/sim2real-b/run-a/reports/sim2real.rrd", "2026-07-01T00:00:00+00:00"),
    ("checkpoints/physical-ai-data-factory/paidf-1/cosmos_augmented/f.png", "2026-07-22T00:00:00+00:00"),
    ("checkpoints/lerobot/default/model.pt", "2026-06-01T00:00:00+00:00"),
]


def test_list_run_categories_enumerates_dynamically() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    cats = list_run_categories("bucket", base_prefix="checkpoints", s3=s3)
    assert set(cats) == {
        "checkpoints/sim2real-b",
        "checkpoints/physical-ai-data-factory",
        "checkpoints/lerobot",
    }


def test_list_all_runs_merges_across_categories_latest_first() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    page = list_all_runs("bucket", base_prefix="checkpoints", limit=50, s3=s3)
    ids = [r.run_id for r in page.runs]
    # All runs across every category, no hardcoded workflow path; newest first.
    assert ids == ["paidf-1", "run-a", "default"]
    assert page.total_runs == 3


def test_find_run_artifacts_locates_run_in_any_category() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    arts = find_run_artifacts("bucket", base_prefix="checkpoints", run_id="paidf-1", s3=s3)
    assert [a.key for a in arts] == [
        "checkpoints/physical-ai-data-factory/paidf-1/cosmos_augmented/f.png"
    ]
    # A run under a different category is also found without a hardcoded prefix.
    assert find_run_artifacts("bucket", base_prefix="checkpoints", run_id="run-a", s3=s3)
    assert find_run_artifacts("bucket", base_prefix="checkpoints", run_id="missing", s3=s3) == []
