from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from npa.workflows.artifacts import (
    AmbiguousRunError,
    Artifact,
    ArtifactDiscoveryError,
    RunResolution,
    RunSummary,
    _merge_staging_resolutions,
    _merge_staging_summaries,
    artifact_media_type,
    artifact_data_role,
    build_fiftyone_dataset,
    decode_run_ref,
    download_s3_uri,
    encode_run_ref,
    find_run_artifacts,
    infer_run_id_from_artifact_key,
    find_run_artifact_page,
    list_all_run_prefixes,
    list_all_runs,
    list_artifacts,
    list_artifacts_page,
    list_run_categories,
    list_runs,
    list_runs_at_prefix_across_buckets,
    render_hint_for_object,
    resolve_run_artifact,
    resolve_run_artifacts,
    select_preferred_artifact,
)


def test_build_fiftyone_dataset_groups_variants_and_summarizes() -> None:
    run = "paidf-demo"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/input/video_0_frame_01.png",
        f"{base}/input/video_0_frame_02.png",
        f"{base}/cosmos_augmented/manifest.json",
        f"{base}/cosmos_augmented/aug-{run}-0/augmented_video.mp4",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/cosmos_augmented/aug-{run}-1/augmented_video.mp4",
        f"{base}/cosmos_augmented/aug-{run}-1/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json",
        f"{base}/labeled_augmented/captions.json",
        f"{base}/grade/vlm_eval_stub.json",
        f"{base}/grade/decision.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {
            "variables": {
                "cloth_color": "green",
                "lighting": "warm lamp light",
                "prompt": "a green cloth",
            }
        },
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json": {
            "variables": {
                "cloth_color": "blue",
                "lighting": "cool overhead light",
                "prompt": "a blue cloth",
            }
        },
        f"{base}/labeled_augmented/captions.json": {
            "captions": [
                {
                    "image": f"aug-{run}-0/frame-00000.png",
                    "caption": "green cloth on a countertop",
                },
                {
                    "image": f"aug-{run}-1/frame-00000.png",
                    "caption": "blue cloth on a sofa",
                },
            ]
        },
        f"{base}/grade/vlm_eval_stub.json": {
            "score": 0.0,
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        },
        f"{base}/grade/decision.json": {"decision": "loop_back"},
        f"{base}/curation/report.json": {
            "multiply": {"mode": "multi-variant", "variant_count": 2}
        },
    }

    dataset = build_fiftyone_dataset(
        keys, run_id=run, read_json=lambda k: payloads.get(k)
    )

    summary = dataset["summary"]
    assert summary["augmented_count"] == 2
    assert summary["variant_count"] == 2
    assert summary["multiply_mode"] == "multi-variant"
    assert summary["grade_score"] == 0.0
    assert summary["grade_decision"] == "loop_back"
    assert summary["input_count"] == 2
    assert summary["original_input_count"] == 2
    assert summary["synthetic_augmented_count"] == 2
    assert set(dataset["fields"]) == {"cloth_color", "lighting"}

    aug = [s for s in dataset["samples"] if s["group"] == "augmented"]
    inp = [s for s in dataset["samples"] if s["group"] == "source"]
    assert len(aug) == 2 and len(inp) == 2
    assert [sample["group"] for sample in dataset["samples"][:2]] == [
        "source",
        "source",
    ]
    assert all(sample["data_role"] == "source_input" for sample in inp)
    assert all(sample["data_role"] == "synthetic_augmented" for sample in aug)
    assert dataset["review"]["real_fiftyone"] is False
    assert "FiftyOne did not run" in dataset["review"]["label"]
    first = aug[0]
    assert first["thumbnail_key"].endswith("frame-00000.png")
    assert first["video_key"].endswith("augmented_video.mp4")
    assert "prompt" not in first["tags"]  # prompt is surfaced separately, not a tag
    assert first["caption"] == "green cloth on a countertop"
    # Report-only curation (no fiftyone block) -> empty curation surface.
    assert summary["curation_engine"] == ""
    assert first["uniqueness"] is None
    assert first["curated"] is None


def test_build_fiftyone_dataset_surfaces_real_fiftyone_curation() -> None:
    run = "paidf-fo"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/cosmos_augmented/aug-{run}-1/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {
            "variables": {"cloth_color": "green"}
        },
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json": {
            "variables": {"cloth_color": "blue"}
        },
        f"{base}/curation/report.json": {
            "multiply": {"mode": "multi-variant", "variant_count": 2},
            "curation_engine": "fiftyone-brain",
            "curated_kept": 1,
            "curated_dropped": 1,
            "fiftyone": {
                "brain": {
                    "uniqueness": {"count": 2, "mean": 0.55, "min": 0.2, "max": 0.9},
                    "near_duplicate_count": 1,
                },
                "selection": {"near_duplicate_count": 1},
                "samples": {
                    f"aug-{run}-0": {
                        "uniqueness": 0.9,
                        "kept": True,
                        "redundant": False,
                    },
                    f"aug-{run}-1": {
                        "uniqueness": 0.2,
                        "kept": False,
                        "redundant": True,
                    },
                },
            },
        },
    }

    dataset = build_fiftyone_dataset(
        keys, run_id=run, read_json=lambda k: payloads.get(k)
    )
    summary = dataset["summary"]
    assert summary["curation_engine"] == "fiftyone-brain"
    assert summary["curated_kept"] == 1
    assert summary["curated_dropped"] == 1
    assert summary["near_duplicate_count"] == 1
    assert summary["uniqueness"]["mean"] == 0.55
    assert dataset["review"]["real_fiftyone"] is True
    assert dataset["review"]["label"] == "Real FiftyOne Brain review"

    aug = sorted(
        (s for s in dataset["samples"] if s["group"] == "augmented"),
        key=lambda s: s["id"],
    )
    assert aug[0]["uniqueness"] == 0.9
    assert aug[0]["curated"] is True
    assert aug[0]["curation_flags"] == []
    assert aug[1]["uniqueness"] == 0.2
    assert aug[1]["curated"] is False
    assert aug[1]["curation_flags"] == ["redundant"]


def test_build_fiftyone_dataset_surfaces_input_captions_video_and_visualization() -> (
    None
):
    run = "paidf-more"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/input/video_0.mp4",
        f"{base}/input/frame_01.png",
        f"{base}/input/frame_02.png",
        f"{base}/labeled_original/captions.json",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/cosmos_augmented/aug-{run}-1/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {
            "variables": {"cloth_color": "green"}
        },
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json": {
            "variables": {"cloth_color": "blue"}
        },
        f"{base}/labeled_original/captions.json": {
            "captions": [
                {"image": "frame_01.png", "caption": "source: robot arm, plain wall"},
                {"image": "frame_02.png", "caption": "source: robot arm mid-fold"},
            ]
        },
        f"{base}/curation/report.json": {
            "curation_engine": "fiftyone-brain",
            "fiftyone": {
                "visualization": [
                    {"id": f"aug-{run}-0", "point": [0.1, 0.2]},
                    {"id": f"aug-{run}-1", "point": [0.9, -0.3]},
                ],
                "samples": {
                    f"aug-{run}-0": {
                        "uniqueness": 0.9,
                        "kept": True,
                        "redundant": False,
                    },
                    f"aug-{run}-1": {
                        "uniqueness": 0.2,
                        "kept": True,
                        "redundant": True,
                    },
                },
            },
        },
    }

    dataset = build_fiftyone_dataset(
        keys, run_id=run, read_json=lambda k: payloads.get(k), bucket="bkt"
    )

    # Visualization surfaced at the top level.
    assert len(dataset["visualization"]) == 2

    inp = [s for s in dataset["samples"] if s["group"] == "source"]
    # Source video is now included as an input sample (with a poster + video_uri).
    videos = [s for s in inp if s["video_uri"]]
    assert len(videos) == 1
    assert videos[0]["video_uri"].endswith("video_0.mp4")
    # Input frames carry their annotate-original captions.
    frame1 = next(s for s in inp if s["id"] == "frame_01.png")
    assert frame1["caption"] == "source: robot arm, plain wall"

    # Augmented samples carry their PCA point.
    aug = sorted(
        (s for s in dataset["samples"] if s["group"] == "augmented"),
        key=lambda s: s["id"],
    )
    assert aug[0]["point"] == [0.1, 0.2]


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
        Artifact(
            "run",
            "run/frame.png",
            "s3://bucket/run/frame.png",
            1,
            "2026-01-01T00:00:00+00:00",
            "image",
            True,
        ),
        Artifact(
            "run",
            "run/trace.rrd",
            "s3://bucket/run/trace.rrd",
            1,
            "2026-01-01T00:00:00+00:00",
            "rerun",
            True,
        ),
        Artifact(
            "run",
            "run/out.mp4",
            "s3://bucket/run/out.mp4",
            1,
            "2026-01-01T00:00:00+00:00",
            "video",
            True,
        ),
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.render == "rerun"


def test_select_preferred_artifact_chooses_run_report_rrd_before_component_images() -> (
    None
):
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
        Artifact(
            "run",
            "run/raw.foo",
            "s3://bucket/run/raw.foo",
            1,
            "2026-01-01T00:00:00+00:00",
            "download",
            False,
        )
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.key.endswith("raw.foo")


def test_list_runs_reports_truncation_explicitly() -> None:
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    _obj("run-1/a.txt"),
                    _obj("run-2/b.txt"),
                    _obj("run-3/c.txt"),
                ]
            },
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
    assert (
        render_hint_for_object(key="x/video.bin", content_type="video/mp4") == "video"
    )
    assert render_hint_for_object(key="x/opaque.new") == "download"


def test_render_hint_maps_mcap_to_lichtblick_render() -> None:
    from npa.workflows.artifacts import is_inline_render

    assert render_hint_for_object(key="run/reports/sim2real.mcap") == "mcap"
    # MCAP is an inline (viewable) render so the artifact browser offers it.
    assert is_inline_render("mcap") is True


def test_render_hint_and_media_type_handle_ppm_as_image() -> None:
    # .ppm (sim2real rollout camera dumps) are classified as images and transcoded to
    # PNG on serve (browsers cannot decode PPM), so they render in the Image pane.
    from npa.workflows.artifacts import artifact_media_type

    for key in ("run/rollout/camera-000.ppm", "run/x.pgm", "run/y.bmp", "run/z.tiff"):
        assert render_hint_for_object(key=key) == "image"
    assert artifact_media_type("camera-000.ppm") == "image/png"
    assert artifact_media_type("x.bmp") == "image/png"


def test_artifact_media_type_prefers_explicit_browser_types() -> None:
    assert artifact_media_type("demo.mp4") == "video/mp4"
    assert artifact_media_type("demo.webm") == "video/webm"
    assert artifact_media_type("shot.png") == "image/png"
    assert artifact_media_type("notes.md").startswith("text/plain")
    assert artifact_media_type("blob.bin") == "application/octet-stream"


def test_list_runs_requires_positive_limit() -> None:
    with pytest.raises(ArtifactDiscoveryError):
        list_runs("bucket", limit=0, s3=_FakeS3([]))


def test_list_runs_started_at_uses_run_start_not_newest_write() -> None:
    """started_at reflects when the run started; last_modified the newest write."""
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    # Run started 2026-07-25 22:26:36Z (encoded in the id); first
                    # artifact a few seconds later, newest artifact two days on.
                    _obj(
                        "sim2real-b/s2r-real-0725t222636z/env/data.json",
                        ts="2026-07-25T22:26:39+00:00",
                    ),
                    _obj(
                        "sim2real-b/s2r-real-0725t222636z/reports/sim2real.rrd",
                        ts="2026-07-27T02:17:31+00:00",
                    ),
                ]
            }
        ]
    )
    page = list_runs("bucket", prefix="sim2real-b", limit=50, s3=s3)
    run = next(r for r in page.runs if r.run_id == "s2r-real-0725t222636z")
    # Newest write is July 27; the displayed start is July 25 (id-encoded time).
    assert run.last_modified == "2026-07-27T02:17:31+00:00"
    assert run.started_at == "2026-07-25T22:26:36+00:00"
    assert run.to_dict()["started_at"] == "2026-07-25T22:26:36+00:00"


def test_list_runs_started_at_falls_back_to_earliest_write() -> None:
    """Runs whose id has no embedded timestamp start at the earliest artifact."""
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    _obj("cat/plain-run/a.json", ts="2026-05-10T08:00:00+00:00"),
                    _obj("cat/plain-run/b.json", ts="2026-05-12T09:00:00+00:00"),
                ]
            }
        ]
    )
    page = list_runs("bucket", prefix="cat", limit=50, s3=s3)
    run = next(r for r in page.runs if r.run_id == "plain-run")
    assert run.started_at == "2026-05-10T08:00:00+00:00"


def test_parse_run_id_timestamps_handles_full_and_yearless_forms() -> None:
    from npa.workflows.artifacts import _parse_run_id_timestamps, _run_started_at

    assert _parse_run_id_timestamps("job-20260725T222636Z") == [
        "2026-07-25T22:26:36+00:00"
    ]
    # Year-less: the hinted year plus the prior year are offered as candidates.
    assert _parse_run_id_timestamps("s2r-real-0725t222636z", year_hint=2026) == [
        "2026-07-25T22:26:36+00:00",
        "2025-07-25T22:26:36+00:00",
    ]
    # No embedded timestamp -> nothing parsed, fall back to the earliest write.
    assert _parse_run_id_timestamps("free-form-run-name") == []
    assert (
        _run_started_at("free-form-run-name", "2026-05-10T08:00:00+00:00")
        == "2026-05-10T08:00:00+00:00"
    )


def test_run_started_at_distrusts_far_off_id_date() -> None:
    from npa.workflows.artifacts import _run_started_at

    # A far-off id date (not just-before the first write) is distrusted.
    assert (
        _run_started_at("legacy-v20200101t000000-run", "2026-05-10T08:00:00+00:00")
        == "2026-05-10T08:00:00+00:00"
    )


def test_run_started_at_picks_real_start_past_a_red_herring_timestamp() -> None:
    from npa.workflows.artifacts import _run_started_at

    # A leading red-herring timestamp (2020) plus the real year-less start; the
    # latest candidate at/just-before the first write (2026-07-25) is chosen.
    started = _run_started_at(
        "legacy-v20200101t000000-s2r-0725t222636z", "2026-07-25T22:26:39+00:00"
    )
    assert started == "2026-07-25T22:26:36+00:00"


def test_run_started_at_handles_year_boundary() -> None:
    from npa.workflows.artifacts import _run_started_at

    # Started 2025-12-31 23:59:00; first artifact 2026-01-01. The prior-year
    # candidate is selected so the start isn't after the first write.
    started = _run_started_at("run-1231t235900z", "2026-01-01T00:00:05+00:00")
    assert started == "2025-12-31T23:59:00+00:00"


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
                        rest = key[len(Prefix) :]
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

    def list_objects_v2(
        self,
        *,
        Bucket=None,  # noqa: N803
        Prefix="",  # noqa: N803
        MaxKeys=1000,  # noqa: N803
        ContinuationToken="",  # noqa: N803
    ):
        del Bucket
        matching = [
            _obj(key, ts=ts) for key, ts in self._keys if key.startswith(Prefix)
        ]
        start = int(ContinuationToken or 0)
        end = min(start + int(MaxKeys), len(matching))
        truncated = end < len(matching)
        return {
            "Contents": matching[start:end],
            "IsTruncated": truncated,
            "NextContinuationToken": str(end) if truncated else "",
        }


class _PaginatedPrefixAwareS3(_PrefixAwareS3):
    """Prefix-aware fake that splits every object/category listing into pages."""

    def __init__(self, keys: list[tuple[str, str]], *, page_size: int = 1):
        super().__init__(keys)
        self.page_size = page_size
        self.object_body_fetches = 0

    def get_object(self, **_kwargs):
        self.object_body_fetches += 1
        raise AssertionError("run listing must not fetch object bodies")

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        store = self._keys
        page_size = self.page_size

        class _P:
            def paginate(self, Bucket=None, Prefix="", Delimiter=None):  # noqa: N803
                if Delimiter:
                    values: list[dict] = []
                    seen: set[str] = set()
                    for key, _ts in store:
                        if not key.startswith(Prefix):
                            continue
                        rest = key[len(Prefix) :]
                        if "/" not in rest:
                            continue
                        segment = rest.split("/", 1)[0]
                        common = Prefix + segment + "/"
                        if segment and common not in seen:
                            seen.add(common)
                            values.append({"Prefix": common})
                    for offset in range(0, len(values), page_size):
                        yield {"CommonPrefixes": values[offset : offset + page_size]}
                else:
                    values = [
                        _obj(key, ts=ts) for key, ts in store if key.startswith(Prefix)
                    ]
                    for offset in range(0, len(values), page_size):
                        yield {"Contents": values[offset : offset + page_size]}

        return _P()


_LAYOUT = [
    ("checkpoints/sim2real-b/run-a/reports/sim2real.rrd", "2026-07-01T00:00:00+00:00"),
    (
        "checkpoints/physical-ai-data-factory/paidf-1/cosmos_augmented/f.png",
        "2026-07-22T00:00:00+00:00",
    ),
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


def test_list_all_runs_ignores_malformed_historical_template_prefixes() -> None:
    s3 = _PrefixAwareS3(
        _LAYOUT
        + [
            (
                "checkpoints/isaac/2026-07-31_${NPA_ISAAC_RUN_ID}/report.json",
                "2026-07-31T08:01:11+00:00",
            )
        ]
    )

    page = list_all_runs("bucket", base_prefix="checkpoints", limit=50, s3=s3)

    assert {item.run_id for item in page.runs} == {"paidf-1", "run-a", "default"}
    assert page.discovery_complete is True


def test_find_run_artifacts_locates_run_in_any_category() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    arts = find_run_artifacts(
        "bucket", base_prefix="checkpoints", run_id="paidf-1", s3=s3
    )
    assert [a.key for a in arts] == [
        "checkpoints/physical-ai-data-factory/paidf-1/cosmos_augmented/f.png"
    ]
    # A run under a different category is also found without a hardcoded prefix.
    assert find_run_artifacts(
        "bucket", base_prefix="checkpoints", run_id="run-a", s3=s3
    )
    assert (
        find_run_artifacts("bucket", base_prefix="checkpoints", run_id="missing", s3=s3)
        == []
    )


def test_nested_storage_root_discovers_and_resolves_both_runs_without_body_fetch() -> (
    None
):
    layout = [
        ("archive/solutions/tool-a/run-one/report.rrd", "2026-08-01T00:00:00+00:00"),
        ("archive/solutions/tool-a/run-one/result.mp4", "2026-08-01T00:00:01+00:00"),
        ("archive/solutions/tool-b/run-two/manifest.json", "2026-08-02T00:00:00+00:00"),
        ("archive/solutions/tool-b/run-two/notes.txt", "2026-08-02T00:00:01+00:00"),
    ]
    s3 = _PaginatedPrefixAwareS3(layout, page_size=1)
    page = list_all_runs("bucket", base_prefix="archive/solutions", limit=100, s3=s3)
    assert {item.run_id for item in page.runs} == {"run-one", "run-two"}
    assert page.total_runs == 2
    assert all(item.run_ref for item in page.runs)
    assert s3.object_body_fetches == 0

    resolved = resolve_run_artifacts(
        ["bucket"],
        base_prefix="archive/solutions",
        run_ref_or_id="run-two",
        s3=s3,
    )
    assert resolved is not None
    assert resolved.source_prefix == "archive/solutions/tool-b"
    assert {item.key for item in resolved.artifacts} == {
        "archive/solutions/tool-b/run-two/manifest.json",
        "archive/solutions/tool-b/run-two/notes.txt",
    }


def test_paginated_discovery_has_no_duplicate_or_lost_runs() -> None:
    layout = [
        (
            f"nested/root/category-{i % 3}/run-{i}/artifact-{j}.json",
            f"2026-08-{i + 1:02d}T00:00:0{j}+00:00",
        )
        for i in range(6)
        for j in range(2)
    ]
    s3 = _PaginatedPrefixAwareS3(layout, page_size=1)
    page = list_all_runs("bucket", base_prefix="nested/root", limit=100, s3=s3)
    assert page.total_runs == 6
    assert len(page.runs) == 6
    assert {item.run_id for item in page.runs} == {f"run-{i}" for i in range(6)}
    assert s3.object_body_fetches == 0


def test_paginated_category_reports_true_total_beyond_global_limit() -> None:
    layout = [
        (
            f"nested/root/only-category/run-{index:03d}/artifact.json",
            f"2026-08-{(index % 28) + 1:02d}T00:00:00+00:00",
        )
        for index in range(101)
    ]
    s3 = _PaginatedPrefixAwareS3(layout, page_size=7)
    page = list_all_runs("bucket", base_prefix="nested/root", limit=100, s3=s3)
    assert len(page.runs) == 100
    assert page.total_runs == 101
    assert page.truncated is True
    assert len({item.run_ref for item in page.runs}) == 100
    assert s3.object_body_fetches == 0


def test_duplicate_run_basenames_are_source_qualified_and_plain_lookup_fails_closed() -> (
    None
):
    layout = [
        ("root/category-a/shared-run/a.rrd", "2026-08-01T00:00:00+00:00"),
        ("root/category-b/shared-run/b.rrd", "2026-08-02T00:00:00+00:00"),
    ]
    s3 = _PrefixAwareS3(layout)
    page = list_all_runs("bucket", base_prefix="root", limit=100, s3=s3)
    duplicates = [item for item in page.runs if item.run_id == "shared-run"]
    assert len(duplicates) == 2
    assert len({item.run_ref for item in duplicates}) == 2
    with pytest.raises(AmbiguousRunError):
        resolve_run_artifacts(
            ["bucket"], base_prefix="root", run_ref_or_id="shared-run", s3=s3
        )

    exact = resolve_run_artifacts(
        ["bucket"],
        base_prefix="root",
        run_ref_or_id=next(
            item.run_ref
            for item in duplicates
            if item.source_prefix.endswith("category-b")
        ),
        s3=s3,
    )
    assert exact is not None
    assert [item.key for item in exact.artifacts] == [
        "root/category-b/shared-run/b.rrd"
    ]


def test_run_ref_must_match_a_server_discovered_exact_source(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    A._run_list_cache_clear()
    artifact = A.Artifact(
        "safe-run",
        "authorized/safe-run/report.json",
        "s3://bucket/authorized/safe-run/report.json",
        1,
        "2026-08-10T00:00:00Z",
        "json",
        True,
    )
    monkeypatch.setattr(
        A,
        "find_run_artifact_matches",
        lambda *_args, **_kwargs: [
            A.RunResolution("safe-run", "bucket", "authorized", [artifact])
        ],
    )

    forged = A.encode_run_ref("bucket", "guessed", "safe-run")
    assert (
        A.resolve_run_artifacts(
            ["bucket"], base_prefix="", run_ref_or_id=forged, s3=object()
        )
        is None
    )


def test_exact_run_ref_reuses_credential_scoped_server_observation(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    A._run_list_cache_clear()
    s3 = _PrefixAwareS3(
        [
            (
                "authorized/safe-run/report.json",
                "2026-08-10T00:00:00+00:00",
            )
        ]
    )
    try:
        page = A.list_runs_cached_multi(
            ["bucket"],
            base_prefix="",
            limit=100,
            contains="safe-run",
            s3=s3,
        )
        assert len(page.runs) == 1
        monkeypatch.setattr(
            A,
            "find_run_artifact_matches",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cached exact selector repeated full discovery")
            ),
        )

        resolved = A.resolve_run_artifacts(
            ["bucket"],
            base_prefix="",
            run_ref_or_id=page.runs[0].run_ref,
            s3=s3,
        )

        assert resolved is not None
        assert resolved.source_prefix == "authorized"
        assert [item.key for item in resolved.artifacts] == [
            "authorized/safe-run/report.json"
        ]
    finally:
        A._run_list_cache_clear()


def test_server_discovered_exact_source_survives_unrelated_truncated_scan(
    monkeypatch,
) -> None:
    import npa.workflows.artifacts as A

    artifact = A.Artifact(
        "safe-run",
        "authorized/safe-run/report.json",
        "s3://bucket/authorized/safe-run/report.json",
        1,
        "2026-08-10T00:00:00Z",
        "json",
        True,
    )
    monkeypatch.setattr(
        A,
        "_list_artifact_run_index",
        lambda *_args, **_kwargs: A.RunListPage(
            runs=[
                A.RunSummary(
                    run_id="safe-run",
                    last_modified="2026-08-10T00:00:00Z",
                    artifact_count=1,
                    has_viewable=True,
                    bucket="bucket",
                    resolved_prefix="authorized",
                    namespaces=("authorized",),
                )
            ],
            truncated=True,
            total_runs=1,
            limit=100,
            discovery_complete=False,
        ),
    )
    monkeypatch.setattr(A, "list_artifacts", lambda *_args, **_kwargs: [artifact])

    matches = A.find_run_artifact_matches(
        "bucket",
        base_prefix="",
        run_id="safe-run",
        exact_source_prefix="authorized",
        s3=object(),
    )

    assert matches == [A.RunResolution("safe-run", "bucket", "authorized", [artifact])]


def test_undiscovered_exact_source_still_fails_closed_on_truncated_scan(
    monkeypatch,
) -> None:
    import npa.workflows.artifacts as A

    monkeypatch.setattr(
        A,
        "_list_artifact_run_index",
        lambda *_args, **_kwargs: A.RunListPage(
            runs=[
                A.RunSummary(
                    run_id="safe-run",
                    last_modified="2026-08-10T00:00:00Z",
                    artifact_count=1,
                    has_viewable=True,
                    bucket="bucket",
                    resolved_prefix="authorized",
                )
            ],
            truncated=True,
            total_runs=1,
            limit=100,
            discovery_complete=False,
        ),
    )

    with pytest.raises(A.ArtifactDiscoveryError, match="incomplete"):
        A.find_run_artifact_matches(
            "bucket",
            base_prefix="",
            run_id="safe-run",
            exact_source_prefix="guessed",
            s3=object(),
        )


def test_plain_run_resolution_fails_when_any_bucket_search_is_incomplete(
    monkeypatch,
) -> None:
    import npa.workflows.artifacts as A

    artifact = A.Artifact(
        "safe-run",
        "authorized/safe-run/report.json",
        "s3://bucket-a/authorized/safe-run/report.json",
        1,
        "2026-08-10T00:00:00Z",
        "json",
        True,
    )

    def find(bucket, **_kwargs):
        if bucket == "bucket-b":
            raise A.ArtifactDiscoveryError("incomplete")
        return [A.RunResolution("safe-run", bucket, "authorized", [artifact])]

    monkeypatch.setattr(A, "find_run_artifact_matches", find)
    with pytest.raises(A.ArtifactDiscoveryError, match="incomplete"):
        A.resolve_run_artifacts(
            ["bucket-a", "bucket-b"],
            base_prefix="",
            run_ref_or_id="safe-run",
            s3=object(),
        )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("run/view.rrd", "rerun"),
        ("run/movie.mp4", "video"),
        ("run/data.json", "json"),
        ("run/readme.txt", "text"),
        ("run/frame.png", "image"),
        ("run/blob.future", "download"),
    ],
)
def test_render_contract_covers_viewers_and_unknown_download(
    key: str, expected: str
) -> None:
    assert render_hint_for_object(key=key) == expected


def test_run_refs_and_ids_reject_traversal_and_malformed_values() -> None:
    ref = encode_run_ref("valid-bucket", "nested/root", "safe-run")
    assert decode_run_ref(ref) == ("valid-bucket", "nested/root", "safe-run")
    for bad in ("npa1_not-base64!", "npa1_", "../run", "folder/run"):
        if bad.startswith("npa1_"):
            with pytest.raises(ArtifactDiscoveryError):
                decode_run_ref(bad)
        else:
            with pytest.raises(ArtifactDiscoveryError):
                resolve_run_artifacts(
                    ["valid-bucket"],
                    base_prefix="nested/root",
                    run_ref_or_id=bad,
                    s3=_PrefixAwareS3([]),
                )
    with pytest.raises(ArtifactDiscoveryError):
        encode_run_ref("valid-bucket", "nested/../escape", "safe-run")


# Runs also live at the BUCKET ROOT under a category (not under the configured
# base root), e.g. scenario-gen-smoke/<run>/... and physical-ai-data-factory/<run>/...
# Discovery must span both roots so these are visible + openable.
_MULTI_ROOT_LAYOUT = _LAYOUT + [
    (
        "scenario-gen-smoke/scenario-gen-smoke-1/npa-workflow/manifest.json",
        "2026-07-23T15:32:22+00:00",
    ),
    (
        "scenario-gen-smoke/scenario-gen-smoke-1/ranked/ranked.json",
        "2026-07-23T15:32:20+00:00",
    ),
    (
        "physical-ai-data-factory/paidf-root-1/reports/final.json",
        "2026-07-19T00:00:00+00:00",
    ),
]


def test_discovery_categories_spans_base_and_bucket_root() -> None:
    from npa.workflows.artifacts import discovery_categories

    cats = discovery_categories(
        "bucket", base_prefix="checkpoints", s3=_PrefixAwareS3(_MULTI_ROOT_LAYOUT)
    )
    # Base-root categories come first, then root-level categories; the base root
    # itself ("checkpoints") is NOT treated as a run parent (its children are cats).
    assert "checkpoints/sim2real-b" in cats
    assert "checkpoints/physical-ai-data-factory" in cats
    assert "scenario-gen-smoke" in cats
    assert "physical-ai-data-factory" in cats
    assert "checkpoints" not in cats


def test_list_all_runs_surfaces_root_level_runs() -> None:
    s3 = _PrefixAwareS3(_MULTI_ROOT_LAYOUT)
    page = list_all_runs("bucket", base_prefix="checkpoints", limit=50, s3=s3)
    ids = [r.run_id for r in page.runs]
    # Root-level runs are discovered alongside checkpoints runs, newest first.
    assert ids[0] == "scenario-gen-smoke-1"
    assert set(ids) == {
        "scenario-gen-smoke-1",
        "paidf-1",
        "paidf-root-1",
        "run-a",
        "default",
    }


def test_lightweight_prefix_index_discovers_runs_without_object_summaries() -> None:
    page = list_all_run_prefixes(
        "bucket",
        base_prefix="checkpoints",
        limit=50,
        s3=_PrefixAwareS3(_MULTI_ROOT_LAYOUT),
    )

    assert {item.run_id for item in page.runs} == {
        "scenario-gen-smoke-1",
        "paidf-1",
        "paidf-root-1",
        "run-a",
        "default",
    }
    assert all(item.summary_complete is True for item in page.runs)
    assert all(item.artifact_count > 0 for item in page.runs)
    assert all(item.last_modified for item in page.runs)
    assert any(item.has_viewable is True for item in page.runs)


def test_lightweight_timestamp_less_runs_use_s3_recency_and_viewability() -> None:
    layout = [
        ("category/plain-old/result.bin", "2026-07-01T00:00:00+00:00"),
        ("category/plain-new/preview.mp4", "2026-08-01T00:00:00+00:00"),
    ]

    page = list_all_run_prefixes(
        "bucket",
        limit=50,
        s3=_PrefixAwareS3(layout),
    )

    assert [item.run_id for item in page.runs] == ["plain-new", "plain-old"]
    assert page.runs[0].last_modified == "2026-08-01T00:00:00+00:00"
    assert page.runs[0].started_at == "2026-08-01T00:00:00+00:00"
    assert page.runs[0].has_viewable is True
    assert page.runs[0].summary_complete is True
    assert page.runs[1].has_viewable is False


def test_artifact_index_follows_all_pages_for_exact_summary() -> None:
    layout = [
        (f"category/large-run/raw/{index:04d}.bin", "2026-08-01T00:00:00+00:00")
        for index in range(1001)
    ]
    layout.append(("category/large-run/preview.mp4", "2026-08-02T00:00:00+00:00"))

    page = list_all_run_prefixes(
        "bucket",
        limit=50,
        s3=_PrefixAwareS3(layout),
    )

    summary = page.runs[0]
    assert summary.run_id == "large-run"
    assert summary.summary_complete is True
    assert summary.has_viewable is True
    assert summary.artifact_count == 1002


def test_find_run_artifacts_locates_root_level_run() -> None:
    s3 = _PrefixAwareS3(_MULTI_ROOT_LAYOUT)
    arts = find_run_artifacts(
        "bucket", base_prefix="checkpoints", run_id="scenario-gen-smoke-1", s3=s3
    )
    keys = sorted(a.key for a in arts)
    assert keys == [
        "scenario-gen-smoke/scenario-gen-smoke-1/npa-workflow/manifest.json",
        "scenario-gen-smoke/scenario-gen-smoke-1/ranked/ranked.json",
    ]


def test_find_run_artifacts_merges_staging_inputs_with_authoritative_outputs() -> None:
    """One run id may have staged inputs and a separate completed output root."""
    run_id = "groot17-8gpu-20260806T024557Z-3dfb0270"
    layout = [
        (f"groot-1-7-finetune/{run_id}/source/runner.py", "2026-08-06T02:40:00+00:00"),
        (f"groot-1-7-finetune/{run_id}/data/episode.mp4", "2026-08-06T02:41:00+00:00"),
        (f"{run_id}/checkpoints/model.safetensors", "2026-08-06T03:01:00+00:00"),
        (f"{run_id}/manifest.json", "2026-08-06T03:02:00+00:00"),
    ]

    artifacts = find_run_artifacts(
        "bucket", base_prefix="", run_id=run_id, s3=_PrefixAwareS3(layout)
    )

    assert {item.key for item in artifacts} == {key for key, _timestamp in layout}
    assert {
        item.role
        for item in artifacts
        if "/source/" in item.key or "/data/" in item.key
    } == {"input"}
    outputs = [item for item in artifacts if item.role == "output"]
    assert {item.key for item in outputs} == {
        f"{run_id}/checkpoints/model.safetensors",
        f"{run_id}/manifest.json",
    }
    checkpoint = next(item for item in outputs if item.key.endswith(".safetensors"))
    assert checkpoint.render == "download"
    assert checkpoint.inline is False


def test_list_all_runs_groups_duplicate_run_namespaces_and_counts_outputs() -> None:
    run_id = "groot17-8gpu-20260806T024557Z-3dfb0270"
    layout = [
        (f"groot-1-7-finetune/{run_id}/source/runner.py", "2026-08-06T02:40:00+00:00"),
        (f"groot-1-7-finetune/{run_id}/data/episode.mp4", "2026-08-06T02:41:00+00:00"),
        (f"{run_id}/checkpoints/model.safetensors", "2026-08-06T03:01:00+00:00"),
        (f"{run_id}/manifest.json", "2026-08-06T03:02:00+00:00"),
    ]

    page = list_all_runs(
        "bucket", base_prefix="", limit=50, contains=run_id, s3=_PrefixAwareS3(layout)
    )

    matching = [run for run in page.runs if run.run_id == run_id]
    assert len(matching) == 1
    assert matching[0].artifact_count == 4
    assert matching[0].output_artifact_count == 2
    assert matching[0].input_artifact_count == 2


def test_staging_merge_deduplicates_same_source_and_preserves_started_at() -> None:
    run_id = "groot-run-20260811T000000Z"
    output = Artifact(
        run_id=run_id,
        key=f"{run_id}/manifest.json",
        s3_uri=f"s3://bucket/{run_id}/manifest.json",
        size=10,
        last_modified="2026-08-11T00:01:00+00:00",
        render="json",
        inline=True,
        relative_key="manifest.json",
    )
    duplicate = RunResolution(run_id, "bucket", "", [output])
    merged_resolutions = _merge_staging_resolutions([duplicate, duplicate])
    assert len(merged_resolutions) == 1
    assert merged_resolutions[0].artifacts == [output]

    primary = RunSummary(
        run_id=run_id,
        last_modified="2026-08-11T00:01:00+00:00",
        started_at="2026-08-11T00:00:00+00:00",
        artifact_count=1,
        has_viewable=True,
        bucket="bucket",
        resolved_prefix="",
        output_artifact_count=1,
        canonical_score=1000,
    )
    staging = RunSummary(
        run_id=run_id,
        last_modified="",
        started_at="",
        artifact_count=1,
        has_viewable=False,
        bucket="bucket",
        resolved_prefix="groot-1-7-finetune",
        input_artifact_count=1,
        canonical_score=0,
    )
    merged_summaries = _merge_staging_summaries([primary, staging])
    assert len(merged_summaries) == 1
    assert merged_summaries[0].started_at == primary.started_at
    assert merged_summaries[0].last_modified == primary.last_modified
    assert merged_summaries[0].artifact_count == 2


def test_staging_resolution_conflicts_remain_fail_closed() -> None:
    run_id = "groot-run-20260811T000000Z"
    first = Artifact(
        run_id=run_id,
        key=f"{run_id}/manifest.json",
        s3_uri=f"s3://bucket/{run_id}/manifest.json",
        size=10,
        last_modified="2026-08-11T00:01:00+00:00",
        render="json",
        inline=True,
        relative_key="manifest.json",
    )
    conflicting = Artifact(**{**first.__dict__, "size": 11})
    matches = [
        RunResolution(run_id, "bucket", "", [first]),
        RunResolution(run_id, "bucket", "", [conflicting]),
    ]
    assert _merge_staging_resolutions(matches) == matches


def test_yaml_artifact_is_renderable_text_and_downloadable() -> None:
    assert render_hint_for_object(key="run/workflow.yaml") == "text"
    assert artifact_media_type("workflow.yaml").startswith("text/plain")


def test_list_runs_skips_bare_files_not_run_dirs() -> None:
    # A file sitting directly under a category is not a run directory.
    s3 = _PrefixAwareS3(
        [
            ("scenario-gen-smoke/records.json", "2026-07-23T00:00:00+00:00"),
            (
                "scenario-gen-smoke/real-run-1/npa-workflow/status.json",
                "2026-07-23T10:00:00+00:00",
            ),
        ]
    )
    page = list_runs("bucket", prefix="scenario-gen-smoke", limit=50, s3=s3)
    ids = [r.run_id for r in page.runs]
    assert ids == ["real-run-1"]
    assert "records.json" not in ids


def test_discovery_categories_excludes_infra_roots() -> None:
    from npa.workflows.artifacts import discovery_categories

    layout = _MULTI_ROOT_LAYOUT + [
        ("npa-agent/session-state/a/b.json", "2026-07-23T00:00:00+00:00"),
        ("npa-agent/tenants/t/chat-sessions/s.json", "2026-07-23T00:00:00+00:00"),
    ]
    cats = discovery_categories(
        "bucket",
        base_prefix="checkpoints",
        exclude={"npa-agent"},
        s3=_PrefixAwareS3(layout),
    )
    assert "npa-agent" not in cats
    assert "scenario-gen-smoke" in cats


def test_discovery_categories_preserves_nested_exclusion_scope() -> None:
    from npa.workflows.artifacts import discovery_categories

    layout = [
        ("npa-agent/session-state/current/state.json", "2026-07-23T00:00:00+00:00"),
        ("npa-agent/customer-runs/run-a/report.json", "2026-07-24T00:00:00+00:00"),
        ("other/run-b/report.json", "2026-07-25T00:00:00+00:00"),
    ]
    cats = discovery_categories(
        "bucket",
        exclude={"npa-agent/session-state"},
        s3=_PrefixAwareS3(layout),
    )

    assert "npa-agent" in cats
    assert "other" in cats


def test_list_all_runs_excludes_infra_roots() -> None:
    layout = _MULTI_ROOT_LAYOUT + [
        ("npa-agent/session-state/a/state.json", "2026-07-23T00:00:00+00:00"),
    ]
    page = list_all_runs(
        "bucket",
        base_prefix="checkpoints",
        limit=50,
        exclude={"npa-agent"},
        s3=_PrefixAwareS3(layout),
    )
    ids = [r.run_id for r in page.runs]
    assert "session-state" not in ids
    assert "scenario-gen-smoke-1" in ids


def test_infrastructure_state_roots_are_never_discovered_as_runs() -> None:
    layout = _MULTI_ROOT_LAYOUT + [
        (
            "terraform-state/environments/production.tfstate",
            "2026-08-01T00:00:00+00:00",
        ),
        ("terraform_state/workspaces/default.tfstate", "2026-08-01T00:01:00+00:00"),
        (
            "checkpoints/sim2real-b/terraform-state/current.tfstate",
            "2026-08-01T00:02:00+00:00",
        ),
        (
            "scenario-gen-smoke/terraform_state/current.tfstate",
            "2026-08-01T00:03:00+00:00",
        ),
    ]

    full = list_all_runs(
        "bucket", base_prefix="checkpoints", limit=50, s3=_PrefixAwareS3(layout)
    )
    light = list_all_run_prefixes(
        "bucket", base_prefix="checkpoints", limit=50, s3=_PrefixAwareS3(layout)
    )

    assert "terraform-state" not in {item.run_id for item in full.runs}
    assert "terraform_state" not in {item.run_id for item in full.runs}
    assert "environments" not in {item.run_id for item in light.runs}
    assert "workspaces" not in {item.run_id for item in light.runs}


def test_infrastructure_only_bucket_returns_no_user_runs() -> None:
    layout = [
        (
            "terraform-state/environments/production.tfstate",
            "2026-08-01T00:00:00+00:00",
        ),
        ("terraform_state/workspaces/default.tfstate", "2026-08-01T00:01:00+00:00"),
    ]

    full = list_all_runs(
        "bucket", limit=50, contains="terraform-state", s3=_PrefixAwareS3(layout)
    )
    light = list_all_run_prefixes(
        "bucket", limit=50, contains="terraform-state", s3=_PrefixAwareS3(layout)
    )

    assert full.runs == []
    assert full.total_runs == 0
    assert light.runs == []
    assert light.total_runs == 0


def test_artifact_index_excludes_category_and_source_cache_roots_across_pages() -> None:
    nested = "byof-solution-e2e-20310102T030405Z"
    flat = "flat-policy-20310103T030405Z"
    pages = [
        {
            "Contents": [
                _obj("tenants/tenant-a/project-a/chat-sessions/session.json"),
                _obj(f"npa-src/{flat}/source/main.py"),
                _obj(f"oss-solutions/solution-family/{nested}/output/future.blobx"),
            ]
        },
        {
            "Contents": [
                _obj(f"{flat}/evaluation/aggregate.json"),
                _obj(f"{flat}/preview.mp4"),
            ]
        },
    ]

    page = list_all_run_prefixes("bucket", limit=50, s3=_FakeS3(pages))
    sources = {(item.run_id, item.resolved_prefix) for item in page.runs}

    assert sources == {
        (nested, "oss-solutions/solution-family"),
        (flat, ""),
    }
    assert page.discovery_complete is True
    assert "tenants" not in {item.run_id for item in page.runs}


def test_duplicate_run_ids_keep_each_exact_source() -> None:
    duplicate = "duplicate-policy-20310103T030405Z"
    layout = [
        (f"{duplicate}/aggregate.json", "2031-01-03T03:05:00+00:00"),
        (f"category/{duplicate}/report.json", "2031-01-03T03:06:00+00:00"),
    ]

    page = list_all_run_prefixes("bucket", limit=50, s3=_PrefixAwareS3(layout))
    matches = [item for item in page.runs if item.run_id == duplicate]

    assert {item.resolved_prefix for item in matches} == {"", "category"}
    assert len(matches) == 2


def test_mixed_category_and_flat_layout_retains_timestamped_parent_run() -> None:
    flat_run = "policy-run-20310405t060708z"
    layout = _MULTI_ROOT_LAYOUT + [
        (f"{flat_run}/evaluation/aggregate.json", "2031-04-05T06:10:00+00:00"),
        (f"{flat_run}/checkpoints/policy.ckpt", "2031-04-05T06:11:00+00:00"),
    ]

    page = list_all_run_prefixes(
        "bucket", base_prefix="checkpoints", limit=50, s3=_PrefixAwareS3(layout)
    )

    run_ids = {item.run_id for item in page.runs}
    assert flat_run in run_ids
    assert "evaluation" not in run_ids
    assert (
        page.runs[[item.run_id for item in page.runs].index(flat_run)].to_dict()[
            "source_type"
        ]
        == "artifact_storage"
    )


def test_flat_run_detection_still_honors_server_side_search() -> None:
    layout = [
        (
            "alpha-run-20310405t060708z/evaluation/report.json",
            "2031-04-05T06:10:00+00:00",
        ),
        (
            "beta-run-20310406t060708z/evaluation/report.json",
            "2031-04-06T06:10:00+00:00",
        ),
    ]

    page = list_all_run_prefixes(
        "bucket", limit=50, contains="beta-run", s3=_PrefixAwareS3(layout)
    )

    assert [item.run_id for item in page.runs] == ["beta-run-20310406t060708z"]


def test_ppm_and_netpbm_are_images_and_need_transcode() -> None:
    from npa.workflows.artifacts import needs_image_transcode, render_hint_for_object

    # Sim-rollout camera frames are saved as .ppm — classified as viewable images.
    assert render_hint_for_object(key="run/actions/rollout/camera-000.ppm") == "image"
    assert render_hint_for_object(key="run/x.bmp") == "image"
    # Browser cannot render these natively → must transcode to PNG on the way out.
    for name in ("camera-000.ppm", "x.pgm", "y.bmp", "z.tiff"):
        assert needs_image_transcode(name) is True
    # Web-native images are served as-is (no transcode).
    for name in ("frame.png", "a.jpg", "b.webp"):
        assert needs_image_transcode(name) is False
    assert render_hint_for_object(key="run/frame.png") == "image"


def test_list_runs_contains_search_survives_limit() -> None:
    # An old run beyond the newest `limit` must still be found by substring search.
    keys = [
        (
            f"cat/run-new-{i:03d}/reports/r.json",
            f"2026-07-{(i % 27) + 1:02d}T00:00:00+00:00",
        )
        for i in range(30)
    ]
    keys.append(
        (
            "cat/rtxpro-staged-2x2-old/actions/train/camera-000.ppm",
            "2026-06-13T01:13:56+00:00",
        )
    )
    s3 = _PrefixAwareS3(keys)
    # Without search, limit=5 returns only the 5 newest → the old run is cut off.
    page = list_runs("bucket", prefix="cat", limit=5, s3=s3)
    assert "rtxpro-staged-2x2-old" not in [r.run_id for r in page.runs]
    # With substring search, the old run is found despite the small limit.
    page = list_runs(
        "bucket", prefix="cat", limit=5, contains="rtxpro-staged-2x2", s3=s3
    )
    assert [r.run_id for r in page.runs] == ["rtxpro-staged-2x2-old"]


def test_list_all_runs_contains_search_across_roots() -> None:
    layout = _MULTI_ROOT_LAYOUT + [
        (
            "sim2real-b/rtxpro-staged-2x2-old/actions/train/camera-000.ppm",
            "2026-06-13T01:13:56+00:00",
        ),
    ]
    page = list_all_runs(
        "bucket",
        base_prefix="checkpoints",
        limit=3,
        contains="rtxpro-staged",
        s3=_PrefixAwareS3(layout),
    )
    assert [r.run_id for r in page.runs] == ["rtxpro-staged-2x2-old"]


def test_exact_search_finds_flat_root_run_in_mixed_layout() -> None:
    run_id = "mixed-flat-run-20300101t010203z"
    layout = _MULTI_ROOT_LAYOUT + [
        (f"{run_id}/eval/report.json", "2026-08-06T01:50:00+00:00"),
        (f"{run_id}/eval/rollout.mp4", "2026-08-06T01:51:00+00:00"),
        (f"{run_id}/checkpoints/policy.ckpt", "2026-08-06T01:52:00+00:00"),
        (f"{run_id}/checkpoints/metrics.json", "2026-08-06T01:53:00+00:00"),
        (f"{run_id}/logs/train.log", "2026-08-06T01:54:00+00:00"),
        (f"{run_id}/raw/future-format.blobx", "2026-08-06T01:55:00+00:00"),
    ]
    s3 = _PrefixAwareS3(layout)

    page = list_all_runs(
        "bucket",
        base_prefix="checkpoints",
        limit=50,
        contains=run_id,
        s3=s3,
    )

    assert [item.run_id for item in page.runs] == [run_id]
    assert page.runs[0].artifact_count == 6
    assert "eval" not in [item.run_id for item in page.runs]
    assert "checkpoints" not in [item.run_id for item in page.runs]
    artifacts = find_run_artifacts(
        "bucket", base_prefix="checkpoints", run_id=run_id, s3=s3
    )
    assert len(artifacts) == 6
    unknown = next(item for item in artifacts if item.key.endswith(".blobx"))
    assert unknown.render == "download"
    assert unknown.inline is False

    lightweight = list_all_run_prefixes(
        "bucket",
        base_prefix="checkpoints",
        limit=50,
        contains=run_id,
        s3=s3,
    )
    assert [item.run_id for item in lightweight.runs] == [run_id]
    assert lightweight.runs[0].summary_complete is True

    resolved_prefix, artifact_page = find_run_artifact_page(
        "bucket",
        base_prefix="checkpoints",
        run_id=run_id,
        s3=s3,
    )
    assert resolved_prefix == ""
    assert len(artifact_page.artifacts) == 6
    assert artifact_page.truncated is False


def test_artifact_pages_preserve_unknown_formats_and_cursor() -> None:
    run_id = "paged-run"
    s3 = _PrefixAwareS3(
        [
            (f"category/{run_id}/a.json", "2030-01-01T00:00:01+00:00"),
            (f"category/{run_id}/b.futureblob", "2030-01-01T00:00:02+00:00"),
            (f"category/{run_id}/c.mp4", "2030-01-01T00:00:03+00:00"),
        ]
    )

    first = list_artifacts_page(
        "bucket",
        run_id,
        prefix="category",
        page_size=2,
        s3=s3,
    )
    second = list_artifacts_page(
        "bucket",
        run_id,
        prefix="category",
        cursor=first.next_cursor,
        page_size=2,
        s3=s3,
    )

    assert first.truncated is True
    assert first.next_cursor
    assert second.truncated is False
    combined = [*first.artifacts, *second.artifacts]
    unknown = next(item for item in combined if item.key.endswith(".futureblob"))
    assert unknown.render == "download"
    assert unknown.inline is False


def test_run_artifact_discovery_object_scan_is_bounded_and_truthful(
    monkeypatch,
) -> None:
    import npa.workflows.artifacts as A

    monkeypatch.setattr(A, "MAX_RUN_DISCOVERY_OBJECTS", 2)
    page = A.list_all_run_prefixes(
        "bucket",
        limit=50,
        s3=_PrefixAwareS3(
            [
                (
                    f"category/run-{index}/report.json",
                    f"2030-01-0{index + 1}T00:00:00+00:00",
                )
                for index in range(5)
            ]
        ),
    )
    assert page.discovery_complete is False
    assert page.truncated is True
    assert page.to_dict()["pagination_complete"] is False


def test_explicit_prefix_discovery_searches_all_accessible_project_buckets() -> None:
    class MultiBucketS3:
        def get_paginator(self, name: str):
            assert name == "list_objects_v2"

            class Paginator:
                def paginate(self, **kwargs):
                    bucket = kwargs["Bucket"]
                    prefix = kwargs["Prefix"]
                    yield {"Contents": [_obj(f"{prefix}run-{bucket}/report.json")]}

            return Paginator()

    page = list_runs_at_prefix_across_buckets(
        ["bucket-a", "bucket-b"],
        prefix="category/",
        bucket_projects={"bucket-a": "project-a", "bucket-b": "project-b"},
        s3=MultiBucketS3(),
    )

    assert {(run.bucket, run.project_id) for run in page.runs} == {
        ("bucket-a", "project-a"),
        ("bucket-b", "project-b"),
    }


# --- Multi-bucket discovery ---------------------------------------------------


def test_list_accessible_buckets_primary_first_deduped() -> None:
    import npa.workflows.artifacts as A

    class _S3:
        def list_buckets(self):
            raise AssertionError("discovery must not enumerate unrelated buckets")

    got = A.list_accessible_buckets(_S3(), primary="primary", extra=["b2"])
    assert got[0] == "primary"
    assert got.count("primary") == 1 and got.count("b2") == 1
    assert set(got) == {"primary", "b2"}


def test_list_accessible_buckets_survives_no_listbuckets_permission() -> None:
    import npa.workflows.artifacts as A

    class _S3:
        def list_buckets(self):
            raise A.BotoCoreError()

    got = A.list_accessible_buckets(_S3(), primary="primary", extra=["x"])
    assert got == ["primary", "x"]  # falls back to primary/extras only


def test_find_run_artifacts_across_buckets_returns_unique_match(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    scanned: list[str] = []

    def fake_find(bucket, *, base_prefix, run_id, s3):
        scanned.append(bucket)
        if bucket == "b2":
            artifact = A.Artifact(
                run_id,
                f"byof/{run_id}/x.json",
                f"s3://b2/byof/{run_id}/x.json",
                1,
                "t",
                "json",
                False,
            )
            return [A.RunResolution(run_id, "b2", f"b2:{run_id}", [artifact])]
        return []

    monkeypatch.setattr(A, "find_run_artifact_matches", fake_find)
    bkt, arts = A.find_run_artifacts_across_buckets(
        ["b1", "b2", "b3"], base_prefix="", run_id="run-x", s3=object()
    )
    assert bkt == "b2" and len(arts) == 1
    assert arts[0].s3_uri == "s3://b2/byof/run-x/x.json"
    assert scanned == ["b1", "b2", "b3"]  # all configured buckets prove uniqueness


def test_list_all_runs_across_buckets_merges_and_tags(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    def fake_all(bucket, *, base_prefix, limit, exclude, contains, s3):
        return A.RunListPage(
            runs=[A.RunSummary(f"run-{bucket}", "2026-06-30T00:00:00+00:00", 1, True)],
            truncated=False,
            total_runs=1,
            limit=limit,
        )

    monkeypatch.setattr(A, "list_all_runs", fake_all)
    page = A.list_all_runs_across_buckets(
        ["b1", "b2"],
        base_prefix="",
        limit=50,
        exclude=None,
        contains="",
        bucket_projects={"b1": "project-a", "b2": "project-b"},
        s3=object(),
    )
    tagged = {(r.bucket, r.project_id, r.run_id) for r in page.runs}
    assert ("b1", "project-a", "run-b1") in tagged
    assert ("b2", "project-b", "run-b2") in tagged
    assert page.total_runs == 2


def test_multi_bucket_run_discovery_preserves_bucket_truncation(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    def fake_light(bucket, **_kwargs):
        return A.RunListPage(
            runs=[A.RunSummary("run-known", "2030-01-01T00:00:00+00:00", 1, True)],
            truncated=True,
            total_runs=A.MAX_RUN_PARENT_CANDIDATES + 1,
            limit=A.MAX_RUN_PARENT_CANDIDATES,
            discovery_complete=True,
        )

    monkeypatch.setattr(A, "list_all_run_prefixes", fake_light)
    page = A.list_all_runs_across_buckets(
        ["bucket"],
        base_prefix="",
        limit=A.MAX_RUN_PARENT_CANDIDATES,
        lightweight=True,
        s3=object(),
    )

    assert page.total_runs == A.MAX_RUN_PARENT_CANDIDATES + 1
    assert page.truncated is True
    assert page.discovery_complete is True


def test_exact_source_discovery_reports_a_truncated_candidate_set_incomplete(
    monkeypatch,
) -> None:
    import npa.workflows.artifacts as A

    source = A.RunSummary(
        "run-known",
        "2026-06-30T00:00:00+00:00",
        1,
        True,
        bucket="bucket",
        project_id="project",
        resolved_prefix="category",
    )
    monkeypatch.setattr(
        A,
        "list_all_runs_across_buckets",
        lambda *_args, **_kwargs: A.RunListPage(
            runs=[source],
            truncated=True,
            total_runs=A.MAX_RUN_PARENT_CANDIDATES + 1,
            limit=A.MAX_RUN_PARENT_CANDIDATES,
            discovery_complete=True,
        ),
    )

    matches, errors, complete = A.find_run_sources_across_buckets(
        ["bucket"], base_prefix="", run_id="run-known", s3=object()
    )

    assert matches == [source]
    assert errors == ()
    assert complete is False


def test_exact_source_discovery_uses_one_bounded_prefix_probe(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    calls: list[dict[str, object]] = []

    class _S3:
        def list_objects_v2(self, **kwargs):
            calls.append(dict(kwargs))
            return {
                "Contents": [
                    {
                        "Key": "category/nested/run-known/report.json",
                        "LastModified": datetime(2026, 8, 10, tzinfo=timezone.utc),
                    }
                ]
            }

    monkeypatch.setattr(
        A,
        "list_all_runs_across_buckets",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact source probe must not scan the whole bucket")
        ),
    )
    matches, errors, complete = A.find_run_sources_across_buckets(
        ["bucket"],
        base_prefix="",
        run_id="run-known",
        exact_prefix="category/nested",
        exclude={"npa-agent/tenants"},
        bucket_projects={"bucket": "project"},
        s3=_S3(),
    )

    assert calls == [
        {
            "Bucket": "bucket",
            "Prefix": "category/nested/run-known/",
            "MaxKeys": 1,
        }
    ]
    assert errors == ()
    assert complete is True
    assert len(matches) == 1
    assert matches[0].bucket == "bucket"
    assert matches[0].project_id == "project"
    assert matches[0].resolved_prefix == "category/nested"
    assert matches[0].summary_complete is False


def test_exact_source_discovery_rejects_excluded_structural_prefix() -> None:
    import npa.workflows.artifacts as A

    class _S3:
        def list_objects_v2(self, **_kwargs):
            raise AssertionError("excluded prefixes must not be probed")

    matches, errors, complete = A.find_run_sources_across_buckets(
        ["bucket"],
        base_prefix="",
        run_id="run-known",
        exact_prefix="npa-agent/tenants",
        s3=_S3(),
    )

    assert matches == []
    assert errors == ()
    assert complete is True


def test_multi_bucket_discovery_keeps_accessible_siblings_when_one_is_denied(
    monkeypatch,
) -> None:
    import npa.workflows.artifacts as A

    def fake_light(bucket, *, base_prefix, limit, exclude, contains, s3):
        if bucket == "denied-bucket":
            raise A.ArtifactDiscoveryError("access denied")
        return A.RunListPage(
            runs=[
                A.RunSummary(
                    f"real-run-{bucket}", "2030-01-01T00:00:00+00:00", 0, False
                )
            ],
            truncated=False,
            total_runs=1,
            limit=limit,
        )

    monkeypatch.setattr(A, "list_all_run_prefixes", fake_light)
    page = A.list_all_runs_across_buckets(
        ["accessible-a", "denied-bucket", "accessible-b"],
        base_prefix="",
        limit=50,
        bucket_projects={"accessible-a": "project-a", "accessible-b": "project-b"},
        lightweight=True,
        s3=object(),
    )

    assert {(item.bucket, item.project_id, item.run_id) for item in page.runs} == {
        ("accessible-a", "project-a", "real-run-accessible-a"),
        ("accessible-b", "project-b", "real-run-accessible-b"),
    }


def test_build_fiftyone_dataset_emits_bucket_qualified_uris() -> None:
    run = "paidf-demo"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/cosmos_augmented/aug-{run}-0/augmented_video.mp4",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {
            "variables": {"cloth_color": "green"}
        }
    }
    ds = build_fiftyone_dataset(
        keys, run_id=run, read_json=lambda k: payloads.get(k), bucket="lerobot-d87cf691"
    )
    aug = [s for s in ds["samples"] if s["group"] == "augmented"][0]
    assert aug["thumbnail_uri"].startswith("s3://lerobot-d87cf691/")
    assert aug["thumbnail_uri"].endswith("frame-00000.png")
    assert aug["video_uri"].endswith("augmented_video.mp4")


def test_dataset_exposes_seeded_source_and_derived_conditioning_clip() -> None:
    run = "paidf-seeded"
    base = f"physical-ai-data-factory/{run}"
    keys = [
        f"{base}/configs/manifest.json",
        f"{base}/input/frame_0000.png",
        f"{base}/input/conditioning.mp4",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/configs/manifest.json": {
            "input_source": {
                "kind": "npa_seeded_fixture",
                "uri": f"s3://bucket/{base}/input/",
                "frame_count": 8,
                "description": "NPA-generated seeded fixture used as this run's input",
            }
        },
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {
            "variables": {"lighting": "warm"}
        },
        f"{base}/curation/report.json": {"curation_engine": "report-only"},
    }

    dataset = build_fiftyone_dataset(
        keys,
        run_id=run,
        read_json=lambda key: payloads.get(key),
        bucket="bucket",
    )

    assert dataset["source"]["kind"] == "npa_seeded_fixture"
    assert dataset["samples"][0]["data_role"] == "synthetic_fixture"
    assert dataset["samples"][1]["data_role"] == "derived_conditioning"
    assert dataset["summary"]["conditioning_count"] == 1
    assert dataset["summary"]["source_input_count"] == 1
    assert dataset["summary"]["original_input_count"] == 0
    assert dataset["summary"]["fixture_count"] == 1
    assert dataset["review"]["label"] == "Artifact summary only — FiftyOne did not run"


def test_dataset_preserves_real_source_conditioning_variant_lineage_and_labels() -> (
    None
):
    run = "paidf-real"
    base = f"physical-ai-data-factory/{run}"
    keys = [
        f"{base}/input/provenance.json",
        f"{base}/input/source.mp4",
        f"{base}/input/conditioning.mp4",
        f"{base}/input/conditioning-frame-0001.png",
        f"{base}/cosmos_augmented/aug-{run}/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}/metadata.json",
        f"{base}/labeled_augmented/captions.json",
        f"{base}/curation/report.json",
    ]
    source = {
        "schema_version": "npa.paidf.input-provenance.v1",
        "source_kind": "upstream_sample",
        "input_origin": "actual_capture",
        "input_origin_label": "Upstream real sample",
        "authoritative_upstream_url": "https://official.example/dataset",
        "immutable_revision": "a" * 40,
        "asset_license": "CC-BY-4.0",
        "asset_attribution": "Example author",
        "sha256": "b" * 64,
        "staged_canonical_s3_uri": f"s3://bucket/{base}/input/",
        "cosmos_conditioning": {
            "enabled": True,
            "staged_uri": f"s3://bucket/{base}/input/conditioning.mp4",
        },
        "derivation": {"kind": "normalized_conditioning_clip"},
    }
    payloads = {
        f"{base}/input/provenance.json": source,
        f"{base}/cosmos_augmented/aug-{run}/metadata.json": {
            "variables": {
                "lighting": "warm lamp light",
                "color_grade": "warm",
                "prompt": "appearance only",
            }
        },
        f"{base}/labeled_augmented/captions.json": {
            "captions": [
                {
                    "image": f"aug-{run}/frame-00000.png",
                    "caption": "robot variant under warm light",
                }
            ]
        },
        f"{base}/curation/report.json": {
            "curation_engine": "fiftyone-brain",
            "fiftyone": {
                "samples": {
                    f"aug-{run}": {
                        "uniqueness": 0.8,
                        "kept": True,
                        "redundant": False,
                    }
                }
            },
        },
    }

    dataset = build_fiftyone_dataset(
        keys,
        run_id=run,
        read_json=lambda key: payloads.get(key),
        bucket="bucket",
    )

    assert [sample["group"] for sample in dataset["samples"]] == [
        "source",
        "conditioning",
        "conditioning",
        "augmented",
    ]
    assert dataset["source"] == source
    assert dataset["samples"][0]["data_role_label"] == "Upstream real sample"
    augmented = dataset["samples"][-1]
    assert augmented["lineage"]["source_sha256"] == "b" * 64
    assert augmented["lineage"]["conditioning_uri"].endswith("conditioning.mp4")
    assert augmented["tags"] == {
        "lighting": "warm lamp light",
        "color_grade": "warm",
    }
    assert augmented["caption"] == "robot variant under warm light"
    assert augmented["uniqueness"] == 0.8
    assert augmented["curated"] is True


def test_artifact_roles_and_run_relative_resolution_are_explicit() -> None:
    run = "paidf-full-id"
    base = f"physical-ai-data-factory/{run}"
    original = Artifact(
        run,
        f"{base}/input/frame_0000.png",
        f"s3://bucket/{base}/input/frame_0000.png",
        10,
        "",
        "image",
        True,
    )
    augmented = Artifact(
        run,
        f"{base}/cosmos_augmented/aug-0/frame-00000.png",
        f"s3://bucket/{base}/cosmos_augmented/aug-0/frame-00000.png",
        10,
        "",
        "image",
        True,
    )
    report = Artifact(
        run,
        f"{base}/reports/sim2real.rrd",
        f"s3://bucket/{base}/reports/sim2real.rrd",
        10,
        "",
        "rerun",
        False,
    )

    assert original.to_dict()["data_role"] == "source_input"
    assert augmented.to_dict()["data_role"] == "synthetic_augmented"
    assert artifact_data_role(report.key, run)["role"] == "pipeline_metadata"
    assert (
        resolve_run_artifact(
            [original, augmented, report],
            run_id=run,
            requested_key="reports/sim2real.rrd",
        )
        is report
    )
    assert infer_run_id_from_artifact_key(report.key) == run
