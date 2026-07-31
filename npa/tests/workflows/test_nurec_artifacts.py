"""A NuRec run must be discoverable and viewable through the agent's artifact API.

The agent's run picker imposes concrete structural rules
(``npa.workflows.artifacts``); these tests pin the NuRec S3 layout and run-id
scheme against them so a future layout change cannot silently make a run
invisible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from npa.workflows.artifacts import (
    list_artifacts,
    list_runs,
    render_hint_for_object,
    select_preferred_artifact,
)

BUCKET = "example-bucket"
CATEGORY = "checkpoints/neural-reconstruction"
RUN_ID = "neural-reconstruction-struktur28-20260731t170500z"
#: The workflow writes exactly this tree.
RUN_KEYS = (
    "ncore/manifest.json",
    "input/camera_images/camera2/000000.jpg",
    "input/camera_images/camera2/000001.jpg",
    "reconstruction/last.usdz",
    "reconstruction/parsed.yaml",
    "reconstruction/metrics.yaml",
    "reconstruction/val/camera2.mp4",
    "novel_views/camera2/000000.png",
    "novel_views/camera2.mp4",
    "reports/final.json",
    "reports/sim2real.rrd",
)


class _FakePaginator:
    def __init__(self, keys: list[tuple[str, str]]) -> None:
        self._keys = keys

    def paginate(self, Bucket=None, Prefix="", Delimiter=None):  # noqa: N803
        if Delimiter:
            prefixes: list[dict] = []
            seen: set[str] = set()
            for key, _ts in self._keys:
                if not key.startswith(Prefix):
                    continue
                remainder = key[len(Prefix) :]
                head = remainder.split(Delimiter, 1)[0]
                if Delimiter in remainder and head not in seen:
                    seen.add(head)
                    prefixes.append({"Prefix": f"{Prefix}{head}{Delimiter}"})
            yield {"CommonPrefixes": prefixes}
            return
        contents = [
            {
                "Key": key,
                "Size": 1024,
                "LastModified": datetime.fromisoformat(ts).astimezone(timezone.utc),
            }
            for key, ts in self._keys
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}


class _FakeS3:
    def __init__(self, keys: list[tuple[str, str]]) -> None:
        self._keys = keys

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self._keys)


def _run_objects(
    run_id: str = RUN_ID, written_at: str = "2026-07-31T17:12:00+00:00"
) -> list[tuple[str, str]]:
    return [(f"{CATEGORY}/{run_id}/{relative}", written_at) for relative in RUN_KEYS]


def test_run_is_listed_as_viewable() -> None:
    s3 = _FakeS3(_run_objects())

    page = list_runs(BUCKET, prefix=CATEGORY, s3=s3)

    assert [run.run_id for run in page.runs] == [RUN_ID]
    run = page.runs[0]
    assert run.artifact_count == len(RUN_KEYS)
    # Without has_viewable the agent hides the run from the picker.
    assert run.has_viewable is True


def test_run_is_dated_by_its_start_timestamp_not_the_newest_write() -> None:
    # The run id embeds submit time; artifacts land minutes later.
    s3 = _FakeS3(_run_objects(written_at="2026-07-31T17:12:00+00:00"))

    run = list_runs(BUCKET, prefix=CATEGORY, s3=s3).runs[0]

    assert run.started_at == "2026-07-31T17:05:00+00:00"
    assert run.last_modified == "2026-07-31T17:12:00+00:00"


def test_run_id_scheme_survives_the_start_time_window_guard() -> None:
    # _run_started_at only trusts an id-encoded time within 3 days of the first
    # write, so a stale-looking id must fall back rather than mis-date the run.
    s3 = _FakeS3(_run_objects(written_at="2026-08-20T00:00:00+00:00"))

    run = list_runs(BUCKET, prefix=CATEGORY, s3=s3).runs[0]

    assert run.started_at == "2026-08-20T00:00:00+00:00"


def test_preferred_artifact_is_the_rerun_recording() -> None:
    s3 = _FakeS3(_run_objects())

    artifacts = list_artifacts(BUCKET, RUN_ID, prefix=CATEGORY, s3=s3)
    preferred = select_preferred_artifact(artifacts)

    assert preferred is not None
    assert preferred.key.endswith("/reports/sim2real.rrd")
    assert preferred.render == "rerun"
    assert preferred.inline is True


def test_run_is_a_directory_not_a_bare_file() -> None:
    # list_runs skips bare files sitting directly under a category prefix.
    s3 = _FakeS3([(f"{CATEGORY}/records.json", "2026-07-31T17:00:00+00:00")])

    assert list_runs(BUCKET, prefix=CATEGORY, s3=s3).runs == []


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (f"{CATEGORY}/{RUN_ID}/reports/sim2real.rrd", "rerun"),
        (f"{CATEGORY}/{RUN_ID}/novel_views/camera2/000000.png", "image"),
        (f"{CATEGORY}/{RUN_ID}/input/camera_images/camera2/000000.jpg", "image"),
        (f"{CATEGORY}/{RUN_ID}/novel_views/camera2.mp4", "video"),
        (f"{CATEGORY}/{RUN_ID}/reports/final.json", "json"),
        (f"{CATEGORY}/{RUN_ID}/reconstruction/metrics.yaml", "text"),
    ],
)
def test_every_advertised_artifact_type_renders_inline(key: str, expected: str) -> None:
    assert render_hint_for_object(key=key) == expected


def test_usdz_is_offered_as_a_download_not_a_broken_inline_preview() -> None:
    # No browser renders USDZ and the agent has no USDZ viewer, so "download" is
    # the honest classification. Viewability comes from the .rrd/.png/.mp4/.json.
    assert render_hint_for_object(key=f"{CATEGORY}/{RUN_ID}/reconstruction/last.usdz") == "download"


@pytest.mark.parametrize(
    "name",
    ["last.usdz", "scene.usd", "scene.usda", "scene.usdc", "gaussians.ply", "mesh.obj"],
)
def test_every_3d_asset_extension_is_explicitly_download_only(name: str) -> None:
    """Pinned explicitly, not left to the mimetypes fallback.

    If Python ever learns a `model/...` type for these, an implicit fallthrough
    could start classifying a reconstruction as something the browser is asked to
    render inline, producing a broken pane.
    """
    from npa.workflows.artifacts import is_model_artifact

    assert is_model_artifact(name) is True
    assert render_hint_for_object(key=f"{CATEGORY}/{RUN_ID}/reconstruction/{name}") == "download"


def test_model_extensions_do_not_swallow_a_viewable_artifact() -> None:
    from npa.workflows.artifacts import is_model_artifact

    for name in ("sim2real.rrd", "frame.png", "clip.mp4", "final.json", "metrics.yaml"):
        assert is_model_artifact(name) is False, name


def test_a_run_of_only_3d_assets_would_not_be_viewable() -> None:
    """Guards the reason the run also publishes renders: a USDZ-only run is not
    viewable, so the workflow must keep emitting the .rrd, PNGs and MP4."""
    s3 = _FakeS3(
        [
            (f"{CATEGORY}/usdz-only-run/reconstruction/last.usdz", "2026-07-31T17:00:00+00:00"),
        ]
    )

    runs = list_runs(BUCKET, prefix=CATEGORY, s3=s3).runs

    assert [run.run_id for run in runs] == ["usdz-only-run"]
    assert runs[0].has_viewable is False


def test_run_id_is_a_single_safe_segment() -> None:
    from npa.workflows.rerun_serve import validate_run_id

    assert validate_run_id(RUN_ID) == RUN_ID
    assert "/" not in RUN_ID


def test_run_id_is_not_rejected_as_a_placeholder() -> None:
    from npa.workflows.artifacts import _parse_run_id_timestamps

    # One unambiguous timestamp, parsed to the exact submit instant.
    assert _parse_run_id_timestamps(RUN_ID) == ["2026-07-31T17:05:00+00:00"]


def test_local_run_directory_matches_the_documented_layout(tmp_path: Path) -> None:
    """The stage prefixes the workflow writes are the ones the viz module reads."""
    from npa.workflows.data_factory_viz import RUN_SUBDIRS

    stages = {key.split("/", 1)[0] for key in RUN_KEYS}

    assert stages <= set(RUN_SUBDIRS)
