"""Unit tests for the NVIDIA Cosmos Curator workbench tool.

The upstream stages themselves are exercised by the image's golden eval (a real
curation run); these tests cover everything around them — availability probing,
upstream's documented pipeline argv, the output-tree ingest, variant staging, and
the report assembly — without needing the upstream checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from npa.workbench.cosmos_curate import (
    CosmosCurateError,
    CuratorAvailability,
    discover_videos,
    ingest_output,
    result_uri_for,
    split_pipeline_argv,
)
from npa.workbench.cosmos_curate import report as report_mod
from npa.workbench.cosmos_curate import upstream as upstream_mod

# One real per-clip metadata document from an upstream ClipWriterStage run
# (npa workbench cosmos-curate curate-videos over a 1280x704 clip), trimmed to the
# fields the ingest reads.
UPSTREAM_META = {
    "span_uuid": "0eefc8e4-2d58-5113-9fba-b68deda2583e",
    "source_video": "/tmp/staged/clip-a.mp4",
    "duration_span": [0.0, 3.0],
    "width_source": 1280,
    "height_source": 704,
    "framerate_source": 24.0,
    "clip_location": "/tmp/out/clips/0eefc8e4-2d58-5113-9fba-b68deda2583e.mp4",
    "width": 1280,
    "height": 704,
    "framerate": 24.0,
    "num_frames": 72,
    "video_codec": "h264",
    "num_bytes": 35265,
    "motion_score": {"global_mean": 0.00031337200198322535, "per_patch_min_256": 0.0},
    "windows": [{"start_frame": 0, "end_frame": 72, "qwen_caption": "A test pattern in motion."}],
    "valid": False,
}


def _write_curator_output(root: Path, *, clips: int = 2) -> None:
    (root / "clips").mkdir(parents=True)
    (root / "metas" / "v0").mkdir(parents=True)
    (root / "processed_videos").mkdir(parents=True)
    for index in range(clips):
        meta = dict(UPSTREAM_META)
        meta["span_uuid"] = f"clip-uuid-{index}"
        meta["duration_span"] = [float(index * 3), float(index * 3 + 3)]
        (root / "clips" / f"clip-uuid-{index}.mp4").write_bytes(b"fake mp4")
        (root / "metas" / "v0" / f"clip-uuid-{index}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
    (root / "processed_videos" / "clip-a.mp4.json").write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def test_availability_without_a_checkout_names_the_env_var() -> None:
    availability = CuratorAvailability()
    assert not availability.can_run_in_process
    assert "NPA_COSMOS_CURATE_SRC" in availability.reason()


def test_availability_without_a_usable_encoder_explains_the_encoder_requirement() -> None:
    availability = CuratorAvailability(
        source="/opt/cosmos-curate", importable=True, ffmpeg="/usr/bin/ffmpeg", encoders=()
    )
    assert not availability.can_run_in_process
    assert "libopenh264" in availability.reason() and "h264_nvenc" in availability.reason()


def test_availability_names_the_python_version_gap() -> None:
    """Upstream needs >=3.12; an older interpreter otherwise fails deep in an import."""

    availability = CuratorAvailability(
        source="/opt/cosmos-curate",
        importable=False,
        import_error="ImportError: cannot import name 'Self' from 'typing'",
        ffmpeg="/usr/bin/ffmpeg",
        encoders=("libopenh264",),
        python_version="3.10.12",
    )
    assert not availability.can_run_in_process
    reason = availability.reason()
    assert "Python >= 3.12" in reason and "3.10.12" in reason


def test_availability_accepts_a_new_enough_python() -> None:
    availability = CuratorAvailability(
        source="/opt/cosmos-curate",
        importable=True,
        ffmpeg="/usr/bin/ffmpeg",
        encoders=("libopenh264",),
        python_version="3.12.8",
    )
    assert availability.python_ok
    assert availability.can_run_in_process


def test_availability_surfaces_an_import_failure() -> None:
    availability = CuratorAvailability(
        source="/opt/cosmos-curate",
        importable=False,
        import_error="ModuleNotFoundError: cosmos_xenna",
        ffmpeg="/usr/bin/ffmpeg",
        encoders=("libopenh264",),
    )
    assert "cosmos_xenna" in availability.reason()


def test_availability_prefers_nvenc_only_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    availability = CuratorAvailability(
        source="/opt/cosmos-curate",
        importable=True,
        ffmpeg="/usr/bin/ffmpeg",
        encoders=("libopenh264", "h264_nvenc"),
    )
    monkeypatch.setattr(upstream_mod, "_has_gpu", lambda: False)
    assert availability.encoder == "libopenh264"
    monkeypatch.setattr(upstream_mod, "_has_gpu", lambda: True)
    assert availability.encoder == "h264_nvenc"
    assert availability.can_run_in_process
    assert availability.reason() == ""


def test_upstream_source_dir_requires_the_pipelines_package(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-checkout"
    empty.mkdir()
    assert upstream_mod.upstream_source_dir(environ={"NPA_COSMOS_CURATE_SRC": str(empty)}) is None

    checkout = tmp_path / "checkout"
    (checkout / "cosmos_curator" / "pipelines").mkdir(parents=True)
    assert upstream_mod.upstream_source_dir(
        environ={"NPA_COSMOS_CURATE_SRC": str(checkout)}
    ) == checkout


# ---------------------------------------------------------------------------
# Upstream's documented container command
# ---------------------------------------------------------------------------


def test_split_pipeline_argv_matches_upstreams_fixed_stride_invocation() -> None:
    argv = split_pipeline_argv(
        input_video_path="s3://bucket/run/cosmos_augmented/",
        output_clip_path="s3://bucket/run/curation/cosmos_curator/",
        fixed_stride_split_duration=3,
        fixed_stride_min_clip_length_s=1.0,
    )
    assert argv[:2] == ["video-pipeline", "split"]
    assert "--input-video-path" in argv and "--output-clip-path" in argv
    assert argv[argv.index("--splitting-algorithm") + 1] == "fixed-stride"
    assert argv[argv.index("--fixed-stride-split-duration") + 1] == "3"
    # Embeddings are a GPU stage; they stay off unless asked for.
    assert "--no-generate-embeddings" in argv
    assert "--embedding-algorithm" not in argv


def test_split_pipeline_argv_keeps_embeddings_when_requested() -> None:
    argv = split_pipeline_argv(
        input_video_path="/in",
        output_clip_path="/out",
        generate_embeddings=True,
        embedding_algorithm="openai",
        captioning_algorithm="qwen",
        limit=1,
    )
    assert "--no-generate-embeddings" not in argv
    assert argv[argv.index("--embedding-algorithm") + 1] == "openai"
    assert argv[argv.index("--captioning-algorithm") + 1] == "qwen"
    assert argv[argv.index("--limit") + 1] == "1"


def test_split_pipeline_argv_omits_stride_flags_for_transnetv2() -> None:
    argv = split_pipeline_argv(
        input_video_path="/in", output_clip_path="/out", splitting_algorithm="transnetv2"
    )
    assert "--fixed-stride-split-duration" not in argv


def test_split_pipeline_argv_rejects_an_unknown_algorithm() -> None:
    with pytest.raises(CosmosCurateError, match="fixed-stride or transnetv2"):
        split_pipeline_argv(input_video_path="/in", output_clip_path="/out", splitting_algorithm="magic")


def test_split_pipeline_argv_requires_both_paths() -> None:
    with pytest.raises(CosmosCurateError, match="input and output paths"):
        split_pipeline_argv(input_video_path="", output_clip_path="/out")


# ---------------------------------------------------------------------------
# Reading upstream's output tree
# ---------------------------------------------------------------------------


def test_ingest_output_reads_upstream_clip_metadata(tmp_path: Path) -> None:
    _write_curator_output(tmp_path, clips=2)
    ingested = ingest_output(tmp_path)

    assert len(ingested["clips"]) == 2
    assert len(ingested["clip_files"]) == 2
    assert ingested["processed_videos"] == ["clip-a.mp4.json"]

    first = ingested["clips"][0]
    assert first.clip_id == "clip-uuid-0"
    assert first.duration_s == 3.0
    assert (first.width, first.height) == (1280, 704)
    assert first.num_frames == 72
    assert first.motion_score_global_mean == pytest.approx(0.000313372, rel=1e-6)
    assert first.caption == "A test pattern in motion."


def test_ingest_output_is_empty_for_a_missing_tree(tmp_path: Path) -> None:
    ingested = ingest_output(tmp_path / "nothing-here")
    assert ingested == {"clips": [], "clip_files": [], "processed_videos": []}


def test_ingest_output_skips_unreadable_metadata(tmp_path: Path) -> None:
    _write_curator_output(tmp_path, clips=1)
    (tmp_path / "metas" / "v0" / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(ingest_output(tmp_path)["clips"]) == 1


def test_discover_videos_finds_nested_clips(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.mp4").write_bytes(b"x")
    (tmp_path / "two.MOV").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")
    assert [path.name for path in discover_videos(tmp_path)] == ["one.mp4", "two.MOV"]


# ---------------------------------------------------------------------------
# Variant staging and report assembly
# ---------------------------------------------------------------------------


def test_stage_variants_names_each_download_after_its_variant(tmp_path: Path) -> None:
    augment = tmp_path / "cosmos_augmented"
    for clip in ("clip-a", "clip b"):
        (augment / clip).mkdir(parents=True)
        (augment / clip / "augmented_video.mp4").write_bytes(b"fake mp4")
    (augment / "no-video").mkdir()

    staged = tmp_path / "staged"
    warnings: list[str] = []
    variants = report_mod._stage_variants(
        str(augment), staged, store=object(), max_variants=0, warnings=warnings
    )
    assert set(variants.values()) == {"clip-a", "clip b"}
    assert sorted(path.name for path in staged.iterdir()) == ["clip-a.mp4", "clip_b.mp4"]
    assert any("no-video" in warning for warning in warnings)


def test_stage_variants_honors_max_variants(tmp_path: Path) -> None:
    augment = tmp_path / "cosmos_augmented"
    for clip in ("clip-a", "clip-b", "clip-c"):
        (augment / clip).mkdir(parents=True)
        (augment / clip / "augmented_video.mp4").write_bytes(b"fake mp4")
    variants = report_mod._stage_variants(
        str(augment), tmp_path / "staged", store=object(), max_variants=2, warnings=[]
    )
    assert len(variants) == 2


def test_curate_augmented_reports_unavailable_without_the_curator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report_mod, "probe_availability", lambda: CuratorAvailability())
    report = report_mod.curate_augmented(
        augment_uri=str(tmp_path / "cosmos_augmented"),
        curated_uri=str(tmp_path / "curated"),
        storage=object(),
    )
    assert report.status == "skipped"
    assert report.engine == "unavailable"
    assert report.clip_count == 0
    assert any("NPA_COSMOS_CURATE_SRC" in warning for warning in report.warnings)
    assert report.to_dict()["schema"] == "npa.cosmos_curate.curation.v1"


def test_curate_augmented_can_require_the_curator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workbench.cosmos_curate import CosmosCurateUnavailable

    monkeypatch.setattr(report_mod, "probe_availability", lambda: CuratorAvailability())
    with pytest.raises(CosmosCurateUnavailable):
        report_mod.curate_augmented(
            augment_uri=str(tmp_path / "cosmos_augmented"),
            curated_uri=str(tmp_path / "curated"),
            require_curator=True,
            storage=object(),
        )


def test_curate_augmented_summarizes_a_real_curator_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report assembly against a stubbed curator run over real-shaped output."""

    from npa.workbench.cosmos_curate.pipeline import CuratorRunResult

    augment = tmp_path / "cosmos_augmented"
    for clip in ("variant-0", "variant-1"):
        (augment / clip).mkdir(parents=True)
        (augment / clip / "augmented_video.mp4").write_bytes(b"fake mp4")

    monkeypatch.setattr(
        report_mod,
        "probe_availability",
        lambda: CuratorAvailability(
            source="/opt/cosmos-curate",
            importable=True,
            ffmpeg="/usr/bin/ffmpeg",
            encoders=("libopenh264",),
        ),
    )

    def fake_curate_videos(*, input_dir: Any, output_dir: Any, **kwargs: Any) -> CuratorRunResult:
        out = Path(output_dir)
        (out / "clips").mkdir(parents=True)
        (out / "metas" / "v0").mkdir(parents=True)
        staged = sorted(Path(input_dir).glob("*.mp4"))
        assert [path.stem for path in staged] == ["variant-0", "variant-1"]
        for index, source in enumerate(staged):
            meta = dict(UPSTREAM_META)
            meta["span_uuid"] = f"clip-{index}"
            meta["source_video"] = str(source)
            (out / "clips" / f"clip-{index}.mp4").write_bytes(b"fake mp4")
            (out / "metas" / "v0" / f"clip-{index}.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
        return CuratorRunResult(
            status="completed",
            engine="cosmos-curator-stages",
            source="/opt/cosmos-curate",
            encoder="libopenh264",
            input_dir=str(input_dir),
            output_dir=str(out),
            input_videos=len(staged),
            clips_written=len(staged),
            clips_filtered=1,
            motion_filter="score-only",
        )

    monkeypatch.setattr(report_mod, "curate_videos", fake_curate_videos)

    curated = tmp_path / "curated"
    report = report_mod.curate_augmented(
        augment_uri=str(augment),
        curated_uri=str(curated),
        report_uri=str(tmp_path / "report.json"),
        storage=object(),
    )
    assert report.status == "completed"
    assert report.engine == "cosmos-curator-stages"
    assert report.variant_count == 2
    assert report.clip_count == 2
    assert report.filtered_count == 1
    assert report.total_duration_s == 6.0
    # Each clip is attributed to the variant it was cut from, not the staging path.
    assert report.per_variant == {"variant-0": 1, "variant-1": 1}
    # The curator's own output tree is published under curated_uri.
    assert (curated / "clips" / "clip-0.mp4").is_file()
    assert (curated / "metas" / "v0" / "clip-0.json").is_file()


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "GPU 0: NVIDIA RTX PRO 6000 (UUID: GPU-abc)\n", True),
        (0, "", False),  # driver answers, no device allocated to this pod
        (9, "", False),  # nvidia-smi present but cannot talk to a driver
    ],
    ids=["device-present", "no-device", "driver-error"],
)
def test_the_gpu_encoder_is_only_chosen_when_a_device_is_really_there(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, expected: bool
) -> None:
    """`nvidia-smi` on PATH does not mean this pod has a GPU.

    A CPU-tier pod on a GPU node has the binary and no device, and picking
    h264_nvenc there fails every encode with CUDA_ERROR_NO_DEVICE — which upstream
    logs per clip instead of raising, so the run reports success having dropped work.
    """

    import subprocess

    from npa.workbench.cosmos_curate import upstream as up

    up._has_gpu.cache_clear()
    monkeypatch.setattr(up.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        up.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], returncode, stdout=stdout, stderr=""),
    )
    try:
        assert up._has_gpu() is expected
    finally:
        up._has_gpu.cache_clear()


def test_variants_whose_names_sanitize_alike_are_staged_separately(tmp_path: Path) -> None:
    """Every unsafe character maps to ``_``, so distinct variants can collide.

    Staged under one name the second download overwrites the first, and its clips are
    then attributed to the wrong variant.
    """

    augment = tmp_path / "cosmos_augmented"
    for clip in ("cam a", "cam+a", "cam_a"):
        (augment / clip).mkdir(parents=True)
        (augment / clip / "augmented_video.mp4").write_bytes(clip.encode())

    staged = tmp_path / "staged"
    staged.mkdir()
    variants = report_mod._stage_variants(
        str(augment), staged, store=object(), max_variants=0, warnings=[]
    )

    assert sorted(variants.values()) == ["cam a", "cam+a", "cam_a"]
    assert len(list(staged.glob("*.mp4"))) == 3
    # Each staged file still holds its own variant's bytes.
    for stem, variant in variants.items():
        assert (staged / f"{stem}.mp4").read_bytes() == variant.encode()


def _weights_env(root: Path) -> dict[str, str]:
    return {"NPA_COSMOS_CURATE_WEIGHTS_DIR": str(root)}


def test_an_interrupted_download_is_not_mistaken_for_a_finished_one(tmp_path: Path) -> None:
    """Files on disk are not evidence of a complete fetch.

    Registry entries carry no file list, so "any file is here" accepted a download
    killed part-way; the next run skipped it and the missing shard surfaced much later
    as a load error inside a stage.
    """

    from npa.workbench.cosmos_curate.models import ModelSpec, model_status

    spec = ModelSpec(key="motion", model_id="org/model")
    partial = tmp_path / "org" / "model"
    partial.mkdir(parents=True)
    (partial / "model-00001-of-00002.safetensors").write_bytes(b"half a download")

    status = model_status([spec], environ=_weights_env(tmp_path))[0]
    assert status.present is False
    assert "no completion stamp" in status.stale_reason


def test_weights_from_another_revision_do_not_satisfy_a_pinned_request(tmp_path: Path) -> None:
    """The pin is only meaningful if a cache at a different commit is re-fetched."""

    from npa.workbench.cosmos_curate.models import (
        ModelSpec,
        model_status,
        write_completion_stamp,
    )

    local = tmp_path / "org" / "model"
    local.mkdir(parents=True)
    (local / "weights.bin").write_bytes(b"weights")
    write_completion_stamp(local, ModelSpec(key="motion", model_id="org/model", revision="old"))

    wanted = ModelSpec(key="motion", model_id="org/model", revision="new")
    status = model_status([wanted], environ=_weights_env(tmp_path))[0]
    assert status.present is False
    assert "wanted new" in status.stale_reason


def test_a_stamped_download_at_the_pinned_revision_is_reused(tmp_path: Path) -> None:
    from npa.workbench.cosmos_curate.models import (
        ModelSpec,
        model_status,
        write_completion_stamp,
    )

    spec = ModelSpec(key="motion", model_id="org/model", revision="abc123")
    local = tmp_path / "org" / "model"
    local.mkdir(parents=True)
    (local / "weights.bin").write_bytes(b"weights")
    write_completion_stamp(local, spec)

    status = model_status([spec], environ=_weights_env(tmp_path))[0]
    assert status.present is True
    assert status.stale_reason == ""
    # The stamp is bookkeeping, not a weight file.
    assert status.file_count == 1


def test_a_cache_predating_stamps_is_kept_when_the_registry_vouches_for_it(
    tmp_path: Path,
) -> None:
    """Do not re-download gigabytes just because the stamp convention is new."""

    from npa.workbench.cosmos_curate.models import ModelSpec, model_status

    spec = ModelSpec(key="motion", model_id="org/model", files=("a.bin", "b.bin"))
    local = tmp_path / "org" / "model"
    local.mkdir(parents=True)
    (local / "a.bin").write_bytes(b"a")
    (local / "b.bin").write_bytes(b"b")

    assert model_status([spec], environ=_weights_env(tmp_path))[0].present is True


def test_a_checkout_at_another_commit_is_refused(tmp_path: Path) -> None:
    """Upstream's constructor kwargs are not an API, so the commit has to match.

    Without this the mismatch imports fine and only shows up later as a TypeError
    from inside an upstream constructor, which reads like a bug in our call.
    """

    from npa.workbench.cosmos_curate.upstream import REVISION_STAMP_FILE, probe_availability

    checkout = tmp_path / "cosmos-curator"
    (checkout / "cosmos_curator" / "pipelines").mkdir(parents=True)
    (checkout / REVISION_STAMP_FILE).write_text("0" * 40, encoding="utf-8")

    availability = probe_availability(environ={"NPA_COSMOS_CURATE_SRC": str(checkout)})
    assert availability.revision == "0" * 40
    assert availability.revision_ok is False
    assert availability.can_run_in_process is False


def test_a_wrong_commit_says_which_commit_and_which_file_to_revisit() -> None:
    """The reason has to be actionable, and has to say so on every interpreter.

    Asserted against a constructed availability rather than a probe: on Python
    < 3.12 the probe reports the version gap first, so driving this through the
    probe would only exercise the message on some interpreters.
    """

    from npa.workbench.cosmos_curate.upstream import PINNED_REVISION, CuratorAvailability

    reason = CuratorAvailability(
        source="/opt/cosmos-curator",
        revision="0" * 40,
        python_version="3.12.3",
        importable=True,
        ffmpeg="/usr/bin/ffmpeg",
        encoders=("libopenh264",),
    ).reason()
    assert PINNED_REVISION in reason
    assert "0" * 40 in reason
    assert "pipeline.py" in reason


def test_a_checkout_without_a_stamp_is_not_refused(tmp_path: Path) -> None:
    """Working from your own clone is supported; only a stated wrong commit is not."""

    from npa.workbench.cosmos_curate.upstream import probe_availability

    checkout = tmp_path / "cosmos-curator"
    (checkout / "cosmos_curator" / "pipelines").mkdir(parents=True)

    availability = probe_availability(environ={"NPA_COSMOS_CURATE_SRC": str(checkout)})
    assert availability.revision == ""
    assert availability.revision_ok is True
    assert "checked out at" not in availability.reason()


def test_the_image_pin_and_the_code_pin_cannot_drift(tmp_path: Path) -> None:
    """The Dockerfile checks out one commit; this code is written against another.

    They are the same fact in two places, so assert it rather than trusting a
    reviewer to notice when one moves.
    """

    import re

    from npa.workbench.cosmos_curate.upstream import PINNED_REVISION

    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "cosmos-curate"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    match = re.search(r"^ARG COSMOS_CURATE_REF=(\S+)", dockerfile, re.MULTILINE)
    assert match, "the Dockerfile no longer pins COSMOS_CURATE_REF"
    assert match.group(1) == PINNED_REVISION, (
        "cosmos-curate/Dockerfile checks out a different commit than "
        "cosmos_curate/upstream.py drives its stages at"
    )


def test_a_moved_upstream_signature_is_reported_as_unavailable() -> None:
    """A stage whose kwargs moved must surface as "cannot run here", not a traceback."""

    from npa.workbench.cosmos_curate.pipeline import _construct

    class MovedOn:
        def __init__(self, *, brand_new_name: str) -> None:
            self.brand_new_name = brand_new_name

    with pytest.raises(CosmosCurateError, match="does not accept the arguments"):
        _construct({"ClipWriterStage": MovedOn}, "ClipWriterStage", output_path="/tmp/x")


def _stub_a_successful_curator_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Stage two variants and a curator that really writes an output tree."""

    from npa.workbench.cosmos_curate.pipeline import CuratorRunResult

    augment = tmp_path / "cosmos_augmented"
    for clip in ("variant-0", "variant-1"):
        (augment / clip).mkdir(parents=True)
        (augment / clip / "augmented_video.mp4").write_bytes(b"fake mp4")

    monkeypatch.setattr(
        report_mod,
        "probe_availability",
        lambda: CuratorAvailability(
            source="/opt/cosmos-curate",
            importable=True,
            ffmpeg="/usr/bin/ffmpeg",
            encoders=("libopenh264",),
        ),
    )

    def fake_curate_videos(*, input_dir: Any, output_dir: Any, **kwargs: Any) -> CuratorRunResult:
        out = Path(output_dir)
        (out / "clips").mkdir(parents=True)
        (out / "metas" / "v0").mkdir(parents=True)
        for index, source in enumerate(sorted(Path(input_dir).glob("*.mp4"))):
            meta = dict(UPSTREAM_META)
            meta["span_uuid"] = f"clip-{index}"
            meta["source_video"] = str(source)
            (out / "clips" / f"clip-{index}.mp4").write_bytes(b"fake mp4")
            (out / "metas" / "v0" / f"clip-{index}.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
        return CuratorRunResult(
            status="completed",
            engine="cosmos-curator-stages",
            source="/opt/cosmos-curate",
            encoder="libopenh264",
            input_dir=str(input_dir),
            output_dir=str(out),
            input_videos=2,
            clips_written=2,
            clips_filtered=0,
            motion_filter="score-only",
        )

    monkeypatch.setattr(report_mod, "curate_videos", fake_curate_videos)
    return augment, tmp_path / "curated"


class _UnwritableStore:
    """Object storage that reads fine but refuses the upload."""

    def download_path(self, uri: str, dest: str) -> str:
        raise AssertionError("the local fixture never downloads")

    def upload_directory(self, local: str, uri: str) -> None:
        raise RuntimeError("AccessDenied")


def test_a_failed_publish_is_reported_degraded_not_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clips that never reached ``curated_uri`` must not be reported as curated.

    The clips live in a temp directory that curate_augmented drops on the way out, so
    a "completed" report with a real clip_count sends the review stage to an empty
    prefix and it silently curates nothing.
    """

    augment, _ = _stub_a_successful_curator_run(tmp_path, monkeypatch)
    report = report_mod.curate_augmented(
        augment_uri=str(augment),
        curated_uri="s3://bucket/curated/",
        report_uri=str(tmp_path / "report.json"),
        storage=_UnwritableStore(),
    )
    assert report.status == "degraded"
    assert report.clip_count == 2  # what it curated is still reported honestly
    assert any("could not publish" in warning for warning in report.warnings), report.warnings


def test_a_failed_publish_raises_when_the_curator_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    augment, _ = _stub_a_successful_curator_run(tmp_path, monkeypatch)
    with pytest.raises(CosmosCurateError, match="could not publish"):
        report_mod.curate_augmented(
            augment_uri=str(augment),
            curated_uri="s3://bucket/curated/",
            report_uri=str(tmp_path / "report.json"),
            storage=_UnwritableStore(),
            require_curator=True,
        )


def test_curate_augmented_requires_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        report_mod,
        "probe_availability",
        lambda: CuratorAvailability(
            source="/opt/cosmos-curate",
            importable=True,
            ffmpeg="/usr/bin/ffmpeg",
            encoders=("libopenh264",),
        ),
    )
    empty = tmp_path / "cosmos_augmented"
    empty.mkdir()
    with pytest.raises(CosmosCurateError, match="no augmented variant videos"):
        report_mod.curate_augmented(
            augment_uri=str(empty), curated_uri=str(tmp_path / "curated"), storage=object()
        )


def test_result_uri_for_appends_the_result_filename() -> None:
    from npa.workbench.cosmos_curate import RESULT_FILENAME

    assert result_uri_for("s3://b/run/curation/") == f"s3://b/run/curation/{RESULT_FILENAME}"
    assert result_uri_for("s3://b/run/curation/custom.json") == "s3://b/run/curation/custom.json"


def test_write_report_round_trips_locally(tmp_path: Path) -> None:
    from npa.workbench.cosmos_curate import write_report

    written = write_report({"clip_count": 3}, result_uri=str(tmp_path / "curation"))
    assert json.loads(Path(written).read_text())["clip_count"] == 3


def test_local_run_needs_no_object_storage_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local --augment-uri must not require an S3 endpoint to be configured."""
    for name in ("AWS_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(report_mod, "probe_availability", lambda: CuratorAvailability())

    report = report_mod.curate_augmented(
        augment_uri=str(tmp_path / "cosmos_augmented"),
        curated_uri=str(tmp_path / "curated"),
    )
    assert report.engine == "unavailable"
