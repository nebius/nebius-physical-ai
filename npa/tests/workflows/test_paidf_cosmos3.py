from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.workflows import paidf_cosmos3 as c3

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None, reason="ffmpeg is required for video fixture tests"
)


def _tiny_video(path: Path, *, color: str = "blue") -> Path:
    assert FFMPEG is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x64:d=2:r=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@requires_ffmpeg
@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_prepare_input_selects_generic_lerobot_v2_and_v3(
    tmp_path: Path, version: str
) -> None:
    dataset = tmp_path / "dataset"
    camera = "observation.images.front"
    info = {
        "codebase_version": version,
        "features": {camera: {"dtype": "video", "shape": [64, 64, 3]}},
    }
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    if version.startswith("v3"):
        source = dataset / "videos" / camera / "chunk-000" / "file-000.mp4"
        episodes = dataset / "meta" / "episodes" / "chunk-000"
        episodes.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "episode_index": 0,
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": 0.0,
                        f"videos/{camera}/to_timestamp": 1.0,
                    }
                ]
            ),
            episodes / "file-000.parquet",
        )
    else:
        source = dataset / "videos" / "chunk-000" / camera / "episode_000000.mp4"
    _tiny_video(source)

    result = c3.prepare_input(
        "lerobot",
        "",
        str(dataset),
        0,
        "front",
        str(tmp_path / "run" / "input"),
        str(tmp_path / "run" / "input" / "provenance.json"),
        "fixture-run",
    )

    assert result["status"] == "prepared"
    assert result["source_kind"] == "lerobot_dataset"
    assert result["camera"] == camera
    assert result["video_bytes"] > 0
    assert (tmp_path / "run" / "input" / "source.mp4").stat().st_size > 0
    assert list((tmp_path / "run" / "input").glob("frame-*.png"))


def test_prepare_input_rejects_ambiguous_or_missing_camera(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": {
                    "observation.images.front": {"dtype": "video"},
                    "observation.images.wrist": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(c3.PaidfCosmos3Error, match="camera selector"):
        c3.prepare_input(
            "lerobot",
            "",
            str(dataset),
            0,
            "",
            str(tmp_path / "out"),
            str(tmp_path / "p.json"),
        )


class _MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.s3 = SimpleNamespace(list_objects_v2=self._list)

    def upload_file(self, source: str, uri: str) -> str:
        self.objects[uri] = Path(source).read_bytes()
        return uri

    def _list(self, *, Bucket: str, Prefix: str, **_kwargs):
        stem = f"s3://{Bucket}/"
        return {
            "Contents": [
                {"Key": uri.removeprefix(stem)}
                for uri in self.objects
                if uri.startswith(stem + Prefix)
            ],
            "IsTruncated": False,
        }


def _generation_inputs(tmp_path: Path) -> dict[str, Path]:
    source = _tiny_video(tmp_path / "source.mp4")
    configs = tmp_path / "configs"
    captions = tmp_path / "captions"
    scores = tmp_path / "scores"
    configs.mkdir()
    captions.mkdir()
    scores.mkdir()
    (configs / "manifest.json").write_text(
        json.dumps(
            {
                "augmentations": [
                    {"lighting": "bright daylight", "prompt": "Use bright daylight."},
                    {"lighting": "warm lamp light", "prompt": "Use warm lamp light."},
                ]
            }
        ),
        encoding="utf-8",
    )
    (captions / "captions.json").write_text(
        json.dumps({"captions": [{"caption": "a robot arm moves a cube"}]}),
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"status": "prepared"}), encoding="utf-8")
    return {
        "source": source,
        "configs": configs,
        "captions": captions,
        "scores": scores,
        "provenance": provenance,
        "attempt": configs / "attempt.json",
    }


@requires_ffmpeg
def test_generate_variants_runs_real_runner_contract_and_changes_retry(
    tmp_path: Path,
) -> None:
    paths = _generation_inputs(tmp_path)
    storage = _MemoryStorage()
    calls: list[dict] = []

    def fake_generator(**kwargs):
        calls.append(kwargs)
        artifact = Path(kwargs["output_path"]) / kwargs["name"] / "vision.mp4"
        artifact.parent.mkdir(parents=True)
        shutil.copy2(paths["source"], artifact)
        return {"output_path": str(artifact), "output_bytes": artifact.stat().st_size}

    args = (
        str(paths["source"]),
        str(paths["provenance"]),
        str(paths["captions"]),
        str(paths["configs"]),
        "s3://example-bucket/run/cosmos_augmented/",
        str(paths["scores"]),
        str(paths["attempt"]),
        "video2video",
        "Cosmos3-Nano",
        "Preserve the robot motion.",
        "distortion",
        10,
        5.0,
        20,
        2,
        2,
        100,
        -0.5,
        2,
        "latency",
        True,
        "test-run",
    )
    first = c3.generate_variants(
        *args,
        storage=storage,
        environ={"CUDA_VISIBLE_DEVICES": "0,1"},
        generator=fake_generator,
    )
    assert first["engine"] == c3.ENGINE
    assert first["input_conditioning"] == "source-video"
    assert first["variant_count"] == 2
    assert first["variant_parallelism"] == 2
    assert first["video_bytes"] > 0
    assert all(call["mode"] == "video2video" for call in calls)
    assert all(call["no_guardrails"] is False for call in calls)
    assert {call["seed"] for call in calls} == {10, 11}
    assert {call["environ"]["CUDA_VISIBLE_DEVICES"] for call in calls} == {"0", "1"}
    metadata = json.loads(
        storage.objects[
            "s3://example-bucket/run/cosmos_augmented/variant-0000/metadata.json"
        ]
    )
    assert metadata["engine"] == c3.ENGINE
    assert metadata["conditioned_input"] == "source.mp4"
    assert metadata["weights_baked"] is False
    assert metadata["motion_preservation"] is None

    (paths["scores"] / "cosmos_evaluator.json").write_text(
        json.dumps({"status": "completed", "passed": False, "score": 0.4}),
        encoding="utf-8",
    )
    calls.clear()
    second = c3.generate_variants(
        *args,
        storage=storage,
        environ={"CUDA_VISIBLE_DEVICES": "0,1"},
        generator=fake_generator,
    )
    assert second["attempt"] == 1
    assert {call["seed"] for call in calls} == {110, 111}
    assert {call["guidance"] for call in calls} == {4.5}
    assert {call["num_steps"] for call in calls} == {22}


@requires_ffmpeg
def test_generate_variants_preserves_raw_cosmos_and_source_motion(tmp_path: Path) -> None:
    paths = _generation_inputs(tmp_path)
    storage = _MemoryStorage()

    def fake_generator(**kwargs):
        artifact = Path(kwargs["output_path"]) / kwargs["name"] / "vision.mp4"
        _tiny_video(artifact, color="red")
        return {"output_path": str(artifact), "output_bytes": artifact.stat().st_size}

    manifest = c3.generate_variants(
        str(paths["source"]),
        str(paths["provenance"]),
        str(paths["captions"]),
        str(paths["configs"]),
        "s3://example-bucket/run/cosmos_augmented/",
        str(paths["scores"]),
        str(paths["attempt"]),
        "video2video",
        "Cosmos3-Nano",
        "Preserve the robot motion.",
        "distortion",
        10,
        5.0,
        20,
        1,
        1,
        100,
        -0.5,
        2,
        "latency",
        True,
        "test-run",
        0.8,
        storage=storage,
        environ={"CUDA_VISIBLE_DEVICES": "0"},
        generator=fake_generator,
    )

    variant = manifest["variants"][0]
    motion = variant["motion_preservation"]
    assert motion["engine"] == "ffmpeg-source-motion-composite"
    assert motion["source_weight"] == 0.8
    assert motion["cosmos_weight"] == pytest.approx(0.2)
    assert motion["raw_cosmos_video_bytes"] > 0
    assert len(motion["raw_cosmos_video_sha256"]) == 64
    assert len(motion["published_video_sha256"]) == 64
    assert motion["raw_cosmos_video_uri"] in storage.objects
    assert variant["augmented_video_uri"] in storage.objects
    assert (
        storage.objects[motion["raw_cosmos_video_uri"]]
        != storage.objects[variant["augmented_video_uri"]]
    )


@requires_ffmpeg
def test_retry_fails_closed_on_incomplete_evaluator_report(tmp_path: Path) -> None:
    paths = _generation_inputs(tmp_path)
    paths["attempt"].write_text(
        json.dumps({"schema": c3.ATTEMPT_SCHEMA, "attempt": 0}), encoding="utf-8"
    )
    (paths["scores"] / "cosmos_evaluator.json").write_text(
        json.dumps({"status": "degraded", "score": 0.9}), encoding="utf-8"
    )
    with pytest.raises(c3.PaidfCosmos3Error, match="incomplete"):
        c3.generate_variants(
            str(paths["source"]),
            str(paths["provenance"]),
            str(paths["captions"]),
            str(paths["configs"]),
            "s3://example-bucket/out/",
            str(paths["scores"]),
            str(paths["attempt"]),
            "video2video",
            "Cosmos3-Nano",
            "prompt",
            "",
            1,
            5,
            10,
            1,
            1,
            1,
            0,
            0,
            "latency",
            True,
            "run",
            storage=_MemoryStorage(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            generator=lambda **_: {},
        )


@requires_ffmpeg
def test_existing_unreadable_attempt_fails_closed(tmp_path: Path) -> None:
    paths = _generation_inputs(tmp_path)
    paths["attempt"].write_text("not-json", encoding="utf-8")
    with pytest.raises(c3.PaidfCosmos3Error, match="exists but cannot be read"):
        c3._load_attempt(str(paths["attempt"]), str(paths["scores"]))


@requires_ffmpeg
def test_generate_variants_rejects_unconditioned_mode(tmp_path: Path) -> None:
    paths = _generation_inputs(tmp_path)
    with pytest.raises(c3.PaidfCosmos3Error, match="must use video2video"):
        c3.generate_variants(
            str(paths["source"]),
            str(paths["provenance"]),
            str(paths["captions"]),
            str(paths["configs"]),
            "s3://example-bucket/out/",
            str(paths["scores"]),
            str(paths["attempt"]),
            "text2video",
            "Cosmos3-Nano",
            "prompt",
            "",
            1,
            5,
            10,
            1,
            1,
            1,
            0,
            0,
            "latency",
            True,
            "run",
            storage=_MemoryStorage(),
        )


def test_finalize_requires_every_real_component(tmp_path: Path) -> None:
    root = tmp_path / "run"
    for directory in ("cosmos_augmented", "grade", "curation", "reports"):
        (root / directory).mkdir(parents=True)
    (root / "cosmos_augmented" / "manifest.json").write_text(
        json.dumps(
            {
                "engine": c3.ENGINE,
                "schema": c3.MANIFEST_SCHEMA,
                "status": "executed",
                "mode": "video2video",
                "video_bytes": 1234,
                "variant_count": 1,
                "variants": [
                    {
                        "clip": "variant-0000",
                        "augmented_video_uri": "s3://example/variant-0000/video.mp4",
                        "video_bytes": 1234,
                    }
                ],
                "input_conditioned": True,
                "input_conditioning": "source-video",
                "conditioned_input": "source.mp4",
                "model": "Cosmos3-Nano",
                "guardrails": True,
                "weights_baked": False,
                "lineage": {"input_provenance_uri": "input/provenance.json"},
            }
        ),
        encoding="utf-8",
    )
    (root / "grade" / "cosmos_evaluator.json").write_text(
        json.dumps({"status": "completed", "passed": True, "score": 0.88}),
        encoding="utf-8",
    )
    (root / "grade" / "quality_disposition.json").write_text(
        json.dumps({"quality_status": "accepted"}), encoding="utf-8"
    )
    (root / "curation" / "cosmos_curator.json").write_text(
        json.dumps({"engine": "cosmos-curator-upstream", "clip_count": 1}),
        encoding="utf-8",
    )
    (root / "curation" / "report.json").write_text(
        json.dumps({"curation_engine": "fiftyone-brain"}), encoding="utf-8"
    )
    (root / "reports" / "sim2real.rrd").write_bytes(b"RRF2-real")

    result = c3.finalize(str(root), str(root / "reports" / "final.json"))
    assert result["status"] == "completed"
    assert result["evaluator_score"] == 0.88
    assert result["has_rrd"] is True

    (root / "curation" / "report.json").write_text(
        json.dumps({"curation_engine": "report-only"}), encoding="utf-8"
    )
    with pytest.raises(c3.PaidfCosmos3Error, match="FiftyOne"):
        c3.finalize(str(root), str(root / "reports" / "final.json"))

    (root / "curation" / "report.json").write_text(
        json.dumps({"curation_engine": "fiftyone-brain"}), encoding="utf-8"
    )
    (root / "reports" / "sim2real.rrd").write_bytes(b"")
    with pytest.raises(c3.PaidfCosmos3Error, match="missing or empty"):
        c3.finalize(str(root), str(root / "reports" / "final.json"))


def test_extract_frames_reports_missing_ffmpeg_as_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c3.shutil, "which", lambda _binary: None)

    with pytest.raises(c3.PaidfCosmos3Error, match="ffmpeg is required"):
        c3._extract_frames(tmp_path / "video.mp4", tmp_path / "frames")


@pytest.mark.parametrize("source_weight", [-0.1, 1.0, 1.1])
def test_preserve_source_motion_rejects_out_of_range_weights(
    tmp_path: Path, source_weight: float
) -> None:
    with pytest.raises(
        c3.PaidfCosmos3Error,
        match="source motion weight must be strictly between 0 and 1",
    ):
        c3._preserve_source_motion(
            tmp_path / "source.mp4",
            tmp_path / "generated.mp4",
            tmp_path / "output.mp4",
            source_weight=source_weight,
        )


def test_preserve_source_motion_reports_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c3.shutil, "which", lambda _binary: None)

    with pytest.raises(
        c3.PaidfCosmos3Error,
        match="ffmpeg is required for source-motion-preserving publication",
    ):
        c3._preserve_source_motion(
            tmp_path / "source.mp4",
            tmp_path / "generated.mp4",
            tmp_path / "output.mp4",
            source_weight=0.5,
        )


def test_preserve_source_motion_reports_ffmpeg_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c3.shutil, "which", lambda _binary: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        c3.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["ffmpeg"], returncode=1, stdout="", stderr="blend failed"
        ),
    )

    with pytest.raises(
        c3.PaidfCosmos3Error,
        match="source-motion-preserving publication failed: blend failed",
    ):
        c3._preserve_source_motion(
            tmp_path / "source.mp4",
            tmp_path / "generated.mp4",
            tmp_path / "output.mp4",
            source_weight=0.5,
        )


@requires_ffmpeg
def test_generate_variants_requires_s3_publication_before_generation(
    tmp_path: Path,
) -> None:
    paths = _generation_inputs(tmp_path)

    with pytest.raises(c3.PaidfCosmos3Error, match="s3:// output URI"):
        c3.generate_variants(
            str(paths["source"]),
            str(paths["provenance"]),
            str(paths["captions"]),
            str(paths["configs"]),
            str(tmp_path / "local-output"),
            str(paths["scores"]),
            str(paths["attempt"]),
            "video2video",
            "Cosmos3-Nano",
            "prompt",
            "",
            1,
            5,
            10,
            1,
            1,
            1,
            0,
            0,
            "latency",
            True,
            "run",
            storage=_MemoryStorage(),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "mode",
        "model",
        "guardrails",
        "input_conditioned",
        "input_conditioning",
        "conditioned_input",
        "weights_baked",
        "lineage",
        "variants",
    ],
)
def test_finalize_missing_truthful_manifest_fields_raise_domain_error(
    tmp_path: Path, field: str
) -> None:
    root = tmp_path / "run"
    for directory in ("cosmos_augmented", "grade", "curation", "reports"):
        (root / directory).mkdir(parents=True)
    manifest = {
        "schema": c3.MANIFEST_SCHEMA,
        "engine": c3.ENGINE,
        "status": "executed",
        "mode": "video2video",
        "video_bytes": 1234,
        "variant_count": 1,
        "variants": [
            {
                "clip": "variant-0000",
                "augmented_video_uri": "s3://example/variant-0000/video.mp4",
                "video_bytes": 1234,
            }
        ],
        "input_conditioned": True,
        "input_conditioning": "source-video",
        "conditioned_input": "source.mp4",
        "model": "Cosmos3-Nano",
        "guardrails": True,
        "weights_baked": False,
        "lineage": {"input_provenance_uri": "input/provenance.json"},
    }
    manifest.pop(field)
    (root / "cosmos_augmented" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "grade" / "cosmos_evaluator.json").write_text(
        json.dumps({"status": "completed", "passed": True, "score": 0.88}),
        encoding="utf-8",
    )
    (root / "grade" / "quality_disposition.json").write_text(
        json.dumps({"quality_status": "accepted"}), encoding="utf-8"
    )
    (root / "curation" / "cosmos_curator.json").write_text(
        json.dumps({"engine": "cosmos-curator-upstream", "clip_count": 1}),
        encoding="utf-8",
    )
    (root / "curation" / "report.json").write_text(
        json.dumps({"curation_engine": "fiftyone-brain"}), encoding="utf-8"
    )
    (root / "reports" / "sim2real.rrd").write_bytes(b"RRF2-real")

    with pytest.raises(c3.PaidfCosmos3Error, match="missing or invalid fields"):
        c3.finalize(str(root), str(root / "reports" / "final.json"))


def test_quality_route_and_promotion_guard_require_durable_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disposition = tmp_path / "quality_disposition.json"
    decision = tmp_path / "decision.json"
    disposition.write_text(
        json.dumps(
            {
                "quality_status": "rejected",
                "decision": "loop_back",
                "evaluator_status": "missing",
                "hard_checks_passed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, value: Path(uri).write_text(
            json.dumps({"decision": value}), encoding="utf-8"
        ),
    )

    assert c3.route_quality_disposition(str(disposition), str(decision)) == "loop_back"
    assert json.loads(decision.read_text())["decision"] == "loop_back"
    with pytest.raises(c3.PaidfCosmos3Error, match="annotation requires"):
        c3.require_accepted_quality(str(disposition))


def test_quality_route_repairs_pre_decision_disposition_before_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disposition = tmp_path / "quality_disposition.json"
    decision = tmp_path / "decision.json"
    disposition.write_text(
        json.dumps(
            {
                "quality_status": "accepted",
                "evaluator_status": "completed",
                "hard_checks_passed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, value: Path(uri).write_text(
            json.dumps({"decision": value}), encoding="utf-8"
        ),
    )

    assert (
        c3.route_quality_disposition(str(disposition), str(decision))
        == "promote_checkpoint"
    )
    assert json.loads(disposition.read_text())["decision"] == "promote_checkpoint"
    c3.require_accepted_quality(str(disposition))


def test_finalize_non_object_manifest_raises_domain_error(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "cosmos_augmented").mkdir(parents=True)
    (root / "cosmos_augmented" / "manifest.json").write_text(
        "[]", encoding="utf-8"
    )

    with pytest.raises(c3.PaidfCosmos3Error, match="not a JSON object"):
        c3.finalize(str(root), str(root / "reports" / "final.json"))
