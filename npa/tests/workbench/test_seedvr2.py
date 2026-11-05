"""Focused contracts for the shared SeedVR2 runtime and evidence operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from npa.sdk.workbench import seedvr2 as seedvr2_sdk
from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError
from npa.workbench.seedvr2 import artifacts, conditioning, quality, runtime
from npa.workbench.seedvr2.service import create_app
from npa.workbench.seedvr2.schemas import (
    MODEL_REVISION,
    QUALITY_BOUNDARY_SCHEMA,
    RESULT_SCHEMA,
    SOURCE_REVISION,
    RestoreRequest,
    VideoArtifactRequest,
)
from npa.workbench.storage_scope import StorageScope, use_storage_scope


INPUT_URI = "s3://example-bucket/input.mp4"
OUTPUT_PREFIX = "s3://example-bucket/run/restoration/"
PROBE_URI = "s3://example-bucket/run/probe.json"
RUNTIME_IDENTITY = {
    "image": "ghcr.io/nebius/npa-seedvr2@sha256:" + "a" * 64,
    "image_digest": "sha256:" + "a" * 64,
    "npa_source_revision": "b" * 40,
    "gpu": {
        "status": "available",
        "name": "NVIDIA H100 80GB HBM3",
        "memory_mib": "81559",
        "driver_version": "580.0",
        "compute_capability": "9.0",
        "mig_mode": "Disabled",
        "count": "1",
    },
    "sequence_parallel_size": 1,
    "color_fix": False,
}


class FakeStorage:
    """Small byte-exact object store used at the production storage boundary."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)

    def read_bytes_with_etag(self, uri: str):
        data = self.objects.get(uri)
        if data is None:
            return None
        return data, hashlib.sha256(data).hexdigest()

    def download_file(self, uri: str, local_path: str) -> str:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[uri])
        return str(target)

    def upload_file(self, local_file: str, uri: str) -> str:
        self.objects[uri] = Path(local_file).read_bytes()
        return uri

    def put_bytes_conditional(
        self, payload: bytes, uri: str, *, if_none_match: bool
    ) -> str:
        assert if_none_match is True
        if uri in self.objects:
            raise RuntimeError("conditional write collision")
        self.objects[uri] = payload
        return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for media contract tests")
    path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x16:rate=5",
            "-frames:v",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _model_files(tmp_path: Path) -> dict[str, Path]:
    names = ("seedvr2_ema_3b.pth", "ema_vae.pth", "pos_emb.pt", "neg_emb.pt")
    files = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = path
    return files


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "seedvr-source"
    (root / "projects").mkdir(parents=True)
    (root / "configs_3b").mkdir()
    (root / "models" / "video_vae_v3").mkdir(parents=True)
    (root / "projects" / "inference_seedvr2_3b.py").write_text("# pinned upstream\n")
    (root / "configs_3b" / "main.yaml").write_text(
        "vae: {dtype: bfloat16, scaling_factor: 0.9152, grouping: false}\n"
    )
    (root / "models" / "video_vae_v3" / "s8_c16_t4_inflation_sd3.yaml").write_text(
        "model: {}\n"
    )
    return root


def _fake_inference(argv, *, cwd, env, stdout, stderr, check):
    assert argv[0].endswith("torchrun")
    assert "--nproc-per-node=1" in argv
    assert (Path(cwd) / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml").is_file()
    assert env.get("HF_TOKEN") is None
    assert env.get("AWS_SECRET_ACCESS_KEY") is None
    work = Path(cwd).parent
    shutil.copyfile(work / "input" / "input.mp4", work / "generated" / "input.mp4")
    stdout.write(b"official inference placeholder for boundary test\n")
    return subprocess.CompletedProcess(argv, 0)


def _restore(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    controls=None,
    gpu=None,
    runner=_fake_inference,
) -> tuple[dict, FakeStorage]:
    storage = FakeStorage({INPUT_URI: source_video.read_bytes()})
    identity = {**RUNTIME_IDENTITY, "gpu": gpu or RUNTIME_IDENTITY["gpu"]}
    _restore_environment(tmp_path, monkeypatch, identity)
    artifacts.probe(
        VideoArtifactRequest(
            input_path=INPUT_URI,
            output_path=PROBE_URI,
            run_id="unit-boundary",
        ),
        storage_factory=lambda: storage,
    )
    result = runtime.restore(
        RestoreRequest(
            input_path=INPUT_URI,
            output_path=OUTPUT_PREFIX,
            run_id="unit-boundary",
            probe_path=PROBE_URI,
            output_height=16,
            output_width=32,
            **(controls or {}),
        ),
        storage_factory=lambda: storage,
        inference_runner=runner,
        model_resolver=lambda: _model_files(tmp_path),
        runtime_identity_resolver=lambda expected="H100": identity,
    )
    return result, storage


def _restore_environment(tmp_path, monkeypatch, identity):
    monkeypatch.setattr(runtime, "BAKED_NPA_ROOT", Path(runtime.__file__).parents[2])
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    source_root = _source_tree(tmp_path)
    monkeypatch.setattr(
        conditioning,
        "SOURCE_MAIN_SHA256",
        runtime._sha256(source_root / "configs_3b/main.yaml"),
    )
    monkeypatch.setattr(runtime, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(artifacts, "SOURCE_ROOT", source_root)
    monkeypatch.setenv("SEEDVR2_PYTHON", "/opt/seedvr2-venv/bin/python")
    monkeypatch.setenv("NPA_TASK_IMAGE", RUNTIME_IDENTITY["image"])
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-inference")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-inference")
    source_revision_path = tmp_path / "npa-source-revision"
    source_revision_path.write_text(RUNTIME_IDENTITY["npa_source_revision"] + "\n")
    monkeypatch.setattr(artifacts, "SOURCE_REVISION_PATH", source_revision_path)
    monkeypatch.setattr(
        artifacts, "_runtime_identity", lambda expected="H100": identity
    )


def test_dry_run_is_deterministic_and_uses_official_torchrun() -> None:
    request = RestoreRequest(
        input_path=INPUT_URI,
        output_path=OUTPUT_PREFIX,
        run_id="repeatable",
        output_height=480,
        output_width=640,
        dry_run=True,
    )
    first = runtime.restore(request)
    second = runtime.restore(request)
    assert first == second
    assert first["status"] == "dry_run"
    assert first["source"]["revision"] == SOURCE_REVISION
    assert first["model"]["revision"] == MODEL_REVISION
    argv = first["argv"]
    assert argv[0] == "/opt/seedvr2-venv/bin/torchrun"
    assert "inference_seedvr2_3b.py" in argv[3]
    assert argv[-2:] == ["--sp_size", "1"]


def test_restore_publishes_readback_verified_artifacts(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    assert result["schema"] == RESULT_SCHEMA
    assert result["status"] == "ok"
    assert (
        result["input"]["sha256"]
        == hashlib.sha256(source_video.read_bytes()).hexdigest()
    )
    assert result["output"]["media"]["frames"] == 3
    assert result["output"]["unique_decoded_frames"] == 3
    assert result["output"]["derived_sensor_truth"] is False
    assert result["input"]["probe"]["uri"] == PROBE_URI
    assert result["runtime"]["image_digest"] == "sha256:" + "a" * 64
    assert result["runtime"]["sequence_parallel_size"] == 1
    assert (
        result["artifact_hashes"]["upstream_log"]
        == hashlib.sha256(storage.objects[OUTPUT_PREFIX + "upstream.log"]).hexdigest()
    )
    for name in ("restored.mp4", "upstream.log", "result.json"):
        assert OUTPUT_PREFIX + name in storage.objects
    delivered = json.loads(storage.objects[OUTPUT_PREFIX + "result.json"])
    assert delivered == result
    assert not list((tmp_path / "runs").glob("*"))


def test_upstream_failure_surfaces_a_bounded_redacted_log_tail(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "upstream.log"

    def failing_runner(argv, *, cwd, env, stdout, stderr, check):
        stdout.write(b"x" * (runtime.UPSTREAM_FAILURE_TAIL_BYTES + 100))
        stdout.write(
            b"\nHF_TOKEN=hf_privateexample123\n"
            b"AWS_SECRET_ACCESS_KEY: very-secret-value\n"
            b"'AWS_SECRET_ACCESS_KEY': 'dict-secret-value'\n"
            b"OPENAI_API_KEY='quoted-secret-value'\n"
            b"Authorization: Bearer authorization-token-value\n"
            b"Authorization: Basic SYNTHETIC_BASIC_123456\n"
            b"'Authorization': 'Digest username=\"user\", response=\"digest-secret\"'\n"
            b"request header Bearer abcdefghijklmnop\n"
            b"ModuleNotFoundError: No module named 'pytorch'\n"
        )
        return subprocess.CompletedProcess(argv, 1)

    with pytest.raises(runtime.SeedVR2Error) as error:
        runtime._execute_upstream(
            ["torchrun", "inference.py"],
            tmp_path,
            log_path,
            failing_runner,
        )

    message = str(error.value)
    assert "[truncated to complete lines within final 16384 bytes]" in message
    assert "ModuleNotFoundError: No module named 'pytorch'" in message
    assert message.count("<redacted>") == 8
    assert "hf_privateexample123" not in message
    assert "very-secret-value" not in message
    assert "dict-secret-value" not in message
    assert "quoted-secret-value" not in message
    assert "authorization-token-value" not in message
    assert "SYNTHETIC_BASIC_123456" not in message
    assert "digest-secret" not in message
    assert "abcdefghijklmnop" not in message
    assert log_path.stat().st_size > runtime.UPSTREAM_FAILURE_TAIL_BYTES


@pytest.mark.parametrize(
    "field,value", [("driver_version", "581.1"), ("memory_mib", "81000")]
)
def test_verify_records_valid_cross_node_inventory_drift(
    tmp_path, source_video, monkeypatch, field, value
):
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    verifier_gpu = {**result["runtime"]["gpu"], field: value}
    monkeypatch.setattr(runtime, "SOURCE_REVISION_PATH", artifacts.SOURCE_REVISION_PATH)
    monkeypatch.setattr(runtime, "_gpu_inventory", lambda: verifier_gpu)
    monkeypatch.setattr(artifacts, "_runtime_identity", runtime._runtime_identity)
    verification = artifacts.verify(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path="s3://example-bucket/run/verification.json",
            run_id=result["run_id"],
        ),
        storage_factory=lambda: storage,
    )
    assert verification["verifier_runtime"]["gpu"] == verifier_gpu
    assert result["runtime"]["gpu"][field] != value
    published = json.loads(storage.objects["s3://example-bucket/run/verification.json"])
    assert published == verification


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "NVIDIA H200"),
        ("compute_capability", "10.0"),
        ("memory_mib", "10000"),
        ("mig_mode", "Enabled"),
        ("count", "2"),
    ],
)
def test_verify_still_rejects_incompatible_verifier_gpu(
    tmp_path, source_video, monkeypatch, field, value
):
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    monkeypatch.setattr(runtime, "SOURCE_REVISION_PATH", artifacts.SOURCE_REVISION_PATH)
    monkeypatch.setattr(
        runtime, "_gpu_inventory", lambda: {**result["runtime"]["gpu"], field: value}
    )
    monkeypatch.setattr(artifacts, "_runtime_identity", runtime._runtime_identity)
    output = "s3://example-bucket/run/verification.json"
    with pytest.raises(runtime.SeedVR2Error, match="verified H100"):
        artifacts.verify(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path=output,
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )
    assert output not in storage.objects


def test_upstream_failure_tail_discards_a_partial_boundary_line(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "boundary.log"
    secret_line = b"HF_TOKEN=" + b"boundary-secret-" * 1200
    log_path.write_bytes(secret_line + b"\nTraceback: retained root cause\n")

    tail = runtime._upstream_failure_tail(log_path)

    assert tail.startswith("[truncated to complete lines within final 16384 bytes]")
    assert "Traceback: retained root cause" in tail
    assert "boundary-secret" not in tail


def test_verify_and_review_recompute_identity_and_make_nonblended_media(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    verification = artifacts.verify(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path="s3://example-bucket/run/verification.json",
            run_id=result["run_id"],
        ),
        storage_factory=lambda: storage,
    )
    assert verification["restored_video_sha256"] == result["output"]["sha256"]
    assert verification["attestation_scope"] == (
        "artifact_and_runtime_consistency_only"
    )
    assert verification["producer_execution_attested"] is False
    assert verification["validated_execution"] == {
        "run_id": result["run_id"],
        "image": result["runtime"]["image"],
        "image_digest": result["runtime"]["image_digest"],
        "npa_source_revision": result["runtime"]["npa_source_revision"],
        "probe_sha256": result["input"]["probe"]["sha256"],
        "upstream_log_sha256": result["artifact_hashes"]["upstream_log"],
    }
    assert verification["verifier_runtime"] == result["runtime"]
    review = artifacts.review(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path="s3://example-bucket/run/review/",
            run_id=result["run_id"],
        ),
        storage_factory=lambda: storage,
    )
    assert review["comparison"]["blending"] is False
    assert review["comparison"]["selected_frame_indices"] == [0, 1, 2]
    for name in ("comparison.mp4", "contact-sheet.png", "review.json", "index.html"):
        assert "s3://example-bucket/run/review/" + name in storage.objects


def test_quality_boundary_binds_exact_media_and_stays_review_only(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    output_uri = "s3://example-bucket/run/quality-boundary.json"

    boundary = quality.assess(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path=output_uri,
            run_id=result["run_id"],
        ),
        storage_factory=lambda: storage,
    )

    assert boundary["schema"] == QUALITY_BOUNDARY_SCHEMA
    assert boundary["status"] == "review_only"
    assert boundary["quality_qualified"] is False
    assert boundary["sensor_truth_qualified"] is False
    assert boundary["downstream_use"] == "review_only"
    assert boundary["source"]["sha256"] == result["input"]["sha256"]
    assert boundary["candidate"]["sha256"] == result["output"]["sha256"]
    assert boundary["candidate"]["derived_sensor_truth"] is False
    assert boundary["timing_evidence"] == {
        "fully_decoded": True,
        "source_frames": 3,
        "candidate_frames": 3,
        "frame_counts_match": True,
        "frame_rates_match": True,
    }
    assert boundary["diagnostics"]["temporal_consistency"]["enforcement"] == "advisory"
    assert boundary["diagnostics"]["motion_hallucination"]["enforcement"] == "advisory"
    assert boundary["diagnostics"]["temporal_consistency"]["configuration"][
        "threshold"
    ] == pytest.approx(0.8)
    assert boundary["diagnostics"]["motion_hallucination"]["configuration"] == {
        "threshold": 0.682,
        "grad_thresh": 10.0,
        "blur_ksize": 7,
        "morph_k": 3,
        "dist_tol_px": 7.0,
        "prefer_upstream": False,
    }
    assert json.loads(storage.objects[output_uri]) == boundary


def test_temporal_validator_allows_bounded_float32_region_reduction_error(
    tmp_path: Path,
) -> None:
    width, height, frames = 320, 240, 3
    raw_frames = random.Random(1).randbytes(width * height * 3 * frames)
    video = tmp_path / "random-noise.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "3",
            "-i",
            "pipe:0",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(video),
        ],
        input=raw_frames,
        check=True,
    )
    payload = json.loads(
        json.dumps(
            quality.check_temporal_consistency(
                clip_id="float32-reduction",
                original_video=video,
                augmented_video=video,
                threshold=quality.TEMPORAL_THRESHOLD,
                regions=quality._temporal_region_configuration(),
                noise_floor=quality.TEMPORAL_NOISE_FLOOR,
                blur_ksize=quality.TEMPORAL_BLUR_KSIZE,
            ).to_dict()
        )
    )
    differences = [
        abs(
            payload["regions"][0][field]
            - sum(region[field] for region in payload["regions"][1:]) / 4.0
        )
        for field in (
            "source_mean_acceleration",
            "augmented_mean_acceleration",
            "residual_mean_acceleration",
        )
    ]
    assert max(differences) > 1e-6

    quality._validate_temporal_diagnostic(
        payload,
        run_id="float32-reduction",
        frame_count=frames,
        frame_width=width,
        frame_height=height,
    )


def test_failing_advisory_diagnostics_never_become_quality_acceptance(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    temporal = quality.check_temporal_consistency(
        clip_id=result["run_id"],
        original_video=source_video,
        augmented_video=source_video,
    ).to_dict()
    temporal.update(passed=False, score=0.1)
    for region in temporal["regions"]:
        region.update(
            source_mean_acceleration=2.0,
            augmented_mean_acceleration=2.0,
            passed=False,
            residual_mean_acceleration=2.5,
            acceleration_ratio=1.0,
            score=0.1,
        )
    motion = quality.check_hallucination(
        clip_id=result["run_id"],
        original_video=source_video,
        augmented_video=source_video,
        prefer_upstream=False,
    ).to_dict()
    motion.update(
        passed=False,
        score=0.0,
        total_hallucinated_dynamic_pixels=1,
        total_augmented_dynamic_pixels=1,
    )
    diagnostic_calls: list[dict[str, object]] = []

    def _temporal_result(**kwargs):
        diagnostic_calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: temporal)

    def _motion_result(**kwargs):
        diagnostic_calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: motion)

    monkeypatch.setattr(
        quality,
        "check_temporal_consistency",
        _temporal_result,
    )
    monkeypatch.setattr(
        quality,
        "check_hallucination",
        _motion_result,
    )

    boundary = quality.assess(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path="s3://example-bucket/run/failed-diagnostics.json",
            run_id=result["run_id"],
        ),
        storage_factory=lambda: storage,
    )

    assert boundary["status"] == "review_only"
    assert boundary["quality_qualified"] is False
    assert boundary["diagnostics"]["temporal_consistency"]["result"]["passed"] is False
    assert boundary["diagnostics"]["motion_hallucination"]["result"]["passed"] is False
    assert len(diagnostic_calls) == 2
    for call in diagnostic_calls:
        assert call["ffmpeg_input_args"] == (
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "mov",
        )
        assert call["subprocess_env"] == runtime._media_environment()


def test_quality_boundary_fails_closed_before_diagnostics_on_incomplete_timing(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    output_uri = "s3://example-bucket/run/incomplete-quality-boundary.json"
    decoded = iter(
        [["source-0", "source-1", "source-2"], ["candidate-0", "candidate-1"]]
    )
    monkeypatch.setattr(quality, "_decoded_frame_hashes", lambda path: next(decoded))
    monkeypatch.setattr(
        quality,
        "check_temporal_consistency",
        lambda **kwargs: pytest.fail("diagnostics must not run on incomplete evidence"),
    )
    monkeypatch.setattr(
        quality,
        "check_hallucination",
        lambda **kwargs: pytest.fail("diagnostics must not run on incomplete evidence"),
    )

    with pytest.raises(runtime.SeedVR2Error, match="timing evidence is incomplete"):
        quality.assess(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path=output_uri,
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )
    assert output_uri not in storage.objects


def test_quality_boundary_fails_closed_when_a_diagnostic_is_incomplete(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    output_uri = "s3://example-bucket/run/incomplete-diagnostic.json"

    def fail_temporal(**kwargs):
        raise CosmosEvaluatorError("decode stopped")

    monkeypatch.setattr(quality, "check_temporal_consistency", fail_temporal)

    with pytest.raises(runtime.SeedVR2Error, match="complete evidence"):
        quality.assess(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path=output_uri,
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )
    assert output_uri not in storage.objects


@pytest.mark.parametrize(
    "case",
    [
        "temporal-missing-field",
        "temporal-wrong-engine",
        "temporal-frame-mismatch",
        "temporal-nonfinite-score",
        "temporal-verdict-contradiction",
        "temporal-score-contradiction",
        "temporal-ratio-contradiction",
        "temporal-upper-triangle-contradiction",
        "temporal-physical-range-contradiction",
        "temporal-cross-region-contradiction",
        "temporal-overprecision",
        "temporal-wrong-blur-type",
        "motion-wrong-threshold",
        "motion-frame-mismatch",
        "motion-inconsistent-counts",
        "motion-impossible-pixel-count",
        "motion-wrong-score-type",
    ],
)
def test_quality_boundary_rejects_semantically_incomplete_diagnostics(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    temporal = quality.check_temporal_consistency(
        clip_id=result["run_id"],
        original_video=source_video,
        augmented_video=source_video,
    ).to_dict()
    motion = quality.check_hallucination(
        clip_id=result["run_id"],
        original_video=source_video,
        augmented_video=source_video,
        prefer_upstream=False,
    ).to_dict()
    if case == "temporal-missing-field":
        temporal.pop("engine")
    elif case == "temporal-wrong-engine":
        temporal["engine"] = "unreviewed-engine"
    elif case == "temporal-frame-mismatch":
        temporal["frame_counts_match"] = False
    elif case == "temporal-nonfinite-score":
        temporal["score"] = float("nan")
    elif case == "temporal-verdict-contradiction":
        temporal["regions"][0].update(
            passed=True,
            residual_mean_acceleration=2.5,
            score=0.1,
        )
        temporal.update(passed=True, score=0.1)
    elif case == "temporal-score-contradiction":
        temporal["regions"][0].update(passed=False, score=0.1)
        temporal.update(passed=False, score=0.1)
    elif case == "temporal-ratio-contradiction":
        temporal["regions"][0]["acceleration_ratio"] += 1.0
    elif case == "temporal-upper-triangle-contradiction":
        temporal["regions"][0].update(
            source_mean_acceleration=0.0,
            augmented_mean_acceleration=0.0,
            passed=False,
            residual_mean_acceleration=2.5,
            acceleration_ratio=0.0,
            score=0.1,
        )
        temporal.update(passed=False, score=0.1)
    elif case == "temporal-physical-range-contradiction":
        temporal["regions"][0].update(
            source_mean_acceleration=1_000_000.0,
            augmented_mean_acceleration=1_000_000.0,
            acceleration_ratio=1.0,
        )
    elif case == "temporal-cross-region-contradiction":
        temporal["regions"][0].update(
            source_mean_acceleration=510.0,
            augmented_mean_acceleration=510.0,
            residual_mean_acceleration=0.0,
            acceleration_ratio=1.0,
            score=1.0,
            passed=True,
        )
        for region in temporal["regions"][1:]:
            region.update(
                source_mean_acceleration=0.0,
                augmented_mean_acceleration=0.0,
                residual_mean_acceleration=0.0,
                acceleration_ratio=0.0,
                score=1.0,
                passed=True,
            )
        temporal.update(score=1.0, passed=True)
    elif case == "temporal-overprecision":
        temporal["regions"][0]["acceleration_ratio"] += 0.0000001
    elif case == "temporal-wrong-blur-type":
        temporal["blur_ksize"] = 7.0
    elif case == "motion-wrong-threshold":
        motion["threshold"] = 0.8
    elif case == "motion-frame-mismatch":
        motion["total_frames"] = 2
    elif case == "motion-inconsistent-counts":
        motion["total_hallucinated_dynamic_pixels"] = 1
        motion["total_augmented_dynamic_pixels"] = 0
    elif case == "motion-impossible-pixel-count":
        motion["total_augmented_dynamic_pixels"] = 1_000_000_000
    elif case == "motion-wrong-score-type":
        motion["score"] = 1
    monkeypatch.setattr(
        quality,
        "check_temporal_consistency",
        lambda **kwargs: SimpleNamespace(to_dict=lambda: temporal),
    )
    monkeypatch.setattr(
        quality,
        "check_hallucination",
        lambda **kwargs: SimpleNamespace(to_dict=lambda: motion),
    )
    output_uri = f"s3://example-bucket/run/{case}.json"

    with pytest.raises(runtime.SeedVR2Error, match="diagnostic evidence"):
        quality.assess(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path=output_uri,
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )

    assert output_uri not in storage.objects


def test_sdk_quality_boundary_uses_the_shared_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    monkeypatch.setattr(
        quality,
        "assess",
        lambda request: (
            seen.append(request) or {"status": "review_only", "run_id": request.run_id}
        ),
    )

    document = seedvr2_sdk.quality_boundary(
        input_path="s3://example-bucket/run/result.json",
        output_path="s3://example-bucket/run/quality-boundary.json",
        run_id="sdk-quality",
    )

    assert document == {"status": "review_only", "run_id": "sdk-quality"}
    assert len(seen) == 1
    assert seen[0].input_path.endswith("/result.json")


@pytest.mark.parametrize(
    ("height", "width"),
    [(479, 640), (480, 639)],
)
def test_dimensions_must_be_divisible_by_sixteen(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        RestoreRequest(
            input_path=INPUT_URI,
            output_path=OUTPUT_PREFIX,
            run_id="invalid",
            output_height=height,
            output_width=width,
        )


def test_dimensions_must_fit_the_reviewed_h100_pixel_budget() -> None:
    with pytest.raises(ValueError, match="must not exceed 1920x1080"):
        RestoreRequest(
            input_path=INPUT_URI,
            output_path=OUTPUT_PREFIX,
            run_id="too-large",
            output_height=1088,
            output_width=1920,
        )


@pytest.mark.parametrize("frame_count", [1, 2])
def test_short_clip_is_rejected_before_gpu_or_model_work_and_publication(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
) -> None:
    short_clip = tmp_path / f"{frame_count}-frame.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source_video),
            "-frames:v",
            str(frame_count),
            "-c:v",
            "copy",
            str(short_clip),
        ],
        check=True,
    )
    storage = FakeStorage({INPUT_URI: short_clip.read_bytes()})
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    error = "at least three decodable frames"
    with pytest.raises(runtime.SeedVR2Error, match=error):
        artifacts.probe(
            VideoArtifactRequest(
                input_path=INPUT_URI, output_path=PROBE_URI, run_id="short-clip"
            ),
            storage_factory=lambda: storage,
        )
    with pytest.raises(runtime.SeedVR2Error, match=error):
        runtime.restore(
            RestoreRequest(
                input_path=INPUT_URI,
                output_path=OUTPUT_PREFIX,
                probe_path=PROBE_URI,
                run_id="short-clip",
                output_height=16,
                output_width=32,
            ),
            storage_factory=lambda: storage,
            model_resolver=lambda: pytest.fail("model resolution must not run"),
            runtime_identity_resolver=lambda expected: pytest.fail(
                "GPU resolution must not run"
            ),
            inference_runner=lambda *args, **kwargs: pytest.fail(
                "inference must not run"
            ),
        )
    assert storage.objects == {INPUT_URI: short_clip.read_bytes()}
    assert list((tmp_path / "runs").glob("*/failure.json"))
    assert runtime._probe_video(source_video)["frames"] == 3


def test_decoded_frame_hashes_exclude_audio_packets(
    tmp_path: Path, source_video: Path
) -> None:
    clip = tmp_path / "video-with-audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_video),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        check=True,
    )

    assert runtime._probe_video(clip)["frames"] == 3
    assert len(runtime._decoded_frame_hashes(clip)) == 3


def test_restore_rejects_source_aspect_ratio_change_before_model_resolution(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage({INPUT_URI: source_video.read_bytes()})
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    with pytest.raises(runtime.SeedVR2Error, match="aspect ratio"):
        runtime.restore(
            RestoreRequest(
                input_path=INPUT_URI,
                output_path=OUTPUT_PREFIX,
                probe_path=PROBE_URI,
                run_id="aspect-ratio",
                output_height=32,
                output_width=32,
            ),
            storage_factory=lambda: storage,
            model_resolver=lambda: pytest.fail("model resolution must not run"),
        )


def test_source_video_must_fit_reviewed_decode_budget() -> None:
    request = RestoreRequest(
        input_path=INPUT_URI,
        output_path=OUTPUT_PREFIX,
        probe_path=PROBE_URI,
        run_id="source-budget",
    )
    with pytest.raises(runtime.SeedVR2Error, match="source video area"):
        runtime._validate_source_geometry(
            request,
            {"width": 3840, "height": 2160},
        )


def test_probe_binds_restore_to_exact_input_bytes(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage({INPUT_URI: source_video.read_bytes()})
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    artifacts.probe(
        VideoArtifactRequest(
            input_path=INPUT_URI,
            output_path=PROBE_URI,
            run_id="probe-binding",
        ),
        storage_factory=lambda: storage,
    )
    storage.objects[INPUT_URI] += b"mutated-after-probe"
    with pytest.raises(runtime.SeedVR2Error, match="does not bind"):
        runtime.restore(
            RestoreRequest(
                input_path=INPUT_URI,
                output_path=OUTPUT_PREFIX,
                probe_path=PROBE_URI,
                run_id="probe-binding",
                output_height=16,
                output_width=32,
            ),
            storage_factory=lambda: storage,
            model_resolver=lambda: pytest.fail("model resolution must not run"),
        )


def test_probe_cannot_be_replayed_across_workflow_runs(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage({INPUT_URI: source_video.read_bytes()})
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    artifacts.probe(
        VideoArtifactRequest(
            input_path=INPUT_URI,
            output_path=PROBE_URI,
            run_id="original-run",
        ),
        storage_factory=lambda: storage,
    )
    with pytest.raises(runtime.SeedVR2Error, match="does not bind"):
        runtime.restore(
            RestoreRequest(
                input_path=INPUT_URI,
                output_path=OUTPUT_PREFIX,
                probe_path=PROBE_URI,
                run_id="replayed-run",
                output_height=16,
                output_width=32,
            ),
            storage_factory=lambda: storage,
            model_resolver=lambda: pytest.fail("model resolution must not run"),
        )


def test_non_dry_restore_requires_probe() -> None:
    with pytest.raises(runtime.SeedVR2Error, match="requires probe_path"):
        runtime.restore(
            RestoreRequest(
                input_path=INPUT_URI,
                output_path=OUTPUT_PREFIX,
                run_id="missing-probe",
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_path", "s3://test-bucket/inputs%2Flow.mp4"),
        ("output_path", "s3://test-bucket/runs%2Funit/restoration/"),
        ("probe_path", "s3://test-bucket/runs/unit/%70robe.json"),
    ],
)
def test_restore_rejects_noncanonical_storage_keys(field: str, value: str) -> None:
    values = {
        "input_path": INPUT_URI,
        "output_path": OUTPUT_PREFIX,
        "probe_path": PROBE_URI,
        "run_id": "canonical",
    }
    values[field] = value
    with pytest.raises(runtime.SeedVR2Error, match="canonical, unescaped"):
        runtime._validate_request(RestoreRequest(**values))


def test_upstream_source_root_is_not_environment_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEEDVR2_SOURCE_ROOT", "/workspace/untrusted-seedvr")
    request = RestoreRequest(
        input_path=INPUT_URI,
        output_path=OUTPUT_PREFIX,
        probe_path=PROBE_URI,
        run_id="source-root",
    )
    argv = runtime.build_restore_argv(request, Path("/workspace/run/upstream"))
    assert argv[3] == "/opt/seedvr2/projects/inference_seedvr2_3b.py"
    assert runtime._inference_environment()["PYTHONPATH"] == "/opt/seedvr2"


def test_inference_environment_is_a_credential_free_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "ECS_CONTAINER_CREDENTIALS_FULL_URI",
        "HF_TOKEN",
        "NEBIUS_IAM_TOKEN",
        "SEEDVR2_TOKEN",
    ):
        monkeypatch.setenv(name, f"secret-{name}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    environment = runtime._inference_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HOME"] == "/workspace"
    assert environment["TMPDIR"] == "/workspace/tmp"
    assert (
        not {
            "AWS_ACCESS_KEY_ID",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "ECS_CONTAINER_CREDENTIALS_FULL_URI",
            "HF_TOKEN",
            "NEBIUS_IAM_TOKEN",
            "SEEDVR2_TOKEN",
        }
        & environment.keys()
    )
    media_environment = runtime._media_environment()
    model_environment = runtime._model_fetch_environment()
    assert media_environment["TMPDIR"] == "/workspace/tmp"
    assert model_environment["TMPDIR"] == "/workspace/tmp"
    assert not {"AWS_ACCESS_KEY_ID", "NEBIUS_IAM_TOKEN"} & media_environment.keys()
    assert not {"AWS_ACCESS_KEY_ID", "NEBIUS_IAM_TOKEN"} & model_environment.keys()
    assert model_environment["HF_TOKEN"] == "secret-HF_TOKEN"


def test_media_probe_refuses_playlist_disguised_as_mp4(tmp_path: Path) -> None:
    playlist = tmp_path / "playlist.mp4"
    playlist.write_text("#EXTM3U\nhttps://metadata.invalid/secret.ts\n")
    with pytest.raises(runtime.SeedVR2Error, match="ffprobe could not decode"):
        runtime._probe_video(playlist)


def _identity_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "BAKED_NPA_ROOT", Path(runtime.__file__).parents[2])
    source_revision_path = tmp_path / "npa-source-revision"
    source_revision_path.write_text("b" * 40 + "\n")
    monkeypatch.setattr(runtime, "SOURCE_REVISION_PATH", source_revision_path)
    monkeypatch.setenv("NPA_TASK_IMAGE", RUNTIME_IDENTITY["image"])
    monkeypatch.setattr(
        runtime, "_gpu_inventory", lambda expected="H100": RUNTIME_IDENTITY["gpu"]
    )
    assert runtime._runtime_identity()["image_digest"] == "sha256:" + "a" * 64
    return source_revision_path


def test_runtime_identity_requires_digest_bound_h100(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_revision_path = _identity_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("NPA_TASK_IMAGE", "ghcr.io/nebius/npa-seedvr2:mutable")
    with pytest.raises(runtime.SeedVR2Error, match="immutable image digest"):
        runtime._runtime_identity()
    monkeypatch.setenv("NPA_TASK_IMAGE", RUNTIME_IDENTITY["image"])
    source_revision_path.write_text("mutable\n")
    with pytest.raises(runtime.SeedVR2Error, match="invalid baked NPA source revision"):
        runtime._runtime_identity()
    source_revision_path.write_text("b" * 40 + "\n")
    monkeypatch.setattr(
        runtime,
        "_gpu_inventory",
        lambda: {**RUNTIME_IDENTITY["gpu"], "name": "NVIDIA H200"},
    )
    with pytest.raises(runtime.SeedVR2Error, match="verified H100"):
        runtime._runtime_identity()
    monkeypatch.setattr(
        runtime,
        "_gpu_inventory",
        lambda: {**RUNTIME_IDENTITY["gpu"], "memory_mib": "10240"},
    )
    with pytest.raises(runtime.SeedVR2Error, match="full-memory"):
        runtime._runtime_identity()
    monkeypatch.setattr(
        runtime,
        "_gpu_inventory",
        lambda: {**RUNTIME_IDENTITY["gpu"], "mig_mode": "Enabled"},
    )
    with pytest.raises(runtime.SeedVR2Error, match="full-memory"):
        runtime._runtime_identity()


def test_review_rejects_result_runtime_forgery(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    forged = json.loads(storage.objects[result["artifacts"]["result"]])
    forged["runtime"]["image"] = "registry.invalid/forged@sha256:" + "f" * 64
    storage.objects[result["artifacts"]["result"]] = json.dumps(forged).encode()
    with pytest.raises(runtime.SeedVR2Error, match="runtime identity"):
        artifacts.review(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path="s3://example-bucket/run/forged-review/",
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )


def test_verify_rejects_result_repository_forgery(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    forged = json.loads(storage.objects[result["artifacts"]["result"]])
    forged["source"]["repository"] = "https://example.invalid/lookalike"
    storage.objects[result["artifacts"]["result"]] = json.dumps(forged).encode()
    with pytest.raises(runtime.SeedVR2Error, match="artifact identity"):
        artifacts.verify(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path="s3://example-bucket/run/forged-verification.json",
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )


def test_verify_rejects_result_media_contract_forgery(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    forged = json.loads(storage.objects[result["artifacts"]["result"]])
    forged["request"]["output_width"] = 16
    forged["argv"][forged["argv"].index("--res_w") + 1] = "16"
    storage.objects[result["artifacts"]["result"]] = json.dumps(forged).encode()
    with pytest.raises(runtime.SeedVR2Error, match="media contract|aspect ratio"):
        artifacts.verify(
            VideoArtifactRequest(
                input_path=result["artifacts"]["result"],
                output_path="s3://example-bucket/run/forged-verification.json",
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )


def test_verify_authorizes_source_uri_from_result(
    tmp_path: Path,
    source_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    forged = json.loads(storage.objects[result["artifacts"]["result"]])
    forged["request"]["input_path"] = "s3://outside/private.mp4"
    forged["input"]["uri"] = "s3://outside/private.mp4"
    storage.objects[result["artifacts"]["result"]] = json.dumps(forged).encode()
    request = VideoArtifactRequest(
        input_path=result["artifacts"]["result"],
        output_path="s3://example-bucket/run/forged-verification.json",
        run_id=result["run_id"],
    )
    scope = StorageScope.from_config(
        s3_roots=[
            "s3://example-bucket/input.mp4",
            "s3://example-bucket/run",
        ]
    )
    with use_storage_scope(scope):
        with pytest.raises(ValueError, match="outside the configured"):
            artifacts.verify(request, storage_factory=lambda: storage)


def test_review_rejects_nested_result_uri_outside_storage_scope(
    tmp_path: Path,
) -> None:
    result_uri = "s3://allowed/run/result.json"
    result = {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "source": {"revision": SOURCE_REVISION},
        "model": {"revision": MODEL_REVISION},
        "input": {"uri": "s3://outside/private.mp4"},
        "output": {},
        "artifacts": {"restored_video": "s3://allowed/run/restored.mp4"},
    }
    storage = FakeStorage({result_uri: json.dumps(result).encode()})
    request = VideoArtifactRequest(
        input_path=result_uri,
        output_path="s3://allowed/run/review/",
        run_id="nested-scope",
    )
    monkeypatch_root = tmp_path / "runs"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NPA_SEEDVR2_WORK_DIR", str(monkeypatch_root))
        scope = StorageScope.from_config(s3_roots=["s3://allowed/"])
        with use_storage_scope(scope):
            with pytest.raises(ValueError, match="outside the configured"):
                artifacts.review(request, storage_factory=lambda: storage)


@pytest.mark.parametrize("section", ["source", "model", "input", "output", "artifacts"])
def test_result_loader_fails_closed_on_non_object_sections(
    tmp_path: Path, section: str
) -> None:
    document = {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "source": {"revision": SOURCE_REVISION},
        "model": {"revision": MODEL_REVISION},
        "input": {},
        "output": {},
        "artifacts": {},
    }
    document[section] = []
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document))
    with pytest.raises(runtime.SeedVR2Error, match="identity or status"):
        artifacts._load_result(path)


def test_review_frame_selection_matches_frozen_anchor_positions() -> None:
    assert artifacts._selected_frames(100) == [0, 16, 32, 48, 60, 68, 76, 84, 96]


def test_runtime_refuses_non_s3_and_non_video_paths() -> None:
    with pytest.raises(runtime.SeedVR2Error, match="exact s3:// MP4"):
        runtime.restore(
            RestoreRequest(
                input_path="s3://example-bucket/input.txt",
                output_path=OUTPUT_PREFIX,
                run_id="invalid",
                dry_run=True,
            )
        )
    with pytest.raises(runtime.SeedVR2Error, match="bucket prefix"):
        runtime.restore(
            RestoreRequest(
                input_path=INPUT_URI,
                output_path="local-output",
                run_id="invalid",
                dry_run=True,
            )
        )


def test_service_requires_auth_and_enforces_request_s3_roots(monkeypatch) -> None:
    seen = []

    def fake_restore(request):
        runtime._validate_request(request)
        seen.append(request)
        return {"status": "ok", "run_id": request.run_id}

    monkeypatch.setattr(runtime, "restore", fake_restore)
    client = TestClient(
        create_app(
            token="secret",
            allowed_s3_roots=["s3://allowed/input", "s3://allowed/output"],
        )
    )
    body = {
        "input_path": "s3://allowed/input/video.mp4",
        "output_path": "s3://allowed/output/run/",
        "run_id": "service",
        "dry_run": True,
    }
    assert client.post("/restore", json=body).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    response = client.post("/restore", json=body, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "run_id": "service"}
    assert len(seen) == 1
    body["output_path"] = "s3://other-bucket/output/"
    response = client.post("/restore", json=body, headers=headers)
    assert response.status_code == 400
    assert "outside the configured" in response.json()["detail"]


def test_service_exposes_the_shared_quality_boundary_operation(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        quality,
        "assess",
        lambda request: (
            seen.append(request) or {"status": "review_only", "run_id": request.run_id}
        ),
    )
    client = TestClient(
        create_app(
            token="secret",
            allowed_s3_roots=["s3://allowed/run"],
        )
    )
    body = {
        "input_path": "s3://allowed/run/result.json",
        "output_path": "s3://allowed/run/quality-boundary.json",
        "run_id": "service-quality",
    }

    response = client.post(
        "/quality-boundary",
        json=body,
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "review_only",
        "run_id": "service-quality",
    }
    assert len(seen) == 1
