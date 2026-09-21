"""Focused contracts for the shared SeedVR2 runtime and evidence operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.service import create_app
from npa.workbench.seedvr2.schemas import (
    MODEL_REVISION,
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
        "__inherit__: models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml\n"
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
) -> tuple[dict, FakeStorage]:
    storage = FakeStorage({INPUT_URI: source_video.read_bytes()})
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    source_root = _source_tree(tmp_path)
    monkeypatch.setattr(runtime, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(artifacts, "SOURCE_ROOT", source_root)
    monkeypatch.setenv("SEEDVR2_PYTHON", "/opt/seedvr2-venv/bin/python")
    monkeypatch.setenv("NPA_TASK_IMAGE", RUNTIME_IDENTITY["image"])
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-inference")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-inference")
    source_revision_path = tmp_path / "npa-source-revision"
    source_revision_path.write_text(RUNTIME_IDENTITY["npa_source_revision"] + "\n")
    monkeypatch.setattr(artifacts, "SOURCE_REVISION_PATH", source_revision_path)
    monkeypatch.setattr(artifacts, "_runtime_identity", lambda: RUNTIME_IDENTITY)
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
        ),
        storage_factory=lambda: storage,
        inference_runner=_fake_inference,
        model_resolver=lambda: _model_files(tmp_path),
        runtime_identity_resolver=lambda: RUNTIME_IDENTITY,
    )
    return result, storage


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


def test_runtime_identity_requires_digest_bound_h100(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_revision_path = tmp_path / "npa-source-revision"
    source_revision_path.write_text("b" * 40 + "\n")
    monkeypatch.setattr(runtime, "SOURCE_REVISION_PATH", source_revision_path)
    monkeypatch.setenv("NPA_TASK_IMAGE", RUNTIME_IDENTITY["image"])
    monkeypatch.setattr(runtime, "_gpu_inventory", lambda: RUNTIME_IDENTITY["gpu"])
    assert runtime._runtime_identity()["image_digest"] == "sha256:" + "a" * 64
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
