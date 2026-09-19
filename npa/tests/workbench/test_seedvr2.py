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


INPUT_URI = "s3://example-bucket/input.mp4"
OUTPUT_PREFIX = "s3://example-bucket/run/restoration/"


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
    (root / "projects" / "inference_seedvr2_3b.py").write_text("# pinned upstream\n")
    return root


def _fake_inference(argv, *, cwd, env, stdout, stderr, check):
    assert argv[0].endswith("torchrun")
    assert "--nproc-per-node=1" in argv
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
    monkeypatch.setenv("SEEDVR2_SOURCE_ROOT", str(_source_tree(tmp_path)))
    monkeypatch.setenv("SEEDVR2_PYTHON", "/opt/seedvr2-venv/bin/python")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-inference")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-inference")
    result = runtime.restore(
        RestoreRequest(
            input_path=INPUT_URI,
            output_path=OUTPUT_PREFIX,
            run_id="unit-boundary",
            output_height=16,
            output_width=32,
        ),
        storage_factory=lambda: storage,
        inference_runner=_fake_inference,
        model_resolver=lambda: _model_files(tmp_path),
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
    assert result["runtime"]["sequence_parallel_size"] == 1
    for name in ("restored.mp4", "upstream.log", "result.json"):
        assert OUTPUT_PREFIX + name in storage.objects
    delivered = json.loads(storage.objects[OUTPUT_PREFIX + "result.json"])
    assert delivered == result
    assert not list((tmp_path / "runs").glob("*"))


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
            run_id="verify",
        ),
        storage_factory=lambda: storage,
    )
    assert verification["restored_video_sha256"] == result["output"]["sha256"]
    review = artifacts.review(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path="s3://example-bucket/run/review/",
            run_id="review",
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
                output_path="/tmp/output",
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
