"""Exercise SAM 3.1 access refusal, cache isolation and video-output contracts."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "npa/docker/workbench/sam3"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"sam31_{name}", SOURCE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = _load("runtime")
segment = _load("segment")


def test_missing_token_refuses_before_network_or_cache(tmp_path, monkeypatch):
    for key in runtime.TOKEN_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NPA_SAM3_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(
        runtime, "_access", lambda _: pytest.fail("network was reached")
    )
    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        runtime._ensure()
    assert not list(tmp_path.iterdir())


def test_denied_access_never_starts_download_or_install(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-only-credential")
    monkeypatch.setenv("NPA_SAM3_CACHE", str(tmp_path / "cache"))

    def deny(_):
        raise RuntimeError("HTTP 403")

    monkeypatch.setattr(runtime, "_access", deny)
    with pytest.raises(RuntimeError, match="403"):
        runtime._ensure()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "status,body,ready", [(403, b"no", False), (200, b"", False), (206, b"p", True)]
)
def test_access_reads_exact_revision_payload(monkeypatch, status, body, ready):
    import requests

    @contextmanager
    def get(url, **kwargs):
        assert runtime.PINS["model_revision"] in url
        assert url.endswith("/sam3.1_multiplex.pt")
        assert kwargs["stream"] is True
        assert kwargs["headers"]["Range"] == "bytes=0-0"
        yield SimpleNamespace(status_code=status, iter_content=lambda _: iter([body]))

    monkeypatch.setattr(requests, "get", get)
    if ready:
        runtime._access("test-only-credential")
    else:
        with pytest.raises(RuntimeError, match="denied or unavailable"):
            runtime._access("test-only-credential")


def test_health_and_refusal_are_offline(tmp_path):
    env = {**os.environ, "NPA_SAM3_CACHE": str(tmp_path / "cache")}
    for key in runtime.TOKEN_KEYS:
        env.pop(key, None)
    for mode in ("health", "assert-refusal"):
        result = subprocess.run(
            [sys.executable, str(SOURCE / "runtime.py"), mode],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert not (tmp_path / "cache").exists()


def test_runtime_install_failure_does_not_leave_reusable_partial_cache(
    tmp_path, monkeypatch
):
    def fail(path):
        (path / "partial").write_text("incomplete")
        raise RuntimeError("install failed")

    monkeypatch.setattr(runtime, "_install", fail)
    with pytest.raises(RuntimeError, match="install failed"):
        runtime._runtime(tmp_path)
    assert not list(tmp_path.iterdir())


def test_inference_child_does_not_receive_hf_credentials(monkeypatch):
    for key in runtime.TOKEN_KEYS:
        monkeypatch.setenv(key, "test-only-credential")
    assert not set(runtime.TOKEN_KEYS) & runtime._child_env().keys()


def test_checkpoint_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"wrong checkpoint")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **_: str(checkpoint)),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.errors",
        SimpleNamespace(HfHubHTTPError=ConnectionError),
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        runtime._checkpoint(tmp_path, "test-only-credential")


def test_stream_rejects_skipped_frame_before_writing(tmp_path):
    predictor = SimpleNamespace(
        handle_stream_request=lambda **_: iter([{"frame_index": 1}])
    )
    with pytest.raises(ValueError, match="skipped or duplicated"):
        segment._stream(predictor, "session", None, None, tmp_path)


@pytest.mark.parametrize("count,nonempty", [(2, 2), (3, 0)])
def test_output_validation_rejects_short_or_empty_result(
    tmp_path, monkeypatch, count, nonempty
):
    monkeypatch.setattr(segment, "_video_info", lambda _: {"nb_read_frames": count})
    records = [{"mask_pixels": int(i < nonempty)} for i in range(count)]
    with pytest.raises(ValueError):
        segment._validate(records, {"nb_read_frames": 3}, tmp_path)


def test_overlay_preserves_ids_and_writes_actual_masks(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((1, 8, 8), dtype=bool)
    mask[:, 2:5, 2:5] = True
    path = tmp_path / "mask.npz"
    record = segment._overlay(
        frame, {"out_binary_masks": mask, "out_obj_ids": [7]}, path
    )
    assert record == {"object_ids": [7], "mask_pixels": 9}
    with np.load(path, allow_pickle=False) as saved:
        assert saved["object_ids"].tolist() == [7]
        assert np.array_equal(saved["masks"], mask)
    assert frame[3, 3].any()
    assert not frame[0, 0].any()


def test_candidate_is_development_only():
    from npa.deploy.images import development_image_for_tool, publicly_publishable_tools

    assert "sam3" not in publicly_publishable_tools()
    assert development_image_for_tool("sam3", git_sha="a" * 40).endswith(
        "/npa-sam3:dev-" + "a" * 40
    )
    assert json.loads((SOURCE / "pins.json").read_text())["model"] == "facebook/sam3.1"


def test_real_overlay_encoding_preserves_frame_count(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("CPU codec test requires ffmpeg and ffprobe")
    source = tmp_path / "input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=4",
            "-frames:v",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    mask = np.zeros((1, 32, 32), dtype=bool)
    mask[:, 8:16, 8:16] = True
    responses = [
        {"frame_index": i, "outputs": {"out_binary_masks": mask, "out_obj_ids": [4]}}
        for i in range(4)
    ]
    predictor = SimpleNamespace(handle_stream_request=lambda **_: iter(responses))
    output = tmp_path / "result"
    (output / "masks").mkdir(parents=True)
    info = segment._video_info(source)
    records = segment._encode(predictor, "fixture", source, output, info)
    assert segment._validate(records, info, output) == {
        "decoded_frames": 4,
        "nonempty_mask_frames": 4,
    }
    assert len(list((output / "masks").glob("*.npz"))) == 4
