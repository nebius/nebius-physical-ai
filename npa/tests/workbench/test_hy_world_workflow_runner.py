"""Offline command-order contracts for the HY-World BYOF stage adapter."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from npa.workbench.hy_world import workflow_runner


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _runtime_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    output = tmp_path / "output"
    stage = tmp_path / "stage"
    cache = tmp_path / "warmed-model-cache"
    output.mkdir()
    stage.mkdir()
    (cache / "source-cached-runtime").mkdir(parents=True)
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(output))
    monkeypatch.setenv("NPA_HY_WORLD_INPUT_STAGE_DIR", str(stage))
    monkeypatch.setenv("NPA_HY_WORLD_RUNTIME_CACHE", str(cache))
    monkeypatch.setenv("BYOF_IMAGE", "registry.invalid/hy-world@sha256:" + "a" * 64)
    monkeypatch.setenv("NPA_HY_WORLD_LLM_ADDR", "private-vllm.internal")
    monkeypatch.setenv("HY_WORLD_PROMPT_B64", _b64("Expand this scene"))
    monkeypatch.setenv(
        "HY_WORLD_INPUT_IMAGE_URI_B64", _b64("s3://input-bucket/photo.png")
    )
    monkeypatch.setenv("HY_WORLD_LLM_PORT_B64", _b64("8000"))
    return stage, cache


def test_stage_checks_bootstrap_before_staging_input_and_reuses_warmed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage, cache = _runtime_environment(monkeypatch, tmp_path)
    events: list[tuple[str, str]] = []
    staged: list[Path] = []

    def download(source: str, destination: Path) -> None:
        events.append(("download", source))
        staged.append(destination)
        destination.write_bytes(b"not-a-real-png-but-long-enough" * 16)

    def validate(path: Path) -> None:
        events.append(("validate", path.name))
        assert path.is_file()

    def check_endpoint(address: str, port: str, model: str) -> None:
        events.append(("vllm", f"{address}:{port}/{model}"))

    def invoke(command: str, environment: dict[str, str]) -> None:
        events.append(("runtime", command))
        if command == "run-image-to-world":
            assert Path(environment["HY_WORLD_INPUT_IMAGE"]).is_file()
            assert environment["NPA_HY_WORLD_LLM_ADDR"] == "private-vllm.internal"
            assert environment["NPA_HY_WORLD_LLM_PORT"] == "8000"
            output = Path(environment["NPA_SMOKE_OUTPUT_DIR"])
            (output / "hy_world_image_to_world.json").write_text("{}", encoding="utf-8")
            reports = output / "reports"
            reports.mkdir()
            (reports / "hy_world_scene.rrd").write_bytes(b"rrd")
            (reports / "hy_world_scene_rrd_manifest.json").write_text(
                "{}", encoding="utf-8"
            )

    monkeypatch.setattr(workflow_runner, "_download_image", download)
    monkeypatch.setattr(workflow_runner, "_validate_image", validate)
    monkeypatch.setattr(workflow_runner, "_run_runtime", invoke)
    monkeypatch.setattr(workflow_runner, "_check_vllm_endpoint", check_endpoint)

    workflow_runner.run()

    assert events == [
        ("runtime", "terms"),
        ("runtime", "health"),
        ("runtime", "bootstrap-integrity"),
        ("vllm", "private-vllm.internal:8000/Qwen/Qwen3-VL-8B-Instruct"),
        ("download", "s3://input-bucket/photo.png"),
        ("validate", "input.png"),
        ("runtime", "run-image-to-world"),
    ]
    assert cache.is_dir(), "a warmed runtime/model cache is never scanned or removed"
    assert staged and not staged[0].exists(), "the disposable staged input is cleaned"
    assert not list(stage.glob("npa-hy-world-input-*"))


def test_private_vllm_endpoint_refuses_a_url_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _runtime_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("NPA_HY_WORLD_LLM_ADDR", "https://private-vllm.internal")
    monkeypatch.setattr(
        workflow_runner,
        "_download_image",
        lambda *_args: pytest.fail("an invalid endpoint must fail before S3 access"),
    )

    with pytest.raises(workflow_runner.WorkflowInputError, match="without a scheme"):
        workflow_runner.run()


def test_vllm_preflight_uses_the_upstream_openai_surface_and_exact_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.close()

    class Opener:
        def open(self, request, timeout: int):
            requests.append((request, timeout))
            return Response(b'{"data":[{"id":"Qwen/Qwen3-VL-8B-Instruct"}]}')

    monkeypatch.setattr(workflow_runner, "build_opener", lambda *_args: Opener())

    workflow_runner._check_vllm_endpoint(
        "private-vllm.internal", "8000", "Qwen/Qwen3-VL-8B-Instruct"
    )

    request, timeout = requests.pop()
    assert request.full_url == "http://private-vllm.internal:8000/v1/models"
    assert request.get_header("Authorization") == "Bearer EMPTY"
    assert timeout == 15


def test_vllm_preflight_rejects_a_different_served_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.close()

    class Opener:
        def open(self, _request, timeout: int):
            assert timeout == 15
            return Response(b'{"data":[{"id":"other-model"}]}')

    monkeypatch.setattr(workflow_runner, "build_opener", lambda *_args: Opener())

    with pytest.raises(workflow_runner.WorkflowInputError, match="pinned Qwen3-VL"):
        workflow_runner._check_vllm_endpoint(
            "private-vllm.internal", "8000", "Qwen/Qwen3-VL-8B-Instruct"
        )
