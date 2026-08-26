"""Packaging and real-component guards for native Cosmos3 Ray Serve."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    UNVALIDATED_PUBLICATION_TOOLS,
    supported_tool_version,
)
from npa.smoke.manifest import container

_SCANNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "scan_image_cosmos3_ray_serve_payload.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "cosmos3_ray_payload_scanner", _SCANNER_PATH
)
assert _SPEC and _SPEC.loader
_SCANNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCANNER
_SPEC.loader.exec_module(_SCANNER)
scan_tarball = _SCANNER.scan_tarball

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/cosmos3-ray-serve"
DOCKERFILE = IMAGE / "Dockerfile"


def test_image_is_registered_as_gpu_accepted_public_service() -> None:
    assert CONTAINER_IMAGE_NAMES["cosmos3-ray-serve"] == "npa-cosmos3-ray-serve"
    assert supported_tool_version("cosmos3-ray-serve") == "ray1-cu130"
    assert "cosmos3-ray-serve" not in UNVALIDATED_PUBLICATION_TOOLS
    contract = yaml.safe_load(
        (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    )
    entry = contract["images"]["cosmos3-ray-serve"]
    assert entry["tier"] == "service"
    assert entry["ports"] == [8000]
    assert entry["redistribution"] == "public"


def test_image_uses_exact_accepted_framework_parent_and_bakes_no_weights() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "npa-cosmos3@sha256:0eb459" in text
    assert "5e67049cd94acb667786f1e6dd0dab821cb90c97" in text
    assert 'npa.serving.backend="cosmos-framework-native-ray-serve"' in text
    assert "vllm" not in text.lower()
    assert "*.safetensors" in text
    assert "NPA_COSMOS3_RAY_GUARDRAILS=true" in text
    assert "ARG COSMOS3_RAY_VERSION=2.46.0" in text
    assert "ARG LINUX_LIBC_DEV_VERSION=6.8.0-138.138" in text
    assert '"linux-libc-dev=${LINUX_LIBC_DEV_VERSION}"' in text
    assert "uv sync --frozen --inexact" in text
    assert "--extra guardrail --extra serve --group cu130" in text
    assert "uv pip install --python /opt/npa/.venv/bin/python" in text


def test_server_invokes_upstream_native_batching_component() -> None:
    server = (ROOT / "npa/src/npa/workbench/cosmos/ray_server.py").read_text()
    verify = (IMAGE / "verify_env.py").read_text()
    assert (
        "from cosmos_framework.inference.ray.serve import OmniModelDeployment" in server
    )
    assert ".generate.remote(sample)" in server
    assert "@ray.serve.batch" in verify
    assert "OmniInference" in verify
    assert '_cuda_getArchFlags() or "").split()' in verify


def test_ray_ingress_keeps_pydantic_models_out_of_frozen_route_metadata() -> None:
    server = (ROOT / "npa/src/npa/workbench/cosmos/ray_server.py").read_text()
    assert '@api.post("/v1/batches")' in server
    assert "response_model=RayBatchResponse" not in server
    assert "body: dict[str, Any]" in server
    assert "request = RayBatchRequest.model_validate(body)" in server
    assert ').model_dump(mode="json")' in server
    assert ") -> FileResponse:" not in server


def test_golden_eval_is_real_model_backed_batching() -> None:
    spec = container("cosmos3-ray-serve")
    assert spec.golden_eval.kind == "server-smoke"
    assert spec.golden_eval.gpu == "required"
    smoke = (IMAGE / "smoke_functional.sh").read_text()
    assert '"samples"' in smoke
    assert "ray-smoke-a" in smoke and "ray-smoke-b" in smoke
    assert "ray-batch" in smoke


def _docker_archive(path: Path, member_name: str, payload: bytes) -> None:
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    layer.seek(0)
    config = {"config": {}, "history": []}
    manifest = [{"Config": "config.json", "RepoTags": [], "Layers": ["layer.tar"]}]
    with tarfile.open(path, mode="w") as archive:
        for name, body in (
            ("manifest.json", json.dumps(manifest).encode()),
            ("config.json", json.dumps(config).encode()),
            ("layer.tar", layer.read()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))


def test_payload_scan_rejects_runtime_model_cache(tmp_path: Path) -> None:
    archive = tmp_path / "image.tar"
    _docker_archive(
        archive,
        "home/ubuntu/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/config.json",
        b"{}",
    )
    report = scan_tarball(archive)
    assert report["verdict"] == "restricted-payload-detected"


def test_payload_scan_allows_framework_source(tmp_path: Path) -> None:
    archive = tmp_path / "image.tar"
    _docker_archive(archive, "opt/cosmos3/cosmos-framework/README.md", b"framework")
    report = scan_tarball(archive)
    assert report["verdict"] == "clean"
