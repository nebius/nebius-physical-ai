"""Tests for the manifest-driven invocation MVP (npa.workbench.manifest).

All tests use a stub backend: no docker, no network, no GPU. They prove the
shared-runtime contract — one execution path, descriptor-only onboarding,
input validation, artifact checks, and per-surface provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.workbench import manifest
from npa.workbench.manifest import (
    Catalog,
    DescriptorError,
    LocalDockerBackend,
    Runtime,
    load_descriptor,
    parse_descriptor,
)
from npa.workbench.manifest import sdk as sdk_surface
from npa.workbench.manifest import yaml_spec
from npa.workbench.manifest.runtime import BackendResult

PACKAGE_DIR = Path(manifest.__file__).resolve().parent


class StubBackend:
    """Pretends to run the container; echoes a fabricated artifact."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, image_pinned, argv, gpu, outputs, payload=None):
        self.calls.append(
            {
                "image": image_pinned,
                "argv": argv,
                "gpu": gpu,
                "outputs": outputs,
                "payload": payload or {},
            }
        )
        artifacts = {}
        for oname, _cpath in outputs:
            artifacts[oname] = json.dumps(
                {
                    "device_ok": True,
                    "device": "STUB-GPU",
                    "elapsed_s": 3.21,
                    "checksum": 1.5,
                    "size": 8192,
                    "seed": 0,
                }
            )
        return BackendResult(
            exit_code=0,
            logs="stub",
            artifacts_raw=artifacts,
            elapsed_s=0.01,
            stdout="stub-stdout",
        )


def make_runtime(tmp_path: Path) -> tuple[Runtime, StubBackend]:
    stub = StubBackend()
    rt = Runtime(
        Catalog(PACKAGE_DIR / "descriptors"),
        {"stub": stub},
        records_dir=tmp_path / "records",
    )
    return rt, stub


def test_package_imports_and_lists_both_tools(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    assert rt.catalog.list() == ["cuda-matmul@0.1.0", "gpu-info@0.1.0"]


def test_argv_rendered_as_list_with_digest_pinned_image(tmp_path: Path) -> None:
    rt, stub = make_runtime(tmp_path)
    res = rt.invoke(
        "cuda-matmul",
        "0.1.0",
        "run",
        {"size": 1024, "seed": 7},
        surface="sdk",
        backend="stub",
    )
    call = stub.calls[-1]
    assert call["image"].startswith("pytorch/pytorch@sha256:")
    assert call["argv"] == [
        "python",
        "/work/matmul.py",
        "--size",
        "1024",
        "--seed",
        "7",
        "--out",
        "/work/result.json",
    ]
    assert call["gpu"] == 1
    assert "/work/matmul.py" in call["payload"]
    assert res.success


def test_success_checks_evaluated_from_artifact(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    res = rt.invoke(
        "cuda-matmul", "0.1.0", "run", {"size": 1024}, surface="sdk", backend="stub"
    )
    assert res.success
    assert res.artifacts["result"]["device_ok"] is True
    assert all(c["ok"] for c in res.checks)


def test_input_validation_rejects_bad_and_unknown_params(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    with pytest.raises(DescriptorError):
        rt.invoke(
            "cuda-matmul",
            "0.1.0",
            "run",
            {"size": "huge"},
            surface="sdk",
            backend="stub",
        )
    with pytest.raises(DescriptorError):
        rt.invoke(
            "cuda-matmul", "0.1.0", "run", {"bogus": 1}, surface="sdk", backend="stub"
        )


def test_second_tool_onboarded_with_zero_code_changes(tmp_path: Path) -> None:
    # gpu-info exists only as a descriptor file; the runtime never branches
    # on tool identity.
    rt, stub = make_runtime(tmp_path)
    res = rt.invoke("gpu-info", "0.1.0", "probe", {}, surface="yaml", backend="stub")
    assert stub.calls[-1]["argv"][0] == "nvidia-smi"
    assert res.surface == "yaml"


def test_yaml_surface_translates_to_invoke(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    res = yaml_spec.invoke_spec(
        rt, PACKAGE_DIR / "specs" / "matmul.yaml", backend="stub"
    )
    assert res.surface == "yaml"
    assert res.success


def test_sdk_surface_translates_to_invoke(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    res = sdk_surface.invoke(
        rt, "cuda-matmul", "0.1.0", "run", {"size": 512}, backend="stub"
    )
    assert res.surface == "sdk"
    assert res.success


def test_api_surface_translates_to_invoke(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from npa.workbench.manifest.api import create_app

    rt, _ = make_runtime(tmp_path)
    client = TestClient(create_app(rt))
    r = client.post(
        "/invoke",
        json={
            "tool": "cuda-matmul",
            "version": "0.1.0",
            "command": "run",
            "inputs": {"size": 512},
            "backend": "stub",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["surface"] == "api"
    assert body["success"] is True
    assert client.get("/tools").json()["tools"] == [
        "cuda-matmul@0.1.0",
        "gpu-info@0.1.0",
    ]


def test_verification_record_carries_evidence_separately(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    res = rt.invoke(
        "cuda-matmul", "0.1.0", "run", {"size": 512}, surface="api", backend="stub"
    )
    rec = res.verification_record()
    assert rec["image_digest"].startswith("sha256:")
    assert rec["surface"] == "api"
    assert rec["artifacts"]["result"]["device_ok"] is True
    assert list((tmp_path / "records").glob("cuda-matmul-*-api.json"))


def test_descriptor_rejects_missing_digest_pin() -> None:
    raw = yaml.safe_load(
        (PACKAGE_DIR / "descriptors" / "cuda_matmul.v1.yaml").read_text()
    )
    raw["image"] = {"repository": "pytorch/pytorch", "tag": "latest"}
    with pytest.raises(DescriptorError):
        parse_descriptor(raw)


def test_local_docker_backend_name() -> None:
    assert LocalDockerBackend().name == "local"


def test_load_descriptor_roundtrip() -> None:
    desc = load_descriptor(PACKAGE_DIR / "descriptors" / "gpu_info.v1.yaml")
    assert desc.id == "gpu-info@0.1.0"
    assert set(desc.commands) == {"probe"}
