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


def _patch_subprocess_run(monkeypatch, tmp_path: Path, image: str) -> dict:
    """Replace subprocess.run in the backends module with a fake docker.

    Returns a dict capturing the container-side argv. The fake honors the
    backend's -v mount by writing artifacts to the host outdir.
    """
    import os
    from types import SimpleNamespace

    from npa.workbench.manifest import backends as be_mod

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        host_out = None
        for i, a in enumerate(cmd):
            if a == "-v" and cmd[i + 1].endswith(":/work/out"):
                host_out = cmd[i + 1].rsplit(":", 1)[0]
        argv = cmd[cmd.index(image) + 1 :]
        captured["argv"] = argv
        for i, a in enumerate(argv):
            if a == "--out" and i + 1 < len(argv) and host_out:
                target = argv[i + 1]
                if target.startswith("/work/out/"):
                    target = os.path.join(host_out, target[len("/work/out/") :])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w") as f:
                    json.dump({"device_ok": True, "elapsed_s": 0.5}, f)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(be_mod.subprocess, "run", fake_run)
    return captured


def test_local_backend_renders_exact_container_argv(
    tmp_path: Path, monkeypatch
) -> None:
    from npa.workbench.manifest import LocalDockerBackend

    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("cuda-matmul", "0.1.0")
    image = desc.image.pinned()
    captured = _patch_subprocess_run(monkeypatch, tmp_path, image)
    be = LocalDockerBackend()
    res = be.run(
        image_pinned=image,
        argv=["python", "/work/matmul.py", "--out", "/work/result.json"],
        gpu=0,
        outputs=[("result", "/work/result.json")],
        payload={"/work/matmul.py": "print('hi')"},
    )
    # Payload path rewritten to the backend's mount; output path rewritten
    # to the artifact dir; everything else byte-identical; no shell.
    assert captured["argv"] == [
        "python",
        "/work/out/payload/matmul.py",
        "--out",
        "/work/out/result.dat",
    ]
    assert json.loads(res.artifacts_raw["result"])["device_ok"] is True
    assert res.stdout == "ok"


def test_local_backend_stdout_output_does_not_corrupt_argv(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test: a source=stdout output has no path; the old code fed
    # "" into str.replace and corrupted every argv token.
    from npa.workbench.manifest import LocalDockerBackend

    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("gpu-info", "0.1.0")
    image = desc.image.pinned()
    captured = _patch_subprocess_run(monkeypatch, tmp_path, image)
    be = LocalDockerBackend()
    argv = ["nvidia-smi", "--query-gpu=name", "--format=csv"]
    be.run(
        image_pinned=image,
        argv=argv,
        gpu=1,
        outputs=[("report", "")],
        payload={},
    )
    assert captured["argv"] == argv


def _parse(logs: str, outputs=None):
    from npa.workbench.manifest.nebius_backend import NebiusBackend

    return NebiusBackend()._parse(logs, outputs or [], 0.1)


def _b64(obj) -> str:
    import base64

    return base64.b64encode(json.dumps(obj).encode()).decode()


def test_parse_well_formed_markers() -> None:
    art = _b64({"device_ok": True})
    logs = f"some log\n@@EXIT:0@@\n@@ARTIFACT:result:{art}@@\nmore\n"
    res = _parse(logs, [("result", "/work/result.json")])
    assert res.exit_code == 0
    assert json.loads(res.artifacts_raw["result"]) == {"device_ok": True}
    assert res.stdout == "some log\nmore"


def test_parse_missing_exit_defaults_to_failure() -> None:
    res = _parse("just logs\n", [])
    assert res.exit_code == 1


def test_parse_truncated_exit_marker_still_reads() -> None:
    # No trailing @@ (log rotated mid-line): body must still parse.
    res = _parse("@@EXIT:3", [])
    assert res.exit_code == 3


def test_parse_missing_artifact_leaves_it_absent() -> None:
    res = _parse("@@EXIT:0@@\n@@ARTIFACT_MISSING:result@@\n", [])
    assert res.exit_code == 0
    assert "result" not in res.artifacts_raw


def test_parse_non_marker_at_sign_lines_pass_through() -> None:
    logs = "@@hello world\n@@EXIT:0@@\n"
    res = _parse(logs, [])
    assert res.exit_code == 0
    assert "@@hello world" in res.stdout


def test_parse_base64_round_trip() -> None:
    payload = {"nested": {"list": [1, 2, 3]}, "unicode": "héllo"}
    logs = f"@@EXIT:0@@\n@@ARTIFACT:result:{_b64(payload)}@@\n"
    res = _parse(logs, [])
    assert json.loads(res.artifacts_raw["result"]) == payload


class _Exit1Backend(StubBackend):
    name = "exit1"

    def run(self, image_pinned, argv, gpu, outputs, payload=None):
        return BackendResult(
            exit_code=1, logs="boom", artifacts_raw={}, elapsed_s=0.01, stdout="boom"
        )


class _RaisingBackend:
    name = "raising"

    def run(self, image_pinned, argv, gpu, outputs, payload=None):
        raise RuntimeError("pod exploded")


class _BadJsonBackend(StubBackend):
    name = "badjson"

    def run(self, image_pinned, argv, gpu, outputs, payload=None):
        return BackendResult(
            exit_code=0,
            logs="x",
            artifacts_raw={"result": "{not json"},
            elapsed_s=0.01,
            stdout="x",
        )


def _runtime_with(tmp_path: Path, backend) -> Runtime:
    return Runtime(
        Catalog(PACKAGE_DIR / "descriptors"),
        {"b": backend},
        records_dir=tmp_path / "records",
    )


def _records(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "records").glob("*.json"))


def test_invoke_writes_record_on_nonzero_exit(tmp_path: Path) -> None:
    rt = _runtime_with(tmp_path, _Exit1Backend())
    res = rt.invoke(
        "cuda-matmul", "0.1.0", "run", {"size": 8}, surface="sdk", backend="b"
    )
    assert res.success is False
    assert res.exit_code == 1
    assert _records(tmp_path), "record must exist even on failure"


def test_invoke_writes_record_on_missing_artifact(tmp_path: Path) -> None:
    class _NoArtifact(StubBackend):
        name = "noartifact"

        def run(self, image_pinned, argv, gpu, outputs, payload=None):
            return BackendResult(
                exit_code=0, logs="x", artifacts_raw={}, elapsed_s=0.01, stdout="x"
            )

    rt = _runtime_with(tmp_path, _NoArtifact())
    res = rt.invoke(
        "cuda-matmul", "0.1.0", "run", {"size": 8}, surface="sdk", backend="b"
    )
    assert res.success is False
    recs = _records(tmp_path)
    assert recs
    rec = json.loads(recs[0].read_text())
    assert rec["success"] is False
    assert any(c["check"] == "artifact:result" for c in rec["checks"])


def test_invoke_writes_record_on_malformed_json(tmp_path: Path) -> None:
    rt = _runtime_with(tmp_path, _BadJsonBackend())
    res = rt.invoke(
        "cuda-matmul", "0.1.0", "run", {"size": 8}, surface="sdk", backend="b"
    )
    assert res.success is False
    recs = _records(tmp_path)
    assert recs
    rec = json.loads(recs[0].read_text())
    assert any(c["check"] == "artifact:result:parse" for c in rec["checks"])


def test_invoke_writes_record_when_backend_raises(tmp_path: Path) -> None:
    rt = _runtime_with(tmp_path, _RaisingBackend())
    res = rt.invoke(
        "cuda-matmul", "0.1.0", "run", {"size": 8}, surface="sdk", backend="b"
    )
    assert res.success is False
    assert res.error is not None and "RuntimeError" in res.error
    recs = _records(tmp_path)
    assert recs
    assert json.loads(recs[0].read_text())["error"] == res.error


def test_record_filenames_never_collide(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    rt.invoke("cuda-matmul", "0.1.0", "run", {"size": 8}, surface="sdk", backend="stub")
    rt.invoke("cuda-matmul", "0.1.0", "run", {"size": 8}, surface="sdk", backend="stub")
    recs = list((tmp_path / "records").glob("cuda-matmul-*-sdk.json"))
    assert len(recs) == 2


def _raw_descriptor() -> dict:
    digest = "sha256:" + "0" * 64
    return {
        "apiVersion": "manifest/v0.1",
        "name": "t",
        "version": "0.1.0",
        "image": {"repository": "r", "tag": "t", "digest": digest},
        "commands": {
            "run": {
                "argv": ["bin", "--size", "{{size}}", "{{flag}}"],
                "params": {
                    "size": {"type": "integer", "default": 4},
                    "note": {"type": "string"},
                    "verbose": {"type": "boolean", "default": False},
                    "flag": {"type": "boolean", "default": True},
                },
                "outputs": {},
            }
        },
    }


def test_optional_none_params_are_not_rendered() -> None:
    spec = parse_descriptor(_raw_descriptor()).commands["run"]
    argv = spec.render_argv(spec.resolve_inputs({}))
    assert argv == ["bin", "--size", "4", "true"]
    assert "None" not in argv


def test_descriptor_rejects_file_output_without_path() -> None:
    raw = _raw_descriptor()
    raw["commands"]["run"]["outputs"] = {"result": {"format": "json"}}
    with pytest.raises(DescriptorError):
        parse_descriptor(raw)
