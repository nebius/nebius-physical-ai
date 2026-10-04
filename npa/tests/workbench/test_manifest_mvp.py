"""Tests for the manifest-driven invocation MVP (npa.workbench.manifest).

All tests use a stub backend: no docker, no network, no GPU. They prove the
shared-runtime contract — one execution path, descriptor-only onboarding,
input validation, artifact checks, and per-surface provenance.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from npa.workbench import manifest
from npa.workbench.manifest import (
    Catalog,
    DescriptorError,
    LocalDockerBackend,
    NebiusBackend,
    Runtime,
    load_descriptor,
    parse_descriptor,
)
from npa.workbench.manifest import sdk as sdk_surface
from npa.workbench.manifest import yaml_spec
from npa.workbench.manifest.runtime import BackendResult
from npa.workbench.manifest.schema import CommandSpec, OutputSpec

PACKAGE_DIR = Path(manifest.__file__).resolve().parent


class StubBackend:
    """Pretends to run the container; echoes a fabricated artifact."""

    name = "stub"
    s3_upload = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(
        self,
        image_pinned,
        argv,
        gpu,
        outputs,
        payload=None,
        memory_gb=0,
        s3_inputs=None,
        s3_output_names=None,
        s3_prefix="",
        s3_env=None,
        pip_packages=None,
        apt_packages=None,
    ):
        self.calls.append(
            {
                "image": image_pinned,
                "argv": argv,
                "gpu": gpu,
                "memory_gb": memory_gb,
                "outputs": outputs,
                "payload": payload or {},
                "s3_inputs": s3_inputs,
                "s3_output_names": s3_output_names,
                "s3_prefix": s3_prefix,
                "pip_packages": pip_packages,
                "apt_packages": apt_packages,
            }
        )
        artifacts = {}
        for oname, _cpath, fmt, _source in outputs:
            if fmt == "binary":
                artifacts[oname] = "stub-base64-blob"
            else:
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
    assert rt.catalog.list() == [
        "cuda-matmul@0.1.0",
        "gpu-info@0.1.0",
        "large-artifact@0.1.0",
        "openvla-predict@0.1.0",
        "pendulum-rtx@0.1.0",
        "pendulum-viz@0.1.0",
    ]


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
        "large-artifact@0.1.0",
        "openvla-predict@0.1.0",
        "pendulum-rtx@0.1.0",
        "pendulum-viz@0.1.0",
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
        outputs=[("result", "/work/result.json", "json", "file")],
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
        outputs=[("report", "", "text", "stdout")],
        payload={},
    )
    assert captured["argv"] == argv


def _parse(logs: str, outputs=None):
    from npa.workbench.manifest.nebius_backend import NebiusBackend

    return NebiusBackend()._parse(logs, 0.1)


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

    def run(self, *a, **k):
        return BackendResult(
            exit_code=1, logs="boom", artifacts_raw={}, elapsed_s=0.01, stdout="boom"
        )


class _RaisingBackend:
    name = "raising"

    def run(self, *a, **k):
        raise RuntimeError("pod exploded")


class _BadJsonBackend(StubBackend):
    name = "badjson"

    def run(self, *a, **k):
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

        def run(self, *a, **k):
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


# ---------------------------------------------------------------------------
# Round 2: S3 transport, schema hardening, Job spec.
# ---------------------------------------------------------------------------


class _S3Stub(StubBackend):
    name = "nebius-like"
    s3_upload = True


def _desc_dict(**over):
    base = {
        "apiVersion": "manifest/v0.1",
        "name": "t",
        "version": "0.1.0",
        "image": {
            "repository": "r",
            "digest": "sha256:" + "ab" * 32,
        },
        "commands": {
            "run": {
                "argv": ["python", "x.py"],
                "params": {},
                "outputs": {
                    "o": {"path": "/work/o", "format": "json", "required": True}
                },
            }
        },
    }
    base.update(over)
    return base


def test_unknown_param_type_rejected() -> None:
    d = _desc_dict()
    d["commands"]["run"]["params"] = {"p": {"type": "uuid"}}
    with pytest.raises(DescriptorError):
        parse_descriptor(d)


def test_check_naming_undeclared_artifact_rejected() -> None:
    d = _desc_dict()
    d["success"] = {"checks": [{"artifact": "ghost", "json_path": "a", "equals": True}]}
    with pytest.raises(DescriptorError, match="undeclared artifact"):
        parse_descriptor(d)


def test_payload_traversal_rejected() -> None:
    d = _desc_dict()
    d["payload_files"] = [{"container_path": "/work/x.py", "host_path": "../evil/x.py"}]
    with pytest.raises(DescriptorError, match="host_path"):
        parse_descriptor(d)
    d["payload_files"] = [{"container_path": "/work/x.py", "host_path": "/abs/x.py"}]
    with pytest.raises(DescriptorError, match="host_path"):
        parse_descriptor(d)


def test_invalid_s3_uri_rejected() -> None:
    d = _desc_dict()
    d["inputs"] = [
        {
            "name": "c",
            "s3_uri": "https://example.com/x",
            "container_path": "/work/c",
        }
    ]
    with pytest.raises(DescriptorError, match="s3://"):
        parse_descriptor(d)


def test_binary_format_parses() -> None:
    d = _desc_dict()
    d["commands"]["run"]["outputs"] = {
        "blob": {"path": "/work/b", "format": "binary", "required": True}
    }
    desc = parse_descriptor(d)
    assert desc.commands["run"].outputs["blob"].format == "binary"


def test_binary_outputs_require_store_on_s3_backend(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    rt.backends["s3stub"] = _S3Stub()
    d = _desc_dict()
    d["commands"]["run"]["outputs"] = {
        "blob": {"path": "/work/b", "format": "binary", "required": True}
    }
    rt.catalog._entries["binonly@0.1.0"] = parse_descriptor(d)
    # invoke never raises: a misconfigured descriptor becomes a failed
    # result with a clear error and a record.
    res = rt.invoke("binonly", "0.1.0", "run", {}, backend="s3stub")
    assert res.success is False
    assert "artifact_store" in res.error


def test_s3_inputs_and_prefix_reach_backend(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    stub = StubBackend()
    rt.backends["stub"] = stub
    res = rt.invoke("large-artifact", "0.1.0", "run", {"size": 1024}, backend="stub")
    call = stub.calls[0]
    assert call["s3_inputs"] == [
        (
            "s3://my-bucket/manifest-mvp/inputs/large_artifact_config.json",
            "/work/config.json",
        )
    ]
    assert call["s3_output_names"] == ["matrix"]
    assert call["s3_prefix"].startswith("s3://my-bucket/manifest-mvp/large-artifact/")
    # Stub echoes a base64 blob for the binary artifact; it stays a string.
    assert res.artifacts["matrix"] == "stub-base64-blob"
    # And the summary check ran against the JSON artifact.
    assert any(c["check"] == "summary.device_ok" and c["ok"] for c in res.checks)


def test_binary_artifact_stays_string_not_parsed() -> None:
    bres = BackendResult(
        exit_code=0,
        logs="",
        artifacts_raw={"m": "s3://b/k/x.npy"},
        elapsed_s=0.1,
        stdout="",
    )
    spec = CommandSpec(
        name="run",
        argv=["x"],
        params={},
        outputs={
            "m": OutputSpec(name="m", path="/w/m", format="binary", required=True)
        },
    )
    artifacts, errors = Runtime._collect_artifacts(spec, bres)
    assert artifacts["m"] == "s3://b/k/x.npy"
    assert errors == []


def test_large_artifact_descriptor_validates(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("large-artifact", "0.1.0")
    assert desc.artifact_store is not None
    assert desc.artifact_store.bucket == "my-bucket"
    assert desc.inputs[0].s3_uri.startswith("s3://")
    assert desc.resources.memory_gb == 16


class _FakeKc(NebiusBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.created: list[dict] = []
        self.kc_calls: list[tuple] = []

    def _kc(self, *args, input_text=None, timeout=120):
        self.last_timeout = timeout
        full = [self.kubectl]
        if self.kube_context:
            full += ["--context", self.kube_context]
        full += ["-n", self.namespace, *args]
        self.kc_calls.append((full, input_text))
        if input_text and '"kind": "Job"' in input_text:
            self.created.append(json.loads(input_text))
        return subprocess.CompletedProcess(
            args=full, returncode=0, stdout="", stderr=""
        )


def _job(gpu: int, memory_gb: int = 0) -> dict:
    be = _FakeKc()
    be._create_job(
        "job-1",
        "cm-1",
        "img@sha256:abc",
        ["python", "a.py"],
        gpu,
        memory_gb,
        [("out", "/work/o", "json", "file")],
        {},
        [],
        [],
        "",
        {},
        [],
        [],
    )
    assert len(be.created) == 1
    return be.created[0]


def test_job_spec_labels_and_ttl() -> None:
    job = _job(gpu=1)
    assert job["kind"] == "Job"
    assert job["spec"]["ttlSecondsAfterFinished"] == 86400
    assert job["spec"]["backoffLimit"] == 0
    labels = job["metadata"]["labels"]
    assert labels["app.kubernetes.io/managed-by"] == "npa"
    assert labels["app.kubernetes.io/part-of"] == "npa-workbench"


def test_node_selector_only_when_gpu_requested() -> None:
    spec_gpu = _job(gpu=1)["spec"]["template"]["spec"]
    assert spec_gpu["nodeSelector"] == {"nvidia.com/gpu.present": "true"}
    assert spec_gpu["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    spec_cpu = _job(gpu=0)["spec"]["template"]["spec"]
    assert "nodeSelector" not in spec_cpu
    assert "resources" not in spec_cpu["containers"][0]


def test_memory_limits_rendered() -> None:
    spec = _job(gpu=1, memory_gb=16)["spec"]["template"]["spec"]
    res = spec["containers"][0]["resources"]
    assert res["limits"]["memory"] == "16Gi"
    assert res["requests"]["memory"] == "16Gi"


def test_kube_context_flag() -> None:
    be = _FakeKc(kube_context="mk8s-prod")
    be._create_configmap("cm-x", {})
    args, _ = be.kc_calls[0]
    assert "--context" in args and "mk8s-prod" in args


def test_parse_s3_markers() -> None:
    be = NebiusBackend()
    logs = (
        "downloading\n"
        "@@S3IN:/work/config.json@@\n"
        "@@ARTIFACT:summary:eyJhIjogMX0=@@\n"
        "@@S3:matrix:s3://bucket/key/result.npy@@\n"
        "@@EXIT:0@@\n"
    )
    res = be._parse(logs, 1.0)
    assert res.artifacts_raw["matrix"] == "s3://bucket/key/result.npy"
    assert res.artifacts_raw["summary"] == '{"a": 1}'
    assert "@@S3:" not in res.stdout
    assert res.exit_code == 0


def test_configmap_returns_per_run_payload_map() -> None:
    be = _FakeKc()
    m1 = be._create_configmap("cm-1", {"/work/a.py": "AAA"})
    m2 = be._create_configmap("cm-2", {"/work/b.py": "BBB"})
    assert m1 != m2  # no shared instance state between runs
    assert m1 == {"/work/a.py": "a.py"}


# ---------------------------------------------------------------------------
# Round 3: environment.pip for real workload dependencies.
# ---------------------------------------------------------------------------


def test_environment_pip_parses() -> None:
    d = _desc_dict()
    d["environment"] = {"pip": ["newton", "matplotlib"]}
    desc = parse_descriptor(d)
    assert desc.environment.pip == ("newton", "matplotlib")


def test_environment_pip_must_be_list_of_strings() -> None:
    d = _desc_dict()
    d["environment"] = {"pip": "newton"}
    with pytest.raises(DescriptorError, match="environment.pip"):
        parse_descriptor(d)
    d["environment"] = {"pip": [123]}
    with pytest.raises(DescriptorError, match="environment.pip"):
        parse_descriptor(d)


def test_environment_defaults_empty() -> None:
    desc = parse_descriptor(_desc_dict())
    assert desc.environment.pip == ()


def test_pip_packages_reach_backend(tmp_path: Path) -> None:
    rt, stub = make_runtime(tmp_path)
    d = _desc_dict()
    d["environment"] = {"pip": ["newton"]}
    rt.catalog._entries["piptest@0.1.0"] = parse_descriptor(d)
    rt.invoke("piptest", "0.1.0", "run", {}, backend="stub")
    assert stub.calls[0]["pip_packages"] == ["newton"]


def test_manifest_pip_env_in_job_spec() -> None:
    be = _FakeKc()
    be._create_job(
        "job-1",
        "cm-1",
        "img@sha256:abc",
        ["python", "a.py"],
        1,
        0,
        [("out", "/work/o", "json", "file")],
        {},
        [],
        [],
        "",
        {},
        ["newton", "matplotlib"],
        [],
    )
    job = be.created[0]
    env = {
        e["name"]: e["value"]
        for e in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert json.loads(env["MANIFEST_PIP"]) == ["newton", "matplotlib"]


def test_parse_pip_markers_stripped() -> None:
    be = NebiusBackend()
    logs = "@@PIP:newton@@\ninstalling\n@@EXIT:0@@\n"
    res = be._parse(logs, 1.0)
    assert "@@PIP:" not in res.stdout
    assert res.exit_code == 0


def test_parse_pipfail_marks_failure() -> None:
    be = NebiusBackend()
    logs = "@@PIP:newton@@\n@@PIPFAIL:newton@@\n"
    res = be._parse(logs, 1.0)
    assert res.exit_code == 3


def test_pendulum_viz_descriptor_validates(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("pendulum-viz", "0.1.0")
    assert set(desc.environment.pip) == {
        "newton",
        "matplotlib",
        "imageio-ffmpeg",
    }
    assert desc.artifact_store is not None
    assert "pendulum-viz@0.1.0" in rt.catalog.list()


def test_openvla_predict_descriptor_validates(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("openvla-predict", "0.1.0")
    assert "transformers==4.40.1" in desc.environment.pip
    assert desc.resources.gpu == 1
    assert desc.resources.memory_gb == 32
    assert desc.artifact_store is not None
    assert desc.commands["run"].outputs["action"].format == "binary"


def test_pendulum_rtx_descriptor_validates(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("pendulum-rtx", "0.1.0")
    assert desc.environment.pip == ("newton",)
    # Multi-file payload: shared sim module + workload.
    cpaths = [c for c, _ in desc.payload_files]
    assert cpaths == ["/work/pendulum_sim.py", "/work/pendulum_rtx.py"]
    assert desc.commands["run"].outputs["video"].format == "binary"


def test_wait_done_returns_on_failed_condition() -> None:
    # Regression: a failed job never gains condition=complete; the wait
    # must detect the Failed condition promptly, not burn the timeout.
    # (kubectl wait with multiple --for flags does not OR them.)
    calls = {"n": 0}

    class _StatusKc(_FakeKc):
        def _kc(self, *args, input_text=None, timeout=120):
            calls["n"] += 1
            conds = [{"type": "Failed", "status": "True"}] if calls["n"] >= 2 else []
            import json as _json

            self.kc_calls.append((args, input_text))
            return subprocess.CompletedProcess(
                args=list(args),
                returncode=0,
                stdout=_json.dumps({"status": {"conditions": conds}}),
                stderr="",
            )

    be = _StatusKc(timeout_s=3600)
    be._wait_done("job-1")
    assert calls["n"] == 2  # polled, then saw Failed


def test_configmap_keys_preserve_basenames_and_dedupe() -> None:
    be = _FakeKc()
    m = be._create_configmap("cm-1", {"/work/a.py": "AAA", "/other/a.py": "BBB"})
    assert m == {"/work/a.py": "a.py", "/other/a.py": "a.py.1"}


def test_manifest_s3_bucket_env_override(tmp_path: Path, monkeypatch) -> None:
    # Committed descriptors name a placeholder bucket; the operator's live
    # bucket arrives via env and applies to the store prefix and input URIs.
    monkeypatch.setenv("MANIFEST_S3_BUCKET", "live-bucket")
    rt, stub = make_runtime(tmp_path)
    rt.invoke("large-artifact", "0.1.0", "run", {"size": 1024}, backend="stub")
    call = stub.calls[0]
    assert call["s3_prefix"].startswith("s3://live-bucket/manifest-mvp/")
    assert call["s3_inputs"] == [
        (
            "s3://live-bucket/manifest-mvp/inputs/large_artifact_config.json",
            "/work/config.json",
        )
    ]


def test_environment_apt_parses_and_reaches_backend(tmp_path: Path) -> None:
    d = _desc_dict()
    d["environment"] = {"apt": ["libx11-6"]}
    desc = parse_descriptor(d)
    assert desc.environment.apt == ("libx11-6",)
    rt, stub = make_runtime(tmp_path)
    rt.catalog._entries["apttest@0.1.0"] = desc
    rt.invoke("apttest", "0.1.0", "run", {}, backend="stub")
    assert stub.calls[0]["apt_packages"] == ["libx11-6"]


def test_manifest_apt_env_in_job_spec() -> None:
    be = _FakeKc()
    be._create_job(
        "job-1",
        "cm-1",
        "img@sha256:abc",
        ["python", "a.py"],
        1,
        0,
        [("out", "/work/o", "json", "file")],
        {},
        [],
        [],
        "",
        {},
        [],
        ["libx11-6"],
    )
    job = be.created[0]
    env = {
        e["name"]: e["value"]
        for e in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert json.loads(env["MANIFEST_APT"]) == ["libx11-6"]


def test_parse_apt_markers() -> None:
    be = NebiusBackend()
    res = be._parse("@@APT:libx11-6@@\nok\n@@EXIT:0@@\n", 1.0)
    assert "@@APT:" not in res.stdout and res.exit_code == 0
    res = be._parse("@@APTFAIL@@\n", 1.0)
    assert res.exit_code == 3


def test_pendulum_rtx_has_apt_and_blender_input(tmp_path: Path) -> None:
    rt, _ = make_runtime(tmp_path)
    desc = rt.catalog.get("pendulum-rtx", "0.1.0")
    assert "libx11-6" in desc.environment.apt
    assert desc.inputs[0].container_path == "/work/blender.tar.xz"
