"""Check the requested conditioning, isolated config bytes, and GPU contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from npa.sdk.workbench import seedvr2 as sdk
from npa.workbench.seedvr2 import artifacts, conditioning, hardware, runtime
from npa.workbench.seedvr2.hardware import validate_gpu
from npa.workbench.seedvr2.schemas import RestoreRequest, VideoArtifactRequest
from test_seedvr2 import (
    RUNTIME_IDENTITY,
    _fake_inference,
    _identity_environment,
    _restore,
    source_video as source_video,
)

B200_GPU = {
    **RUNTIME_IDENTITY["gpu"],
    "name": "NVIDIA B200",
    "compute_capability": "10.0",
    "memory_mib": "183359",
}


@pytest.fixture
def config_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "configs_3b").mkdir(parents=True)
    payload = (
        b"vae: {dtype: bfloat16, scaling_factor: 0.9152}\ndiffusion: {steps: 50}\n"
    )
    (source / "configs_3b/main.yaml").write_bytes(payload)
    (source / "configs_3b/extra.yaml").write_bytes(b"keep: identical\n")
    monkeypatch.setattr(
        conditioning, "SOURCE_MAIN_SHA256", hashlib.sha256(payload).hexdigest()
    )
    return source


@pytest.mark.parametrize("mode", ["sample", "posterior-mode"])
def test_owned_configuration_changes_only_sampling(config_source, tmp_path, mode):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = (config_source / "configs_3b/main.yaml").read_bytes()
    conditioning.prepare_configuration(config_source, workspace, mode)
    target = workspace / "configs_3b/main.yaml"
    assert not target.is_symlink()
    assert (
        target.stat().st_ino != (config_source / "configs_3b/main.yaml").stat().st_ino
    )
    expected = yaml.safe_load(original)
    if mode == "posterior-mode":
        expected["vae"]["use_sample"] = False
    else:
        assert target.read_bytes() == original
    assert yaml.safe_load(target.read_bytes()) == expected
    assert (workspace / "configs_3b/extra.yaml").read_bytes() == b"keep: identical\n"
    assert (config_source / "configs_3b/main.yaml").read_bytes() == original
    conditioning.verify_configuration(config_source, workspace, mode)


@pytest.mark.parametrize("mutation", ["changed", "extra", "symlink", "root-link"])
def test_changed_workspace_cannot_be_published(config_source, tmp_path, mutation):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    conditioning.prepare_configuration(config_source, workspace, "sample")
    config = workspace / "configs_3b"
    if mutation == "changed":
        (config / "main.yaml").write_text("vae: {use_sample: false}\n")
    elif mutation == "extra":
        (config / "injected.yaml").write_text("altered: true\n")
    elif mutation == "symlink":
        (config / "main.yaml").unlink()
        (config / "main.yaml").symlink_to(config_source / "configs_3b/main.yaml")
    else:
        config.rename(workspace / "retained")
        config.symlink_to(workspace / "retained", target_is_directory=True)
    with pytest.raises(ValueError):
        conditioning.verify_configuration(config_source, workspace, "sample")


def test_drifted_source_rejected_before_copy(config_source, tmp_path):
    (config_source / "configs_3b/main.yaml").write_text("vae: {}\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="hash mismatch"):
        conditioning.prepare_configuration(config_source, workspace, "sample")
    assert not (workspace / "configs_3b").exists()


def test_defaults_and_sdk_request_reach_shared_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "restore", lambda request: calls.append(request) or {})
    paths = dict(
        input_path="s3://example-bucket/input.mp4",
        output_path="s3://example-bucket/result/",
        run_id="contract",
    )
    sdk.restore(**paths)
    sdk.restore(**paths, conditioning_mode="posterior-mode", expected_gpu="B200")
    assert (calls[0].conditioning_mode, calls[0].expected_gpu) == ("sample", "H100")
    assert (calls[1].conditioning_mode, calls[1].expected_gpu) == (
        "posterior-mode",
        "B200",
    )
    assert calls[0].seed == calls[1].seed == 666


@pytest.mark.parametrize(
    "changes", [{"conditioning_mode": "mean"}, {"expected_gpu": "H200"}]
)
def test_unknown_contract_rejected(changes):
    with pytest.raises(ValidationError):
        RestoreRequest(input_path="x", output_path="y", run_id="contract", **changes)


@pytest.mark.parametrize(
    "expected,capability,memory", [("H100", "9.0", "81559"), ("B200", "10.0", "183359")]
)
def test_actual_gpu_contracts(expected, capability, memory):
    gpu = {
        **RUNTIME_IDENTITY["gpu"],
        "name": "NVIDIA " + expected,
        "compute_capability": capability,
        "memory_mib": memory,
    }
    validate_gpu(gpu, expected)
    for change in (
        {"count": "2"},
        {"mig_mode": "Enabled"},
        {"memory_mib": "20000"},
        {"name": "NVIDIA H200"},
        {"compute_capability": "8.9"},
        {"status": "unavailable"},
    ):
        with pytest.raises(ValueError, match="full-memory"):
            validate_gpu({**gpu, **change}, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("conditioning_mode", "posterior-mode"),
        ("effective_main_sha256", "0" * 64),
        ("vae_use_sample", False),
    ],
)
def test_reviewer_rejects_config_provenance_forgery(
    tmp_path, source_video, monkeypatch, field, value
):
    result, storage = _restore(tmp_path, source_video, monkeypatch)
    forged = copy.deepcopy(result)
    forged["configuration"][field] = value
    uri = result["artifacts"]["result"]
    storage.objects[uri] = json.dumps(forged).encode()
    with pytest.raises(runtime.SeedVR2Error, match="configuration identity"):
        artifacts.review(
            VideoArtifactRequest(
                input_path=uri,
                output_path="s3://example-bucket/review/",
                run_id=result["run_id"],
            ),
            storage_factory=lambda: storage,
        )


def test_posterior_mode_b200_reaches_restore_and_verifier(
    tmp_path, source_video, monkeypatch
):
    seen = []

    def runner(argv, **kwargs):
        document = yaml.safe_load(
            (Path(kwargs["cwd"]) / "configs_3b/main.yaml").read_text()
        )
        seen.append(document["vae"]["use_sample"])
        return _fake_inference(argv, **kwargs)

    result, storage = _restore(
        tmp_path,
        source_video,
        monkeypatch,
        controls={"expected_gpu": "B200", "conditioning_mode": "posterior-mode"},
        gpu=B200_GPU,
        runner=runner,
    )
    assert seen == [False]
    assert result["configuration"]["vae_use_sample"] is False
    assert result["runtime"]["gpu"] == B200_GPU
    verified = artifacts.verify(
        VideoArtifactRequest(
            input_path=result["artifacts"]["result"],
            output_path="s3://example-bucket/verified.json",
            run_id=result["run_id"],
        ),
        storage_factory=lambda: storage,
    )
    assert verified["verifier_runtime"]["gpu"] == B200_GPU


def test_post_inference_config_mutation_is_rejected(
    tmp_path, source_video, monkeypatch
):
    def runner(argv, **kwargs):
        completed = _fake_inference(argv, **kwargs)
        (Path(kwargs["cwd"]) / "configs_3b/main.yaml").write_text(
            "vae: {use_sample: false}\n"
        )
        return completed

    with pytest.raises(runtime.SeedVR2Error, match="private evidence"):
        _restore(tmp_path, source_video, monkeypatch, runner=runner)


@pytest.mark.parametrize(
    "case", ["valid", "missing", "hopper-only", "empty", "invalid"]
)
def test_b200_requires_actual_assigned_device_and_baked_inventory(
    tmp_path, monkeypatch, case
):
    _identity_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, "_gpu_inventory", lambda: B200_GPU)
    directory = tmp_path / "arches"
    directory.mkdir()
    monkeypatch.setattr(hardware, "ARCHES_ROOT", directory)
    report = {"extension.so": {"sass": ["sm_90", "sm_100"], "ptx": [], "bytes": 100}}
    if case == "hopper-only":
        report["extension.so"]["sass"] = ["sm_90"]
    if case == "empty":
        report = {}
    for name in ("flash-attn", "apex"):
        if case != "missing":
            (directory / (name + ".json")).write_text(
                "bad" if case == "invalid" else json.dumps(report)
            )
    if case == "valid":
        assert runtime._runtime_identity("B200")["gpu"] == B200_GPU
        with pytest.raises(runtime.SeedVR2Error, match="H100"):
            runtime._runtime_identity()
    else:
        with pytest.raises(runtime.SeedVR2Error, match="B200"):
            runtime._runtime_identity("B200")
