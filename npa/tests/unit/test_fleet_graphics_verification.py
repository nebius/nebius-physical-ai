"""Exercise reusable Fleet CUDA and graphics qualification without a live cluster."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.cluster.gpu_health import GpuHealthError
from npa.fleet import graphics_verification as verification
from npa.fleet.spec import ClusterSpec, FleetSpec, NodePoolSpec, ProjectSpec


def _cluster(name="render", count=2):
    return ClusterSpec(
        name=name,
        cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="48vcpu-192gb"),
        gpu_nodes=NodePoolSpec(
            count=count,
            platform="gpu-rtx6000-a",
            preset="8gpu-192vcpu-1744gb",
        ),
        gpu_workload_profile="rtx-rendering",
    )


def _spec(clusters=None):
    return FleetSpec(
        name="graphics-test",
        tenant_id="tenant-test",
        region="test-region",
        profile="operator-test",
        projects=[
            ProjectSpec(
                name="team",
                project_id="project-test",
                clusters=clusters or [_cluster()],
            )
        ],
    )


def _identity():
    evidence = json.dumps({"private_project": "never-publish"})
    return SimpleNamespace(
        kubeconfig=Path("/private/kubeconfig"),
        evidence_sha256="a" * 64,
        evidence_json=evidence,
    )


def _healthy(config):
    cuda = [{"vectoradd": "passed"} for _ in range(config.expected_gpu_nodes)]
    graphics = [
        {"glx": "loaded", "egl": "loaded", "nvidia_device": "enumerated"}
        for _ in range(config.expected_gpu_nodes)
    ]
    return {"status": "healthy", "cuda_smokes": cuda, "graphics_smokes": graphics}


def _prepare(monkeypatch):
    observed = []
    monkeypatch.setattr(
        verification, "resolve_fleet_identity", lambda *args, **kwargs: _identity()
    )
    monkeypatch.setattr(verification, "require_bin", lambda name: name)

    def validate(*args, **kwargs):
        observed.append(kwargs["config"])
        return _healthy(kwargs["config"])

    monkeypatch.setattr(verification, "validate_gpu_health", validate)
    return observed


def test_all_rtx_workers_pass_cuda_glx_egl_and_vulkan(tmp_path, monkeypatch):
    observed = _prepare(monkeypatch)
    report = verification.verify_graphics(_spec(), evidence_dir=tmp_path)
    assert report["passed"] is True
    assert report["verified_clusters"] == 1
    assert report["gpu_workers"] == 2
    assert report["gpus"] == 16
    assert report["cuda_workers"] == 2
    assert report["glx_workers"] == 2
    assert report["egl_workers"] == 2
    assert report["vulkan_workers"] == 2
    assert observed[0].graphics_smoke is True
    assert observed[0].cuda_smoke is True
    assert observed[0].driver_mode == "operator"


def test_public_report_omits_private_identity_and_receipts_are_owner_only(
    tmp_path, monkeypatch
):
    _prepare(monkeypatch)
    report = verification.verify_graphics(_spec(), evidence_dir=tmp_path)
    assert "never-publish" not in json.dumps(report)
    receipts = list(tmp_path.rglob("target-*.json"))
    assert len(receipts) == 1
    assert receipts[0].stat().st_mode & 0o077 == 0
    assert "never-publish" in receipts[0].read_text()


def test_invalid_target_fails_without_running_health(tmp_path, monkeypatch):
    cluster = _cluster()
    cluster.gpu_workload_profile = ""
    observed = _prepare(monkeypatch)
    report = verification.verify_graphics(_spec([cluster]), evidence_dir=tmp_path)
    assert report["passed"] is False
    assert report["failures"] == ["invalid_graphics_target"]
    assert observed == []


def test_gpu_failure_is_sanitized_and_preserved_privately(tmp_path, monkeypatch):
    _prepare(monkeypatch)
    failure = GpuHealthError(
        "graphics failed on private-node",
        evidence={"node": "private-node"},
    )
    monkeypatch.setattr(
        verification,
        "validate_gpu_health",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    report = verification.verify_graphics(_spec(), evidence_dir=tmp_path)
    assert report["failures"] == ["graphics_validation_failed"]
    assert "private-node" not in json.dumps(report)
    assert "private-node" in next(tmp_path.rglob("target-*.json")).read_text()


def test_parallel_results_preserve_declaration_order(tmp_path, monkeypatch):
    observed = _prepare(monkeypatch)
    clusters = [_cluster("one", 1), _cluster("two", 2), _cluster("three", 3)]
    report = verification.verify_graphics(
        _spec(clusters),
        evidence_dir=tmp_path,
        concurrency=3,
    )
    assert [item["target_index"] for item in report["clusters"]] == [0, 1, 2]
    assert report["gpu_workers"] == 6
    assert sorted(config.expected_gpu_nodes for config in observed) == [1, 2, 3]


@pytest.mark.parametrize(
    "options",
    [{"concurrency": 0}, {"stabilization_seconds": -1}, {"timeout_minutes": 0}],
)
def test_invalid_options_fail_before_target_access(tmp_path, options):
    with pytest.raises(verification.GraphicsVerificationError):
        verification.verify_graphics(_spec(), evidence_dir=tmp_path, **options)
