from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows import sim2real_health as health
from npa.workflows.sim2real_health import (
    DoctorProbes,
    KubeResult,
    check_cluster,
    check_config,
    check_coherence,
    check_registry,
    check_s3,
    check_tokens,
    coherence_failures,
    run_preflight,
)
from npa.workflows.sim2real_loop import build_config_from_env

REPO_ROOT = Path(__file__).resolve().parents[3]


class _Creds:
    def __init__(self, *, hf="", ngc="", ak="", sk=""):
        self.hf_token = hf
        self.ngc_api_key = ngc
        self.s3_access_key_id = ak
        self.s3_secret_access_key = sk


def _config(**overrides):
    return build_config_from_env(run_id="health-test", **overrides)


def test_coherence_passes_on_real_repo() -> None:
    assert coherence_failures(REPO_ROOT) == []
    assert check_coherence(REPO_ROOT).status == health.PASS


def test_sdk_accepts_every_seam_as_config_field() -> None:
    field_names = {f for f in vars(_config()).keys()}
    for seam in health.SIM2REAL_SEAMS:
        assert seam.config_field in field_names
        # build_config_from_env applies the override keyed by config-field name.
        applied = build_config_from_env(**{seam.config_field: getattr(_config(), seam.config_field)})
        assert hasattr(applied, seam.config_field)


def test_config_fails_without_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("NPA_S3_BUCKET", "NPA_SIM2REAL_BUCKET", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    result = check_config(_config(s3_bucket=""))
    assert result.status == health.FAIL
    assert any("s3_bucket" in d for d in result.details)


def test_empty_bucket_override_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: `npa workbench health sim2real` passes --s3-bucket "" (the flag
    # default), which must NOT clobber NPA_SIM2REAL_BUCKET / S3_BUCKET from the env.
    for key in ("NPA_S3_BUCKET", "NPA_SIM2REAL_BUCKET", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "env-bucket")
    config = build_config_from_env(run_id="health-test", s3_bucket="")
    assert config.s3_bucket == "env-bucket"
    assert check_config(config).status in {health.PASS, health.WARN}

    monkeypatch.delenv("NPA_SIM2REAL_BUCKET", raising=False)
    monkeypatch.setenv("S3_BUCKET", "alias-bucket")
    assert build_config_from_env(run_id="health-test", s3_bucket="").s3_bucket == "alias-bucket"


def test_explicit_bucket_override_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "env-bucket")
    config = build_config_from_env(run_id="health-test", s3_bucket="explicit-bucket")
    assert config.s3_bucket == "explicit-bucket"


def test_config_warns_on_derived_optional_seams() -> None:
    result = check_config(_config(s3_bucket="real-bucket", trigger_dataset_uri="", assets_uri="", scene_spec_uri=""))
    assert result.status == health.WARN


def test_config_passes_when_seams_resolve() -> None:
    result = check_config(
        _config(
            s3_bucket="real-bucket",
            trigger_dataset_uri="s3://real-bucket/trig/",
            assets_uri="s3://real-bucket/assets/",
        )
    )
    assert result.status == health.PASS


def test_config_fails_on_invalid_schema() -> None:
    result = check_config(_config(s3_bucket="real-bucket", threshold=5.0))
    assert result.status == health.FAIL


def test_s3_skips_without_endpoint_or_creds() -> None:
    probes = DoctorProbes(credentials=_Creds())
    assert check_s3(_config(s3_bucket="b", s3_endpoint=""), probes=probes).status == health.SKIP

    probes_no_keys = DoctorProbes(credentials=_Creds())
    res = check_s3(_config(s3_bucket="b", s3_endpoint="https://endpoint.example"), probes=probes_no_keys)
    assert res.status == health.SKIP


def test_s3_pass_and_fail_with_injected_client() -> None:
    class _OkClient:
        def list_checkpoints(self, uri):
            return []

    class _BadClient:
        def list_checkpoints(self, uri):
            raise RuntimeError("NoSuchBucket")

    creds = _Creds(ak="a", sk="s")
    ok = check_s3(
        _config(s3_bucket="b", s3_endpoint="https://endpoint.example"),
        probes=DoctorProbes(credentials=creds, s3_client_factory=_OkClient),
    )
    assert ok.status == health.PASS

    bad = check_s3(
        _config(s3_bucket="b", s3_endpoint="https://endpoint.example"),
        probes=DoctorProbes(credentials=creds, s3_client_factory=_BadClient),
    )
    assert bad.status == health.FAIL
    assert "NoSuchBucket" in " ".join(bad.details)


def test_registry_warns_on_unqualified_images() -> None:
    # Default reference images are bare npa-* names, not registry-qualified.
    result = check_registry(_config(), probes=DoctorProbes(image_inspector=lambda i: True))
    assert result.status == health.WARN


def test_registry_inspects_qualified_images() -> None:
    qualified = {
        "augment_image": "reg.example/npa-envgen:0.1.2",
        "policy_image": "reg.example/npa-reference-policy:0.1.2",
        "trainer_image": "reg.example/npa-lerobot-vlm-rl:0.1.1",
        "vlm_image": "reg.example/npa-cosmos3-reason:3.0.1",
        "eval_image": "reg.example/npa-loop-eval:0.1.2",
    }
    cfg = _config(**qualified)
    ok = check_registry(cfg, probes=DoctorProbes(image_inspector=lambda i: True))
    assert ok.status == health.PASS

    missing = check_registry(
        cfg, probes=DoctorProbes(image_inspector=lambda i: i.endswith("0.1.1"))
    )
    assert missing.status == health.FAIL

    no_tool = check_registry(cfg, probes=DoctorProbes(image_inspector=lambda i: None))
    assert no_tool.status == health.SKIP


def test_tokens_warns_when_missing_and_passes_when_present() -> None:
    warn = check_tokens(_config(), probes=DoctorProbes(credentials=_Creds()))
    assert warn.status == health.WARN
    ok = check_tokens(_config(), probes=DoctorProbes(credentials=_Creds(hf="hf_x", ngc="nv_x")))
    assert ok.status == health.PASS


def _kube_nodes(gpu_count: int, nodes: int = 1) -> str:
    items = [
        {"status": {"allocatable": {"nvidia.com/gpu": str(gpu_count)}}}
        for _ in range(nodes)
    ]
    return json.dumps({"items": items})


def test_cluster_skips_without_runner() -> None:
    assert check_cluster(_config(), probes=DoctorProbes()).status == health.SKIP


def test_cluster_pass_counts_schedulable_gpus() -> None:
    def runner(args):
        if args[:2] == ["config", "current-context"]:
            return KubeResult(0, "prod-cluster")
        if args[:3] == ["auth", "can-i", "create"]:
            return KubeResult(0, "yes")
        if args[:2] == ["get", "nodes"]:
            return KubeResult(0, _kube_nodes(8, nodes=2))
        return KubeResult(1, "", "unexpected")

    result = check_cluster(_config(), probes=DoctorProbes(kube_runner=runner))
    assert result.status == health.PASS
    assert "16 schedulable" in result.summary


def test_cluster_fails_on_zero_gpus() -> None:
    def runner(args):
        if args[:2] == ["config", "current-context"]:
            return KubeResult(0, "prod-cluster")
        if args[:3] == ["auth", "can-i", "create"]:
            return KubeResult(0, "yes")
        if args[:2] == ["get", "nodes"]:
            return KubeResult(0, _kube_nodes(0, nodes=3))
        return KubeResult(1, "", "x")

    result = check_cluster(_config(), probes=DoctorProbes(kube_runner=runner))
    assert result.status == health.FAIL
    assert "0 schedulable" in result.summary


def test_cluster_fails_on_unpinned_context() -> None:
    def runner(args):
        if args[:2] == ["config", "current-context"]:
            return KubeResult(1, "", "error: current-context is not set")
        return KubeResult(0, "yes")

    result = check_cluster(_config(), probes=DoctorProbes(kube_runner=runner))
    assert result.status == health.FAIL
    assert "context" in result.summary.lower()


def test_cluster_fails_without_pod_permission() -> None:
    def runner(args):
        if args[:2] == ["config", "current-context"]:
            return KubeResult(0, "prod-cluster")
        if args[:3] == ["auth", "can-i", "create"]:
            return KubeResult(1, "no")
        return KubeResult(0, "")

    result = check_cluster(_config(), probes=DoctorProbes(kube_runner=runner))
    assert result.status == health.FAIL


def test_run_preflight_selects_requested_checks() -> None:
    results = run_preflight(
        _config(s3_bucket="b"),
        repo_root=REPO_ROOT,
        probes=DoctorProbes(),
        checks=["config", "coherence"],
    )
    assert [r.name for r in results] == ["config", "three-tier-coherence"]


@pytest.mark.parametrize("gpu_resource", ["nvidia.com/gpu"])
def test_count_schedulable_handles_bad_json(gpu_resource: str) -> None:
    assert health._count_schedulable_gpus("not-json", gpu_resource) == (0, 0)
