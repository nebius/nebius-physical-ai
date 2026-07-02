"""Unit tests for BYOF live infra resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.byof.live import (
    byof_onboard_skill_path,
    byof_ubuntu_validation_repo,
    byof_validation_repo,
    resolve_byof_kubernetes_target,
    resolve_byof_project,
    resolve_byof_resource_yaml,
)


def test_byof_validation_repo_defaults_to_leisaac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_BYOF_REPO_URL", raising=False)
    monkeypatch.delenv("NPA_BYOF_VALIDATION_REPO_URL", raising=False)
    url, ref = byof_validation_repo()
    assert "leisaac" in url
    assert ref == "main"


def test_byof_validation_repo_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_BYOF_REPO_URL", "https://github.com/example/demo.git")
    monkeypatch.setenv("NPA_BYOF_REPO_REF", "v1.0.0")
    url, ref = byof_validation_repo()
    assert url.endswith("demo.git")
    assert ref == "v1.0.0"


def test_resolve_byof_kubernetes_target_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_BYOF_K8S_CONTEXT", "customer-context")
    monkeypatch.setenv("NPA_BYOF_KUBECONFIG", "/tmp/customer-kubeconfig")
    target = resolve_byof_kubernetes_target("rtxpro")
    assert target.context == "customer-context"
    assert target.kubeconfig == "/tmp/customer-kubeconfig"


def test_resolve_byof_kubernetes_target_from_cluster_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NPA_BYOF_K8S_CONTEXT", raising=False)
    monkeypatch.delenv("NPA_BYOF_KUBECONFIG", raising=False)
    monkeypatch.delenv("KUBECONFIG", raising=False)
    cluster_dir = tmp_path / "clusters" / "customer-mk8s"
    cluster_dir.mkdir(parents=True)
    kubeconfig = cluster_dir / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    (cluster_dir / "cluster.json").write_text(
        (
            "{"
            '"name": "customer-mk8s",'
            '"cluster_id": "mk8s-test",'
            '"project_id": "project-test",'
            '"region": "us-central1",'
            '"node_count": 1,'
            '"node_platform": "cpu-d3",'
            '"node_preset": "4vcpu-16gb",'
            '"k8s_version": "1.33",'
            '"subnet_id": "subnet-test",'
            '"created_at": "2026-01-01T00:00:00Z",'
            f'"kubeconfig_path": "{kubeconfig}"'
            "}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NPA_BYOF_CLUSTER_NAME", "customer-mk8s")
    monkeypatch.setattr("npa.cluster.state.CLUSTERS_DIR", tmp_path / "clusters", raising=False)
    target = resolve_byof_kubernetes_target("rtxpro")
    assert target.context == "customer-mk8s"
    assert target.kubeconfig == str(kubeconfig)


def test_resolve_byof_resource_yaml_rtxpro_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_BYOF_RESOURCE_YAML", raising=False)

    def _fake_block(_project: str | None) -> dict[str, object]:
        return {"gpu_profile": "rtxpro"}

    monkeypatch.setattr("npa.workflows.byof.live._project_kubernetes_block", _fake_block)
    path = resolve_byof_resource_yaml("rtxpro", smoke=True)
    assert path.endswith("isaac-lab-rl-train-rtxpro-smoke.yaml")


def test_resolve_byof_resource_yaml_datagen_rtxpro_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_BYOF_RESOURCE_YAML", raising=False)

    def _fake_block(_project: str | None) -> dict[str, object]:
        return {"gpu_profile": "rtxpro"}

    monkeypatch.setattr("npa.workflows.byof.live._project_kubernetes_block", _fake_block)
    path = resolve_byof_resource_yaml("rtxpro", smoke=True, workload="datagen")
    assert path.endswith("byof-datagen-rtxpro-smoke.yaml")


def test_resolve_byof_resource_yaml_container_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_BYOF_RESOURCE_YAML", raising=False)
    path = resolve_byof_resource_yaml("rtxpro", smoke=True, workload="container-verify")
    assert path.endswith("byof-container-smoke-rtxpro.yaml")


def test_byof_ubuntu_validation_repo_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_BYOF_REPO_URL", raising=False)
    monkeypatch.delenv("NPA_BYOF_UBUNTU_VALIDATION_REPO_URL", raising=False)
    url, ref = byof_ubuntu_validation_repo()
    assert "hellogitworld" in url
    assert ref == "master"


def test_byof_onboard_skill_path() -> None:
    from npa.workflows.byof.live import byof_onboard_skill_path, load_byof_onboard_skill_text

    assert byof_onboard_skill_path() == "skills/workflows/byof-onboard/SKILL.md"
    assert "run_byof_repo.py" in load_byof_onboard_skill_text()


def test_resolve_byof_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_E2E_PROJECT", "rtxpro")
    assert resolve_byof_project() == "rtxpro"
