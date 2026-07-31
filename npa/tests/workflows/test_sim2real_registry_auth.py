"""Tests for Nebius registry pull-secret refresh."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.registry_auth import (
    docker_config_json,
    ensure_nebius_registry_pull_secret,
    mint_nebius_registry_token,
)


def test_mint_nebius_registry_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.subprocess.run",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="token-abc\n", stderr=""),
    )
    assert mint_nebius_registry_token() == "token-abc"


def test_mint_nebius_registry_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-pod contexts may have the token but not the ``nebius`` CLI."""
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "env-token")

    def _fail_run(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("nebius CLI must not be invoked when env token is set")

    monkeypatch.setattr("npa.workflows.sim2real.registry_auth.subprocess.run", _fail_run)
    assert mint_nebius_registry_token() == "env-token"


def test_sibling_refresh_is_best_effort_on_mint_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing in-pod ``nebius`` CLI must not crash the orchestrator."""
    from npa.workflows.sim2real import engine

    def _raise(*images, **kwargs):
        raise RuntimeError("Could not mint Nebius registry token")

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_registry_pull_secret_for_images",
        _raise,
    )
    config = Sim2RealLoopConfig(run_id="run-best-effort", k8s_context="ctx")
    # Must not raise.
    engine._refresh_registry_pull_secret_for_sibling_job(
        "cr.us-central1.nebius.cloud/reg/npa-lerobot-vlm-rl:1.0",
        config=config,
        namespace="default",
    )


def test_apply_secret_reports_missing_kubectl_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-pod orchestrator often has no kubectl; that must stay in contract.

    Callers treat this refresh as best-effort and catch ``RuntimeError``, so a
    bare ``FileNotFoundError`` from ``subprocess.run`` would escape them and abort
    the run instead of degrading to an ImagePullBackOff on the sibling.
    """

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )

    def _no_kubectl(cmd, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "kubectl")

    monkeypatch.setattr("npa.workflows.sim2real.registry_auth.subprocess.run", _no_kubectl)
    with pytest.raises(RuntimeError, match="registry pull secret"):
        ensure_nebius_registry_pull_secret(registry_server="cr.eu-north1.nebius.cloud")


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(2, "No such file or directory", "kubectl"), ValueError("boom")],
)
def test_sibling_refresh_never_aborts_the_run(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """The refresh is advertised as best-effort, so no failure type may propagate."""

    from npa.workflows.sim2real import engine

    def _raise(*images, **kwargs):
        raise error

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_registry_pull_secret_for_images",
        _raise,
    )
    config = Sim2RealLoopConfig(run_id="run-best-effort-any", k8s_context="ctx")
    engine._refresh_registry_pull_secret_for_sibling_job(
        "cr.us-central1.nebius.cloud/reg/npa-lerobot-vlm-rl:1.0",
        config=config,
        namespace="default",
    )


def test_docker_config_json_uses_iam_username() -> None:
    payload = docker_config_json(registry_server="cr.eu-north1.nebius.cloud", token="tok")
    entry = payload["auths"]["cr.eu-north1.nebius.cloud"]
    assert entry["username"] == "iam"
    assert entry["password"] == "tok"


def test_ensure_nebius_registry_pull_secret_applies_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )
    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured["input"] = kwargs.get("input", "")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("npa.workflows.sim2real.registry_auth.subprocess.run", fake_run)
    ensure_nebius_registry_pull_secret(
        registry_server="cr.eu-north1.nebius.cloud",
        k8s_context="demo-context",
    )
    payload = json.loads(captured["input"])
    assert payload["metadata"]["name"] == "npa-nebius-registry"


def test_refresh_registry_pull_secret_helper_forwards_k8s_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workflows.sim2real import engine

    captured: dict[str, object] = {}

    def fake_ensure(*images, **kwargs):
        captured["images"] = images
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_registry_pull_secret_for_images",
        fake_ensure,
    )
    config = Sim2RealLoopConfig(
        run_id="run-registry-helper",
        k8s_namespace="sim2real",
        k8s_kubeconfig="/tmp/kubeconfig",
        k8s_context="npa-rtxpro-mk8s",
    )
    engine._refresh_registry_pull_secret_for_sibling_job(
        "cr.eu-north1.nebius.cloud/reg/npa-lerobot-vlm-rl:1.0",
        config=config,
        namespace="sim2real",
    )
    assert captured["images"] == (
        "cr.eu-north1.nebius.cloud/reg/npa-lerobot-vlm-rl:1.0",
    )
    assert captured["kwargs"] == {
        "namespace": "sim2real",
        "kubeconfig": "/tmp/kubeconfig",
        "k8s_context": "npa-rtxpro-mk8s",
    }


def test_sibling_kubernetes_job_refreshes_registry_pull_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Long Sim2Real runs must re-mint pull secrets before each sibling Job."""
    from npa.workflows.sim2real import engine

    refresh_calls: list[tuple] = []

    def fake_refresh(image, *, config, namespace):
        refresh_calls.append((image, config.k8s_context, namespace))

    monkeypatch.setattr(engine, "_refresh_registry_pull_secret_for_sibling_job", fake_refresh)
    monkeypatch.setattr(engine, "_ensure_sibling_source_env", lambda config, env: env)
    monkeypatch.setattr(
        engine,
        "_component_job_manifest",
        lambda *args, **kwargs: {"kind": "Job", "metadata": {"name": "job"}},
    )
    monkeypatch.setattr(
        engine,
        "_kubectl",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(engine, "_log_sibling_job_applied", lambda *args, **kwargs: "uid")
    monkeypatch.setattr(engine, "_wait_kubernetes_job", lambda *args, **kwargs: "complete")
    monkeypatch.setattr(
        engine,
        "_component_pod_info",
        lambda *args, **kwargs: {"image_digests": []},
    )
    monkeypatch.setattr(engine, "_cleanup_component_job", lambda *args, **kwargs: MagicMock(stdout="", stderr=""))
    monkeypatch.setattr(engine, "_download_component_output", lambda *args, **kwargs: None)

    config = Sim2RealLoopConfig(
        run_id="run-registry-refresh",
        k8s_namespace="sim2real",
        k8s_kubeconfig="/tmp/kubeconfig",
        k8s_context="npa-rtxpro-mk8s",
    )
    output_json = tmp_path / "out.json"
    output_json.write_text("{}", encoding="utf-8")

    engine._run_kubernetes_image_component(
        "cr.eu-north1.nebius.cloud/reg/npa-lerobot-vlm-rl:1.0",
        component="train",
        env={},
        output_json=output_json,
        output_uri="s3://bucket/out.json",
        config=config,
        timeout_s=30,
    )

    assert refresh_calls == [
        (
            "cr.eu-north1.nebius.cloud/reg/npa-lerobot-vlm-rl:1.0",
            "npa-rtxpro-mk8s",
            "sim2real",
        )
    ]


def test_indexed_sibling_kubernetes_job_refreshes_registry_pull_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workflows.sim2real import engine

    refresh_calls: list[tuple] = []

    def fake_refresh(image, *, config, namespace):
        refresh_calls.append((image, config.k8s_context, namespace))

    monkeypatch.setattr(engine, "_refresh_registry_pull_secret_for_sibling_job", fake_refresh)
    monkeypatch.setattr(engine, "_ensure_sibling_source_env", lambda config, env: env)
    monkeypatch.setattr(
        engine,
        "_indexed_component_job_manifest",
        lambda *args, **kwargs: {"kind": "Job", "metadata": {"name": "job"}},
    )
    monkeypatch.setattr(
        engine,
        "_kubectl",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(engine, "_log_sibling_job_applied", lambda *args, **kwargs: "uid")
    monkeypatch.setattr(engine, "_wait_kubernetes_job", lambda *args, **kwargs: "complete")
    monkeypatch.setattr(
        engine,
        "_component_pod_info",
        lambda *args, **kwargs: {"image_digests": []},
    )
    monkeypatch.setattr(engine, "_cleanup_component_job", lambda *args, **kwargs: MagicMock(stdout="", stderr=""))

    config = Sim2RealLoopConfig(
        run_id="run-indexed-refresh",
        k8s_namespace="default",
        k8s_context="ctx",
    )
    engine._run_kubernetes_indexed_image_component(
        "cr.eu-north1.nebius.cloud/reg/npa-cosmos2-transfer:1.0",
        component="augment",
        env={},
        config=config,
        completions=2,
        parallelism=2,
        timeout_s=30,
    )
    assert refresh_calls == [
        (
            "cr.eu-north1.nebius.cloud/reg/npa-cosmos2-transfer:1.0",
            "ctx",
            "default",
        )
    ]
