"""Tests for Nebius registry pull-secret refresh."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.k8s_client import JobSnapshot
from npa.workflows.sim2real.registry_auth import (
    docker_config_json,
    ensure_nebius_registry_pull_secret,
    mint_nebius_registry_token,
)


def test_mint_nebius_registry_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Delegated to the canonical npa.clients.nebius_auth helper; with no ambient
    # token the profile-scoped CLI exchange is used.
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.setattr(
        "npa.clients.nebius_auth.subprocess.run",
        lambda *args, **kwargs: MagicMock(
            returncode=0, stdout="token-abc\n", stderr=""
        ),
    )
    assert mint_nebius_registry_token() == "token-abc"


def test_mint_nebius_registry_token_falls_back_to_env_without_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-pod contexts may have the token injected but not the ``nebius`` CLI.

    The canonical helper tries a fresh profile-scoped exchange first (so a stale
    token can't poison the pull secret on operator VMs) and falls back to the
    injected ``NEBIUS_IAM_TOKEN`` when the CLI is unavailable — the in-pod case.
    """
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "env-token")

    def _no_cli(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "nebius")

    monkeypatch.setattr("npa.clients.nebius_auth.subprocess.run", _no_cli)
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


def test_apply_secret_reports_structured_client_failure_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core-API setup failures stay inside the registry-refresh contract."""

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth._docker_helper_credential",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )
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
    payload = docker_config_json(
        registry_servers=["cr.eu-north1.nebius.cloud"], token="tok"
    )
    entry = payload["auths"]["cr.eu-north1.nebius.cloud"]
    assert entry["username"] == "iam"
    assert entry["password"] == "tok"


def test_ensure_nebius_registry_pull_secret_applies_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN_FILE", raising=False)
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth._docker_helper_credential",
        lambda *args, **kwargs: None,
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def apply_secret(self, payload):
            captured["payload"] = payload

    def fake_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return FakeClient()

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        fake_client,
    )
    ensure_nebius_registry_pull_secret(
        registry_server="cr.eu-north1.nebius.cloud",
        k8s_context="demo-context",
    )
    payload = captured["payload"]
    assert payload["metadata"]["name"] == "npa-nebius-registry"
    assert captured["client_kwargs"] == {
        "namespace": "default",
        "kubeconfig": "",
        "context": "demo-context",
        "bearer_token": "",
    }


@pytest.mark.parametrize("ambient_kind", ["env", "file"])
def test_stale_ambient_token_is_replaced_for_exec_mk8s_kubeconfig_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ambient_kind: str,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(
        """
current-context: other-context
contexts:
- name: other-context
  context: {user: cert-user}
- name: fresh-context
  context: {user: mk8s-user}
users:
- name: cert-user
  user: {token: static-token}
- name: mk8s-user
  user:
    exec:
      command: nebius
      args: [iam, get-access-token]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN_FILE", raising=False)
    if ambient_kind == "env":
        monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-token")
    else:
        token_file = tmp_path / "token"
        token_file.write_text("stale-token", encoding="utf-8")
        monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth._docker_helper_credential",
        lambda *args, **kwargs: None,
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def apply_secret(self, payload):
            captured["payload"] = payload

    def fake_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return FakeClient()

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        fake_client,
    )
    ensure_nebius_registry_pull_secret(
        registry_server="cr.us-central1.nebius.cloud",
        kubeconfig=str(kubeconfig),
        k8s_context="fresh-context",
    )
    assert captured["client_kwargs"] == {
        "namespace": "default",
        "kubeconfig": str(kubeconfig),
        "context": "fresh-context",
        "bearer_token": "fresh-token",
    }


@pytest.mark.parametrize(
    "static_user",
    [
        {"token": "configured-static-token"},
        {
            "client-certificate-data": "certificate",
            "client-key-data": "private-key",
        },
    ],
)
@pytest.mark.parametrize("ambient_kind", ["env", "file"])
def test_ambient_token_does_not_override_or_mint_for_static_kubeconfig_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    static_user: dict[str, str],
    ambient_kind: str,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(
        json.dumps(
            {
                "current-context": "static-context",
                "contexts": [
                    {"name": "static-context", "context": {"user": "static-user"}}
                ],
                "users": [{"name": "static-user", "user": static_user}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN_FILE", raising=False)
    if ambient_kind == "env":
        monkeypatch.setenv("NEBIUS_IAM_TOKEN", "ambient-token")
    else:
        token_file = tmp_path / "ambient-token"
        token_file.write_text("ambient-token", encoding="utf-8")
        monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth._docker_helper_credential",
        lambda *args, **kwargs: ("registry-user", "registry-token"),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: pytest.fail(
            "static kubeconfig auth must not invoke the Nebius CLI token path"
        ),
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def apply_secret(self, payload):
            captured["payload"] = payload

    def fake_client(**kwargs):
        captured["client_kwargs"] = kwargs
        return FakeClient()

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        fake_client,
    )
    ensure_nebius_registry_pull_secret(
        registry_server="cr.us-central1.nebius.cloud",
        kubeconfig=str(kubeconfig),
        k8s_context="static-context",
    )

    assert captured["client_kwargs"] == {
        "namespace": "default",
        "kubeconfig": str(kubeconfig),
        "context": "static-context",
        "bearer_token": "",
    }


def test_ensure_materializes_configured_docker_credential_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN_FILE", raising=False)
    docker_config = tmp_path / "docker"
    docker_config.mkdir()
    (docker_config / "config.json").write_text(
        json.dumps(
            {
                "auths": {"cr.eu-north1.nebius.cloud": {}},
                "credHelpers": {"cr.eu-north1.nebius.cloud": "nebius-agent-sa"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: pytest.fail("configured Docker helper must be preferred"),
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "docker-credential-nebius-agent-sa":
            assert kwargs["input"] == "cr.eu-north1.nebius.cloud\n"
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ServerURL": "cr.eu-north1.nebius.cloud",
                        "Username": "iam",
                        "Secret": "helper-token",
                    }
                ),
                stderr="",
            )
        pytest.fail("only the configured Docker credential helper may use subprocess")

    class FakeClient:
        def apply_secret(self, payload):
            captured["payload"] = payload

    monkeypatch.setattr("npa.workflows.sim2real.registry_auth.subprocess.run", fake_run)
    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: FakeClient(),
    )
    ensure_nebius_registry_pull_secret(
        registry_server="cr.eu-north1.nebius.cloud",
        k8s_context="npa-rtxpro-mk8s",
    )
    payload = captured["payload"]
    docker_payload = json.loads(base64.b64decode(payload["data"][".dockerconfigjson"]))
    entry = docker_payload["auths"]["cr.eu-north1.nebius.cloud"]
    assert entry["username"] == "iam"
    assert entry["password"] == "helper-token"


def test_ensure_materializes_direct_docker_auth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A launcher-local `docker login` credential must not be overwritten."""

    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN_FILE", raising=False)

    docker_config = tmp_path / "docker"
    docker_config.mkdir()
    direct_auth = base64.b64encode(b"runtime-sa:project-registry-token").decode()
    (docker_config / "config.json").write_text(
        json.dumps({"auths": {"cr.us-central1.nebius.cloud": {"auth": direct_auth}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: pytest.fail("direct Docker auth must be preferred"),
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def apply_secret(self, payload):
            captured["payload"] = payload

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: FakeClient(),
    )
    ensure_nebius_registry_pull_secret(
        registry_server="cr.us-central1.nebius.cloud",
        k8s_context="npa-s2r-b8edcb22",
    )
    payload = captured["payload"]
    docker_payload = json.loads(base64.b64decode(payload["data"][".dockerconfigjson"]))
    entry = docker_payload["auths"]["cr.us-central1.nebius.cloud"]
    assert entry["username"] == "runtime-sa"
    assert entry["password"] == "project-registry-token"


def test_malformed_direct_docker_auth_falls_back_to_fresh_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docker_config = tmp_path / "docker"
    docker_config.mkdir()
    (docker_config / "config.json").write_text(
        json.dumps({"auths": {"cr.us-central1.nebius.cloud": {"auth": "not-base64!"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.mint_nebius_registry_token",
        lambda **kwargs: "fresh-token",
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def apply_secret(self, payload):
            captured["payload"] = payload

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: FakeClient(),
    )
    ensure_nebius_registry_pull_secret(registry_server="cr.us-central1.nebius.cloud")
    payload = captured["payload"]
    docker_payload = json.loads(base64.b64decode(payload["data"][".dockerconfigjson"]))
    assert (
        docker_payload["auths"]["cr.us-central1.nebius.cloud"]["password"]
        == "fresh-token"
    )


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

    monkeypatch.setattr(
        engine, "_refresh_registry_pull_secret_for_sibling_job", fake_refresh
    )
    snapshot = JobSnapshot(
        name="job",
        namespace="sim2real",
        uid="uid",
        resource_version="1",
        state="complete",
        active=0,
        succeeded=1,
        failed=0,
        deleting=False,
        condition_type="Complete",
        condition_reason="CompletionsReached",
        condition_message="",
        pods=(),
    )

    class FakeClient:
        def snapshot(self, *args, **kwargs):
            return snapshot

        def pod_logs(self, *args, **kwargs):
            return "structured-client-complete"

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: FakeClient(),
    )
    monkeypatch.setattr(
        engine,
        "run_gpu_job_with_fallback",
        lambda **kwargs: {
            "job_name": "job",
            "job_uid": "uid",
            "selected_product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            "image_digests": [],
        },
    )
    monkeypatch.setattr(
        engine, "_download_component_output", lambda *args, **kwargs: None
    )

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

    monkeypatch.setattr(
        engine, "_refresh_registry_pull_secret_for_sibling_job", fake_refresh
    )
    snapshot = JobSnapshot(
        name="job",
        namespace="default",
        uid="uid",
        resource_version="1",
        state="complete",
        active=0,
        succeeded=2,
        failed=0,
        deleting=False,
        condition_type="Complete",
        condition_reason="CompletionsReached",
        condition_message="",
        pods=(),
    )

    class FakeClient:
        def snapshot(self, *args, **kwargs):
            return snapshot

        def pod_logs(self, *args, **kwargs):
            return "structured-client-complete"

    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: FakeClient(),
    )
    monkeypatch.setattr(
        engine,
        "run_gpu_job_with_fallback",
        lambda **kwargs: {
            "job_name": "job",
            "job_uid": "uid",
            "selected_product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            "image_digests": [],
        },
    )

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
