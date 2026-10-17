from __future__ import annotations

from types import SimpleNamespace

import pytest

from npa.workflows.sim2real.smolvla_service_lifecycle import (
    SmolVLAServiceError,
    _run,
    build_manifests,
    resource_names,
)


def _args(**updates):
    values = {
        "command": "start",
        "run_id": "policy-interchange-run-1",
        "checkpoint_role": "base",
        "checkpoint_uri": "hf://lerobot/smolvla_base@" + "a" * 40,
        "bind_adapter_uri": "s3://example/profile/base-adapter.json",
        "runtime_image": "registry.example.invalid/lerobot@sha256:" + "b" * 64,
        "source_commit": "c" * 40,
        "namespace": "default",
        "service_account": "run-base-controller",
        "auth_secret": "run-policy-auth",
        "hf_token_secret": "",
        "gpu_count": 1,
        "server_cpu": "8",
        "server_memory": "64Gi",
        "policy_cache_pvc": "run-policy-cache",
        "gpu_node_selector_key": "nebius.com/gpu-name",
        "gpu_node_selector_value": "RTXPRO6000",
        "hardware_topology": "single-rtxpro6000",
        "ready_timeout_seconds": 1200.0,
        "output_uri": "s3://example/run/start.json",
        "component_root_uri": "s3://example/run",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example.invalid")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "task-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "task-secret")


def test_base_service_is_digest_pinned_private_and_binds_exact_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    manifests = build_manifests(_args())
    deployment = manifests["deployment"]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    shell = container["command"][2]
    assert container["image"].endswith("b" * 64)
    assert "snapshot_download" in shell and "a" * 40 in shell
    assert ".complete" in shell and "if test -f" in shell
    assert "bind-interchange" in shell
    assert deployment["metadata"]["labels"]["app.kubernetes.io/part-of"] == (
        "sim2real-policy-interchange"
    )
    assert manifests["service"]["spec"]["type"] == "ClusterIP"
    assert container["env"][1]["valueFrom"]["secretKeyRef"]["key"] == "auth_token"
    volume = deployment["spec"]["template"]["spec"]["volumes"][0]
    assert volume == {
        "name": "policy",
        "persistentVolumeClaim": {"claimName": "run-policy-cache"},
    }
    assert "emptyDir" not in volume


def test_adapted_private_checkpoint_uses_ephemeral_owned_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    manifests = build_manifests(
        _args(
            checkpoint_role="adapted",
            checkpoint_uri="s3://example/run/checkpoint/",
            bind_adapter_uri="",
        )
    )
    assert (
        manifests["secret"]["metadata"]["labels"]["npa.nebius.ai/checkpoint-role"]
        == "adapted"
    )
    shell = manifests["deployment"]["spec"]["template"]["spec"]["containers"][0][
        "command"
    ][2]
    assert "list_objects_v2" in shell
    assert "bind-interchange" not in shell


def test_cleanup_builds_identity_without_requiring_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key, raising=False)
    manifests = build_manifests(_args(command="cleanup"))
    assert manifests["secret"]["data"] == {}


def test_service_rejects_mutable_image_and_unpinned_hf_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    with pytest.raises(SmolVLAServiceError, match="digest-pinned"):
        build_manifests(_args(runtime_image="registry.example.invalid/lerobot:latest"))
    with pytest.raises(SmolVLAServiceError, match="40-hex"):
        build_manifests(_args(checkpoint_uri="hf://lerobot/smolvla_base@main"))
    with pytest.raises(SmolVLAServiceError, match="policy cache PVC"):
        build_manifests(_args(policy_cache_pvc=""))


def test_resource_names_are_deterministic_and_role_separated() -> None:
    base = resource_names("Run/With Unsafe Chars", "base")
    adapted = resource_names("Run/With Unsafe Chars", "adapted")
    assert base["deployment"] != adapted["deployment"]
    assert len(base["deployment"]) <= 63


def test_start_receipt_binds_loaded_checkpoint_and_private_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _credentials(monkeypatch)
    import npa.workflows.sim2real.smolvla_service_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "_clients", lambda: (object(), object()))
    monkeypatch.setattr(lifecycle, "_assert_absent", lambda *_: None)
    monkeypatch.setattr(
        lifecycle, "_create", lambda *_args, created: created.update({"secret"})
    )
    health = {
        "checkpoint": {"status": "loadable", "tensor_count": 123},
        "interchange": {
            "checkpoint_sha256": "c" * 64,
            "normalization_identity": "d" * 64,
            "manifest_sha256": "e" * 64,
        },
    }
    monkeypatch.setattr(lifecycle, "_ready", lambda *_: ("private.svc", health))
    monkeypatch.setattr(lifecycle, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        lifecycle, "publish_component_record", lambda *_args, **_kwargs: None
    )
    receipt = _run(_args())
    assert receipt["public_ingress"] is False
    assert receipt["application_authentication"] == "kubernetes_secret_bearer"
    assert receipt["checkpoint"]["tensor_count"] == 123
    assert receipt["interchange"]["manifest_sha256"] == "e" * 64
    assert receipt["policy_cache_pvc_sha256"]
