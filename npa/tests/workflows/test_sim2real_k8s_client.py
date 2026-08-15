"""Typed Kubernetes reconciliation tests; no stderr prose classification."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest

from npa.workflows.sim2real.k8s_client import (
    RUN_ID_LABEL,
    SOURCE_SHA_LABEL,
    SPEC_DIGEST_ANNOTATION,
    KubernetesJobClient,
    KubernetesReconcileError,
    _kubeconfig_with_bearer_token,
    kubeconfig_uses_nebius_iam_auth,
    job_spec_digest,
)


def _condition(kind: str, status: str, reason: str = "") -> NS:
    return NS(type=kind, status=status, reason=reason, message="diagnostic only")


def _job(
    *, condition: str = "Complete", annotations: dict[str, str] | None = None
) -> NS:
    return NS(
        metadata=NS(
            name="job",
            namespace="default",
            uid="job-uid",
            resource_version="42",
            deletion_timestamp=None,
            labels={"kueue.x-k8s.io/queue-name": "sim2real-gpu"},
            annotations=annotations or {},
        ),
        status=NS(
            conditions=[_condition(condition, "True", "BackoffLimitExceeded")],
            active=0,
            succeeded=1 if condition == "Complete" else 0,
            failed=1 if condition == "Failed" else 0,
        ),
    )


def _pod(*, owner_uid: str = "job-uid") -> NS:
    terminated = NS(reason="Completed", exit_code=0, signal=0)
    return NS(
        metadata=NS(
            name="job-pod",
            uid="pod-uid",
            deletion_timestamp=None,
            owner_references=[NS(kind="Job", controller=True, uid=owner_uid)],
        ),
        spec=NS(
            node_name="gpu-node",
            containers=[NS(resources=NS(requests={"nvidia.com/gpu": "1", "cpu": "2"}))],
        ),
        status=NS(
            phase="Succeeded",
            conditions=[_condition("PodScheduled", "True")],
            container_statuses=[
                NS(
                    name="component",
                    image="registry/image@sha256:" + "a" * 64,
                    image_id="containerd://registry/image@sha256:" + "a" * 64,
                    restart_count=1,
                    state=NS(waiting=None, terminated=terminated),
                )
            ],
        ),
    )


class _Batch:
    def __init__(self, job: NS | None = None) -> None:
        self.job = job or _job()
        self.created: list[dict[str, Any]] = []
        self.read_error: Exception | None = None
        self.read_count = 0

    def read_namespaced_job(self, _name: str, _namespace: str) -> NS:
        self.read_count += 1
        if self.read_error is not None:
            error, self.read_error = self.read_error, None
            raise error
        return self.job

    def create_namespaced_job(self, _namespace: str, body: dict[str, Any]) -> NS:
        self.created.append(body)
        return NS(metadata=NS(uid="created-uid"))

    def patch_namespaced_job(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Core:
    def __init__(self, pod: NS | None = None) -> None:
        self.pod = pod or _pod()

    def list_namespaced_pod(self, *_args: Any, **_kwargs: Any) -> NS:
        return NS(items=[self.pod])


class _Custom:
    def list_namespaced_custom_object(
        self, *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        return {
            "items": [
                {
                    "metadata": {
                        "name": "job-workload",
                        "ownerReferences": [{"kind": "Job", "uid": "job-uid"}],
                    },
                    "status": {
                        "conditions": [
                            {
                                "type": "QuotaReserved",
                                "status": "True",
                                "reason": "QuotaReserved",
                            },
                            {
                                "type": "Admitted",
                                "status": "True",
                                "reason": "Admitted",
                            },
                        ],
                        "admission": {
                            "podSetAssignments": [
                                {"flavors": {"nvidia.com/gpu": "sim2real-rtx-pro-6000"}}
                            ]
                        },
                    },
                }
            ]
        }


def _manifest() -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "job", "namespace": "default"},
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{"name": "component", "image": "image"}],
                }
            }
        },
    }


def test_bearer_override_replaces_only_selected_exec_credential() -> None:
    payload = {
        "current-context": "selected",
        "contexts": [
            {"name": "selected", "context": {"user": "selected-user"}},
            {"name": "other", "context": {"user": "other-user"}},
        ],
        "users": [
            {
                "name": "selected-user",
                "user": {
                    "exec": {"command": "nebius", "args": ["iam", "get-access-token"]},
                    "tokenFile": "/tmp/stale",
                    "client-certificate-data": "stale-certificate",
                    "client-key-data": "stale-private-key",
                },
            },
            {"name": "other-user", "user": {"exec": {"command": "other"}}},
        ],
    }
    result = _kubeconfig_with_bearer_token(
        payload, context="selected", bearer_token="fresh-token"
    )
    selected = result["users"][0]["user"]
    assert selected == {"token": "fresh-token"}
    assert result["users"][1] == payload["users"][1]
    assert "token" not in payload["users"][0]["user"]


def test_selected_context_controls_nebius_auth_detection(tmp_path) -> None:
    path = tmp_path / "kubeconfig"
    path.write_text(
        """
current-context: static-context
contexts:
- name: static-context
  context: {user: static-user}
- name: mk8s-context
  context: {user: mk8s-user}
users:
- name: static-user
  user:
    client-certificate-data: certificate
    client-key-data: private-key
- name: mk8s-user
  user:
    exec:
      command: /usr/local/bin/nebius
      args: [iam, get-access-token]
""".strip(),
        encoding="utf-8",
    )

    assert (
        kubeconfig_uses_nebius_iam_auth(kubeconfig=str(path), context="mk8s-context")
        is True
    )
    assert (
        kubeconfig_uses_nebius_iam_auth(kubeconfig=str(path), context="static-context")
        is False
    )
    assert kubeconfig_uses_nebius_iam_auth(kubeconfig=str(path), context="") is False


def test_bearer_override_rejects_non_nebius_static_user() -> None:
    payload = {
        "current-context": "selected",
        "contexts": [{"name": "selected", "context": {"user": "cert-user"}}],
        "users": [
            {
                "name": "cert-user",
                "user": {
                    "client-certificate-data": "certificate",
                    "client-key-data": "private-key",
                },
            }
        ],
    }

    with pytest.raises(KubernetesReconcileError, match="requires a selected Nebius"):
        _kubeconfig_with_bearer_token(
            payload, context="selected", bearer_token="must-not-be-injected"
        )
    assert payload["users"][0]["user"] == {
        "client-certificate-data": "certificate",
        "client-key-data": "private-key",
    }


def test_bearer_override_supports_nebius_auth_provider_without_cert_coexistence() -> (
    None
):
    payload = {
        "current-context": "selected",
        "contexts": [{"name": "selected", "context": {"user": "provider-user"}}],
        "users": [
            {
                "name": "provider-user",
                "user": {
                    "auth-provider": {"name": "nebius-iam"},
                    "client-certificate": "/tmp/stale.crt",
                    "client-key": "/tmp/stale.key",
                },
            }
        ],
    }

    result = _kubeconfig_with_bearer_token(
        payload, context="selected", bearer_token="fresh-token"
    )

    assert result["users"][0]["user"] == {"token": "fresh-token"}


def test_from_environment_keeps_static_cert_auth_when_bearer_is_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(
        """
current-context: cert-context
contexts:
- name: cert-context
  context: {user: cert-user}
users:
- name: cert-user
  user:
    client-certificate-data: certificate
    client-key-data: private-key
""".strip(),
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []

    fake_configuration = NS()
    fake_client = NS(
        Configuration=lambda: fake_configuration,
        ApiClient=lambda configuration: configuration,
        BatchV1Api=lambda _client: "batch",
        CoreV1Api=lambda _client: "core",
        CustomObjectsApi=lambda _client: "custom",
    )
    fake_config = NS(
        load_incluster_config=lambda **_kwargs: pytest.fail("not in cluster"),
        load_kube_config=lambda **kwargs: calls.append(("file", kwargs)),
        load_kube_config_from_dict=lambda *_args, **_kwargs: pytest.fail(
            "static cert auth must not be rewritten to an Authorization bearer token"
        ),
    )
    fake_kubernetes = NS(
        client=fake_client,
        config=fake_config,
        watch=NS(Watch=lambda: None),
    )
    monkeypatch.setitem(__import__("sys").modules, "kubernetes", fake_kubernetes)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    client = KubernetesJobClient.from_environment(
        kubeconfig=str(kubeconfig),
        context="cert-context",
        bearer_token="ambient-token-must-not-be-used",
    )

    assert client.batch == "batch"
    assert calls == [
        (
            "file",
            {
                "config_file": str(kubeconfig),
                "context": "cert-context",
                "persist_config": False,
                "client_configuration": fake_configuration,
            },
        )
    ]


def test_snapshot_uses_structured_job_pod_container_and_owner_fields() -> None:
    client = KubernetesJobClient(batch_api=_Batch(), core_api=_Core())
    snapshot = client.snapshot("job")
    assert snapshot.state == "complete"
    assert snapshot.condition_type == "Complete"
    assert snapshot.condition_reason == "BackoffLimitExceeded"
    assert snapshot.selected_nodes == ["gpu-node"]
    assert snapshot.pods[0].owner_uid == "job-uid"
    assert snapshot.pods[0].resource_requests["nvidia.com/gpu"] == "1"
    assert snapshot.pods[0].containers[0].terminated_reason == "Completed"
    assert snapshot.pods[0].containers[0].exit_code == 0


def test_snapshot_records_structured_kueue_admission() -> None:
    client = KubernetesJobClient(
        batch_api=_Batch(), core_api=_Core(), custom_api=_Custom()
    )
    admission = client.snapshot("job").kueue
    assert admission.workload_name == "job-workload"
    assert admission.quota_reserved is True
    assert admission.admitted is True
    assert admission.assigned_flavors == {"nvidia.com/gpu": "sim2real-rtx-pro-6000"}


def test_kueue_authorization_error_preserves_api_status_and_reason() -> None:
    class ApiError(Exception):
        status = 403
        reason = "Forbidden"

    class ForbiddenCustom:
        def list_namespaced_custom_object(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ApiError()

    client = KubernetesJobClient(
        batch_api=_Batch(), core_api=_Core(), custom_api=ForbiddenCustom()
    )
    with pytest.raises(
        KubernetesReconcileError,
        match="list_namespaced_workload failed: status=403 reason=Forbidden",
    ):
        client.snapshot("job")


def test_owner_uid_mismatch_is_never_adopted_from_prose() -> None:
    client = KubernetesJobClient(
        batch_api=_Batch(), core_api=_Core(_pod(owner_uid="other"))
    )
    with pytest.raises(KubernetesReconcileError, match="owner UID"):
        client.snapshot("job")


def test_create_or_adopt_requires_exact_identity_and_spec_digest() -> None:
    manifest = _manifest()
    source_sha = "1" * 40
    runtime = "registry/image@sha256:" + "a" * 64
    prospective = _manifest()
    prospective["metadata"].setdefault("labels", {}).update(
        {RUN_ID_LABEL: "run", SOURCE_SHA_LABEL: source_sha}
    )
    prospective["spec"]["template"].setdefault("metadata", {}).setdefault(
        "labels", {}
    ).update({RUN_ID_LABEL: "run", SOURCE_SHA_LABEL: source_sha})
    prospective["metadata"].setdefault("annotations", {}).update(
        {"sim2real.npa.dev/runtime-image": runtime}
    )
    digest = job_spec_digest(prospective)
    existing = _job(annotations={SPEC_DIGEST_ANNOTATION: digest})
    existing.metadata.labels.update({RUN_ID_LABEL: "run", SOURCE_SHA_LABEL: source_sha})
    client = KubernetesJobClient(batch_api=_Batch(existing), core_api=_Core())
    uid, adopted = client.create_or_adopt(
        manifest, run_id="run", source_sha=source_sha, runtime_image=runtime
    )
    assert (uid, adopted) == ("job-uid", True)

    existing.metadata.labels[SOURCE_SHA_LABEL] = "2" * 40
    with pytest.raises(KubernetesReconcileError, match="collision"):
        client.create_or_adopt(
            manifest, run_id="run", source_sha=source_sha, runtime_image=runtime
        )


def test_credential_401_reloads_and_reconciles_same_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ApiError(Exception):
        status = 401
        reason = "Unauthorized"

    monkeypatch.setenv("NPA_SIM2REAL_K8S_API_RETRY_SECONDS", "0")
    batch = _Batch()
    batch.read_error = ApiError()
    refreshes: list[str] = []
    client = KubernetesJobClient(
        batch_api=batch,
        core_api=_Core(),
        credential_refresh=lambda: refreshes.append("reload"),
    )
    snapshot = client.snapshot("job")
    assert snapshot.uid == "job-uid"
    assert batch.read_count == 2
    assert refreshes == ["reload"]


def test_arbitrary_statusless_error_is_not_a_transport_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_K8S_API_RETRY_SECONDS", "0")
    batch = _Batch()
    batch.read_error = RuntimeError("timed out waiting for condition ImagePullBackOff")
    client = KubernetesJobClient(batch_api=batch, core_api=_Core())
    with pytest.raises(KubernetesReconcileError, match="status=0 reason=RuntimeError"):
        client.snapshot("job")
    assert batch.read_count == 1


def test_registry_secret_rotation_recovers_401_in_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ApiError(Exception):
        status = 401
        reason = "Unauthorized"

    class Core(_Core):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0
            self.patches: list[dict[str, Any]] = []

        def read_namespaced_secret(self, name: str, namespace: str) -> NS:
            self.reads += 1
            if self.reads == 1:
                raise ApiError()
            return NS(metadata=NS(name=name, namespace=namespace))

        def patch_namespaced_secret(
            self, name: str, namespace: str, body: dict[str, Any]
        ) -> None:
            self.patches.append(body)

    monkeypatch.setenv("NPA_SIM2REAL_K8S_API_RETRY_SECONDS", "0")
    core = Core()
    refreshes: list[str] = []
    client = KubernetesJobClient(
        batch_api=_Batch(),
        core_api=core,
        credential_refresh=lambda: refreshes.append("reload"),
    )
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "pull", "namespace": "default"},
        "data": {".dockerconfigjson": "redacted-test-data"},
    }
    client.apply_secret(secret)
    assert core.reads == 2
    assert refreshes == ["reload"]
    assert core.patches == [secret]


def test_registry_secret_is_created_from_structured_404() -> None:
    class ApiError(Exception):
        status = 404
        reason = "Not Found"

    class Core(_Core):
        def __init__(self) -> None:
            super().__init__()
            self.created: list[dict[str, Any]] = []

        def read_namespaced_secret(self, name: str, namespace: str) -> NS:
            raise ApiError()

        def create_namespaced_secret(
            self, namespace: str, body: dict[str, Any]
        ) -> None:
            self.created.append(body)

    core = Core()
    client = KubernetesJobClient(batch_api=_Batch(), core_api=core)
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "pull", "namespace": "default"},
        "data": {".dockerconfigjson": "redacted-test-data"},
    }
    client.apply_secret(secret)
    assert core.created == [secret]
