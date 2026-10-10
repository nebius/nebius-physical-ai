"""Exercise endpoint setup ownership, address retention, and client independence."""

import copy
import json
import stat
import ssl

import pytest
from pydantic import ValidationError

from npa.workbench.team import endpoint_allocation, setup, setup_service_selection
from npa.workbench.team.endpoint_manifests import (
    OWNER_LABEL,
    endpoint_service,
    https_manifests,
)
from npa.workbench.team.errors import BackendError, ConflictError
from npa.workbench.team.setup_models import SetupRequest
from npa.workbench.team.setup_receipt import SetupReceipt


@pytest.fixture
def setup_request(tmp_path):
    return SetupRequest(
        project_id="project-test",
        cluster_id="cluster-test",
        kubeconfig=tmp_path / "kubeconfig",
        context="selected",
        namespace="management",
        image="example/team@sha256:" + "a" * 64,
        configuration_secret="configuration",
        state_claim="state",
        tls_secret="tls",
        endpoint="https://team.example.test",
        node_selector={"pool": "cpu"},
    )


class Infrastructure:
    def __init__(self, setup_request, monkeypatch):
        self.setup_request = setup_request
        self.objects = {
            ("secret", "configuration"): {"metadata": {"name": "configuration"}},
            ("secret", "tls"): {"data": {"tls.crt": "certificate", "tls.key": "key"}},
            ("persistentvolumeclaim", "state"): {"metadata": {"name": "state"}},
        }
        self.mutations = []
        self.allocation = None
        self.next_uid = 0
        monkeypatch.setattr(setup, "resource", self.resource)
        monkeypatch.setattr(setup, "kubernetes", self.kubernetes)
        monkeypatch.setattr(endpoint_allocation, "kubernetes", self.kubernetes)
        monkeypatch.setattr(setup_service_selection, "kubernetes", self.kubernetes)
        monkeypatch.setattr(setup, "provider", self.provider)
        monkeypatch.setattr(endpoint_allocation, "provider", self.provider)
        monkeypatch.setattr(setup, "verify_certificate", lambda *args: None)
        monkeypatch.setattr(
            setup,
            "verify_endpoint",
            lambda *args: {
                "boundary_statuses": {
                    "/health": 200,
                    "/v1/me": 401,
                    "/api/health": 404,
                },
                "dns_points_to_lb": True,
            },
        )

    def resource(self, setup_request, kind, name):
        return copy.deepcopy(self.objects.get((kind.lower(), name)))

    def kubernetes(self, setup_request, *args, document=None):
        assert setup_request.context == "selected"
        if args[:2] == ("config", "view"):
            return {
                "clusters": [{"cluster": {"server": "https://selected.example.test"}}]
            }
        if args[:2] == ("get", "namespace"):
            return {"metadata": {"labels": {}}}
        if args[:2] == ("get", "services"):
            return {
                "items": [
                    copy.deepcopy(value)
                    for (kind, _), value in self.objects.items()
                    if kind == "service"
                ]
            }
        if args[:2] == ("get", "nodes"):
            return {
                "items": [
                    {
                        "metadata": {"labels": {"pool": "cpu"}},
                        "status": {
                            "allocatable": {},
                            "conditions": [{"type": "Ready", "status": "True"}],
                        },
                    }
                ]
            }
        if args[0] == "patch":
            return self._patch_service(args[2], document)
        assert args[0] in {"apply", "create"}
        return self._create_resource(document)

    def _patch_service(self, name, document):
        self.mutations.append(("patch", copy.deepcopy(document)))
        service = self.objects[("service", name)]
        for field in ("annotations", "labels"):
            service["metadata"].setdefault(field, {}).update(
                document["metadata"].get(field, {})
            )
        service["spec"].update(document.get("spec", {}))
        return copy.deepcopy(service)

    def _create_resource(self, document):
        self.mutations.append(("apply", copy.deepcopy(document)))
        self.next_uid += 1
        installed = copy.deepcopy(document)
        metadata = installed["metadata"]
        metadata.update(uid=f"uid-{self.next_uid}", resourceVersion="1", generation=1)
        if installed["kind"] == "Deployment":
            installed["status"] = {"observedGeneration": 1, "availableReplicas": 1}
        if installed["kind"] == "Service":
            installed["status"] = {"loadBalancer": {"ingress": [{"ip": "203.0.113.8"}]}}
            if self.allocation is None:
                self._create_allocation(metadata)
        self.objects[(installed["kind"].lower(), metadata["name"])] = installed
        return copy.deepcopy(installed)

    def _create_allocation(self, metadata):
        self.allocation = {
            "metadata": {
                "id": "allocation-test",
                "parent_id": "project-test",
                "resource_version": "1",
                "labels": {
                    "nebius.com/managed-by": "mk8s",
                    "nebius.com/cluster-id": "cluster-test",
                    "nebius.com/service-name": metadata["name"],
                    "nebius.com/service-namespace": "management",
                    "nebius.com/service-uid": metadata["uid"],
                    "operator-label": "preserve",
                },
            },
            "status": {"details": {"allocated_cidr": "203.0.113.8/32"}},
        }

    def provider(self, *args):
        if args[:3] == ("mk8s", "cluster", "get"):
            return {
                "metadata": {"parent_id": "project-test"},
                "status": {
                    "control_plane": {
                        "endpoints": {
                            "public_endpoint": "https://selected.example.test"
                        }
                    }
                },
            }
        if args[:3] == ("vpc", "allocation", "list"):
            return {"items": [copy.deepcopy(self.allocation)]}
        if args[:3] == ("vpc", "allocation", "get"):
            return copy.deepcopy(self.allocation)
        assert args[:3] == ("vpc", "allocation", "update")
        self.mutations.append(("provider-update", args))
        for index, argument in enumerate(args):
            if argument == "--labels-remove":
                self.allocation["metadata"]["labels"].pop(args[index + 1])
        return copy.deepcopy(self.allocation)


@pytest.fixture
def infrastructure(setup_request, monkeypatch):
    return Infrastructure(setup_request, monkeypatch)


def test_fresh_setup_creates_https_lb_and_retains_address_without_personal_client(
    setup_request, infrastructure, tmp_path
):
    receipt = tmp_path / "installation.json"
    result = setup.setup_control_plane(setup_request, receipt)
    assert result["status"] == "ready"
    assert result["personal_client_required"] is False
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    recorded = json.loads(receipt.read_text())
    assert recorded["allocation_id"] == "allocation-test"
    service = infrastructure.objects[("service", "npa-team")]
    assert (
        service["metadata"]["annotations"][endpoint_allocation.ALLOCATION_ANNOTATION]
        == "allocation-test"
    )
    assert [p["port"] for p in service["spec"]["ports"]] == [443]
    assert infrastructure.allocation["metadata"]["labels"] == {
        "operator-label": "preserve"
    }
    assert "token" not in json.dumps(recorded).lower()


def test_retry_does_not_redeploy_restart_or_allocate_another_lb(
    setup_request, infrastructure, tmp_path
):
    path = tmp_path / "installation.json"
    setup.setup_control_plane(setup_request, path)
    first = copy.deepcopy(infrastructure.mutations)
    setup.setup_control_plane(setup_request, path)
    assert infrastructure.mutations == first


def test_service_recreation_uses_retained_address(
    setup_request, infrastructure, tmp_path
):
    path = tmp_path / "installation.json"
    setup.setup_control_plane(setup_request, path)
    del infrastructure.objects[("service", "npa-team")]
    setup.setup_control_plane(setup_request, path)
    service = infrastructure.objects[("service", "npa-team")]
    assert (
        service["metadata"]["annotations"][endpoint_allocation.ALLOCATION_ANNOTATION]
        == "allocation-test"
    )
    assert service["status"]["loadBalancer"]["ingress"][0]["ip"] == "203.0.113.8"


@pytest.mark.parametrize(
    "field,value",
    [("project_id", "other"), ("context", "other"), ("namespace", "other")],
)
def test_receipt_cannot_retarget_an_existing_setup(
    setup_request, infrastructure, tmp_path, field, value
):
    path = tmp_path / "installation.json"
    setup.setup_control_plane(setup_request, path)
    before = copy.deepcopy(infrastructure.mutations)
    with pytest.raises(ConflictError):
        setup.setup_control_plane(setup_request.model_copy(update={field: value}), path)
    assert infrastructure.mutations == before


def test_unknown_existing_deployment_is_never_adopted_by_name(
    setup_request, infrastructure, tmp_path
):
    deployment = https_manifests(setup_request, "different-owner")[0]
    deployment["metadata"]["uid"] = "unknown"
    infrastructure.objects[("deployment", "npa-team")] = deployment
    with pytest.raises(ConflictError, match="exact Service UID"):
        setup.setup_control_plane(setup_request, tmp_path / "installation.json")
    assert not infrastructure.mutations


def test_existing_installation_requires_exact_service_uid_and_preserves_deployment(
    setup_request, infrastructure, tmp_path
):
    original = tmp_path / "original.json"
    setup.setup_control_plane(setup_request, original)
    deployment = copy.deepcopy(infrastructure.objects[("deployment", "npa-team")])
    service = infrastructure.objects[("service", "npa-team")]
    for document in (deployment, service):
        document["metadata"].pop("labels", None)
    infrastructure.objects[("deployment", "npa-team")] = copy.deepcopy(deployment)
    selected = setup_request.model_copy(
        update={
            "existing_service_uid": service["metadata"]["uid"],
            "existing_deployment_uid": deployment["metadata"]["uid"],
        }
    )
    before = copy.deepcopy(infrastructure.mutations)
    setup.setup_control_plane(selected, tmp_path / "adopted.json")
    assert infrastructure.objects[("deployment", "npa-team")] == deployment
    assert infrastructure.mutations == before


def test_discovery_reuses_public_service_and_leaves_internal_service_private(
    setup_request, infrastructure, tmp_path
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    public = infrastructure.objects.pop(("service", "npa-team"))
    public["metadata"]["name"] = "shared-https"
    infrastructure.objects[("service", "shared-https")] = public
    internal = copy.deepcopy(public)
    internal["metadata"].update(name="npa-team", uid="internal-uid")
    internal["spec"]["type"] = "ClusterIP"
    infrastructure.objects[("service", "npa-team")] = internal
    selected = setup_request.model_copy(
        update={
            "existing_service_uid": public["metadata"]["uid"],
            "existing_deployment_uid": infrastructure.objects[
                ("deployment", "npa-team")
            ]["metadata"]["uid"],
        }
    )
    path = tmp_path / "adopted.json"
    before = copy.deepcopy(infrastructure.mutations)
    setup.setup_control_plane(selected, path)
    setup.setup_control_plane(selected, path)
    assert json.loads(path.read_text())["service_name"] == "shared-https"
    assert infrastructure.objects[("service", "npa-team")] == internal
    assert infrastructure.mutations == before
    del infrastructure.objects[("service", "shared-https")]
    setup.setup_control_plane(selected, path)
    assert (
        infrastructure.objects[("service", "shared-https")]["metadata"]["annotations"][
            endpoint_allocation.ALLOCATION_ANNOTATION
        ]
        == "allocation-test"
    )


def test_discovery_refuses_internal_uid_when_public_service_already_exists(
    setup_request, infrastructure, tmp_path
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    before = copy.deepcopy(infrastructure.mutations)
    selected = setup_request.model_copy(update={"existing_service_uid": "internal"})
    with pytest.raises(ConflictError, match="public Service UID"):
        setup.setup_control_plane(selected, tmp_path / "adopted.json")
    assert infrastructure.mutations == before


def test_discovery_refuses_ambiguous_public_endpoints(
    setup_request, infrastructure, tmp_path
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    other = copy.deepcopy(infrastructure.objects[("service", "npa-team")])
    other["metadata"].update(name="other", uid="other-uid")
    infrastructure.objects[("service", "other")] = other
    before = copy.deepcopy(infrastructure.mutations)
    with pytest.raises(ConflictError, match="multiple endpoints"):
        setup.setup_control_plane(setup_request, tmp_path / "adopted.json")
    assert infrastructure.mutations == before


def test_recreation_refuses_changed_retained_allocation_before_mutation(
    setup_request, infrastructure, tmp_path
):
    path = tmp_path / "installation.json"
    setup.setup_control_plane(setup_request, path)
    del infrastructure.objects[("service", "npa-team")]
    infrastructure.allocation["metadata"]["parent_id"] = "other"
    before = copy.deepcopy(infrastructure.mutations)
    with pytest.raises(ConflictError, match="allocation project"):
        setup.setup_control_plane(setup_request, path)
    assert infrastructure.mutations == before


def test_missing_private_registry_secret_blocks_setup_before_mutation(
    setup_request, infrastructure, tmp_path
):
    selected = setup_request.model_copy(update={"image_pull_secrets": ("registry",)})
    with pytest.raises(BackendError, match="image pull Secrets"):
        setup.setup_control_plane(selected, tmp_path / "installation.json")
    assert not infrastructure.mutations


def test_service_uid_alone_cannot_adopt_an_unowned_deployment(
    setup_request, infrastructure, tmp_path
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    service = infrastructure.objects[("service", "npa-team")]
    deployment = infrastructure.objects[("deployment", "npa-team")]
    deployment["metadata"].pop("labels")
    selected = setup_request.model_copy(
        update={"existing_service_uid": service["metadata"]["uid"]}
    )
    before = copy.deepcopy(infrastructure.mutations)
    with pytest.raises(ConflictError, match="Deployment UID"):
        setup.setup_control_plane(selected, tmp_path / "adopted.json")
    assert infrastructure.mutations == before


def test_selected_deployment_can_be_preserved_when_public_service_is_absent(
    setup_request, infrastructure, tmp_path
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    deployment = infrastructure.objects[("deployment", "npa-team")]
    deployment["metadata"].pop("labels")
    del infrastructure.objects[("service", "npa-team")]
    infrastructure.allocation = None
    selected = setup_request.model_copy(
        update={"existing_deployment_uid": deployment["metadata"]["uid"]}
    )
    before = len(infrastructure.mutations)
    assert (
        setup.setup_control_plane(selected, tmp_path / "adopted.json")["status"]
        == "ready"
    )
    assert infrastructure.objects[("deployment", "npa-team")] == deployment
    created = [
        doc for kind, doc in infrastructure.mutations[before:] if kind == "apply"
    ]
    assert [doc["kind"] for doc in created] == ["Service"]


def test_deployment_disappearance_is_an_identity_failure_without_indefinite_wait(
    setup_request, infrastructure, tmp_path
):
    receipt = SetupReceipt(tmp_path / "installation.json", setup_request)
    receipt.save(deployment_uid="expected")
    with pytest.raises(ConflictError, match="Deployment identity changed"):
        setup._wait_deployment(setup_request, receipt)
    assert not infrastructure.mutations


def test_explicit_clusterip_conversion_preserves_uid_ports_and_operator_annotations(
    setup_request, infrastructure, tmp_path
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    service = infrastructure.objects[("service", "npa-team")]
    deployment = infrastructure.objects[("deployment", "npa-team")]
    service["spec"]["type"] = "ClusterIP"
    service["metadata"]["annotations"]["operator-label"] = "preserve"
    selected = setup_request.model_copy(
        update={
            "existing_service_uid": service["metadata"]["uid"],
            "existing_deployment_uid": deployment["metadata"]["uid"],
        }
    )
    before = copy.deepcopy(service)
    setup.setup_control_plane(selected, tmp_path / "adopted.json")
    assert service["metadata"]["uid"] == before["metadata"]["uid"]
    assert service["spec"]["ports"] == before["spec"]["ports"]
    assert service["metadata"]["annotations"] == before["metadata"]["annotations"]
    assert service["spec"]["type"] == "LoadBalancer"
    patches = [doc for kind, doc in infrastructure.mutations if kind == "patch"]
    assert (
        patches[-1]["metadata"]["resourceVersion"]
        == before["metadata"]["resourceVersion"]
    )


def test_conversion_rejects_uid_change_without_detaching_an_allocation(
    setup_request, infrastructure, tmp_path, monkeypatch
):
    setup.setup_control_plane(setup_request, tmp_path / "original.json")
    service = infrastructure.objects[("service", "npa-team")]
    service["spec"]["type"] = "ClusterIP"
    selected = setup_request.model_copy(
        update={
            "existing_service_uid": service["metadata"]["uid"],
            "existing_deployment_uid": infrastructure.objects[
                ("deployment", "npa-team")
            ]["metadata"]["uid"],
        }
    )
    actual = infrastructure.kubernetes

    def changed_uid(request, *args, **kwargs):
        result = actual(request, *args, **kwargs)
        if args[0] == "patch":
            result["metadata"]["uid"] = "replaced"
        return result

    monkeypatch.setattr(setup, "kubernetes", changed_uid)
    before = copy.deepcopy(infrastructure.allocation)
    with pytest.raises(ConflictError, match="identity changed"):
        setup.setup_control_plane(selected, tmp_path / "adopted.json")
    assert infrastructure.allocation == before


def test_unreachable_network_is_retryable_but_tls_failure_is_terminal(
    setup_request, infrastructure, tmp_path, monkeypatch
):
    path = tmp_path / "installation.json"

    def unavailable(*args):
        raise OSError("network unreachable")

    monkeypatch.setattr(setup, "verify_endpoint", unavailable)
    assert (
        setup.setup_control_plane(setup_request, path)["status"]
        == "endpoint-awaiting-connectivity"
    )

    def bad_tls(*args):
        raise ssl.SSLCertVerificationError("private certificate detail")

    monkeypatch.setattr(setup, "verify_endpoint", bad_tls)
    with pytest.raises(BackendError, match="TLS verification failed") as error:
        setup.setup_control_plane(setup_request, path)
    assert "private certificate detail" not in str(error.value)
    saved = json.loads(path.read_text())
    assert saved["phase"] == "endpoint-tls-failed"
    assert saved["external_https_verified"] is False


def test_terminal_provider_allocation_condition_fails_without_a_time_limit(
    setup_request, infrastructure, tmp_path
):
    path = tmp_path / "installation.json"
    setup.setup_control_plane(setup_request, path)
    service = infrastructure.objects[("service", "npa-team")]
    service["status"] = {
        "conditions": [
            {"type": "LoadBalancerReady", "status": "False", "reason": "QuotaExceeded"}
        ]
    }
    with pytest.raises(BackendError, match="terminal LoadBalancer"):
        setup.setup_control_plane(setup_request, path)


@pytest.mark.parametrize(
    "change", ["project", "cluster", "service", "namespace", "address"]
)
def test_retention_refuses_foreign_provider_identity(
    setup_request, infrastructure, tmp_path, monkeypatch, change
):
    receipt = SetupReceipt(tmp_path / "installation.json", setup_request)
    service = infrastructure.kubernetes(
        setup_request,
        "apply",
        document=endpoint_service(setup_request, receipt.data["installation_id"]),
    )
    allocation = infrastructure.allocation
    if change == "project":
        allocation["metadata"]["parent_id"] = "other"
    elif change == "address":
        allocation["status"]["details"]["allocated_cidr"] = "203.0.113.9/32"
    else:
        key = {
            "cluster": "cluster-id",
            "service": "service-uid",
            "namespace": "service-namespace",
        }[change]
        allocation["metadata"]["labels"]["nebius.com/" + key] = "other"
    before = len(infrastructure.mutations)
    with pytest.raises(ConflictError):
        endpoint_allocation.retain_allocation(setup_request, receipt, service)
    assert len(infrastructure.mutations) == before


def test_retention_recovers_after_detach_before_service_patch(
    setup_request, infrastructure, tmp_path, monkeypatch
):
    path = tmp_path / "installation.json"
    actual = endpoint_allocation._link
    monkeypatch.setattr(
        endpoint_allocation,
        "_link",
        lambda *args: (_ for _ in ()).throw(BackendError("transport interrupted")),
    )
    with pytest.raises(BackendError):
        setup.setup_control_plane(setup_request, path)
    assert json.loads(path.read_text())["phase"] == "allocation-retention-intent"
    assert (
        "nebius.com/managed-by" not in infrastructure.allocation["metadata"]["labels"]
    )
    monkeypatch.setattr(endpoint_allocation, "_link", actual)
    assert setup.setup_control_plane(setup_request, path)["status"] == "ready"


def test_operator_network_timeout_preserves_retryable_server_setup(
    setup_request, infrastructure, tmp_path, monkeypatch
):
    path = tmp_path / "installation.json"

    def unavailable(*args):
        raise TimeoutError("cannot connect")

    monkeypatch.setattr(setup, "verify_endpoint", unavailable)
    result = setup.setup_control_plane(setup_request, path)
    assert result["status"] == "endpoint-awaiting-connectivity"
    assert result["external_https_verified"] is False
    assert json.loads(path.read_text())["allocation_id"] == "allocation-test"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://team.example.test",
        "https://user:secret@team.example.test",
        "https://team.example.test:8443",
        "https://team.example.test/api",
    ],
)
def test_setup_rejects_plaintext_or_ambiguous_endpoints(setup_request, endpoint):
    with pytest.raises(ValidationError):
        SetupRequest.model_validate(
            {**setup_request.model_dump(), "endpoint": endpoint}
        )


def test_setup_manifest_exposes_neither_plaintext_gateway_nor_native_scheduler(
    setup_request,
):
    deployment, configuration, policy = https_manifests(setup_request, "installation")
    pod = deployment["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"pool": "cpu"}
    assert pod["automountServiceAccountToken"] is False
    proxy = pod["containers"][2]
    assert proxy["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"], "add": ["NET_BIND_SERVICE"]},
    }
    assert "--host 127.0.0.1" in pod["containers"][1]["args"][0]
    assert "admin off" in configuration["data"]["Caddyfile"]
    assert policy["spec"]["ingress"] == [{"ports": [{"protocol": "TCP", "port": 8444}]}]
    service = endpoint_service(setup_request, "installation")
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["metadata"]["labels"][OWNER_LABEL] == "installation"


def test_receipt_rejects_symlinks_and_world_readable_files(setup_request, tmp_path):
    path = tmp_path / "installation.json"
    SetupReceipt(path, setup_request)
    path.chmod(0o644)
    with pytest.raises(ConflictError):
        SetupReceipt(path, setup_request)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(ConflictError):
        SetupReceipt(link, setup_request)
