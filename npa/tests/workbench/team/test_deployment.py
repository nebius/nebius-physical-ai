"""Check enrollment drift, private scheduler routing, and personal cloud provisioning."""

import base64
import json
import shlex
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from npa.workbench.team import enrollment, nebius_storage
from npa.workbench.team.administration import stop_run
from npa.workbench.team.deployment import server_config, scheduler_user
from npa.workbench.team.errors import BackendError
from npa.workbench.team.ledger import TeamLedger
from npa.workbench.team.manifests import admission_manifests, execution_manifests
from npa.workbench.team.models import SubmitRequest, TeamConfig
from npa.workbench.team.pod_deployment import service_manifests
from npa.workbench.team.service import binding_snapshot


def test_no_default_port_autostart_or_public_backend(config):
    for endpoint in ("http://127.0.0.1:46580", "https://scheduler.example.test:46581"):
        document = config.model_dump(mode="json")
        document["sky_endpoint"] = endpoint
        with pytest.raises(ValidationError):
            TeamConfig.model_validate(document)


@pytest.mark.parametrize("field", ["state_dir", "sky_python"])
def test_service_rejects_paths_that_change_when_its_working_directory_changes(
    config, field
):
    document = config.model_dump(mode="json")
    document[field] = "relative-path"
    with pytest.raises(ValidationError, match="paths must be absolute"):
        TeamConfig.model_validate(document)


def test_each_cluster_has_its_own_controller_and_no_private_user_registration(
    config, binding
):
    other = replace(binding, cluster="west", connection=config.clusters["west"])
    assert scheduler_user(other) != scheduler_user(binding)
    settings = server_config(config)
    assert settings["rbac"]["default_role"] == "user"
    assert settings["jobs"]["controller"]["consolidation_mode"] is False
    assert all(
        "allowed_users" not in value for value in settings["workspaces"].values()
    )
    control = next(
        value
        for value in settings["kubernetes"]["context_configs"].values()
        if "post_provision_runcmd" in value
    )
    command = shlex.split(control["post_provision_runcmd"][0])
    encoded = command[command.index("%s") + 1]
    kubeconfig = json.loads(base64.b64decode(encoded))
    assert kubeconfig["clusters"][0]["cluster"]["server"] == "https://192.0.2.1:443"
    assert "tokenFile" in kubeconfig["users"][0]["user"]
    assert "token" not in kubeconfig["users"][0]["user"]


def test_service_exposes_only_gateway_and_copies_credentials_privately():
    documents = service_manifests(
        namespace="management",
        image="example/team:test",
        secret="configuration",
        claim="state",
    )
    deployment, service, policy = documents
    pod = deployment["spec"]["template"]["spec"]
    scheduler = pod["containers"][1]
    assert "--port 46581" in scheduler["args"][0]
    assert "--host 127.0.0.1" in scheduler["args"][0]
    assert [port["port"] for port in service["spec"]["ports"]] == [8443]
    assert "chmod 600" in pod["initContainers"][0]["args"][0]
    assert pod["automountServiceAccountToken"] is False


def test_enrollment_requires_current_quota_and_admission(binding, monkeypatch):
    desired = admission_manifests() + execution_manifests(binding)
    inventory = {(item["kind"], item["metadata"]["name"]): item for item in desired}
    admission = desired[0]
    admission["metadata"]["generation"] = 1
    admission["status"] = {"observedGeneration": 1, "typeChecking": {}}

    def transport(binding, *arguments, **kwargs):
        if arguments[:2] == ("auth", "can-i"):
            return SimpleNamespace(returncode=1, stdout="no\n")
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(inventory[(arguments[1], arguments[2])])
        )

    monkeypatch.setattr(enrollment, "kubectl", transport)

    # Namespace-qualified resources can have the same name; use the exact namespace.
    def exact(binding, *arguments, **kwargs):
        if arguments[0] == "auth":
            return transport(binding, *arguments, **kwargs)
        namespace = (
            arguments[arguments.index("--namespace") + 1]
            if "--namespace" in arguments
            else None
        )
        item = next(
            item
            for item in desired
            if item["kind"] == arguments[1]
            and item["metadata"]["name"] == arguments[2]
            and item["metadata"].get("namespace") == namespace
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(item))

    monkeypatch.setattr(enrollment, "kubectl", exact)
    enrollment.verify_enrollment(binding)
    admission["status"].pop("typeChecking")
    probes = []
    monkeypatch.setattr(
        enrollment,
        "_check_admission_requests",
        lambda selected: probes.append(selected),
    )
    enrollment.verify_enrollment(binding)
    assert probes == [binding]
    quota = next(item for item in desired if item["kind"] == "ResourceQuota")
    quota["spec"]["hard"]["requests.nvidia.com/gpu"] = "99"
    with pytest.raises(BackendError):
        enrollment.verify_enrollment(binding)


@pytest.mark.parametrize("denial", ["", "Forbidden: caller cannot create pods"])
def test_admission_proof_rejects_allowed_or_unrelated_denials(
    binding, monkeypatch, denial
):
    calls = []

    def transport(selected, *arguments, **kwargs):
        assert arguments[:2] == ("create", "--dry-run=server")
        pod = json.loads(kwargs["input"])
        calls.append(pod)
        return SimpleNamespace(
            returncode=int(len(calls) > 1 and bool(denial)),
            stdout="",
            stderr=denial if len(calls) > 1 else "",
        )

    monkeypatch.setattr(enrollment, "kubectl", transport)
    with pytest.raises(BackendError, match="denial could not be verified"):
        enrollment._check_admission_requests(binding)
    assert len(calls) == 2


def test_admission_proof_requires_positive_admission(binding, monkeypatch):
    monkeypatch.setattr(
        enrollment,
        "kubectl",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="invalid"
        ),
    )
    with pytest.raises(BackendError, match="valid worker dry run"):
        enrollment._check_admission_requests(binding)


def test_operator_can_cancel_after_offboarding(config, actor, binding, workflow):
    ledger = TeamLedger(config.state_dir)
    request = SubmitRequest(
        workspace="robotics", cluster="east", idempotency_key="stop", workflow=workflow
    )
    record, _ = ledger.create(actor, request, binding_snapshot(binding))
    config = config.model_copy(update={"disabled_subjects": (actor.subject,)})

    def backend(*args):
        return SimpleNamespace(cancel_run=lambda: True)

    assert (
        stop_run(config, record["id"], backend_factory=backend)["status"] == "cancelled"
    )


def test_personal_nebius_group_never_reuses_shared_project_group(monkeypatch, tmp_path):
    receipt = nebius_storage._Receipt(tmp_path)
    environment = SimpleNamespace(project_id="test-project", tenant_id="test-tenant")
    observed = {}

    def ensure(**kwargs):
        observed.update(kwargs)
        kwargs["on_resource_created"]("iam_group", {"id": "personal-group"})
        return SimpleNamespace(
            compatibility_fallback=False,
            scope_id="test-bucket",
            group_id="personal-group",
        )

    monkeypatch.setattr(
        nebius_storage.nebius, "ensure_storage_capability_binding", ensure
    )
    nebius_storage._binding(
        environment, "unique-personal-group", "test-account", "test-bucket", receipt
    )
    assert observed["binding_group_name"] == "unique-personal-group"
    assert observed["allow_editors_fallback"] is False
