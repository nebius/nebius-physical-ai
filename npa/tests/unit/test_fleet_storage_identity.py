"""Mock provider and registration boundaries for Fleet storage qualification."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from npa.cluster.state import ClusterState, save_cluster_state
from npa.fleet import storage_identity as identity
from npa.fleet.spec import ClusterSpec, FleetSpec, NodePoolSpec, ProjectSpec


def _spec():
    cluster = ClusterSpec(name="training", enable_filestore=True,
                          cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="2vcpu-8gb"))
    project = ProjectSpec(name="team", project_id="project-test", clusters=[cluster])
    return FleetSpec(name="storage-test", tenant_id="tenant-test", region="test-region",
                     profile="operator-test", projects=[project])


def _kubeconfig(context):
    return {"apiVersion": "v1", "kind": "Config", "current-context": context,
            "contexts": [{"name": context, "context": {"cluster": "test", "user": "test"}}],
            "clusters": [{"name": "test", "cluster": {
                "server": "https://cluster.example.invalid",
                "certificate-authority-data": "dGVzdC1jZXJ0aWZpY2F0ZQ=="}}],
            "users": [{"name": "test", "user": {"exec": {
                "command": "nebius", "args": ["--profile", "operator-test", "iam", "get-access-token"],
                "apiVersion": "client.authentication.k8s.io/v1beta1"}}}]}


def _write_document(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))
    path.chmod(0o600)


def _registration(root, spec):
    project, cluster = spec.cluster_targets()[0]
    context = f"fleet-{spec.name}-{project.key()}-{cluster.name}"
    kubeconfig = root / context / "kubeconfig"
    registered = ClusterState(context, "cluster-test", project.project_id, spec.region, 1,
                              "cpu-d3", "2vcpu-8gb", "test", "subnet-test", "test",
                              kubeconfig_path=str(kubeconfig), provider_name=cluster.name)
    save_cluster_state(registered, base_dir=root, metadata={
        "managed_by": "npa fleet", "fleet": spec.name, "project_key": project.key()})
    _write_document(kubeconfig, _kubeconfig(context))
    return registered, kubeconfig


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    spec = _spec()
    root = tmp_path / "clusters"
    monkeypatch.setattr(identity.cluster_state, "CLUSTERS_DIR", root)
    monkeypatch.setattr(identity.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(identity, "require_bin", lambda binary: binary)
    _write_document(tmp_path / ".nebius" / "config.yaml", {
        "default": spec.profile, "profiles": {spec.profile: {"tenant-id": spec.tenant_id}}})
    registered, kubeconfig = _registration(root, spec)
    project = {"metadata": {"id": "project-test", "parent_id": "tenant-test", "name": "team"},
               "spec": {"region": "test-region"}, "status": {"container_state": "ACTIVE"}}
    cluster = {"metadata": {"id": "cluster-test", "parent_id": "project-test", "name": "training"},
               "status": {"state": "RUNNING"}}
    runtime = {"spec": spec, "registered": registered, "kubeconfig": kubeconfig,
               "project": project, "cluster": cluster, "fresh": _kubeconfig(registered.name),
               "calls": [], "temporary": []}
    monkeypatch.setattr(identity, "run_capture", _provider(runtime))
    return runtime


def _provider(runtime):
    def run(arguments, *, check, env):
        runtime["calls"].append((arguments, env))
        if "get-credentials" in arguments:
            path = Path(arguments[arguments.index("--kubeconfig") + 1])
            runtime["temporary"].append(path)
            _write_document(path, runtime["fresh"])
            return subprocess.CompletedProcess(arguments, 0, "", "")
        key = "project" if "iam" in arguments else "cluster"
        return subprocess.CompletedProcess(arguments, 0, json.dumps(runtime[key]), "")
    return run


def _resolve(runtime, **kwargs):
    spec = runtime["spec"]
    project, cluster = spec.cluster_targets()[0]
    return identity.resolve_storage_identity(spec, project, cluster, **kwargs)


def test_identity_proves_registered_provider_and_connection(prepared):
    result = _resolve(prepared)
    assert result.kubeconfig == prepared["kubeconfig"]
    assert result.project_id == "project-test"
    assert result.cluster_id == "cluster-test"
    assert len(result.evidence_sha256) == 64
    assert "project-test" not in repr(result)
    assert len(prepared["calls"]) == 3
    assert all(not path.exists() for path in prepared["temporary"])


def test_identity_pins_profile_without_mutating_environment(prepared, monkeypatch):
    for name in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE", "TF_VAR_iam_token"):
        monkeypatch.setenv(name, "unit-only-placeholder")
    _resolve(prepared)
    for arguments, environment in prepared["calls"]:
        assert arguments[1:3] == ["--profile", "operator-test"]
        assert environment["NEBIUS_PROFILE"] == "operator-test"
        assert "NEBIUS_IAM_TOKEN" not in environment
        assert "NPA_NEBIUS_IAM_TOKEN" not in environment
        assert "NEBIUS_IAM_TOKEN_FILE" not in environment
        assert "TF_VAR_iam_token" not in environment


@pytest.mark.parametrize("section,key,value", [
    ("metadata", "id", "project-other"), ("metadata", "parent_id", "tenant-other"),
    ("metadata", "deleted_at", "test"), ("spec", "region", "other-region"),
    ("status", "container_state", "DELETING"), ("status", "suspension_state", "SUSPENDED"),
])
def test_provider_project_mismatch_fails_before_credentials(prepared, section, key, value):
    prepared["project"][section][key] = value
    with pytest.raises(identity.StorageIdentityError, match="project identity"):
        _resolve(prepared)
    assert not prepared["temporary"]


@pytest.mark.parametrize("section,key,value", [
    ("metadata", "id", "cluster-other"), ("metadata", "parent_id", "project-other"),
    ("metadata", "name", "other"), ("metadata", "deleted_at", "test"),
    ("status", "state", "DELETING"),
])
def test_provider_cluster_mismatch_fails_before_credentials(prepared, section, key, value):
    prepared["cluster"][section][key] = value
    with pytest.raises(identity.StorageIdentityError, match="cluster identity"):
        _resolve(prepared)
    assert not prepared["temporary"]


@pytest.mark.parametrize("section,key,value", [
    ("cluster", "server", "https://other.example.invalid"),
    ("cluster", "certificate-authority-data", "b3RoZXItY2VydGlmaWNhdGU="),
    ("exec", "args", ["--profile", "other", "iam", "get-access-token"]),
    ("exec", "command", "other-command"),
])
def test_stale_kubeconfig_fails_and_cleans_temporary_credentials(prepared, section, key, value):
    fresh = prepared["fresh"]
    target = fresh["clusters"][0]["cluster"] if section == "cluster" else fresh["users"][0]["user"]["exec"]
    target[key] = value
    with pytest.raises(identity.StorageIdentityError, match="stale"):
        _resolve(prepared)
    assert all(not path.exists() for path in prepared["temporary"])


@pytest.mark.parametrize("change", ["context", "tls", "certificate", "user", "multiple"])
def test_untrusted_registered_kubeconfig_is_rejected(prepared, change):
    document = copy.deepcopy(prepared["fresh"])
    if change == "context":
        document["current-context"] = "other"
    elif change == "tls":
        document["clusters"][0]["cluster"]["insecure-skip-tls-verify"] = True
    elif change == "certificate":
        del document["clusters"][0]["cluster"]["certificate-authority-data"]
    elif change == "user":
        document["users"][0]["user"] = {"token": "unit-only-placeholder"}
    else:
        document["contexts"].append(document["contexts"][0])
    _write_document(prepared["kubeconfig"], document)
    with pytest.raises(identity.StorageIdentityError):
        _resolve(prepared)
    assert not prepared["temporary"]


@pytest.mark.parametrize("change", ["missing", "permissions", "project", "metadata", "node_count"])
def test_untrusted_registration_is_rejected(prepared, change):
    registered = prepared["registered"]
    root = identity.cluster_state.CLUSTERS_DIR
    if change == "missing":
        prepared["kubeconfig"].unlink()
    elif change == "permissions":
        prepared["kubeconfig"].chmod(0o644)
    elif change == "metadata":
        identity.cluster_state.metadata_file(registered.name).write_text("{}")
    else:
        if change == "project":
            registered.project_id = "project-other"
        else:
            registered.node_count = 2
        save_cluster_state(registered, base_dir=root)
    with pytest.raises(identity.StorageIdentityError):
        _resolve(prepared)


def test_profile_tenant_mismatch_fails_without_provider_access(prepared):
    prepared["spec"].tenant_id = "tenant-other"
    with pytest.raises(identity.StorageIdentityError, match="tenant differ"):
        _resolve(prepared)
    assert not prepared["calls"]


def test_provider_failure_never_exposes_raw_output(prepared, monkeypatch):
    monkeypatch.setattr(identity, "run_capture", lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 1, "private-receipt", "private-receipt"))
    with pytest.raises(identity.StorageIdentityError) as error:
        _resolve(prepared)
    assert "private-receipt" not in str(error.value)


def test_partial_provider_payload_fails_closed(prepared):
    del prepared["project"]["status"]
    with pytest.raises(identity.StorageIdentityError, match="incomplete"):
        _resolve(prepared)


def test_disabled_target_selection_requires_no_registration():
    spec = _spec()
    spec.projects[0].clusters[0].enable_filestore = False
    assert identity.resolve_storage_targets(spec) == spec.cluster_targets()


def test_target_selection_matches_keys_and_overridden_display_names():
    spec = _spec()
    assert identity.resolve_storage_targets(spec, only_projects=["team"])
    assert identity.resolve_storage_targets(spec, only_projects=["custom-team"], project_prefix="custom-")
    assert identity.resolve_storage_targets(spec, only_clusters=["training"])


@pytest.mark.parametrize("selectors", [
    {"only_projects": ["absent"]}, {"only_clusters": ["absent"]},
    {"only_projects": ["team", "absent"]}, {"only_clusters": ["training", "absent"]},
])
def test_unknown_selector_fails_closed(selectors):
    with pytest.raises(identity.StorageIdentityError, match="selector"):
        identity.resolve_storage_targets(_spec(), **selectors)


def test_cluster_selection_is_scoped_by_project():
    spec = _spec()
    other = ClusterSpec(name="other-cluster", cpu_nodes=NodePoolSpec(count=1))
    spec.projects.append(ProjectSpec(name="other", clusters=[other]))
    with pytest.raises(identity.StorageIdentityError, match="selected projects"):
        identity.resolve_storage_targets(spec, only_projects=["team"], only_clusters=["other-cluster"])


def test_generated_credential_paths_are_unique(prepared):
    first = _resolve(prepared)
    second = _resolve(prepared)
    assert prepared["temporary"][0] != prepared["temporary"][1]
    assert first.evidence_sha256 == second.evidence_sha256
    assert all(not path.exists() for path in prepared["temporary"])


def test_existing_filesystem_attachment_is_enabled(prepared):
    cluster = prepared["spec"].projects[0].clusters[0]
    cluster.enable_filestore = False
    cluster.existing_filestore = "filesystem-test"
    assert _resolve(prepared).kubeconfig == prepared["kubeconfig"]


def test_invalid_fleet_path_is_rejected_before_registration(prepared):
    prepared["spec"].name = "../unowned"
    with pytest.raises(identity.StorageIdentityError, match="declaration is invalid"):
        _resolve(prepared)
    assert not prepared["calls"]


def test_storage_client_uses_verified_snapshot_and_private_environment(prepared, monkeypatch):
    from kubernetes import config

    verified = _resolve(prepared)
    observed = []
    @contextmanager
    def client_from_snapshot(snapshot, *, persist_config, temp_file_path):
        observed.append(snapshot)
        assert persist_config is False
        directory = Path(temp_file_path)
        assert directory.is_dir() and directory.stat().st_mode & 0o077 == 0
        (directory / "certificate").write_text("public-unit-certificate")
        yield "test-client"
        observed.append(directory)
    monkeypatch.setattr(config, "new_client_from_config_dict", client_from_snapshot)
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "ambient-unit-placeholder")
    monkeypatch.setenv("NEBIUS_PROFILE", "ambient-other")
    prepared["kubeconfig"].write_text("changed after verification")
    with identity.storage_client(verified) as client:
        assert client == "test-client"
    environment = {entry["name"]: entry["value"] for entry in observed[0]["users"][0]["user"]["exec"]["env"]}
    assert environment["NEBIUS_PROFILE"] == "operator-test"
    assert environment["NPA_NEBIUS_PROFILE"] == "operator-test"
    assert all(environment[name] == "" for name in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN",
                                                   "NEBIUS_IAM_TOKEN_FILE", "TF_VAR_iam_token"))
    assert os.environ["NEBIUS_IAM_TOKEN"] == "ambient-unit-placeholder"
    assert os.environ["NEBIUS_PROFILE"] == "ambient-other"
    assert "env" not in json.loads(verified.configuration_json)["users"][0]["user"]["exec"]
    assert not observed[-1].exists()


def test_storage_client_rejects_missing_verified_snapshot():
    verified = identity.StorageIdentity(Path("test"), "test", "test", "test", "test", "test", "a" * 64)
    with pytest.raises(identity.StorageIdentityError, match="snapshot"):
        with identity.storage_client(verified):
            pytest.fail("an unverified snapshot must never open a client")


def test_private_identity_receipt_reproduces_hash_without_auth_configuration(prepared):
    verified = _resolve(prepared)
    assert hashlib.sha256(verified.evidence_json.encode()).hexdigest() == verified.evidence_sha256
    evidence = json.loads(verified.evidence_json)
    assert evidence["project"] == prepared["project"]
    assert evidence["cluster"] == prepared["cluster"]
    assert len(evidence["connection_sha256"]) == 64
    assert "certificate-authority-data" not in verified.evidence_json
    assert "get-access-token" not in verified.evidence_json
    assert "configuration_json" not in repr(verified)
    assert "evidence_json" not in repr(verified)


def test_repeated_clients_get_independent_ca_files_without_global_cache(prepared, monkeypatch):
    from kubernetes.config.exec_provider import ExecProvider

    monkeypatch.setattr(ExecProvider, "run", lambda *args, **kwargs: {"token": "unit-token-placeholder"})
    verified = _resolve(prepared)
    with identity.storage_client(verified) as first:
        first_path = Path(first.configuration.ssl_ca_cert)
        assert first_path.read_bytes() == b"test-certificate"
        with identity.storage_client(verified) as second:
            second_path = Path(second.configuration.ssl_ca_cert)
            assert second_path != first_path
            assert second_path.read_bytes() == first_path.read_bytes()
        assert not second_path.exists()
        assert first_path.is_file()
    assert not first_path.exists()
    with identity.storage_client(verified) as third:
        assert Path(third.configuration.ssl_ca_cert).is_file()


@pytest.mark.parametrize("environment", [None, [], [{"name": "CUSTOM_SETTING", "value": "unit"}]])
def test_optional_provider_exec_environment_is_supported(prepared, monkeypatch, environment):
    from kubernetes.config.exec_provider import ExecProvider

    monkeypatch.setattr(ExecProvider, "run", lambda *args, **kwargs: {"token": "unit-token-placeholder"})
    prepared["fresh"]["users"][0]["user"]["exec"]["env"] = environment
    _write_document(prepared["kubeconfig"], prepared["fresh"])
    verified = _resolve(prepared)
    with identity.storage_client(verified) as api:
        assert Path(api.configuration.ssl_ca_cert).is_file()
