"""Resolve Fleet storage targets and prove their registered provider identity."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from npa.cluster import state as cluster_state
from npa.cluster_backends.process import BackendCommandError, require_bin, run_capture
from npa.fleet.spec import ClusterSpec, FleetSpec, FleetSpecError, ProjectSpec


class StorageIdentityError(RuntimeError):
    """A selected storage target lacks trustworthy, current identity evidence."""


@dataclass(frozen=True)
class StorageIdentity:
    """Carry private connection identity and its publication-safe evidence hash.

    Args:
        kubeconfig: Verified registered kubeconfig, never a global context.
        project_id: Immutable provider project identity.
        cluster_id: Immutable provider cluster identity.
        tenant_id: Verified parent tenant identity.
        region: Verified provider region.
        profile: Explicit authentication profile for every provider operation.
        evidence_sha256: Hash binding the current identity evidence.
        configuration_json: Immutable verified connection snapshot for API clients.
        evidence_json: Private provider proof and a connection hash for durable receipts.
    """

    kubeconfig: Path = field(repr=False)
    project_id: str = field(repr=False)
    cluster_id: str = field(repr=False)
    tenant_id: str = field(repr=False)
    region: str = field(repr=False)
    profile: str = field(repr=False)
    evidence_sha256: str
    configuration_json: str = field(default="", repr=False, compare=False)
    evidence_json: str = field(default="", repr=False, compare=False)


def resolve_fleet_targets(
    spec: FleetSpec, *, only_projects: list[str] | None = None,
    only_clusters: list[str] | None = None, project_prefix: str | None = None,
    profile: str | None = None,
) -> list[tuple[ProjectSpec, ClusterSpec]]:
    """Select exact declared Fleet targets.

    Args:
        spec: Desired Fleet declaration.
        only_projects: Project keys or display names to include.
        only_clusters: Cluster names within the selected projects.
        project_prefix: Override used when matching project display names.
        profile: Authentication override, accepted consistently by Fleet APIs.
    Returns:
        Selected project and cluster pairs in declaration order.
    Raises:
        StorageIdentityError: A selector does not match the selected scope.
    """
    del profile
    _validate_spec(spec)
    prefix = spec.project_prefix if project_prefix is None else project_prefix
    projects = _selected_projects(spec, only_projects, prefix)
    targets = [(project, cluster) for project in projects for cluster in project.clusters]
    if only_clusters:
        names = {cluster.name for _, cluster in targets}
        if set(only_clusters) - names:
            raise StorageIdentityError("cluster selector has no target in selected projects")
        targets = [(project, cluster) for project, cluster in targets
                   if cluster.name in only_clusters]
    if not targets:
        raise StorageIdentityError("verification selected no Fleet targets")
    return targets


def resolve_storage_targets(
    spec: FleetSpec, *, only_projects: list[str] | None = None,
    only_clusters: list[str] | None = None, project_prefix: str | None = None,
    profile: str | None = None,
) -> list[tuple[ProjectSpec, ClusterSpec]]:
    """Select storage-verification targets, including disabled filesystems.

    Args:
        spec: Desired Fleet declaration.
        only_projects: Project keys or display names to include.
        only_clusters: Cluster names within the selected projects.
        project_prefix: Override used when matching project display names.
        profile: Authentication override accepted consistently by Fleet APIs.
    Returns:
        Selected project and cluster pairs in declaration order.
    Raises:
        StorageIdentityError: A selector does not match the selected scope.
    """
    return resolve_fleet_targets(
        spec, only_projects=only_projects, only_clusters=only_clusters,
        project_prefix=project_prefix, profile=profile,
    )


def _selected_projects(spec, only_projects, prefix):
    if not only_projects:
        return spec.projects
    names = {name for project in spec.projects
             for name in (project.key(), project.display_name(prefix))}
    if set(only_projects) - names:
        raise StorageIdentityError("project selector has no declared Fleet target")
    return [project for project in spec.projects if project.key() in only_projects
            or project.display_name(prefix) in only_projects]


def resolve_fleet_identity(
    spec: FleetSpec, project: ProjectSpec, cluster: ClusterSpec, *,
    profile: str | None = None, project_prefix: str | None = None,
) -> StorageIdentity:
    """Bind a registered kubeconfig to a freshly verified Fleet target.

    Args:
        spec: Desired Fleet declaration.
        project: Selected project declaration.
        cluster: Selected Kubernetes declaration.
        profile: Authentication profile override.
        project_prefix: Override for provider project display-name verification.
    Returns:
        Private immutable identity plus a publication-safe evidence hash.
    Raises:
        StorageIdentityError: Registration, provider, or kubeconfig proof fails.
    """
    _validate_spec(spec)
    if project not in spec.projects or cluster not in project.clusters:
        raise StorageIdentityError("target identity is outside the Fleet declaration")
    if cluster.backend_name() != "mk8s":
        raise StorageIdentityError("target identity requires the mk8s backend")
    registered = _registered_target(spec, project, cluster)
    selected_profile, tenant = _profile_scope(spec, profile)
    region = project.region or spec.region or registered.region
    binary = _provider_binary()
    project_proof = _verify_project(binary, selected_profile, registered, tenant, region)
    _verify_project_name(spec, project, project_proof, project_prefix)
    cluster_proof = _verify_cluster(binary, selected_profile, registered, cluster)
    kubeconfig = _registered_kubeconfig(registered)
    connection = _verify_connection(binary, selected_profile, registered, kubeconfig)
    evidence = _identity_evidence(project_proof, cluster_proof, connection)
    digest = hashlib.sha256(evidence.encode()).hexdigest()
    snapshot = _connection_snapshot(registered.name, connection)
    return StorageIdentity(kubeconfig, registered.project_id, registered.cluster_id,
                           tenant, region, selected_profile, digest, snapshot, evidence)


def resolve_storage_identity(
    spec: FleetSpec, project: ProjectSpec, cluster: ClusterSpec, *,
    profile: str | None = None, project_prefix: str | None = None,
) -> StorageIdentity:
    """Bind a filesystem-enabled Fleet target to verified provider identity.

    Args:
        spec: Desired Fleet declaration.
        project: Selected project declaration.
        cluster: Selected filesystem-enabled Kubernetes declaration.
        profile: Authentication profile override.
        project_prefix: Override for provider project display-name verification.
    Returns:
        Private immutable identity plus a publication-safe evidence hash.
    Raises:
        StorageIdentityError: Storage or provider identity validation fails.
    """
    enabled = cluster.enable_filestore or bool(cluster.existing_filestore)
    if not enabled:
        raise StorageIdentityError("storage identity requires an enabled filesystem")
    return resolve_fleet_identity(
        spec, project, cluster, profile=profile, project_prefix=project_prefix,
    )


def _identity_evidence(project, cluster, connection):
    connection_bytes = json.dumps(connection, sort_keys=True).encode()
    evidence = {"project": project, "cluster": cluster,
                "connection_sha256": hashlib.sha256(connection_bytes).hexdigest()}
    return json.dumps(evidence, sort_keys=True)


@contextmanager
def storage_client(identity: StorageIdentity) -> Iterator[Any]:
    """Open a verified Kubernetes snapshot with isolated profile authentication.

    Args:
        identity: Freshly verified Fleet storage identity and connection snapshot.
    Returns:
        Context manager yielding the private Kubernetes API client.
    Raises:
        StorageIdentityError: The verified snapshot is unavailable or malformed.
        kubernetes.config.ConfigException: Kubernetes authentication cannot load.
    """
    from kubernetes import config

    try:
        snapshot = json.loads(identity.configuration_json)
    except ValueError as exc:
        raise StorageIdentityError("verified connection snapshot is unavailable") from exc
    if not isinstance(snapshot, dict):
        raise StorageIdentityError("verified connection snapshot is malformed")
    _isolate_exec_environment(snapshot, identity.profile)
    with tempfile.TemporaryDirectory(prefix="npa-storage-client-") as directory:
        _scope_certificate(snapshot, Path(directory))
        with config.new_client_from_config_dict(snapshot, persist_config=False,
                                               temp_file_path=directory) as api:
            yield api


def _scope_certificate(snapshot, directory):
    try:
        connection = snapshot["clusters"][0]["cluster"]
        certificate = base64.b64decode(connection.pop("certificate-authority-data"), validate=True)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise StorageIdentityError("verified certificate authority is malformed") from exc
    path = directory / "certificate-authority"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(certificate)
    # The client globally caches embedded CA temp paths even after their deletion.
    connection["certificate-authority"] = str(path)


def _connection_snapshot(context, connection):
    configuration = {"apiVersion": "v1", "kind": "Config", "current-context": context,
                     "contexts": [{"name": context, "context": connection["context"]}],
                     "clusters": [{"name": connection["context"]["cluster"],
                                   "cluster": connection["cluster"]}],
                     "users": [{"name": connection["context"]["user"],
                                "user": connection["user"]}]}
    return json.dumps(configuration, sort_keys=True)


def _isolate_exec_environment(snapshot, profile):
    try:
        execution = snapshot["users"][0]["user"]["exec"]
        environment = {entry["name"]: entry["value"] for entry in execution.get("env") or []}
    except (KeyError, IndexError, TypeError) as exc:
        raise StorageIdentityError("verified exec identity is malformed") from exc
    for name in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE",
                 "TF_VAR_iam_token"):
        environment[name] = ""
    environment.update(NEBIUS_PROFILE=profile, NPA_NEBIUS_PROFILE=profile)
    execution["env"] = [{"name": name, "value": value} for name, value in environment.items()]


def _validate_spec(spec):
    try:
        spec.validate()
    except FleetSpecError as exc:
        raise StorageIdentityError("Fleet declaration is invalid") from exc


def _read_mapping(path: Path, *, yaml_document: bool = False) -> dict[str, Any]:
    try:
        text = path.read_text()
        payload = yaml.safe_load(text) if yaml_document else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise StorageIdentityError("identity configuration is unavailable or malformed") from exc
    if not isinstance(payload, dict):
        raise StorageIdentityError("identity configuration is not an object")
    return payload


def _registered_target(spec, project, cluster):
    context = f"fleet-{spec.name}-{project.key()}-{cluster.name}"
    payload = _read_mapping(cluster_state.state_file(context))
    metadata = _read_mapping(cluster_state.metadata_file(context))
    required = {"managed_by": "npa fleet", "fleet": spec.name, "project_key": project.key()}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise StorageIdentityError("registered target does not belong to this Fleet selection")
    try:
        registered = cluster_state.ClusterState.from_dict(payload)
    except (ValueError, RuntimeError) as exc:
        raise StorageIdentityError("registered cluster identity is malformed") from exc
    matches = (registered.name == context and registered.provider_name == cluster.name
               and registered.node_count == cluster.cpu_count() + cluster.gpu_count()
               and bool(registered.project_id) and bool(registered.cluster_id))
    if not matches or (project.project_id and registered.project_id != project.project_id):
        raise StorageIdentityError("registered cluster identity differs from the Fleet declaration")
    return registered


def _profile_scope(spec, explicit):
    configuration = _read_mapping(Path.home() / ".nebius" / "config.yaml", yaml_document=True)
    selected = explicit or spec.profile or os.environ.get("NPA_NEBIUS_PROFILE")
    selected = selected or os.environ.get("NEBIUS_PROFILE") or configuration.get("default")
    profiles = configuration.get("profiles", {})
    entry = profiles.get(selected, {}) if isinstance(profiles, dict) else {}
    tenant = entry.get("tenant-id") if isinstance(entry, dict) else None
    if not isinstance(selected, str) or not selected or not isinstance(tenant, str) or not tenant:
        raise StorageIdentityError("authentication profile has no authoritative tenant scope")
    if spec.tenant_id and spec.tenant_id != tenant:
        raise StorageIdentityError("authentication profile and Fleet tenant differ")
    return selected, tenant


def _provider_binary():
    try:
        return require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    except BackendCommandError as exc:
        raise StorageIdentityError("provider CLI is unavailable") from exc


def _provider_environment(profile):
    environment = os.environ.copy()
    for name in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE",
                 "TF_VAR_iam_token"):
        environment.pop(name, None)
    environment.update(NEBIUS_PROFILE=profile, NPA_NEBIUS_PROFILE=profile)
    return environment


def _provider_command(binary, profile, arguments):
    try:
        result = run_capture([binary, "--profile", profile, *arguments], check=False,
                             env=_provider_environment(profile))
    except BackendCommandError as exc:
        raise StorageIdentityError("provider identity operation could not complete") from exc
    if result.returncode:
        raise StorageIdentityError("provider identity operation failed")
    return result


def _provider_document(binary, profile, arguments):
    result = _provider_command(binary, profile, [*arguments, "--format", "json"])
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise StorageIdentityError("provider identity response is malformed") from exc
    if not isinstance(payload, dict):
        raise StorageIdentityError("provider identity response is not an object")
    return payload


def _mapping(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict):
        raise StorageIdentityError("provider identity evidence is incomplete")
    return value


def _verify_project(binary, profile, registered, tenant, region):
    payload = _provider_document(binary, profile,
                                 ["iam", "project", "get", "--id", registered.project_id])
    metadata = _mapping(payload, "metadata")
    status = _mapping(payload, "status")
    declared = _mapping(payload, "spec")
    actual_region = declared.get("region") or status.get("region") or metadata.get("region")
    state = status.get("container_state") or status.get("project_state")
    matches = (metadata.get("id") == registered.project_id
               and (metadata.get("parent_id") or metadata.get("parentId")) == tenant
               and bool(region) and actual_region == region and registered.region == region
               and state == "ACTIVE" and status.get("suspension_state", "NONE") == "NONE")
    if not matches or _deleting(metadata):
        raise StorageIdentityError("provider project identity or active state does not match")
    return payload


def _deleting(metadata):
    return any(metadata.get(key) for key in
               ("deleted_at", "deletedAt", "deletion_timestamp", "deletionTimestamp"))


def _verify_project_name(spec, project, payload, prefix):
    if project.project_id:
        return
    selected_prefix = spec.project_prefix if prefix is None else prefix
    if payload["metadata"].get("name") != project.display_name(selected_prefix):
        raise StorageIdentityError("provider project name differs from the Fleet declaration")


def _verify_cluster(binary, profile, registered, cluster):
    payload = _provider_document(binary, profile,
                                 ["mk8s", "cluster", "get", "--id", registered.cluster_id])
    metadata = _mapping(payload, "metadata")
    status = _mapping(payload, "status")
    matches = (metadata.get("id") == registered.cluster_id
               and (metadata.get("parent_id") or metadata.get("parentId")) == registered.project_id
               and metadata.get("name") == cluster.name and status.get("state") == "RUNNING")
    if not matches or _deleting(metadata):
        raise StorageIdentityError("provider cluster identity or active state does not match")
    return payload


def _registered_kubeconfig(registered):
    expected = cluster_state.kubeconfig_file(registered.name)
    selected = Path(registered.kubeconfig_path).expanduser()
    if not registered.kubeconfig_path or selected.resolve() != expected.resolve():
        raise StorageIdentityError("registered kubeconfig path does not match its owned context")
    if not selected.is_file() or selected.stat().st_mode & 0o077:
        raise StorageIdentityError("registered kubeconfig is unavailable or not owner-private")
    return selected


def _verify_connection(binary, profile, registered, kubeconfig):
    actual = _connection_document(kubeconfig, registered.name)
    with tempfile.TemporaryDirectory(prefix="npa-storage-identity-") as directory:
        fresh = Path(directory) / "kubeconfig"
        arguments = ["mk8s", "cluster", "get-credentials", "--id", registered.cluster_id,
                     "--external", "--force", "--kubeconfig", str(fresh),
                     "--context-name", registered.name]
        _provider_command(binary, profile, arguments)
        expected = _connection_document(fresh, registered.name)
    if actual != expected:
        raise StorageIdentityError("registered endpoint, certificate, or exec identity is stale")
    return actual


def _single_named_entry(payload, key, name):
    entries = payload.get(key)
    if not isinstance(entries, list) or len(entries) != 1:
        raise StorageIdentityError("registered kubeconfig must contain exactly one target")
    entry = entries[0]
    if not isinstance(entry, dict) or entry.get("name") != name:
        raise StorageIdentityError("kubeconfig context references a different identity")
    return entry


def _connection_document(path, context_name):
    payload = _read_mapping(path, yaml_document=True)
    if payload.get("current-context") != context_name:
        raise StorageIdentityError("kubeconfig current context differs from registration")
    context = _single_named_entry(payload, "contexts", context_name)
    context = _mapping(context, "context")
    cluster = _single_named_entry(payload, "clusters", context.get("cluster"))
    user = _single_named_entry(payload, "users", context.get("user"))
    connection = _mapping(cluster, "cluster")
    authentication = _mapping(user, "user")
    if not connection.get("certificate-authority-data") or connection.get("insecure-skip-tls-verify"):
        raise StorageIdentityError("kubeconfig lacks verified certificate authority data")
    if not str(connection.get("server", "")).startswith("https://"):
        raise StorageIdentityError("kubeconfig does not use a TLS endpoint")
    if set(authentication) != {"exec"} or not isinstance(authentication["exec"], dict):
        raise StorageIdentityError("kubeconfig must use provider exec authentication")
    return {"context": context, "cluster": connection, "user": authentication}
