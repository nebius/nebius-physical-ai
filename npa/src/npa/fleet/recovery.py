"""Adopt one verified interrupted Fleet apply without Terraform mutation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

from npa.cluster_backends.mk8s_execution import _context_name
from npa.cluster_backends.mk8s_model import MK8sProjectIdentity, as_mk8s_desired
from npa.cluster_backends.mk8s_render import render_tfvars
from npa.fleet.operation_lock import FleetOperationLockError, target_operation_lock
from npa.fleet.release import MK8sReleaseError, select_installation_release


class FleetRecoveryError(RuntimeError):
    """An interrupted Fleet apply lacks the proof required for safe adoption."""


_ENV_SIDECAR = ".npa-fleet-env.json"
_OWNER_MARKER = ".npa-backend-owner.json"
_STATE_FILE = "terraform.tfstate"
_TERRAFORM_STATE_PATHS = frozenset(
    {_STATE_FILE, "terraform.tfstate.backup", ".terraform.tfstate.lock.info"}
)


@dataclass(frozen=True)
class _RecoveryTarget:
    spec: Any
    project: Any
    cluster: Any
    prefix: str
    fleet_root: Path
    install_dir: Path


@dataclass(frozen=True)
class _RecoveryInputs:
    sidecar: dict[str, Any]
    saved: dict[str, str]
    release: Any
    desired: Any
    cluster_id: str


@dataclass(frozen=True)
class _ProviderSession:
    nebius_bin: str
    environment: dict[str, str]
    project: MK8sProjectIdentity


def recover_interrupted_fleet_apply(
    spec,
    *,
    only_projects: list[str] | None = None,
    only_clusters: list[str] | None = None,
    project_prefix: str | None = None,
    profile: str | None = None,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Adopt one terminal MK8s apply only after exact local and live checks."""
    target = _target(
        spec,
        only_projects=only_projects,
        only_clusters=only_clusters,
        project_prefix=project_prefix,
        work_root=work_root,
    )
    try:
        with target_operation_lock(target.install_dir):
            return _recover_locked(target, profile=profile)
    except FleetOperationLockError as exc:
        raise FleetRecoveryError(str(exc)) from exc


def _target(spec, *, only_projects, only_clusters, project_prefix, work_root):
    from npa.fleet import lifecycle

    spec.validate()
    prefix = spec.project_prefix if project_prefix is None else project_prefix
    project, cluster = _selected_target(spec, only_projects, only_clusters, prefix)
    if cluster.backend_name() != "mk8s":
        raise FleetRecoveryError("interrupted Fleet adoption supports only mk8s")
    root = Path(work_root or lifecycle._default_work_root()).expanduser()
    fleet_root = root / spec.name
    install_dir = fleet_root / project.key() / cluster.name
    _assert_target_path(fleet_root, install_dir)
    return _RecoveryTarget(spec, project, cluster, prefix, fleet_root, install_dir)


def _recover_locked(target: _RecoveryTarget, *, profile: str | None) -> dict[str, Any]:
    workdir = target.install_dir / "k8s-training"
    _assert_no_live_apply_owner(workdir)
    inputs = _recovery_inputs(target, workdir, profile=profile)
    had_materialized_cluster_id = bool(inputs.cluster_id)
    if had_materialized_cluster_id:
        _assert_context_adoptable(inputs.saved["context"], inputs)
    session, inputs = _verify_provider(target, inputs)
    if not inputs.cluster_id:
        raise FleetRecoveryError("provider verification returned no cluster identity")
    if not had_materialized_cluster_id:
        _assert_context_adoptable(inputs.saved["context"], inputs)
    # Provider verification can take long enough for a concurrent apply to begin.
    # Recheck immediately before writing ordinary Fleet registration state.
    _assert_no_live_apply_owner(workdir)
    return _persist_recovery(target, inputs, session)


def _recovery_inputs(
    target: _RecoveryTarget, workdir: Path, *, profile: str | None
) -> _RecoveryInputs:
    from npa.fleet import lifecycle

    sidecar = _read_mapping(target.install_dir / _ENV_SIDECAR, "Fleet apply sidecar")
    _assert_owner_marker(target)
    saved = _saved_identity(sidecar, target, profile)
    release = _saved_release(target.install_dir)
    desired = as_mk8s_desired(target.cluster)
    ssh_public_key = lifecycle._resolve_ssh_public_key(target.spec.ssh_public_key)
    _assert_materialized_spec(workdir, desired, ssh_public_key)
    cluster_id = _terraform_cluster_id_or_unmaterialized(
        workdir, saved["project_id"], target.cluster.name
    )
    _assert_state_subnet(workdir, saved["subnet_id"])
    _assert_unregistered(target.fleet_root, target.project.key(), target.cluster.name)
    return _RecoveryInputs(sidecar, saved, release, desired, cluster_id)


def _verify_provider(
    target: _RecoveryTarget, inputs: _RecoveryInputs
) -> tuple[_ProviderSession, _RecoveryInputs]:
    from npa.fleet import lifecycle

    session = _provider_session(target, inputs.saved)
    ssh_public_key = lifecycle._resolve_ssh_public_key(target.spec.ssh_public_key)
    cluster_id = lifecycle._resolve_verified_unchanged_target(
        project=session.project,
        cluster=inputs.desired,
        prefix=target.prefix,
        tenant_id=inputs.saved["tenant_id"],
        region=inputs.saved["region"],
        ssh_public_key=ssh_public_key,
        fleet_root=target.fleet_root,
        nebius_bin=session.nebius_bin,
        profile=inputs.saved["profile"],
        env=session.environment,
    )
    if not cluster_id:
        raise FleetRecoveryError(
            "provider project, terminal cluster, or exact topology could not be verified"
        )
    if inputs.cluster_id and cluster_id != inputs.cluster_id:
        raise FleetRecoveryError(
            "provider verification returned a different immutable cluster identity"
        )
    return session, replace(inputs, cluster_id=cluster_id)


def _provider_session(
    target: _RecoveryTarget, saved: dict[str, str]
) -> _ProviderSession:
    from npa.fleet import lifecycle

    environment = lifecycle._nebius_cli_env()
    if saved["profile"]:
        environment["NEBIUS_PROFILE"] = saved["profile"]
        environment["NPA_NEBIUS_PROFILE"] = saved["profile"]
    project = MK8sProjectIdentity(
        project_key=target.project.key(),
        project_id=saved["project_id"],
        project_name=target.project.name,
        expected_provider_name=target.project.display_name(target.prefix),
    )
    binary = lifecycle._require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    return _ProviderSession(binary, environment, project)


def _persist_recovery(
    target: _RecoveryTarget, inputs: _RecoveryInputs, session: _ProviderSession
) -> dict[str, Any]:
    from npa.fleet import lifecycle

    kubeconfig = target.install_dir / "kubeconfig"
    lifecycle._write_kubeconfig(
        session.nebius_bin,
        inputs.cluster_id,
        kubeconfig,
        inputs.saved["context"],
        session.environment,
        inputs.saved["profile"],
    )
    lifecycle._persist_npa_cluster_identity(
        context=inputs.saved["context"],
        cluster_id=inputs.cluster_id,
        project_id=inputs.saved["project_id"],
        region=inputs.saved["region"],
        cluster=inputs.desired,
        subnet_id=inputs.saved["subnet_id"],
        kubeconfig_path=kubeconfig,
        fleet_name=target.spec.name,
        project_key=target.project.key(),
    )
    recovered = _recovered_record(target, inputs, kubeconfig)
    lifecycle._write_json_file(
        target.install_dir / _ENV_SIDECAR,
        {
            **inputs.sidecar,
            "cluster_id": inputs.cluster_id,
            "status": recovered["status"],
        },
    )
    lifecycle._upsert_fleet_state(
        target.fleet_root, _fleet_metadata(target, inputs), [recovered]
    )
    return recovered


def _recovered_record(target, inputs, kubeconfig) -> dict[str, Any]:
    return {
        "backend": "mk8s",
        "project_key": target.project.key(),
        "project_id": inputs.saved["project_id"],
        "cluster_name": target.cluster.name,
        "cluster_id": inputs.cluster_id,
        "region": inputs.saved["region"],
        "kube_context": inputs.saved["context"],
        "kubeconfig": str(kubeconfig),
        "status": "recovered-pending-validation",
        "mk8s_recipe_release": inputs.release.persistence(),
    }


def _fleet_metadata(target, inputs) -> dict[str, str]:
    return {
        "name": target.spec.name,
        "tenant_id": inputs.saved["tenant_id"],
        "region": inputs.saved["region"],
        "project_prefix": target.prefix,
        "profile": inputs.saved["profile"],
    }


def _selected_target(spec, only_projects, only_clusters, prefix):
    selected = []
    for project in spec.projects:
        if only_projects and not _project_matches(project, only_projects, prefix):
            continue
        for cluster in project.clusters:
            if not only_clusters or cluster.name in only_clusters:
                selected.append((project, cluster))
    if len(selected) != 1:
        raise FleetRecoveryError(
            "interrupted Fleet adoption requires exactly one declared target"
        )
    return selected[0]


def _project_matches(project, selected: list[str], prefix: str) -> bool:
    return project.key() in selected or project.display_name(prefix) in selected


def _assert_target_path(fleet_root: Path, install_dir: Path) -> None:
    try:
        install_dir.resolve().relative_to(fleet_root.resolve())
    except ValueError as exc:
        raise FleetRecoveryError(
            "Fleet target path escapes the Fleet work root"
        ) from exc


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetRecoveryError(f"{label} is unavailable or malformed") from exc
    if not isinstance(payload, dict):
        raise FleetRecoveryError(f"{label} is not an object")
    return payload


def _assert_owner_marker(target: _RecoveryTarget) -> None:
    marker = _read_mapping(
        target.install_dir / _OWNER_MARKER, "Fleet backend owner marker"
    )
    expected = {
        "fleet_name": target.spec.name,
        "project_key": target.project.key(),
        "cluster_name": target.cluster.name,
        "backend": "mk8s",
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise FleetRecoveryError("Fleet backend owner marker does not match target")


def _saved_identity(sidecar, target: _RecoveryTarget, profile: str | None):
    expected = _expected_sidecar_identity(target, profile, sidecar)
    if not _sidecar_matches(sidecar, target, expected):
        raise FleetRecoveryError(
            "interrupted Fleet sidecar does not match selected spec"
        )
    return {
        "tenant_id": str(sidecar["tenant_id"]),
        "project_id": str(sidecar["project_id"]),
        "region": str(sidecar["region"]),
        "subnet_id": str(sidecar.get("subnet_id") or ""),
        "profile": str(sidecar.get("profile") or ""),
        "context": expected["context"],
    }


def _expected_sidecar_identity(target, profile, sidecar) -> dict[str, str]:
    from npa.fleet import lifecycle

    saved_profile = str(sidecar.get("profile") or "")
    selected_profile = (
        profile if profile is not None else (target.spec.profile or saved_profile)
    )
    expected_region = lifecycle._resolve_region(
        target.project.region or target.spec.region
    )
    expected_tenant = lifecycle._resolve_tenant_id(
        "nebius", target.spec.tenant_id, selected_profile
    )
    expected_context = _context_name(
        target.spec.name, target.project.key(), target.cluster.name
    )
    return {
        "profile": selected_profile,
        "tenant_id": expected_tenant,
        "region": expected_region,
        "context": expected_context,
    }


def _sidecar_matches(
    sidecar, target: _RecoveryTarget, expected: dict[str, str]
) -> bool:
    checks = (
        sidecar.get("backend") == "mk8s",
        sidecar.get("status") == "provisioning",
        not sidecar.get("cluster_id"),
        bool(sidecar.get("project_id")),
        bool(sidecar.get("tenant_id")),
        bool(sidecar.get("region")),
        sidecar.get("cluster_name") == target.cluster.name,
        sidecar.get("context") == expected["context"],
        sidecar.get("profile", "") == expected["profile"],
        sidecar.get("tenant_id") == expected["tenant_id"],
        sidecar.get("region") == expected["region"],
        not target.cluster.subnet_id
        or sidecar.get("subnet_id") == target.cluster.subnet_id,
        not target.project.project_id
        or sidecar.get("project_id") == target.project.project_id,
    )
    return all(checks)


def _saved_release(install_dir: Path):
    try:
        return select_installation_release(install_dir)
    except MK8sReleaseError as exc:
        raise FleetRecoveryError(
            "Fleet target has no valid immutable release record"
        ) from exc


def _assert_materialized_spec(workdir: Path, cluster, ssh_public_key: str) -> None:
    try:
        saved = (workdir / "terraform.tfvars").read_text(encoding="utf-8")
        expected = render_tfvars(
            cluster, ssh_public_key=ssh_public_key, recipe_dir=workdir
        )
    except (OSError, ValueError) as exc:
        raise FleetRecoveryError(
            "materialized Fleet Terraform inputs are unavailable"
        ) from exc
    if _normalize_tfvars(saved) != _normalize_tfvars(expected):
        raise FleetRecoveryError("materialized inputs differ from the selected spec")


def _normalize_tfvars(text: str) -> str:
    from npa.cluster_backends.mk8s_execution import _normalize_tfvars_assignments

    return _normalize_tfvars_assignments(text)


def _terraform_cluster_id(workdir: Path, project_id: str, cluster_name: str) -> str:
    state = _read_mapping(workdir / _STATE_FILE, "Terraform state")
    candidate = _root_cluster_attributes(state.get("resources"))
    if len(candidate) != 1:
        raise FleetRecoveryError(
            "Terraform state does not contain one exact mk8s cluster"
        )
    return _materialized_cluster_id(candidate[0], project_id, cluster_name)


def _terraform_cluster_id_or_unmaterialized(
    workdir: Path, project_id: str, cluster_name: str
) -> str:
    """Allow only one root instance with no attributes for verified recovery."""

    state = _read_mapping(workdir / _STATE_FILE, "Terraform state")
    instances = _root_cluster_instances(state.get("resources"))
    if len(instances) != 1:
        raise FleetRecoveryError(
            "Terraform state does not contain one exact mk8s cluster"
        )
    attributes = instances[0].get("attributes")
    if attributes is None and "attributes" not in instances[0]:
        return ""
    if not isinstance(attributes, dict):
        raise FleetRecoveryError(
            "Terraform state does not contain one exact mk8s cluster"
        )
    return _materialized_cluster_id(attributes, project_id, cluster_name)


def _materialized_cluster_id(
    candidate: dict[str, Any], project_id: str, cluster_name: str
) -> str:
    if (
        candidate.get("parent_id") != project_id
        or candidate.get("name") != cluster_name
    ):
        raise FleetRecoveryError(
            "Terraform cluster identity differs from selected target"
        )
    cluster_id = candidate.get("id")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise FleetRecoveryError("Terraform state has no usable cluster identity")
    return cluster_id


def _root_cluster_attributes(resources: object) -> list[dict[str, Any]]:
    attributes = []
    for instance in _root_cluster_instances(resources):
        candidate = instance.get("attributes")
        if isinstance(candidate, dict):
            attributes.append(candidate)
    return attributes


def _root_cluster_instances(resources: object) -> list[dict[str, Any]]:
    if not isinstance(resources, list):
        raise FleetRecoveryError("Terraform state has no resource inventory")
    instances = []
    for resource in resources:
        if not _is_root_cluster_resource(resource):
            continue
        for instance in resource["instances"]:
            if isinstance(instance, dict) and not instance.get("deposed"):
                instances.append(instance)
    return instances


def _assert_state_subnet(workdir: Path, sidecar_subnet: str) -> None:
    state = _read_mapping(workdir / _STATE_FILE, "Terraform state")
    subnets = _state_node_group_subnets(state.get("resources"))
    if len(subnets) != 1 or not sidecar_subnet or sidecar_subnet not in subnets:
        raise FleetRecoveryError(
            "Terraform node-group subnet differs from the Fleet sidecar"
        )


def _state_node_group_subnets(resources: object) -> set[str]:
    if not isinstance(resources, list):
        raise FleetRecoveryError("Terraform state has no resource inventory")
    subnets = set()
    for resource in resources:
        if not _is_node_group_resource(resource):
            continue
        for instance in resource["instances"]:
            if isinstance(instance, dict) and instance.get("deposed"):
                continue
            subnet = _state_instance_subnet(instance)
            if subnet:
                subnets.add(subnet)
            else:
                raise FleetRecoveryError(
                    "Terraform node-group state has no exact subnet"
                )
    return subnets


def _is_node_group_resource(resource: object) -> bool:
    return bool(
        isinstance(resource, dict)
        and resource.get("mode") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_node_group"
        and isinstance(resource.get("instances"), list)
    )


def _state_instance_subnet(instance: object) -> str:
    if not isinstance(instance, dict) or instance.get("deposed"):
        return ""
    attributes = instance.get("attributes")
    template = attributes.get("template") if isinstance(attributes, dict) else None
    interfaces = (
        template.get("network_interfaces") if isinstance(template, dict) else None
    )
    if not isinstance(interfaces, list) or len(interfaces) != 1:
        return ""
    subnet = interfaces[0].get("subnet_id") if isinstance(interfaces[0], dict) else None
    return subnet if isinstance(subnet, str) and subnet else ""


def _is_root_cluster_resource(resource: object) -> bool:
    return bool(
        isinstance(resource, dict)
        and resource.get("mode") == "managed"
        and resource.get("type") == "nebius_mk8s_v1_cluster"
        and not resource.get("module")
        and isinstance(resource.get("instances"), list)
    )


def _assert_unregistered(fleet_root: Path, project_key: str, cluster_name: str) -> None:
    from npa.fleet import lifecycle

    entries = lifecycle._load_fleet_state(fleet_root).get("clusters", [])
    if not isinstance(entries, list):
        raise FleetRecoveryError("persisted Fleet state is malformed")
    if any(
        isinstance(entry, dict)
        and entry.get("project_key") == project_key
        and entry.get("cluster_name") == cluster_name
        for entry in entries
    ):
        raise FleetRecoveryError("selected Fleet target is already registered")


def _assert_context_adoptable(context: str, inputs: _RecoveryInputs) -> None:
    from npa.cluster.state import load_cluster_state

    existing = load_cluster_state(context)
    if existing and (
        existing.cluster_id != inputs.cluster_id
        or existing.project_id != inputs.saved["project_id"]
    ):
        raise FleetRecoveryError(
            "existing NPA context has a different immutable identity"
        )


def _assert_no_live_apply_owner(
    workdir: Path, *, proc_root: Path = Path("/proc")
) -> None:
    if not proc_root.is_dir():
        raise FleetRecoveryError("cannot inspect local Fleet apply ownership")
    expected = workdir.expanduser().resolve()
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        _assert_process_does_not_apply(process, expected)


def _assert_process_does_not_apply(process: Path, workdir: Path) -> None:
    try:
        cwd = (process / "cwd").resolve()
    except OSError:
        cwd = None
    try:
        arguments = (process / "cmdline").read_bytes().split(b"\0")
    except OSError as exc:
        if cwd == workdir:
            raise FleetRecoveryError(
                "could not inspect a process owning the Fleet workdir"
            ) from exc
        arguments = []
    values = [item.decode("utf-8", "surrogateescape") for item in arguments if item]
    if _is_terraform_owner(values, cwd, workdir):
        raise FleetRecoveryError("a live Terraform apply still owns this Fleet target")
    _assert_no_state_lock_owner(process, workdir, values, cwd)


def _is_terraform_owner(arguments: list[str], cwd: Path | None, workdir: Path) -> bool:
    joined = " ".join(arguments)
    chdir_targets_workdir = any(
        value.startswith("-chdir=")
        and Path(value.removeprefix("-chdir=")).expanduser().resolve() == workdir
        for value in arguments
    )
    owns_workdir = cwd == workdir or chdir_targets_workdir
    is_terraform = "terraform" in joined
    is_target_provider = ".terraform/providers/" in joined and str(workdir) in joined
    return (owns_workdir and is_terraform) or is_target_provider


def _assert_no_state_lock_owner(
    process: Path, workdir: Path, arguments: list[str], cwd: Path | None
) -> None:
    try:
        descriptors = list((process / "fd").iterdir())
    except OSError as exc:
        if _may_own_target(arguments, cwd, workdir):
            raise FleetRecoveryError(
                "could not inspect a process that may own Fleet state"
            ) from exc
        return
    for descriptor in descriptors:
        try:
            held_path = descriptor.resolve()
        except OSError as exc:
            if _may_own_target(arguments, cwd, workdir):
                raise FleetRecoveryError(
                    "could not inspect a process that may own Fleet state"
                ) from exc
            continue
        if held_path.parent == workdir and held_path.name in _TERRAFORM_STATE_PATHS:
            raise FleetRecoveryError(
                "a process still owns this Fleet target Terraform state lock"
            )


def _may_own_target(arguments: list[str], cwd: Path | None, workdir: Path) -> bool:
    joined = " ".join(arguments)
    return cwd == workdir or str(workdir) in joined or ".terraform/providers/" in joined
