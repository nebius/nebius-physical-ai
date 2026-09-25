"""Validate original native evidence for an explicit legacy MK8s recovery."""

from __future__ import annotations

import hashlib
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import tarfile


class AbsenceRecoveryError(ValueError):
    """Evidence cannot authorize terminal recovery of this exact operation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AbsenceRecoveryError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned_bytes(entry: dict) -> bytes:
    path = Path(entry["path"])
    require(
        path.is_absolute() and not path.is_symlink() and path.is_file(),
        "Evidence must be an absolute regular file",
    )
    data = path.read_bytes()
    require(digest(data) == entry["sha256"], "Evidence bytes changed")
    return data


def pinned_json(manifest: dict, key: str) -> dict:
    value = json.loads(pinned_bytes(manifest[key]))
    require(isinstance(value, dict), "Evidence document must be an object")
    return value


def _native_members(data: bytes) -> tuple[dict, dict, str]:
    names = (".npa-fleet-env.json", "k8s-training/terraform.tfstate")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        values = []
        raw_state = b""
        for name in names:
            members = [m for m in archive.getmembers() if m.name == name]
            require(
                len(members) == 1 and members[0].isfile(),
                "Native evidence member is missing, duplicated or nonregular",
            )
            raw = archive.extractfile(members[0]).read()
            values.append(json.loads(raw))
            if name.endswith("terraform.tfstate"):
                raw_state = raw
    return values[0], values[1], digest(raw_state)


def _producer_binding(manifest: dict, journal: dict) -> dict:
    start = pinned_json(manifest, "producer_start")
    result = pinned_json(manifest, "producer_result")
    authority = pinned_json(manifest, "profile_binding")
    require(start["started_at"] == result["started_at"], "Attempt changed")
    require(
        type(result["exit"]) is int
        and result["exit"] != 0
        and result["primary_invoked_process_joined"] is True,
        "Only a joined failed provision can be reconciled",
    )
    require(
        start["profile_binding_sha256"] == manifest["profile_binding"]["sha256"],
        "Authority is not bound by the original producer",
    )
    _producer_outputs(manifest, start, result, journal)
    argv = start["argv"]
    require(argv[1:3] == ["cluster", "up"], "Wrong producing command")
    require(
        argv[argv.index("--context") + 1] == journal["requested_name"],
        "Original context does not bind this operation",
    )
    require(
        argv[argv.index("--project") + 1] == journal["project_alias"],
        "Original project alias does not bind this operation",
    )
    for key in ("project_id", "tenant_id", "region"):
        require(authority[key] == journal[key], "Original authority scope mismatch")
    require(
        result["candidate_job_submitted"] is False,
        "This recovery contract does not cover submitted workloads",
    )
    _attempt_timing(start, result, journal)
    return start


def _producer_outputs(manifest: dict, start: dict, result: dict, journal: dict) -> None:
    require(
        start["runner_sha256"] == digest(pinned_bytes(manifest["producer_runner"])),
        "Original runner source changed",
    )
    for stream in ("stdout", "stderr"):
        require(
            result[stream + "_sha256"] == digest(pinned_bytes(manifest[stream])),
            "Original producer output changed",
        )
    stderr = pinned_bytes(manifest["stderr"]).decode("utf-8", errors="replace")
    require(
        "Provisioning operation: " + journal["operation_id"] in stderr,
        "Original process output does not bind this operation identity",
    )


def _attempt_timing(start: dict, result: dict, journal: dict) -> None:
    def parse(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    created, started = parse(journal["created_at"]), parse(start["started_at"])
    finished, updated = parse(result["finished_at"]), parse(journal["updated_at"])
    require(
        created <= started < finished, "Producer predates this operation generation"
    )
    require(
        updated.replace(microsecond=0) == finished.replace(microsecond=0),
        "Original journal does not describe this finished attempt",
    )


def _source_binding(manifest: dict, revision: str) -> dict:
    repository = Path(manifest["producer_repository"])
    paths = (
        "npa/src/npa/cli/cluster/terraform_lifecycle.py",
        "npa/src/npa/cluster_backends/mk8s.py",
        "npa/src/npa/cluster_backends/mk8s_execution.py",
    )
    require(
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
        "Producer revision must be a full commit identity",
    )
    result = {}
    for path in paths:
        data = subprocess.check_output(
            ["git", "-C", str(repository), "show", revision + ":" + path]
        )
        result[path] = digest(data)
    return result


def _resources(state: dict, journal: dict) -> list[dict]:
    kinds = {
        "nebius_mk8s_v1_cluster": "cluster",
        "nebius_mk8s_v1_node_group": "node-group",
        "nebius_applications_v1alpha1_k8s_release": "application-release",
    }
    local = {"random_string", "random_password", "time_static", "terraform_data"}
    resources = []
    for item in state["resources"]:
        if item["mode"] != "managed" or item["type"] in local:
            continue
        for instance in item["instances"]:
            require(item["type"] in kinds, "Uncovered managed resource in native state")
            attrs = instance["attributes"]
            resources.append(
                {
                    "kind": kinds[item["type"]],
                    "id": attrs["id"],
                    "name": attrs.get("name"),
                    "parent_id": attrs["parent_id"],
                    "cluster_id": attrs.get("cluster_id"),
                }
            )
    clusters = [r for r in resources if r["kind"] == "cluster"]
    require(len(clusters) == 1, "Native state must bind exactly one cluster")
    cluster = clusters[0]
    require(
        cluster["parent_id"] == journal["project_id"]
        and cluster["name"] == journal["requested_name"],
        "Foreign native cluster",
    )
    _check_children(resources, cluster, journal["project_id"])
    return resources


def _check_children(resources: list[dict], cluster: dict, project: str) -> None:
    identities = [r["id"] for r in resources]
    require(
        all(isinstance(i, str) and i for i in identities)
        and len(identities) == len(set(identities)),
        "Invalid or duplicated resource ID",
    )
    for resource in resources:
        if resource["kind"] == "node-group":
            require(resource["parent_id"] == cluster["id"], "Foreign node group")
            require(
                isinstance(resource["name"], str) and bool(resource["name"]),
                "Native node group name is missing",
            )
        elif resource["kind"] == "application-release":
            require(
                resource["parent_id"] == project
                and resource["cluster_id"] == cluster["id"],
                "Foreign application",
            )


def load_legacy_evidence(manifest: dict) -> dict:
    require(manifest["schema_version"] == 1, "Unsupported recovery evidence schema")
    journal = pinned_json(manifest, "original_journal")
    require(journal["operation_id"] == manifest["operation_id"], "Wrong operation")
    require(
        journal["command"] == "npa cluster up"
        and journal["resource_type"] == "cluster",
        "Not a standalone cluster operation",
    )
    start = _producer_binding(manifest, journal)
    state, state_hash, resources = _bound_native(manifest, journal, start)
    return {
        "journal": journal,
        "resources": resources,
        "authority": pinned_json(manifest, "profile_binding"),
        "producer_source": start["source_revision"],
        "producer_files": _source_binding(manifest, start["source_revision"]),
        "state_sha256": state_hash,
        "state_lineage": state["lineage"],
        "legacy_binding": "Original project/name, producer attempt and cleanup hash chain; no original operation_id was added to native records.",
    }


def _bound_native(manifest: dict, journal: dict, start: dict):
    intent = pinned_json(manifest, "cleanup_intent")
    require(
        intent["original_journal_sha256"] == manifest["original_journal"]["sha256"],
        "Cleanup did not retain this original journal",
    )
    require(
        intent["source_revision"] == start["source_revision"],
        "Producer source mismatch",
    )
    archive = pinned_bytes(manifest["backend_archive"])
    require(
        intent["backend_archive_sha256"] == digest(archive), "Cleanup archive mismatch"
    )
    sidecar, state, state_hash = _native_members(archive)
    require(intent["original_state_sha256"] == state_hash, "Native state mismatch")
    for key in ("project_id", "tenant_id", "region"):
        require(sidecar[key] == journal[key], "Foreign native sidecar")
    require(
        sidecar["cluster_name"] == journal["requested_name"],
        "Wrong native cluster name",
    )
    require(sidecar["backend"] == "mk8s", "Wrong native backend")
    resources = _resources(state, journal)
    return state, state_hash, resources
