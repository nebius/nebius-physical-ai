"""Verify retained native managed-job identities and their covered naming scope."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import tempfile

import yaml

from npa.cluster.absent_evidence import digest, pinned_bytes, require


def _managed_ids(error: str, run: str) -> set[int]:
    prefix = f"exact managed-job name {run!r} maps to multiple immutable IDs: "
    matches = re.findall(re.escape(prefix) + r"([0-9]+(?:, [0-9]+)+)", error)
    require(len(matches) == 1, "Original ambiguity is missing or unsupported")
    values = [int(value) for value in matches[0].split(", ")]
    require(all(value > 0 for value in values), "Invalid native managed ID")
    require(len(set(values)) == len(values), "Duplicated native managed ID")
    return set(values)


def _workflow_profiles(ledger: dict) -> None:
    steps = ledger["workflow"]["steps"]
    require(isinstance(steps, list) and bool(steps), "Original task profiles missing")
    for step in steps:
        profile = step["resources_profile"]
        require(profile.get("cloud") == "kubernetes", "Uncovered task cloud")
        require(
            set(profile) <= {"cloud", "cpus", "memory", "kubernetes", "gpus"},
            "Uncovered task resource override",
        )
        kubernetes = profile.get("kubernetes", {})
        require(set(kubernetes) <= {"pod_config"}, "Uncovered task configuration")
        pod = kubernetes.get("pod_config", {})
        require(set(pod) <= {"spec"}, "Uncovered task metadata override")
        require(
            set(pod.get("spec", {})) <= {"imagePullSecrets"},
            "Uncovered task Pod configuration",
        )


def _config(data: bytes, context: str) -> dict:
    config = yaml.safe_load(data)
    require(
        set(config) == {"jobs", "kubernetes", "allowed_clouds"}
        and config["allowed_clouds"] == ["kubernetes"]
        and config["kubernetes"] == {"allowed_contexts": [context]},
        "Native config permits uncovered cloud, context or metadata",
    )
    jobs = config["jobs"]
    require(set(jobs) == {"controller"}, "Uncovered jobs configuration")
    require(
        set(jobs["controller"]) == {"resources"}, "Uncovered controller configuration"
    )
    resources = jobs["controller"]["resources"]
    require(
        resources.get("cloud") == "kubernetes"
        and resources.get("region") == context
        and set(resources) <= {"cloud", "cpus", "memory", "region", "autostop"},
        "Uncovered native controller resources",
    )
    return config


def _wrapper(entry, root: Path, run: str, user: str, configs: dict) -> int:
    path = Path(entry["path"])
    require(
        path.parent == root / "home/.sky/jobs_controller"
        and re.fullmatch(re.escape(run) + r"-[0-9a-f]{4}\.yaml", path.name),
        "Native wrapper does not belong to the original root/run",
    )
    wrapper = yaml.safe_load(pinned_bytes(entry))
    script = wrapper["run"]
    ids = re.findall(r"^job_ids_array=\(([1-9][0-9]*)\s*\)$", script, re.MULTILINE)
    require(len(ids) == 1, "Missing or ambiguous native wrapper job ID")
    identities = re.findall(r"export SKYPILOT_USER_ID='([^']+)'", script)
    require(identities == [user], "Native controller user changed")
    mounts = wrapper["file_mounts"]
    config_paths = [
        value for key, value in mounts.items() if key.endswith(".config_yaml")
    ]
    require(
        len(config_paths) == 1 and config_paths[0] in configs, "Native config missing"
    )
    require("--pool" not in script, "Pool jobs are not covered")
    return int(ids[0])


def _controller(manifest: dict, root: Path, logical: str, context: str) -> dict:
    entry = manifest["native_state"]
    require(
        Path(entry["path"]) == root / "sky-runtime/.sky/state.db",
        "Wrong original native state path",
    )
    rows = _controller_rows(manifest, logical)
    require(len(rows) == 1, "Original native controller configuration missing")
    config = yaml.safe_load(rows[0][0])
    provider = config["provider"]
    require(
        all(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value)
            for value in (config["cluster_name"], provider["namespace"])
        ),
        "Invalid native Kubernetes identity",
    )
    require(
        provider["type"] == "external"
        and provider["module"] == "sky.provision.kubernetes"
        and provider["context"] == context,
        "Foreign native controller",
    )
    services = [item["metadata"]["name"] for item in provider.get("services", [])]
    require(
        all(name.startswith(config["cluster_name"] + "-") for name in services),
        "Uncovered native Service name",
    )
    return {
        "name": config["cluster_name"],
        "namespace": provider["namespace"],
        "services": services,
        "yaml_sha256": digest(rows[0][0].encode()),
    }


def _state_wal(manifest: dict) -> bytes | None:
    path = Path(manifest["native_state"]["path"] + "-wal")
    entry = manifest.get("native_wal")
    if path.exists() and path.stat().st_size:
        require(entry and entry["path"] == str(path), "Original SQLite WAL is unbound")
        return pinned_bytes(entry)
    require(entry is None, "Original SQLite WAL changed")
    return None


def _controller_rows(manifest: dict, logical: str) -> list:
    data = pinned_bytes(manifest["native_state"])
    wal = _state_wal(manifest)
    with tempfile.TemporaryDirectory(prefix="npa-sky-absence-state-") as directory:
        path = Path(directory) / "state.db"
        path.write_bytes(data)
        path.chmod(0o600)
        if wal is not None:
            path.with_name("state.db-wal").write_bytes(wal)
            path.with_name("state.db-wal").chmod(0o600)
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            return connection.execute(
                "SELECT yaml FROM cluster_yaml WHERE cluster_name = ?", (logical,)
            ).fetchall()
        finally:
            connection.close()


def _native_jobs(manifest: dict, trace: dict, run: str, user: str) -> list[int]:
    configs = {
        entry["path"]: _config(pinned_bytes(entry), trace["context"])
        for entry in manifest["native_configs"]
    }
    return [
        _wrapper(entry, Path(trace["root"]), run, user, configs)
        for entry in manifest["native_wrappers"]
    ]


def native_scope(manifest: dict, journal: dict, ledger: dict, trace: dict) -> dict:
    """Derive the original resource suffix scope from native producer evidence.

    Args:
        manifest: Pinned native files, without invented Pod UIDs.
        journal: Original failed operation generation.
        ledger: Original NPA submission ledger.
        trace: Validated original root and target selection.
    Returns:
        Exact managed IDs, stable user, and original controller name/namespace.
    Raises:
        ValueError: Evidence is incomplete, ambiguous, or outside the supported shape.
    """
    run = journal["requested_name"]
    ids = _managed_ids(journal["last_error"], run)
    require(
        ids == _managed_ids(ledger["launch"]["reconciliation_error"], run),
        "Native ID sets differ",
    )
    _workflow_profiles(ledger)
    root = Path(trace["root"])
    user = "npa-" + digest(str(root).encode())[:12]
    found = _native_jobs(manifest, trace, run, user)
    require(
        set(found) == ids and len(found) == len(ids),
        "Missing, repeated, or extra native job wrapper",
    )
    controller = _controller(manifest, root, trace["controller"], trace["context"])
    require(
        user in controller["name"], "Original controller does not bind the native user"
    )
    return {
        "user": user,
        "managed_ids": sorted(ids),
        "context": trace["context"],
        "controller": controller,
    }
