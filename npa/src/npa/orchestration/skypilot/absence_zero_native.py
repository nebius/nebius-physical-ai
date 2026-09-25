"""Require empty original native state and derive the pinned no-job naming scope."""

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import yaml

from npa.cluster.absent_evidence import digest, pinned_bytes, require
from npa.orchestration.skypilot.absence_native import _config, _workflow_profiles
from npa.orchestration.skypilot.absence_target import _selection


_EMPTY_QUERIES = {
    "clusters": "SELECT COUNT(*) FROM clusters",
    "cluster_yaml": "SELECT COUNT(*) FROM cluster_yaml",
    "spot": "SELECT COUNT(*) FROM spot",
    "job_info": "SELECT COUNT(*) FROM job_info",
}


def _empty_database(manifest: dict, root: Path, filename: str, tables: tuple) -> None:
    entry = manifest["empty_native_databases"][filename]
    path = root / "sky-runtime/.sky" / filename
    require(entry["path"] == str(path), "Foreign native database")
    data = pinned_bytes(entry)
    wal = _original_wal(manifest, path)
    with tempfile.TemporaryDirectory(prefix="npa-sky-zero-state-") as directory:
        copy = Path(directory) / filename
        copy.write_bytes(data)
        copy.chmod(0o600)
        if wal is not None:
            copy.with_name(copy.name + "-wal").write_bytes(wal)
            copy.with_name(copy.name + "-wal").chmod(0o600)
        connection = sqlite3.connect(copy.as_uri() + "?mode=ro", uri=True)
        try:
            for table in tables:
                # These are fixed schema names, never manifest-derived SQL.
                count = connection.execute(_EMPTY_QUERIES[table]).fetchone()[0]
                require(count == 0, "Original native state has an accepted identity")
        finally:
            connection.close()


def _original_wal(manifest: dict, path: Path) -> bytes | None:
    wal = path.with_name(path.name + "-wal")
    entry = manifest.get("empty_native_wals", {}).get(path.name)
    if wal.exists() and wal.stat().st_size:
        require(entry and entry["path"] == str(wal), "Unbound native WAL")
        return pinned_bytes(entry)
    require(entry is None, "Original native WAL changed")
    return None


def _native_name(logical: str, user: str) -> str:
    normalized = re.sub(r"[._]", "-", logical).lower()
    if len(normalized) <= 42 - len(user) - 1:
        return normalized + "-" + user
    number = int(hashlib.md5(logical.encode(), usedforsecurity=False).hexdigest(), 16)
    encoded = ""
    while number:
        number, remainder = divmod(number, 36)
        encoded = "0123456789abcdefghijklmnopqrstuvwxyz"[remainder] + encoded
    prefix = normalized[: 42 - 2 - 1 - len(user) - 1].rstrip("-")
    return prefix + "-" + encoded[:2] + "-" + user


def zero_native_scope(manifest: dict, journal: dict, ledger: dict, trace: dict) -> dict:
    """Derive ordinary Sky0.12.2 controller scope without inventing managed IDs.

    Args:
        manifest: Pinned original state and configuration files.
        journal: Original failed operation generation.
        ledger: Exact persisted original submit receipt.
        trace: Bound original command and isolated root.
    Returns:
        Native controller/namespace and stable user with no managed IDs.
    Raises:
        ValueError: State has identities, changed configuration, or unsupported naming.
    """
    root = Path(trace["root"])
    user = "npa-" + digest(str(root).encode())[:12]
    require(trace["controller"] == "sky-jobs-controller-" + user, "Controller differs")
    _workflow_profiles(ledger)
    _original_config(manifest, journal, trace)
    require(
        set(manifest["empty_native_databases"]) == {"state.db", "spot_jobs.db"},
        "Incomplete original native state",
    )
    _empty_database(manifest, root, "state.db", ("clusters", "cluster_yaml"))
    _empty_database(manifest, root, "spot_jobs.db", ("spot", "job_info"))
    return _controller_scope(manifest, trace, user)


def _controller_scope(manifest, trace, user) -> dict:
    selected = _selection(
        pinned_bytes(manifest["original_kubeconfig"]), trace["context"]
    )
    namespace = selected[0].get("namespace", "default")
    name = _native_name(trace["controller"], user)
    require(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", namespace),
        "Invalid original namespace",
    )
    return {
        "user": user,
        "managed_ids": [],
        "context": trace["context"],
        "controller": {
            "name": name,
            "namespace": namespace,
            "services": [name + "-head", name + "-head-ssh"],
        },
    }


def _original_config(manifest: dict, journal: dict, trace: dict) -> None:
    path = Path(trace["root"]) / "submissions" / journal["requested_name"]
    path /= "skypilot-config.yaml"
    entry = manifest["native_submission_config"]
    require(entry["path"] == str(path), "Wrong original submission configuration")
    config = yaml.safe_load(pinned_bytes(entry))
    _local_api(manifest, trace, config.pop("api_server"))
    _pull_secrets(config["kubernetes"].pop("pod_config"))
    _config(yaml.safe_dump(config).encode(), trace["context"])


def _local_api(manifest: dict, trace: dict, config: dict) -> None:
    entry = manifest["original_api_daemon"]
    require(
        entry["path"] == str(Path(trace["root"]) / "local-api/daemon.json"),
        "Foreign original API daemon",
    )
    daemon = json.loads(pinned_bytes(entry))
    port = daemon["port"]
    require(
        daemon["root"] == str(Path(trace["root"]) / "local-api")
        and type(port) is int
        and 0 < port < 65536,
        "Original isolated API binding differs",
    )
    require(
        config == {"endpoint": f"http://127.0.0.1:{port}"},
        "Original API endpoint differs from isolated native daemon",
    )


def _pull_secrets(pod: dict) -> None:
    require(
        set(pod) == {"spec"} and set(pod["spec"]) == {"imagePullSecrets"},
        "Uncovered global Pod configuration",
    )
    secrets = pod["spec"]["imagePullSecrets"]
    require(isinstance(secrets, list) and bool(secrets), "Original pull names missing")
    for entry in secrets:
        require(
            set(entry) == {"name"}
            and isinstance(entry["name"], str)
            and re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", entry["name"]),
            "Uncovered global pull selector",
        )
