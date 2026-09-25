"""Bind a never-accepted submit to its original command, process and source records."""

from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

from npa.cluster.absent_evidence import digest, pinned_bytes, require


def _json(manifest: dict, key: str) -> dict:
    value = json.loads(pinned_bytes(manifest[key]))
    require(isinstance(value, dict), "Original producer object required")
    return value


def _option(arguments: list[str], *names: str) -> str:
    positions = [i for i, value in enumerate(arguments) if value in names]
    require(len(positions) == 1, "Missing or repeated producer selector")
    index = positions[0] + 1
    require(index < len(arguments), "Producer selector has no value")
    return arguments[index]


def _process(manifest: dict, freeze: dict, journal: dict) -> None:
    process = _json(manifest, "producer_process")
    require(
        type(process.get("returncode")) is int and process["returncode"] != 0,
        "Original process did not fail",
    )
    for name in ("stdout", "stderr"):
        key = "producer_response" if name == "stdout" else "producer_stderr"
        require(
            digest(pinned_bytes(manifest[key])) == process[name + "_sha256"],
            "Original process output changed",
        )
    start, end = (
        datetime.fromisoformat(value) for value in (freeze["at"], process["at"])
    )
    created, updated = (
        datetime.fromisoformat(journal[key].replace("Z", "+00:00"))
        for key in ("created_at", "updated_at")
    )
    require(start <= created <= updated <= end, "Wrong original attempt interval")
    require((end - updated).total_seconds() < 1, "Original completion differs")


def _source(manifest: dict, freeze: dict) -> None:
    require(
        digest(pinned_bytes(manifest["producer_runner"])) == freeze["runner_sha256"],
        "Original runner changed",
    )
    require(
        digest(pinned_bytes(manifest["producer_workflow"]))
        == freeze["shipped_workflow_sha256"],
        "Original workflow changed",
    )
    from npa.orchestration.skypilot.absence_zero_source import PRODUCER_SOURCE_SHA256

    require(
        set(manifest["producer_modules"]) == set(PRODUCER_SOURCE_SHA256),
        "Incomplete original producer source",
    )
    for name, expected in PRODUCER_SOURCE_SHA256.items():
        require(
            digest(pinned_bytes(manifest["producer_modules"][name])) == expected,
            "Unreviewed original reconciliation implementation",
        )
    _git_source(manifest, freeze)


def _git_source(manifest: dict, freeze: dict) -> None:
    require(
        all(re.fullmatch(r"[0-9a-f]{40}", freeze[key]) for key in ("source", "tree")),
        "Invalid original source identity",
    )
    checkout = Path(freeze["argv"][0]).parents[3]
    command = ["git", "--no-pager", "-C", str(checkout)]
    tree = subprocess.check_output(
        command + ["rev-parse", freeze["source"] + "^{tree}"]
    )
    require(tree.decode().strip() == freeze["tree"], "Original source tree differs")
    for name, entry in manifest["producer_modules"].items():
        require(entry["path"] == str(checkout / name), "Foreign producer module path")
        original = subprocess.check_output(
            command + ["show", freeze["source"] + ":" + name]
        )
        require(original == pinned_bytes(entry), "Original source bytes differ")


def validate_zero_trace(manifest: dict, journal: dict, response: dict) -> dict:
    """Validate original argv and completion without executing the old runner.

    Args:
        manifest: Pinned original records and selectors.
        journal: Original failed journal generation.
        response: Original terminal CLI response.
    Returns:
        Original root, context, controller and evidence paths.
    Raises:
        ValueError: Any original binding is absent or inconsistent.
    """
    freeze = _json(manifest, "producer_freeze")
    _process(manifest, freeze, journal)
    _source(manifest, freeze)
    arguments = freeze["argv"]
    require(
        arguments[1:6] == ["-m", "npa.cli.main", "workbench", "workflow", "submit"]
        and len(arguments) > 6
        and arguments[6] == manifest["producer_workflow"]["path"],
        "Wrong original CLI command",
    )
    return _selectors(manifest, journal, response, freeze, arguments[7:])


def _selectors(manifest, journal, response, freeze, arguments) -> dict:
    root = _option(arguments, "--isolated-config-dir")
    require(
        Path(root).is_absolute()
        and str(Path(root).resolve()) == root
        and root == manifest["isolated_root"],
        "Original root differs",
    )
    require(
        _option(arguments, "--project") == journal["project_alias"]
        and _option(arguments, "--run-id", "--resume-run") == journal["requested_name"]
        and _option(arguments, "--controller-backend") == "kubernetes"
        and "--pool" not in arguments,
        "Wrong original operation selectors",
    )
    context = _option(arguments, "--infra")
    controller = response["launch_transaction"]["controller"]
    require(context == "k8s/" + controller["selected_context"], "Context differs")
    return _environment_scope(manifest, journal, freeze, root, context, controller)


def _environment_scope(manifest, journal, freeze, root, context, controller) -> dict:
    environment = _json(manifest, "producer_environment")
    require(
        not any(
            key in environment
            for key in (
                "NPA_OPERATION_JOURNAL_DIR",
                "NPA_PARENT_LIFECYCLE_OPERATION",
                "NPA_CURRENT_OPERATION_ID",
                "SKYPILOT_USER_ID",
            )
        ),
        "Producer selected an unsupported ownership override",
    )
    require(
        freeze["root_user_id"] == "npa-" + digest(root.encode())[:12], "User differs"
    )
    ledger = Path(environment["NPA_CONFIG_DIR"]) / "workflow-submissions"
    ledger /= journal["project_alias"]
    return {
        "root": root,
        "context": context[4:],
        "controller": controller["name"],
        "ledger_path": str(ledger / (journal["requested_name"] + ".json")),
        "kubeconfig_path": environment["KUBECONFIG"],
        "source": freeze["source"],
        "tree": freeze["tree"],
    }
