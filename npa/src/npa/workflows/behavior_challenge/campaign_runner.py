"""Execute fixed campaign partitions and recover original per-case evidence."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat

from npa.clients.storage import StorageClient, StoragePreconditionFailed

from .artifacts import inspect_rollout
from .campaign import (
    aggregate_panel,
    bind_inspected_rollout,
    declare_panel,
    validate_panel,
    validate_partition,
)
from .case_store import CaseAlreadyStarted, CaseStore
from .execution import _run_case, _runtime_environment
from .nonreporting_train import (
    TRAIN_PANEL_SCHEMA,
    aggregate_train_outputs,
    bind_train_rollout_record,
    train_evaluator_argv,
    validate_train_panel,
    validate_train_partition,
    verify_train_evaluator_source,
)
from .policy import POLICY_FIELDS, managed_policy
from .protocol import evaluator_argv, file_digest, verify_upstream
from .train_experience import validate_finalized_experience


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _is_train_panel(panel) -> bool:
    return isinstance(panel, dict) and panel.get("schema") == TRAIN_PANEL_SCHEMA


def _validate_execution_panel(panel):
    if _is_train_panel(panel):
        return validate_train_panel(panel)
    return validate_panel(panel)


def _validate_execution_partition(partition, panel):
    if _is_train_panel(panel):
        return validate_train_partition(partition, panel)
    return validate_partition(partition, panel)


def _bind_execution_rollout(panel, record):
    if _is_train_panel(panel):
        return bind_train_rollout_record(panel, record)
    return bind_inspected_rollout(panel, record)


def _upstream_commit(panel):
    if _is_train_panel(panel):
        return panel["protocol"]["science_lineage"]["behavior_upstream_commit"]
    return panel["upstream_commit"]


def _put_original(storage, payload, uri):
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        pass
    saved = storage.read_bytes_with_etag(uri)
    if saved is None or saved[0] != payload:
        raise ValueError("Immutable campaign artifact differs from uploaded bytes")


def _record_originals(store, version, output, record):
    experience = _train_experience_bundle(store, version, output)
    if experience is not None:
        _record_train_experience_requirement(store, version, experience)
    prefix = store.artifact_prefix(version)
    _record_case_provenance(store, version, output)
    _put_original(store.storage, _json_bytes(record), f"{prefix}/validation.json")
    for relative, digest in record["files"].items():
        source = output / relative
        if file_digest(source) != digest:
            raise ValueError("Original artifact changed after inspection")
        _put_original(store.storage, source.read_bytes(), f"{prefix}/{relative}")
    _publish_train_experience(store, version, experience)
    return store.complete(version, record)


def _record_train_experience(store, version, output) -> None:
    experience = _train_experience_bundle(store, version, output)
    if experience is None:
        return
    _record_train_experience_requirement(store, version, experience)
    _publish_train_experience(store, version, experience)


def _train_experience_bundle(
    store, version, output, *, verify_official_source: bool = True
):
    root = output / "train-experience"
    manifest_path = root / "experience-manifest.json"
    if not root.exists():
        if version.record.get("train_experience_config") is not None:
            raise ValueError("Enabled TRAIN experience root is absent")
        return None
    if root.is_symlink() or not root.is_dir():
        raise ValueError("TRAIN experience root must be a real directory")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Enabled TRAIN experience lacks its success manifest")
    official_output = output if verify_official_source else None
    manifest = validate_finalized_experience(root, official_output=official_output)
    config_identity = _expected_train_experience_config(version, root)
    payload = manifest_path.read_bytes()
    rows = _validate_train_experience_manifest(store, version, manifest)
    _validate_train_experience_inventory(root, manifest_path, rows)
    requirement = {
        "schema": "npa.behavior.train-experience-publication-requirement.v2",
        "status": "complete_train_experience_required",
        "config": config_identity,
        "manifest": {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    return root, manifest_path, rows, requirement


def _expected_train_experience_config(version, root: Path) -> dict[str, object]:
    expected = version.record.get("train_experience_config")
    path = root / "config.json"
    actual = (
        {"bytes": path.stat().st_size, "sha256": file_digest(path)}
        if path.is_file() and not path.is_symlink()
        else None
    )
    if not isinstance(expected, dict) or expected != actual:
        raise ValueError("TRAIN experience config differs from durable case start")
    return expected


def _validate_train_experience_manifest(store, version, manifest):
    if not isinstance(manifest, dict):
        raise ValueError("TRAIN experience success manifest differs")
    rows = manifest.get("members")
    config = manifest.get("config")
    if (
        manifest.get("schema") != "npa.behavior.train-experience-manifest.v1"
        or manifest.get("status") != "complete_exact_train_experience"
        or manifest.get("development_or_report_used") is not False
        or manifest.get("privileged_state_entered_policy_inputs") is not False
        or not isinstance(config, dict)
        or config.get("panel_sha256") != store.panel_id
        or config.get("case") != version.record["case"]
        or not isinstance(rows, list)
    ):
        raise ValueError("TRAIN experience success manifest differs")
    names = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(names) != len(rows) or len(set(names)) != len(names):
        raise ValueError("TRAIN experience member names differ")
    return rows


def _validate_train_experience_inventory(root, manifest_path, rows) -> None:
    names = []
    for row in rows:
        names.append(_experience_member(root, row).relative_to(root).as_posix())
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("TRAIN experience inventory contains a symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in paths
        if path.is_file() and path != manifest_path
    }
    if actual != set(names):
        raise ValueError("TRAIN experience member inventory differs")


def _record_train_experience_requirement(store, version, experience) -> None:
    requirement = experience[3]
    prefix = store.artifact_prefix(version) + "/train-experience"
    _put_original(
        store.storage,
        _json_bytes(requirement),
        f"{prefix}/publication-requirement.json",
    )


def _publish_train_experience(store, version, experience) -> None:
    if experience is None:
        return
    root, manifest_path, rows, _requirement = experience
    prefix = store.artifact_prefix(version) + "/train-experience"
    for row in rows:
        source = _experience_member(root, row)
        _put_original(store.storage, source.read_bytes(), f"{prefix}/{row['path']}")
    _put_original(
        store.storage, manifest_path.read_bytes(), f"{prefix}/experience-manifest.json"
    )


def _experience_member(root: Path, row: object) -> Path:
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise ValueError("TRAIN experience member row differs")
    relative = Path(row["path"])
    source = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or source.is_symlink()
        or not source.is_file()
        or not source.resolve().is_relative_to(root.resolve())
        or file_digest(source) != row["sha256"]
        or source.stat().st_size != row["bytes"]
    ):
        raise ValueError("TRAIN experience member differs")
    return source


def _record_case_provenance(store, version, output):
    prefix = store.artifact_prefix(version)
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and not path.is_symlink():
            _put_original(
                store.storage, path.read_bytes(), f"{prefix}/provenance/{path.name}"
            )
            files[path.name] = file_digest(path)
    _put_original(store.storage, _json_bytes(files), f"{prefix}/provenance.json")


def _raw_case_payload(output, relative):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with ExitStack() as stack:
        directory = os.open(output, flags | os.O_DIRECTORY)
        stack.callback(os.close, directory)
        parent = os.open(relative.parent, flags | os.O_DIRECTORY, dir_fd=directory)
        stack.callback(os.close, parent)
        descriptor = os.open(relative.name, flags, dir_fd=parent)
        stack.callback(os.close, descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Raw campaign artifact must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()


def _record_raw_case_artifacts(store, version, output):
    case = version.record["case"]
    stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
    prefix = store.artifact_prefix(version) + "/raw"
    files = {}
    for relative in (Path(f"json/{stem}.json"), Path(f"videos/{stem}.mp4")):
        try:
            payload = _raw_case_payload(output, relative)
        except FileNotFoundError:
            continue
        _put_original(store.storage, payload, f"{prefix}/{relative.as_posix()}")
        files[relative.as_posix()] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if files:
        manifest = {
            "schema": "npa.behavior.raw-case-artifacts.v1",
            "status": "unvalidated",
            "panel_id": store.panel_id,
            "claim_id": version.record["claim_id"],
            "case": case,
            "all_prescribed_files_present": len(files) == 2,
            "files": files,
        }
        _put_original(store.storage, _json_bytes(manifest), f"{prefix}/manifest.json")


def _restore_case_provenance(store, version, output):
    prefix = store.artifact_prefix(version)
    saved = store.storage.read_bytes_with_etag(f"{prefix}/provenance.json")
    if saved is None:
        raise CaseAlreadyStarted("Original policy and evaluator provenance is missing")
    for name, digest in json.loads(saved[0]).items():
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid original provenance filename")
        destination = output / name
        store.storage.download_file(f"{prefix}/provenance/{name}", str(destination))
        if file_digest(destination) != digest:
            raise ValueError("Original provenance failed SHA-256 verification")


def _restore_originals(store, version, output, panel):
    prefix = store.artifact_prefix(version)
    saved = store.storage.read_bytes_with_etag(f"{prefix}/validation.json")
    if saved is None:
        raise CaseAlreadyStarted("Recover this claim's original worker output")
    record = json.loads(saved[0])
    _bind_execution_rollout(panel, record)
    if any(record.get(key) != value for key, value in version.record["case"].items()):
        raise ValueError("Recovery manifest differs from the prescribed case")
    _restore_case_provenance(store, version, output)
    for relative, digest in record["files"].items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        store.storage.download_file(f"{prefix}/{relative}", str(destination))
        if file_digest(destination) != digest:
            raise ValueError("Recovered original artifact failed SHA-256 verification")
    _restore_train_experience(store, version, output)
    inspected = inspect_rollout(output, version.record["case"])
    if inspected != record:
        raise ValueError("Recovered video or metrics differ from original inspection")
    return inspected


def _restore_train_experience(store, version, output) -> None:
    prefix = store.artifact_prefix(version) + "/train-experience"
    saved = store.storage.read_bytes_with_etag(f"{prefix}/publication-requirement.json")
    if saved is None:
        if version.record.get("train_experience_config") is not None:
            raise CaseAlreadyStarted("Complete required TRAIN experience publication")
        return
    requirement = _validate_experience_requirement(json.loads(saved[0]))
    if requirement["config"] != version.record.get("train_experience_config"):
        raise ValueError("Recovered TRAIN experience config requirement differs")
    manifest = store.storage.read_bytes_with_etag(f"{prefix}/experience-manifest.json")
    if manifest is None or _payload_identity(manifest[0]) != requirement["manifest"]:
        raise CaseAlreadyStarted("Complete required TRAIN experience publication")
    value = json.loads(manifest[0])
    rows = _validate_train_experience_manifest(store, version, value)
    root = output / "train-experience"
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = store.storage.read_bytes_with_etag(f"{prefix}/{row['path']}")
        if payload is None or _payload_identity(payload[0]) != _row_identity(row):
            raise CaseAlreadyStarted("Complete required TRAIN experience publication")
        _restore_experience_member(root, row["path"], payload[0])
    _restore_experience_member(root, "experience-manifest.json", manifest[0])
    restored = _train_experience_bundle(
        store, version, output, verify_official_source=False
    )
    if restored is None or restored[3] != requirement:
        raise ValueError("Recovered TRAIN experience requirement differs")


def _validate_experience_requirement(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "status", "config", "manifest"}
        or value.get("schema")
        != "npa.behavior.train-experience-publication-requirement.v2"
        or value.get("status") != "complete_train_experience_required"
        or not _valid_payload_identity(value.get("config"))
        or not isinstance(value.get("manifest"), dict)
        or set(value["manifest"]) != {"bytes", "sha256"}
        or type(value["manifest"].get("bytes")) is not int
        or value["manifest"]["bytes"] <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(value["manifest"].get("sha256"))) is None
    ):
        raise ValueError("TRAIN experience publication requirement differs")
    return value


def _valid_payload_identity(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"bytes", "sha256"}
        and type(value.get("bytes")) is int
        and value["bytes"] > 0
        and isinstance(value.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
    )


def _payload_identity(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _row_identity(row: dict) -> dict[str, object]:
    return {name: row[name] for name in ("bytes", "sha256")}


def _restore_experience_member(root: Path, relative_name: str, payload: bytes) -> None:
    relative = Path(relative_name)
    destination = root / relative
    if relative.is_absolute() or ".." in relative.parts or root.is_symlink():
        raise ValueError("Recovered TRAIN experience path differs")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("Recovered TRAIN experience path differs")
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError("Recovered TRAIN experience path differs")
    if destination.is_symlink():
        raise ValueError("Recovered TRAIN experience path differs")
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Recovered TRAIN experience path differs")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise ValueError("Recovered TRAIN experience member differs")
        return
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def recover_case(store, version, output: Path, panel: dict) -> dict:
    """Recover an original rollout without invoking its evaluator again.

    Args:
        store: Case store with conditional object writes.
        version: Original started or completed case version.
        output: Empty local recovery directory.
        panel: Frozen policy panel for identity validation.
    Returns:
        Direct inspection of the downloaded original JSON and decoded video.
    Raises:
        CaseAlreadyStarted: No original validation manifest was published.
        ValueError: Saved hashes, identity, or decoded evidence differ.
        StorageError: Original objects cannot be downloaded or committed.
    """
    record = _restore_originals(store, version, output, panel)
    if version.record["state"] == "started":
        store.complete(version, record)
    elif version.record["state"] != "complete" or version.record["rollout"] != record:
        raise ValueError("Completed case receipt differs from original evidence")
    return record


def run_partition(
    panel, partition, worker_index, store, workspace, execute_case, *, prepare_case=None
):
    """Run unstarted prescribed cases and verify every reused original artifact.

    Args:
        panel: Frozen policy and case declaration.
        partition: Deterministic ownership declaration.
        worker_index: Worker ordinal from the partition.
        store: Durable case store bound to the panel.
        workspace: Persistent, panel-specific output directory.
        execute_case: Prepared evaluator callback accepting case and output path.
        prepare_case: Optional context factory starting a fresh policy before each case.
    Returns:
        Validated original rollout records assigned to this worker.
    Raises:
        ValueError: Panel, partition, worker, or original evidence differs.
        CaseAlreadyStarted: A begun case has no recoverable original artifacts.
        Exception: Startup failures retain a reclaimable claim; evaluator failures
            retain the started marker.
    """
    _validate_execution_panel(panel)
    _validate_execution_partition(partition, panel)
    if store.panel_id != panel["panel_id"]:
        raise ValueError("Case store is bound to another panel")
    if (
        type(worker_index) is not int
        or not 0 <= worker_index < partition["worker_count"]
    ):
        raise ValueError("Invalid campaign worker index")
    assigned = set(partition["workers"][worker_index]["case_ids"])
    cases = [case for case in panel["cases"] if case["case_id"] in assigned]
    return [
        _run_or_recover(
            store, panel, case, workspace, worker_index, execute_case, prepare_case
        )
        for case in cases
    ]


def _run_or_recover(
    store, panel, case, workspace, worker_index, execute_case, prepare_case
):
    existing = store.read(case)
    if existing and existing.record["state"] in {"started", "complete"}:
        output = _case_directory(workspace, existing)
        return _recover_persistent_case(store, existing, output, panel)
    claim = store.claim(case, f"worker-{worker_index}")
    output = _case_directory(workspace, claim)
    if claim.record["state"] == "complete":
        return recover_case(store, claim, output, panel)
    preparation = prepare_case(case, output) if prepare_case else nullcontext()
    try:
        with preparation as train_experience_config:
            started = store.start(
                claim, train_experience_config=train_experience_config
            )
            execute_case(case, output)
            (output / "evaluator-exit.json").write_bytes(_json_bytes(case))
        record = inspect_rollout(output, case)
        _bind_execution_rollout(panel, record)
        _record_originals(store, started, output, record)
    except BaseException:
        _preserve_failed_case(store, claim, output)
        raise
    return record


def _preserve_failed_case(store, claim, output):
    for preserve in (_record_raw_case_artifacts, _record_case_provenance):
        try:
            preserve(store, claim, output)
        except Exception:
            logging.getLogger(__name__).warning(
                "Case evidence upload also failed; retain its original workspace"
            )


def _case_directory(workspace, version):
    output = Path(workspace) / version.record["claim_id"]
    output.mkdir(parents=True, exist_ok=True)
    return output


def _recover_persistent_case(store, version, output, panel):
    if version.record["state"] == "started":
        case = version.record["case"]
        stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
        paths = (output / f"json/{stem}.json", output / f"videos/{stem}.mp4")
        exit_record = output / "evaluator-exit.json"
        finished = (
            exit_record.is_file() and json.loads(exit_record.read_bytes()) == case
        )
        if finished and all(path.is_file() for path in paths):
            record = inspect_rollout(output, case)
            _bind_execution_rollout(panel, record)
            _record_originals(store, version, output, record)
    current = store.read(version.record["case"])
    return recover_case(store, current, output, panel)


def aggregate_stored_panel(panel, store, workspace: Path) -> dict:
    """Compare receipts only after downloading and decoding all original cases.

    Args:
        panel: Frozen complete panel declaration.
        store: Its durable case store.
        workspace: Local artifact verification directory.
    Returns:
        Complete aggregate with explicit byte-verification provenance.
    Raises:
        ValueError: Any case is missing, unfinished, or fails artifact validation.
    """
    _validate_execution_panel(panel)
    if store.panel_id != panel["panel_id"]:
        raise ValueError("Case store is bound to another panel")
    receipts = []
    case_outputs = {}
    for case in panel["cases"]:
        version = store.read(case)
        if version is None or version.record["state"] != "complete":
            raise ValueError("Complete panel required before aggregation")
        output = _case_directory(workspace, version)
        record = recover_case(store, version, output, panel)
        if _is_train_panel(panel):
            _bind_execution_rollout(panel, record)
            case_outputs[case["case_id"]] = output
        else:
            receipts.append(bind_inspected_rollout(panel, record))
    aggregate = (
        aggregate_train_outputs(panel, case_outputs)
        if _is_train_panel(panel)
        else aggregate_panel(panel, receipts)
    )
    return {
        "schema": "npa.behavior.verified-panel.v1",
        "aggregate": aggregate,
        "verification": {
            "all_original_bytes_downloaded_and_hashed": True,
            "all_original_videos_fully_decoded": True,
            "case_count": len(panel["cases"]),
        },
    }


def _read_json(storage, uri, destination):
    storage.download_file(uri, str(destination))
    return json.loads(destination.read_bytes())


def _worker_declarations(args, storage, workspace):
    panel = _read_json(storage, args.panel_uri, workspace / "panel.json")
    partition = _read_json(storage, args.partition_uri, workspace / "partition.json")
    _validate_execution_panel(panel)
    _validate_execution_partition(partition, panel)
    if _is_train_panel(panel):
        return panel, partition
    registry_path = args.upstream_root / "docs/challenge/task_data.json"
    registry = [item["id"] for item in json.loads(registry_path.read_bytes())["tasks"]]
    if panel != declare_panel(
        panel["policy"],
        registry,
        panel["selected_tasks"],
        panel["split"],
        upstream_commit=panel["upstream_commit"],
    ):
        raise ValueError("Frozen panel differs from the actual official registry")
    return panel, partition


def _managed_plan(panel, case):
    if _is_train_panel(panel):
        protocol = panel["protocol"]
        revision = _upstream_commit(panel)
        return {
            "upstream_commit": revision,
            "recipe": {
                "tasks": [protocol["task"]],
                "split": "train",
                "upstream_commit": revision,
                "policy_checkpoint_sha256": panel["policy_binding"]["artifacts"][
                    "checkpoint"
                ]["sha256"],
            },
            "cases": [case],
        }
    return {
        "upstream_commit": panel["upstream_commit"],
        "recipe": {
            "tasks": [case["task"]],
            "split": panel["split"],
            "upstream_commit": panel["upstream_commit"],
            "policy_checkpoint_sha256": panel["policy"]["artifacts"]["checkpoint"][
                "sha256"
            ],
        },
        "cases": [case],
    }


@contextmanager
def _prepared_evaluator(args, panel, workspace):
    from .serving_identity import verify_serving_identity

    simulator = _simulator_preparation(args, workspace)
    if not all(getattr(args, field, None) for field in POLICY_FIELDS):
        raise ValueError("Campaign execution requires a verified managed policy")
    if (
        not _is_train_panel(panel)
        and args.policy_kind == "rlc-specialist"
        and panel["split"] == "report"
    ):
        from .rlc_specialist_admission import verify_specialist_report_token

        verify_specialist_report_token(args, panel)
    else:
        identity = (
            panel["policy_binding"] if _is_train_panel(panel) else panel["policy"]
        )
        verify_serving_identity(args, identity)
    task = (
        [panel["protocol"]["task"]]
        if _is_train_panel(panel)
        else panel["selected_tasks"]
    )
    if len(task) != 1:
        raise ValueError("Managed campaign worker currently serves one task per panel")

    def execute(case, output):
        revision = _upstream_commit(panel)
        verify_upstream(args.upstream_root, revision)
        command = (
            train_evaluator_argv(
                panel["protocol"],
                case,
                root=args.upstream_root,
                python=args.evaluator_python,
                host=args.host,
                port=args.port,
                output=output,
            )
            if _is_train_panel(panel)
            else evaluator_argv(
                case,
                root=args.upstream_root,
                python=args.evaluator_python,
                host=args.host,
                port=args.port,
                output=output,
                upstream_commit=revision,
            )
        )
        case_environment = dict(environment)
        if getattr(args, "train_experience", False):
            if not _is_train_panel(panel):
                raise ValueError("TRAIN experience recording requires a TRAIN panel")
            command = _train_experience_evaluator_argv(command, output)
            case_environment["NPA_TRAIN_EXPERIENCE_ROOT"] = str(
                output / "train-experience"
            )
        _run_case(command, args, output, case, case_environment)
        verify_upstream(args.upstream_root, revision)

    def prepare(case, output):
        return managed_policy(args, _managed_plan(panel, case), output)

    with simulator as simulator_environment:
        environment = _runtime_environment(args)
        environment.update(simulator_environment)
        yield execute, prepare


def _train_experience_evaluator_argv(command: list[str], output: Path) -> list[str]:
    if len(command) < 3 or command[1:3] != ["-m", "omnigibson.eval.eval"]:
        raise ValueError("Official TRAIN evaluator command shape differs")
    return [command[0], str(output / "train_experience_evaluator.py"), *command[3:]]


def _simulator_preparation(args, workspace):
    from .simulator_startup import (
        build_evaluation_context,
        prepared_evaluator_environment,
        validate_simulator_startup_receipt,
    )

    receipt_path = getattr(args, "simulator_startup_receipt", None)
    if receipt_path is None:
        return nullcontext({})
    context = build_evaluation_context(
        args.upstream_root,
        Path(args.evaluator_python),
        Path(args.data_root),
        workspace,
    )
    receipt = validate_simulator_startup_receipt(receipt_path, context)
    return prepared_evaluator_environment(receipt, workspace)


def _publish_worker_provenance(storage, workspace, receipt_uri):
    prefix = receipt_uri.removesuffix(".json") + "/provenance"
    for path in sorted(workspace.iterdir()):
        if path.is_file() and not path.is_symlink():
            _put_original(storage, path.read_bytes(), f"{prefix}/{path.name}")


def _validate_train_experience_scope(args, panel) -> None:
    from .policy_prompt import prompt_override

    experience = getattr(args, "train_experience", False)
    train_policy_kinds = {"comet-native", "comet-trained"}
    if (
        _is_train_panel(panel)
        and args.policy_kind == "comet-trained"
        and not experience
    ):
        raise ValueError("Trained Comet TRAIN execution requires experience recording")
    if experience and (
        not _is_train_panel(panel) or args.policy_kind not in train_policy_kinds
    ):
        raise ValueError("TRAIN experience requires an admitted Comet TRAIN panel")
    if getattr(args, "train_experience_depth", False) and not experience:
        raise ValueError("TRAIN experience depth requires recording")
    if _is_train_panel(panel) and args.policy_kind not in train_policy_kinds:
        raise ValueError("TRAIN campaign execution requires an admitted Comet adapter")
    prompt_override(args, train_panel=_is_train_panel(panel))


def _execute_partition(args, panel, partition, storage, workspace):
    _validate_train_experience_scope(args, panel)
    _validate_worker_startup_binding(args, workspace)
    _train_evaluator_preclaim(args, panel)
    if _is_train_panel(panel):
        if args.policy_kind == "comet-native":
            _native_train_preclaim(args, panel, workspace)
        else:
            _trained_comet_preclaim(args, panel, workspace)
    elif getattr(args, "policy_kind", None) == "comet-trained":
        _trained_comet_preclaim(args, panel, workspace)
    try:
        _prepare_worker_startup(args, workspace)
        _specialist_preclaim(args, panel, storage, workspace)
        store = CaseStore(storage, args.output_path, panel["panel_id"])
        with _prepared_evaluator(args, panel, workspace) as (execute, prepare):
            records = run_partition(
                panel,
                partition,
                args.worker_index,
                store,
                workspace,
                execute,
                prepare_case=prepare,
            )
    except BaseException:
        try:
            _publish_worker_provenance(storage, workspace, args.worker_receipt_uri)
        except Exception:
            logging.getLogger(__name__).warning(
                "Provenance upload also failed; retain the worker workspace for recovery"
            )
        raise
    else:
        _publish_worker_provenance(storage, workspace, args.worker_receipt_uri)
        return records


def _train_evaluator_preclaim(args, panel) -> None:
    if not _is_train_panel(panel):
        return
    verify_train_evaluator_source(panel["protocol"], args.upstream_root)


def _validate_worker_startup_binding(args, workspace: Path) -> None:
    """Reject a startup binding that cannot belong to this worker."""

    spec_path = getattr(args, "simulator_startup_spec", None)
    receipt_path = getattr(args, "simulator_startup_receipt", None)
    if spec_path is not None and receipt_path is not None:
        raise ValueError("simulator startup spec conflicts with startup receipt")
    for label, path in (("specification", spec_path), ("receipt", receipt_path)):
        if path is not None and not Path(path).is_absolute():
            raise ValueError(f"simulator startup {label} path must be absolute")
    if spec_path is not None:
        from .simulator_startup import load_simulator_startup_spec

        spec = load_simulator_startup_spec(Path(spec_path))
        if spec.apps.owner_root.resolve(strict=True) != workspace.resolve(strict=True):
            raise ValueError(
                "simulator startup specification owner_root must equal the "
                "campaign worker workspace"
            )
        return
    if receipt_path is not None:
        from .simulator_startup import (
            build_evaluation_context,
            validate_simulator_startup_receipt,
        )

        context = build_evaluation_context(
            args.upstream_root,
            Path(args.evaluator_python),
            Path(args.data_root),
            workspace,
        )
        validate_simulator_startup_receipt(Path(receipt_path), context)


def _native_train_preclaim(args, panel, workspace) -> None:
    from .native_comet_checkpoint import validate_native_train_admission
    from .native_training_checkpoint import atomic_json
    from .serving_identity import serving_artifact

    binding = getattr(args, "policy_native_binding", None)
    input_root = getattr(args, "policy_native_input_root", None)
    if binding is None or input_root is None:
        raise ValueError("Native TRAIN requires its binding and input root")
    admission = validate_native_train_admission(
        Path(binding),
        Path(input_root),
        Path(args.policy_checkpoint),
        panel,
        expected_verified_checkpoint=Path(args.policy_archive),
        expected_serving_identity=serving_artifact(args),
    )
    args.policy_native_panel = panel
    args.policy_native_admission = admission
    atomic_json(Path(workspace) / "native-train-admission.json", admission)


def _trained_comet_preclaim(args, panel, workspace) -> None:
    from .native_training_checkpoint import atomic_json
    from .serving_identity import serving_artifact
    from .trained_comet_checkpoint import validate_trained_comet_admission

    input_root = getattr(args, "policy_trained_input_root", None)
    if input_root is None:
        raise ValueError("Trained Comet requires its provider-read input root")
    admission = validate_trained_comet_admission(
        Path(args.policy_archive),
        Path(input_root),
        Path(args.policy_checkpoint),
        panel,
        expected_serving_identity=serving_artifact(args),
    )
    args.policy_trained_panel = panel
    args.policy_trained_execution_admission = admission
    atomic_json(Path(workspace) / "trained-comet-admission.json", admission)


def _specialist_preclaim(args, panel, storage, workspace) -> None:
    if (
        getattr(args, "policy_kind", None) != "rlc-specialist"
        or panel["split"] != "report"
    ):
        return
    from .rlc_specialist_admission import (
        verify_specialist_preclaim_endpoint,
        verify_specialist_report_admission,
    )

    verify_specialist_report_admission(
        args, panel, storage, workspace / "report-admission"
    )
    verify_specialist_preclaim_endpoint(args, panel)


def evaluate_partition(args) -> dict:
    """Run a campaign worker through the standard Workbench workflow runtime.

    Args:
        args: Internal worker arguments including frozen panel and persistent paths.
    Returns:
        Worker completion receipt; no partial panel score is emitted.
    Raises:
        ValueError: Frozen policy, protocol, runtime, or evidence checks fail.
        Exception: Policy, evaluator, or storage fails with durable state retained.
    """
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    _validate_worker_startup_binding(args, workspace)
    storage = StorageClient.from_environment()
    panel, partition = _worker_declarations(args, storage, workspace)
    verify_upstream(args.upstream_root, _upstream_commit(panel))
    records = _execute_partition(args, panel, partition, storage, workspace)
    verify_upstream(args.upstream_root, _upstream_commit(panel))
    result = {
        "panel_id": panel["panel_id"],
        "partition_sha256": partition["partition_sha256"],
        "worker_index": args.worker_index,
        "completed": len(records),
        "case_ids": [record["case_id"] for record in records],
    }
    _put_original(storage, _json_bytes(result), args.worker_receipt_uri)
    return result


def _prepare_worker_startup(args, workspace: Path) -> None:
    spec_path = getattr(args, "simulator_startup_spec", None)
    receipt_path = getattr(args, "simulator_startup_receipt", None)
    if spec_path is None or receipt_path is not None:
        return
    from .simulator_startup import write_simulator_startup_receipt
    from .simulator_startup import validate_simulator_startup_receipt

    receipt_path = workspace / "simulator-startup.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        validate_simulator_startup_receipt(receipt_path, expected_spec_path=spec_path)
    else:
        write_simulator_startup_receipt(spec_path, receipt_path)
    args.simulator_startup_receipt = receipt_path


def aggregate_campaign_worker(args) -> dict:
    """Publish a complete panel after direct original-artifact verification.

    Args:
        args: Internal stage arguments for the panel, state, workspace and receipt.
    Returns:
        Verified panel aggregate for subsequent campaign comparison.
    Raises:
        ValueError: The panel is incomplete or original evidence differs.
        StorageError: Inputs or output receipt cannot be accessed.
    """
    args.workspace.mkdir(parents=True, exist_ok=True)
    storage = StorageClient.from_environment()
    panel = _read_json(storage, args.panel_uri, args.workspace / "panel.json")
    store = CaseStore(
        storage, args.output_path, _validate_execution_panel(panel)["panel_id"]
    )
    result = aggregate_stored_panel(panel, store, args.workspace)
    _put_original(storage, _json_bytes(result), args.receipt_uri)
    return result
