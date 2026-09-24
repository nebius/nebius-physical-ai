"""Execute fixed campaign partitions and recover original per-case evidence."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import logging
from pathlib import Path

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
from .policy import POLICY_FIELDS, managed_policy
from .protocol import evaluator_argv, file_digest, verify_upstream


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _put_original(storage, payload, uri):
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        pass
    saved = storage.read_bytes_with_etag(uri)
    if saved is None or saved[0] != payload:
        raise ValueError("Immutable campaign artifact differs from uploaded bytes")


def _record_originals(store, version, output, record):
    prefix = store.artifact_prefix(version)
    _record_case_provenance(store, version, output)
    _put_original(store.storage, _json_bytes(record), f"{prefix}/validation.json")
    for relative, digest in record["files"].items():
        source = output / relative
        if file_digest(source) != digest:
            raise ValueError("Original artifact changed after inspection")
        _put_original(store.storage, source.read_bytes(), f"{prefix}/{relative}")
    return store.complete(version, record)


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
    bind_inspected_rollout(panel, record)
    if any(record.get(key) != value for key, value in version.record["case"].items()):
        raise ValueError("Recovery manifest differs from the prescribed case")
    _restore_case_provenance(store, version, output)
    for relative, digest in record["files"].items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        store.storage.download_file(f"{prefix}/{relative}", str(destination))
        if file_digest(destination) != digest:
            raise ValueError("Recovered original artifact failed SHA-256 verification")
    inspected = inspect_rollout(output, version.record["case"])
    if inspected != record:
        raise ValueError("Recovered video or metrics differ from original inspection")
    return inspected


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
    validate_panel(panel)
    validate_partition(partition, panel)
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
        with preparation:
            started = store.start(claim)
            execute_case(case, output)
            (output / "evaluator-exit.json").write_bytes(_json_bytes(case))
    except BaseException:
        _preserve_failed_case(store, claim, output)
        raise
    record = inspect_rollout(output, case)
    bind_inspected_rollout(panel, record)
    _record_originals(store, started, output, record)
    return record


def _preserve_failed_case(store, claim, output):
    try:
        _record_case_provenance(store, claim, output)
    except Exception:
        logging.getLogger(__name__).warning(
            "Case provenance upload also failed; retain its original workspace"
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
            bind_inspected_rollout(panel, record)
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
    validate_panel(panel)
    if store.panel_id != panel["panel_id"]:
        raise ValueError("Case store is bound to another panel")
    receipts = []
    for case in panel["cases"]:
        version = store.read(case)
        if version is None or version.record["state"] != "complete":
            raise ValueError("Complete panel required before aggregation")
        record = recover_case(
            store, version, _case_directory(workspace, version), panel
        )
        receipts.append(bind_inspected_rollout(panel, record))
    return {
        "schema": "npa.behavior.verified-panel.v1",
        "aggregate": aggregate_panel(panel, receipts),
        "verification": {
            "all_original_bytes_downloaded_and_hashed": True,
            "all_original_videos_fully_decoded": True,
            "case_count": len(receipts),
        },
    }


def _read_json(storage, uri, destination):
    storage.download_file(uri, str(destination))
    return json.loads(destination.read_bytes())


def _worker_declarations(args, storage, workspace):
    panel = _read_json(storage, args.panel_uri, workspace / "panel.json")
    partition = _read_json(storage, args.partition_uri, workspace / "partition.json")
    validate_panel(panel)
    validate_partition(partition, panel)
    registry_path = args.upstream_root / "docs/challenge/task_data.json"
    registry = [item["id"] for item in json.loads(registry_path.read_bytes())["tasks"]]
    if panel != declare_panel(
        panel["policy"], registry, panel["selected_tasks"], panel["split"]
    ):
        raise ValueError("Frozen panel differs from the actual official registry")
    return panel, partition


def _managed_plan(panel, case):
    return {
        "recipe": {
            "tasks": [case["task"]],
            "split": panel["split"],
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
    if args.policy_kind == "rlc-specialist" and panel["split"] == "report":
        from .rlc_specialist_admission import verify_specialist_report_token

        verify_specialist_report_token(args, panel)
    else:
        verify_serving_identity(args, panel["policy"])
    task = panel["selected_tasks"]
    if len(task) != 1:
        raise ValueError("Managed campaign worker currently serves one task per panel")

    def execute(case, output):
        verify_upstream(args.upstream_root)
        command = evaluator_argv(
            case,
            root=args.upstream_root,
            python=args.evaluator_python,
            host=args.host,
            port=args.port,
            output=output,
        )
        _run_case(command, args, output, case, environment)
        verify_upstream(args.upstream_root)

    def prepare(case, output):
        return managed_policy(args, _managed_plan(panel, case), output)

    with simulator as simulator_environment:
        environment = _runtime_environment(args)
        environment.update(simulator_environment)
        yield execute, prepare


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


def _execute_partition(args, panel, partition, storage, workspace):
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
    verify_upstream(args.upstream_root)
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    storage = StorageClient.from_environment()
    panel, partition = _worker_declarations(args, storage, workspace)
    records = _execute_partition(args, panel, partition, storage, workspace)
    verify_upstream(args.upstream_root)
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
    store = CaseStore(storage, args.output_path, validate_panel(panel)["panel_id"])
    result = aggregate_stored_panel(panel, store, args.workspace)
    _put_original(storage, _json_bytes(result), args.receipt_uri)
    return result
