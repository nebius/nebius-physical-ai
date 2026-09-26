"""Run or preflight a provenance-bound native Comet/OpenPI training recipe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from npa.workflows.behavior_challenge.comet_training_data import (
    validate_data_reconstruction,
)
from npa.workflows.behavior_challenge.native_training import (
    NativeTrainingPlan,
    run_native_training,
)
from npa.workflows.behavior_challenge.native_training_checkpoint import (
    atomic_json,
    file_identity,
)


def _implementation(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("npa_comet_openpi_runtime", path)
    if spec is None or spec.loader is None:
        raise ValueError("Comet runtime implementation is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _admission(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("admission receipt identity differs")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "npa.behavior.comet-native-training-admission.v1"
        or value.get("status") != "qualified_inputs_bound_for_native_training"
    ):
        raise ValueError("training admission contract differs")
    if not isinstance(value.get("source_files"), dict) or not value["source_files"]:
        raise ValueError("source identity inventory is absent")
    validate_data_reconstruction(value.get("data_reconstruction"))
    for name in ("minimum_materialization_free_bytes", "minimum_checkpoint_free_bytes"):
        amount = value.get(name)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError(f"{name} must be a positive integer")
    value["identity"] = identity
    return value


def parser() -> argparse.ArgumentParser:
    """Build the scientific parser.

    Args: None. Returns: Shared preflight/training parser.
    Raises: None.
    """
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("operation", choices=("preflight", "train"))
    for name in (
        "source-root",
        "dataset-root",
        "split",
        "parent-checkpoint",
        "workspace",
        "admission",
        "runtime-implementation",
        "workflow-input-contract",
        "checkpoint-root",
        "output",
    ):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument("--admission-sha256", required=True)
    value.add_argument("--split-sha256", required=True)
    value.add_argument("--config-name", required=True)
    value.add_argument("--output-prefix", required=True)
    value.add_argument("--final-step", type=int, required=True)
    value.add_argument("--milestones", required=True)
    value.add_argument("--resume-selection", type=Path)
    return value


def _runtime_args(
    args: argparse.Namespace, selection: dict[str, Any]
) -> argparse.Namespace:
    args.resume_checkpoint = (
        None
        if selection["resume_checkpoint"] is None
        else Path(selection["resume_checkpoint"])
    )
    receipt = (
        None
        if selection["resume_receipt"] is None
        else json.loads(Path(selection["resume_receipt"]).read_text())
    )
    args.resume_receipt = receipt
    args.cursor = selection["cursor"]
    args.durable_milestones = {
        "0": _admission(args.admission, args.admission_sha256)["released_parent"]
    }
    if receipt is not None:
        durable = selection.get("durable_milestones")
        if not isinstance(durable, dict) or durable.get(
            str(selection["logical_update"])
        ) != {**receipt, "provider_manifest": selection["provider_manifest"]}:
            raise ValueError("resume durable milestone chain differs")
        args.durable_milestones = durable
    return args


def main() -> None:
    """Execute real input verification or native training.

    Args: None. Returns: None.
    Raises: Propagates input, admission, resume, or training errors.
    """
    args = parser().parse_args()
    admission = _admission(args.admission, args.admission_sha256)
    args.workflow_inputs = json.loads(args.workflow_input_contract.read_text())
    if (
        args.workflow_inputs.get("schema")
        != "npa.behavior.comet-native-workflow-inputs.v1"
    ):
        raise ValueError("workflow input contract differs")
    implementation = _implementation(args.runtime_implementation)
    if args.operation == "preflight":
        implementation.verify_inputs_only(args, admission, args.output)
        return
    if args.resume_selection is None:
        raise ValueError("training requires a control-produced resume selection")
    selection = json.loads(args.resume_selection.read_text())
    runtime = implementation.CometOpenPIRuntime(
        _runtime_args(args, selection), admission
    )
    milestones = tuple(int(value) for value in args.milestones.split(","))
    plan = NativeTrainingPlan(args.final_step, milestones)
    receipt = run_native_training(runtime, plan, args.checkpoint_root)
    atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()
