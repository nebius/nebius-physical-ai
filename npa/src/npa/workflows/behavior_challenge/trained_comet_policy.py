"""Prepare a parity-qualified trained Comet export for one panel case."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .campaign import canonical_digest
from .comet_policy import CONFIG_NAME, SOURCE_COMMIT, verify_source
from .native_comet_policy import case_seed, finalize_process
from .native_comet_server import validate_load_qualification
from .native_training_checkpoint import atomic_json, file_identity
from .serving_identity import serving_artifact
from .train_experience import write_experience_config
from .train_prompt import effective_prompt_binding
from .trained_comet_checkpoint import validate_trained_comet_admission


def _stage_adapters(output: Path, *, train_experience: bool = False) -> dict[str, dict]:
    names = [
        "trained_comet_checkpoint.py",
        "trained_comet_policy.py",
        "trained_comet_producer.py",
        "native_comet_server.py",
        "comet_policy.py",
        "evaluator_versions.py",
        "evaluator_wire.py",
    ]
    if train_experience:
        names.extend(
            ("train_experience.py", "train_experience_evaluator.py", "train_prompt.py")
        )
    rows = {}
    for name in names:
        source = Path(__file__).with_name(name)
        target = output / name
        shutil.copyfile(source, target)
        rows[name] = file_identity(target)
    if train_experience:
        monitor = output / "semantic_monitor"
        monitor.mkdir()
        for name in ("__init__.py", "collector.py", "interface.py", "schema.py"):
            source = Path(__file__).with_name("semantic_monitor") / name
            target = monitor / name
            shutil.copyfile(source, target)
            rows[f"semantic_monitor/{name}"] = file_identity(target)
    return rows


def _case(args, plan: dict, admission: dict) -> dict:
    split = plan.get("recipe", {}).get("split")
    if split not in {"train", "development", "report"}:
        raise ValueError("Trained Comet policy split differs")
    if split == "train" and not getattr(args, "train_experience", False):
        raise ValueError("Trained Comet TRAIN requires experience recording")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("Trained Comet policy requires one case")
    case = cases[0]
    if (
        plan.get("recipe", {}).get("tasks") != [admission["task"]]
        or case.get("task") != admission["task"]
        or case.get("split") != plan["recipe"]["split"]
        or case.get("rollout_id") != 0
        or getattr(args, "policy_task_name", None) != admission["task"]
    ):
        raise ValueError("Trained Comet task or case binding differs")
    return case


def _rng_identity(admission: dict) -> dict:
    return {name: admission["rng_contract"][name] for name in ("bytes", "sha256")}


def _task_mapping(root: Path, task_id: int, task: str) -> None:
    mapping = json.loads((root / "scripts/task_mapping.json").read_text())
    row = mapping.get(task)
    if (
        not isinstance(row, dict)
        or row.get("task_index") != task_id
        or not isinstance(row.get("task"), str)
        or not row["task"].strip()
    ):
        raise ValueError("Trained Comet source task mapping differs")


def _server_command(args, plan: dict, output: Path, admission: dict) -> list[str]:
    case = plan["cases"][0]
    seed = case_seed(_rng_identity(admission), case)
    command = _server_identity_arguments(args, plan, output, admission, case, seed)
    if admission["trace"]["enabled"]:
        command.extend(("--action-trace", str(output / "native-actions.jsonl")))
    if getattr(args, "train_experience", False):
        command.extend(("--train-experience-root", str(output / "train-experience")))
    return command


def _write_experience_config(
    args, panel, case, output, admission, prompt_binding
) -> None:
    value = {
        "schema": "npa.behavior.train-experience-config.v2",
        "status": "train_only_recording_enabled",
        "split": "train",
        "cadence": "model_decision_observation_with_all_applied_actions",
        "action_horizon": 32,
        "include_depth": getattr(args, "train_experience_depth", False),
        "case": {
            name: case[name]
            for name in ("case_id", "task", "instance_id", "rollout_id", "split")
        },
        "panel_sha256": panel["panel_id"],
        "policy_identity_sha256": panel["policy_binding_sha256"],
        "checkpoint_sha256": admission["serving_tree_sha256"],
        "rng_contract_sha256": admission["rng_contract"]["sha256"],
        "source_commit": SOURCE_COMMIT,
        "prompt_binding": prompt_binding,
    }
    write_experience_config(output / "train-experience", value)


def _server_identity_arguments(args, plan, output, admission, case, seed) -> list[str]:
    command = [
        str(args.policy_python),
        str(output / "native_comet_server.py"),
        "--source-root",
        str(args.policy_root),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--manager-step",
        str(admission["manager_step"]),
        "--asset-id",
        admission["normalization"]["asset_id"],
        "--task-id",
        str(admission["task_id"]),
        "--task-name",
        admission["task"],
        "--policy-label",
        "comet-trained-selected",
        "--port",
        str(args.port),
        "--upstream-commit",
        plan["upstream_commit"],
        "--case-id",
        case["case_id"],
        "--instance-id",
        str(case["instance_id"]),
        "--rollout-id",
        str(case["rollout_id"]),
        "--case-seed",
        str(seed),
        "--checkpoint-sha256",
        admission["serving_tree_sha256"],
        "--rng-contract-sha256",
        admission["rng_contract"]["sha256"],
        "--trace-configuration-sha256",
        canonical_digest(admission["trace"]),
        "--process-receipt",
        str(output / "native-process.json"),
    ]
    prompt = getattr(args, "policy_prompt_override", None)
    if prompt is not None:
        command.extend(("--task-prompt-override", prompt))
    return command


def _expected_qualification(
    admission: dict,
    case: dict,
    seed: int,
    prompt_binding: dict | None = None,
    prompt: str | None = None,
) -> dict[str, object]:
    value = {
        "schema": "npa.behavior.comet-native-serving-load-qualification.v1",
        "status": "checkpoint_loaded_with_explicit_case_rng",
        "case_id": case["case_id"],
        "task": admission["task"],
        "instance_id": case["instance_id"],
        "rollout_id": case["rollout_id"],
        "case_seed": seed,
        "checkpoint_content_sha256": admission["serving_tree_sha256"],
        "manager_step": admission["manager_step"],
        "asset_id": admission["normalization"]["asset_id"],
        "source_commit": SOURCE_COMMIT,
        "config_name": CONFIG_NAME,
        "rng_contract_sha256": admission["rng_contract"]["sha256"],
        "trace_configuration_sha256": canonical_digest(admission["trace"]),
        "inference_count": 0,
    }
    if prompt_binding is not None:
        value["prompt_binding"] = prompt_binding
    if prompt is not None:
        value["task_prompt_override"] = prompt
    return value


def _qualify(command: list[str], args, output: Path, expected: dict) -> Path:
    qualification = output / "trained-load-qualification.json"
    subprocess.run(
        [*command, "--qualify-only", "--qualification-output", str(qualification)],
        cwd=args.policy_root,
        check=True,
    )
    validate_load_qualification(json.loads(qualification.read_text()), expected)
    return qualification


def _validated_admission(args) -> dict:
    admission = validate_trained_comet_admission(
        Path(args.policy_archive),
        args.policy_trained_input_root,
        args.policy_checkpoint,
        args.policy_trained_panel,
        expected_serving_identity=serving_artifact(args),
    )
    if admission != args.policy_trained_execution_admission:
        raise ValueError("Trained Comet admission changed after worker preflight")
    return admission


def _record_provenance(output, admission, case, seed, qualification, adapters, command):
    evidence = {
        "schema": "npa.behavior.comet-trained-policy.v1",
        "status": "discarded_selected_export_load_complete_serving_not_started",
        "admission": admission,
        "case": case,
        "case_seed": seed,
        "qualification": file_identity(qualification),
        "adapters": adapters,
        "command": command,
    }
    atomic_json(output / "policy-provenance.json", evidence)


def prepare_policy(args, plan: dict, output: Path) -> list[str]:
    """Revalidate selected parity, qualify a load, and stage serving.

    Args:
        args: Managed-policy runtime paths and settings.
        plan: One-case frozen DEV, REPORT, or recorded TRAIN execution plan.
        output: Fresh case evidence directory.
    Returns:
        Serving-process argv for the managed supervisor.
    Raises:
        ValueError: Admission, source, task, or qualification differs.
        subprocess.CalledProcessError: The discarded load process fails.
    """
    verify_source(args.policy_root)
    admission = _validated_admission(args)
    case = _case(args, plan, admission)
    _task_mapping(args.policy_root, admission["task_id"], admission["task"])
    output.mkdir(parents=True, exist_ok=True)
    experience = getattr(args, "train_experience", False)
    prompt_binding = (
        effective_prompt_binding(
            args.policy_root,
            admission["task"],
            admission["task_id"],
            getattr(args, "policy_prompt_override", None),
        )
        if experience
        else None
    )
    adapters = _stage_adapters(output, train_experience=experience)
    command = _server_command(args, plan, output, admission)
    seed = case_seed(_rng_identity(admission), case)
    qualification = _qualify(
        command,
        args,
        output,
        _expected_qualification(
            admission,
            case,
            seed,
            prompt_binding,
            getattr(args, "policy_prompt_override", None),
        ),
    )
    if experience:
        qualified_prompt = json.loads(qualification.read_text())["prompt_binding"]
        _write_experience_config(
            args,
            args.policy_trained_panel,
            case,
            output,
            admission,
            qualified_prompt,
        )
    _record_provenance(output, admission, case, seed, qualification, adapters, command)
    return command


__all__ = ["finalize_process", "prepare_policy"]
