"""Prepare a released Comet params checkpoint for recorded TRAIN inference."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .campaign import canonical_digest
from .comet_policy import COMET12_PROFILE, CONFIG_NAME, SOURCE_COMMIT, verify_source
from .native_comet_policy import case_seed, finalize_process
from .native_comet_server import validate_load_qualification
from .native_training_checkpoint import atomic_json, file_identity
from .released_comet_checkpoint import validate_released_comet_admission
from .serving_identity import serving_artifact
from .train_experience import write_experience_config
from .train_prompt import effective_prompt_binding


def _stage_adapters(output: Path) -> dict[str, dict]:
    names = (
        "released_comet_checkpoint.py",
        "released_comet_policy.py",
        "native_comet_server.py",
        "comet_policy.py",
        "comet12-checkpoint.json",
        "evaluator_versions.py",
        "evaluator_wire.py",
        "train_experience.py",
        "train_experience_evaluator.py",
        "train_official_q.py",
        "train_prompt.py",
    )
    rows = {}
    for name in names:
        source = Path(__file__).with_name(name)
        target = output / name
        shutil.copyfile(source, target)
        rows[name] = file_identity(target)
    monitor = output / "semantic_monitor"
    monitor.mkdir()
    for name in ("__init__.py", "collector.py", "interface.py", "schema.py"):
        source = Path(__file__).with_name("semantic_monitor") / name
        target = monitor / name
        shutil.copyfile(source, target)
        rows[f"semantic_monitor/{name}"] = file_identity(target)
    return rows


def _case(args, plan: dict, admission: dict) -> dict:
    cases = plan.get("cases")
    if (
        plan.get("recipe", {}).get("split") != "train"
        or not isinstance(cases, list)
        or len(cases) != 1
        or cases[0].get("task") != admission["task"]
        or cases[0].get("rollout_id") != 0
        or plan["recipe"].get("tasks") != [admission["task"]]
        or getattr(args, "policy_task_name", None) != admission["task"]
        or not getattr(args, "train_experience", False)
    ):
        raise ValueError("Released Comet requires one recorded TRAIN case")
    return cases[0]


def _server_command(args, plan, output, admission, case, seed) -> list[str]:
    command = [
        str(args.policy_python),
        str(output / "native_comet_server.py"),
        "--checkpoint-layout",
        "released",
        "--released-profile",
        COMET12_PROFILE.kind,
        "--source-root",
        str(args.policy_root),
        "--checkpoint",
        str(args.policy_checkpoint),
        "--task-id",
        str(admission["task_id"]),
        "--task-name",
        admission["task"],
        "--policy-label",
        "comet-released-train",
        "--port",
        str(args.port),
        "--upstream-commit",
        plan["upstream_commit"],
        *_identity_arguments(admission, case, seed),
        "--process-receipt",
        str(output / "native-process.json"),
    ]
    return [
        *command,
        "--train-experience-root",
        str(output / "train-experience"),
    ]


def _identity_arguments(admission, case, seed) -> list[str]:
    return [
        "--case-id",
        case["case_id"],
        "--instance-id",
        str(case["instance_id"]),
        "--rollout-id",
        str(case["rollout_id"]),
        "--case-seed",
        str(seed),
        "--checkpoint-sha256",
        admission["checkpoint"]["archive"]["sha256"],
        "--rng-contract-sha256",
        admission["evidence"]["rng_contract"]["identity"]["sha256"],
        "--trace-configuration-sha256",
        canonical_digest(admission["trace"]),
    ]


def _qualification_expected(admission, case, seed, prompt_binding) -> dict:
    return {
        "schema": "npa.behavior.comet-released-serving-load-qualification.v1",
        "status": "checkpoint_loaded_with_explicit_case_rng",
        "case_id": case["case_id"],
        "task": admission["task"],
        "instance_id": case["instance_id"],
        "rollout_id": case["rollout_id"],
        "case_seed": seed,
        "checkpoint_content_sha256": admission["checkpoint"]["archive"]["sha256"],
        "manager_step": None,
        "asset_id": None,
        "source_commit": SOURCE_COMMIT,
        "config_name": CONFIG_NAME,
        "rng_contract_sha256": admission["evidence"]["rng_contract"]["identity"][
            "sha256"
        ],
        "trace_configuration_sha256": canonical_digest(admission["trace"]),
        "inference_count": 0,
        "prompt_binding": prompt_binding,
    }


def _experience_config(args, panel, case, admission, prompt_binding) -> dict:
    return {
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
        "checkpoint_sha256": admission["checkpoint"]["archive"]["sha256"],
        "rng_contract_sha256": admission["evidence"]["rng_contract"]["identity"][
            "sha256"
        ],
        "source_commit": SOURCE_COMMIT,
        "prompt_binding": prompt_binding,
    }


def _validated_admission(args) -> dict:
    admission = validate_released_comet_admission(
        Path(args.policy_released_binding),
        Path(args.policy_released_input_root),
        Path(args.policy_archive),
        Path(args.policy_checkpoint),
        args.policy_released_panel,
        expected_serving_identity=serving_artifact(args),
    )
    if admission != args.policy_released_admission:
        raise ValueError("Released Comet admission changed after worker preflight")
    return admission


def prepare_policy(args, plan: dict, output: Path) -> list[str]:
    """Revalidate a released params checkpoint and stage recorded TRAIN serving."""
    verify_source(args.policy_root)
    admission = _validated_admission(args)
    case = _case(args, plan, admission)
    prompt = effective_prompt_binding(
        args.policy_root, admission["task"], admission["task_id"], None
    )
    output.mkdir(parents=True, exist_ok=True)
    adapters = _stage_adapters(output)
    seed = case_seed(admission["evidence"]["rng_contract"]["identity"], case)
    command = _server_command(args, plan, output, admission, case, seed)
    receipt, qualification = _qualify(args, command, output)
    prompt_expected = _qualification_expected(admission, case, seed, prompt)
    validate_load_qualification(receipt, prompt_expected)
    _record_experience(
        args,
        output,
        admission,
        case,
        seed,
        receipt,
        qualification,
        adapters,
        command,
    )
    return command


def _qualify(args, command, output) -> tuple[dict, Path]:
    qualification = output / "released-load-qualification.json"
    subprocess.run(
        [*command, "--qualify-only", "--qualification-output", str(qualification)],
        cwd=args.policy_root,
        check=True,
    )
    return json.loads(qualification.read_text()), qualification


def _record_experience(
    args, output, admission, case, seed, receipt, qualification, adapters, command
):
    write_experience_config(
        output / "train-experience",
        _experience_config(
            args,
            args.policy_released_panel,
            case,
            admission,
            receipt["prompt_binding"],
        ),
    )
    atomic_json(
        output / "policy-provenance.json",
        {
            "schema": "npa.behavior.comet-released-policy.v1",
            "status": "released_params_load_qualified_serving_not_started",
            "admission": admission,
            "case": case,
            "case_seed": seed,
            "qualification": file_identity(qualification),
            "adapters": adapters,
            "command": command,
            "full_train_state_claimed": False,
            "optimizer_state_claimed": False,
        },
    )


__all__ = ["finalize_process", "prepare_policy"]
