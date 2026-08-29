"""Stateless stage adapters for the canonical compositional Sim2Real workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.workflows.sim2real.stage14_finalize import (
    download_plan as _stage14_download_plan,  # noqa: F401 - compatibility import
    finalize_in_work as _stage14_in_work,
)
from npa.workflows.sim2real.stage9_evaluator import (
    validate_stage7_cosmos3_coverage as _validate_stage7_cosmos3_coverage,
)
from npa.workflows.sim2real.stage9_replay import (
    existing_replay as _stage9_existing_replay,
)
from npa.workflows.sim2real.workflow_io import (
    aggregate_parallel_provenance,
    list_prefix,
    publish_component_lane_record,
    publish_component_record,
    read_json,
    storage,
    write_json,
    write_loop_output,
)


def _set_env(values: dict[str, Any]) -> None:
    for key, value in values.items():
        os.environ[key] = str(value)


def _root(args: argparse.Namespace) -> str:
    return str(args.root_uri).rstrip("/")


def _work(stage: int) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"npa-s2r-stage-{stage:02d}-"))


def _parse_bool(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean (1/0, true/false, yes/no, on/off), got {value!r}"
    )


def _authoritative_scene_args(root: str) -> list[str]:
    """Bind EnvGen and split to the exact Stage 2 task/scene contract."""

    return [
        "--scene-spec-uri",
        f"{root}/stage_02_assets/consumed_scene_spec.json",
    ]


def _stage1(args: argparse.Namespace) -> None:
    from npa.workflows.sim2real.task_contract import (
        build_task_contract,
        validate_seed_dataset_manifest,
    )

    root, work = _root(args), _work(1)
    objects = list_prefix(args.trigger_uri.rstrip("/") + "/")
    if not objects:
        raise RuntimeError("Stage 1 trigger prefix is empty")
    trigger_bucket = urlparse(args.trigger_uri).netloc
    objects_with_bucket = [dict(item, Bucket=trigger_bucket) for item in objects]
    contract = build_task_contract(
        task_id=args.task_id,
        dataset_id=args.dataset_id,
        dataset_uri=args.trigger_uri,
    )
    seed = read_json(args.seed_manifest_uri, directory=work)
    seed_proof = validate_seed_dataset_manifest(
        seed,
        contract=contract,
        trigger_objects=objects_with_bucket,
    )
    payload = {
        "schema": "npa.sim2real.trigger.v1",
        "stage": 1,
        "run_id": args.run_id,
        "trigger_dataset_uri": args.trigger_uri,
        "trigger_dataset_id": args.dataset_id,
        "task_id": args.task_id,
        "object_count": len(objects),
        "object_bytes": sum(int(item.get("Size") or 0) for item in objects),
        "seed_dataset_proof": seed_proof,
        "task_contract": contract,
    }
    uri = f"{root}/stage_01_trigger/trigger.json"
    write_json(uri, payload, directory=work)
    publish_component_record(
        root_uri=root,
        stage=1,
        name="stage_01_trigger",
        tier="WORKS",
        evidence="Validated a non-empty task-aligned Isaac seed dataset and manifest.",
        artifacts={
            "trigger": uri,
            "task_contract_digest": contract["task_contract_digest"],
        },
    )


def _stage2(args: argparse.Namespace) -> None:
    from npa.workflows.sim2real.robot_contract import (
        materialize_robot_contract,
        stock_robot_contract,
    )

    root, work = _root(args), _work(2)
    trigger = read_json(f"{root}/stage_01_trigger/trigger.json", directory=work)
    contract = dict(trigger["task_contract"])
    scene: dict[str, Any] = {
        "schema": "npa.sim2real.consumed_scene_spec.v1",
        "task_id": contract["task_id"],
        "dataset_id": contract["dataset"]["id"],
        "task_contract_digest": contract["task_contract_digest"],
        "task_contract": contract,
        "assets_uri": args.assets_uri,
        "scene_spec_uri": args.scene_spec_uri,
        "camera_names": contract["cameras"],
        "cameras": contract["cameras"],
        "robot": contract["embodiment"],
        "object": contract["object"],
        "success": contract["success"],
    }
    for uri in (args.assets_uri, args.scene_spec_uri):
        if uri and uri.startswith("s3://") and not list_prefix(uri):
            raise RuntimeError(f"Stage 2 configured asset is inaccessible: {uri}")
    contract_uri = f"{root}/stage_02_assets/task-contract.json"
    scene_uri = f"{root}/stage_02_assets/consumed_scene_spec.json"
    robot_uri = f"{root}/stage_02_assets/consumed_robot_spec.json"
    if args.robot_spec_uri:
        robot = materialize_robot_contract(
            robot_spec_uri=args.robot_spec_uri,
            root_uri=root,
            work_dir=work / "robot",
            client=storage(),
        )
        scene["robot_spec_uri"] = robot_uri
        scene["robot"] = robot["embodiment"]
        scene["embodiment_digest"] = robot["embodiment_digest"]
    else:
        robot = stock_robot_contract()
    write_json(contract_uri, contract, directory=work)
    write_json(scene_uri, scene, directory=work)
    write_json(robot_uri, robot, directory=work)
    publish_component_record(
        root_uri=root,
        stage=2,
        name="stage_02_assets",
        tier="WORKS",
        evidence="Consumed the normalized Isaac task, asset, robot, camera, and strict-success contract.",
        artifacts={
            "task_contract": contract_uri,
            "scene": scene_uri,
            "robot_contract": robot_uri,
            "embodiment_digest": robot.get("embodiment_digest", "stock_franka"),
        },
    )


def _stage3(args: argparse.Namespace) -> None:
    from npa.workflows.sim2real.workflow_transfer import (
        run_cosmos2_transfer_component_from_s3,
    )

    root = _root(args)
    _set_env(
        {
            "NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS": "1",
            "NPA_SIM2REAL_AUGMENT_MODE": "real",
        }
    )
    result_uri = f"{root}/augment/cosmos2-transfer-result.json"
    frames_uri = f"{root}/augment/frames/"
    result = run_cosmos2_transfer_component_from_s3(
        input_uri=args.trigger_uri,
        output_uri=result_uri,
        augmented_frames_uri=frames_uri,
        assets_uri=args.assets_uri,
        scene_spec_uri=args.scene_spec_uri,
        image=os.environ["NPA_TASK_IMAGE"],
        run_id=args.run_id,
    )
    manifest = dict(result.get("manifest") or {})
    if manifest.get("mode") != "cosmos_transfer2.5_gpu" or not manifest.get(
        "input_conditioned"
    ):
        raise RuntimeError(
            "Stage 3 did not produce real input-conditioned Cosmos Transfer output"
        )
    frames = manifest.get("frames")
    if (
        not isinstance(frames, list)
        or not frames
        or int(manifest.get("frame_count") or 0) != len(frames)
    ):
        raise RuntimeError("Stage 3 did not publish its exact non-empty frame lineage")
    publish_component_record(
        root_uri=root,
        stage=3,
        name="stage_03_augment",
        tier="WORKS",
        evidence="Real Cosmos Transfer 2.5 generated task-conditioned augmentation on the workflow GPU.",
        artifacts={
            "manifest": f"{root}/augment/manifest.json",
            "result": result_uri,
            "frames": frames_uri,
            "frame_count": int(manifest.get("frame_count") or 0),
        },
        require_gpu=True,
    )


def _stage4(args: argparse.Namespace) -> None:
    from npa.workflows import sim2real_envgen
    from npa.workflows.sim2real.workflow_io import image_provenance

    root, work = _root(args), _work(4)
    result = sim2real_envgen.main(
        [
            "raw-shard",
            "--run-id",
            args.run_id,
            "--output-uri",
            root + "/",
            "--env-count",
            str(args.env_count),
            "--shard-index",
            str(args.shard_index),
            "--shard-count",
            str(args.shard_count),
            "--seed",
            str(args.seed),
            "--augmented-frames-uri",
            f"{root}/augment/manifest.json",
            *_authoritative_scene_args(root),
            "--output-dir",
            str(work),
        ]
    )
    if result != 0:
        raise RuntimeError("Stage 4 raw environment shard failed")
    proof = {
        "schema": "npa.sim2real.envgen_shard_execution.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "provenance": image_provenance(require_gpu=True),
    }
    write_json(
        f"{root}/envs/raw/provenance-{args.shard_index:05d}.json",
        proof,
        directory=work / "proof",
    )
    publish_component_lane_record(
        root_uri=root,
        stage=4,
        lane=f"shard-{args.shard_index:05d}",
        evidence="This standard-workflow leaf generated one declared raw-environment shard.",
        artifacts={
            "raw_envs": f"{root}/envs/raw/",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "provenance": f"{root}/envs/raw/provenance-{args.shard_index:05d}.json",
        },
        execution_provenance=proof["provenance"],
    )


def _stage5(args: argparse.Namespace) -> None:
    from npa.workflows import sim2real_envgen

    root, work = _root(args), _work(5)
    argv = [
        "split",
        "--run-id",
        args.run_id,
        "--output-uri",
        root + "/",
        "--env-count",
        str(args.env_count),
        "--train-fraction",
        str(args.train_fraction),
        "--shard-count",
        str(args.shard_count),
        "--seed",
        str(args.seed),
        "--augmented-frames-uri",
        f"{root}/augment/manifest.json",
        *_authoritative_scene_args(root),
        "--output-dir",
        str(work),
    ]
    if sim2real_envgen.main(argv) != 0:
        raise RuntimeError("Stage 5 environment split failed")
    split_uri = f"{root}/envs/manifest/split-manifest.json"
    split = read_json(split_uri, directory=work / "audit")
    if (
        split.get("disjoint") is not True
        or any((split.get("config_digest_leakage") or {}).values())
        or any(
            int(split.get(key) or 0) < 1
            for key in ("train_count", "validation_count", "gold_heldout_count")
        )
    ):
        raise RuntimeError("Stage 5 did not seal train/validation/gold splits")
    split_digest = hashlib.sha256(
        json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shard_provenance = [
        read_json(
            f"{root}/envs/raw/provenance-{index:05d}.json",
            directory=work / f"shard-proof-{index}",
        )
        for index in range(args.shard_count)
    ]
    expected_envgen_image = args.envgen_image.removeprefix("docker:")
    if not expected_envgen_image or any(
        str(item.get("provenance", {}).get("image") or "") != expected_envgen_image
        for item in shard_provenance
    ):
        raise RuntimeError(
            "Stage 4 shard provenance does not match the configured immutable EnvGen image"
        )
    shard_lane_records = [
        read_json(
            f"{root}/components/lanes/stage_04/shard-{index:05d}.json",
            directory=work / f"shard-lane-{index}",
        )
        for index in range(args.shard_count)
    ]
    if any(
        int(item.get("stage") or 0) != 4
        or item.get("lane") != f"shard-{index:05d}"
        or int(item.get("artifacts", {}).get("shard_index", -1)) != index
        or int(item.get("artifacts", {}).get("shard_count") or 0) != args.shard_count
        or item.get("artifacts", {}).get("image")
        != shard_provenance[index]["provenance"]["image"]
        for index, item in enumerate(shard_lane_records)
    ):
        raise RuntimeError(
            "Stage 4 lane records do not match the declared shard fan-out"
        )
    joined_provenance = aggregate_parallel_provenance(
        [item["provenance"] for item in shard_provenance], stage=4
    )
    publish_component_record(
        root_uri=root,
        stage=4,
        name="stage_04_envs_raw",
        tier="WORKS",
        evidence="Parallel standard-workflow GPU states generated every raw environment shard.",
        artifacts={
            "raw_envs": f"{root}/envs/raw/",
            "shard_count": args.shard_count,
            "shard_provenance": shard_provenance,
            "lane_records": shard_lane_records,
        },
        require_gpu=True,
        execution_provenance=joined_provenance,
    )
    publish_component_record(
        root_uri=root,
        stage=5,
        name="stage_05_envs_train",
        tier="WORKS",
        evidence="Created and sealed disjoint train, validation, and untouched gold scenario sets.",
        artifacts={
            "manifest": split_uri,
            "train_envs": f"{root}/envs/train/envs.jsonl",
            "validation_envs": f"{root}/envs/validation/envs.jsonl",
            "gold_envs": f"{root}/envs/gold-heldout/envs.jsonl",
            "split_digest": split_digest,
        },
    )


def _stage6(args: argparse.Namespace) -> None:
    root, work = _root(args), _work(6)
    split = read_json(f"{root}/envs/manifest/split-manifest.json", directory=work)
    payload = {
        "schema": "npa.sim2real.token_scenario_manifest.v1",
        "run_id": args.run_id,
        "task_contract_uri": f"{root}/stage_02_assets/task-contract.json",
        "augmentation_manifest_uri": f"{root}/augment/manifest.json",
        "split_manifest_uri": f"{root}/envs/manifest/split-manifest.json",
        "train_envs_uri": f"{root}/envs/train/envs.jsonl",
        "validation_envs_uri": f"{root}/envs/validation/envs.jsonl",
        "gold_envs_uri": f"{root}/envs/gold-heldout/envs.jsonl",
        "split_digest": hashlib.sha256(
            json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "sealed_gold": True,
    }
    uri = f"{root}/tokens/manifest.json"
    write_json(uri, payload, directory=work)
    publish_component_record(
        root_uri=root,
        stage=6,
        name="stage_06_tokens",
        tier="WORKS",
        evidence="Published explicit S3 token/scenario handoff URIs with sealed gold lineage.",
        artifacts={"tokens": uri, "split_digest": payload["split_digest"]},
    )


def _common_isaac_env(args: argparse.Namespace, *, split_uri: str) -> dict[str, Any]:
    from npa.workflows.sim2real.isaac_stage_contract import common_environment

    return common_environment(args, split_uri=split_uri)


def _assert_embodiment_evidence(
    *, root: str, payload: dict[str, Any], stage: str
) -> dict[str, Any]:
    from npa.workflows.sim2real.isaac_stage_contract import verify_evidence

    return verify_evidence(root=root, payload=payload, stage=stage)


def _stage7(args: argparse.Namespace) -> None:
    from npa.workflows.sim2real import byo_isaac_policy_rollout

    root, work = _root(args), _work(7)
    output = (
        work
        / "actions"
        / f"outer-{args.outer_iteration:02d}"
        / f"iter-{args.inner_iteration:02d}"
    )
    payload_path = work / "rollouts-result.json"
    env = _common_isaac_env(args, split_uri=f"{root}/envs/train/envs.jsonl")
    env.update(
        {
            "NPA_SIM2REAL_OUTPUT_JSON": payload_path,
            "NPA_SIM2REAL_OUTPUT_DIR": output,
            "NPA_SIM2REAL_ROLLOUT_COUNT": args.rollout_count,
            "NPA_SIM2REAL_STEPS_PER_ROLLOUT": args.steps_per_rollout,
            "NPA_SIM2REAL_ROLLOUT_TAG": (
                f"outer-{args.outer_iteration:02d}-iter-{args.inner_iteration:02d}"
            ),
        }
    )
    _set_env(env)
    if byo_isaac_policy_rollout.main() != 0:
        raise RuntimeError("Stage 7 Isaac rollout adapter failed")
    payload = json.loads(payload_path.read_text())
    embodiment = _assert_embodiment_evidence(
        root=root, payload=payload, stage="Stage 7 rollout"
    )
    inv = dict(payload.get("component_invocation") or {})
    if inv.get("mode") != "npa_workflow_skypilot_task":
        raise RuntimeError("Stage 7 did not execute in its workflow-owned Isaac task")
    destination = (
        f"{root}/actions/train/outer-{args.outer_iteration:02d}/"
        f"iter-{args.inner_iteration:02d}/"
    )
    storage().upload_directory(str(output), destination)
    write_loop_output(
        destination + "rollouts-result.json",
        payload,
        work / "out",
        args.outer_iteration,
        args.inner_iteration,
    )
    publish_component_record(
        root_uri=root,
        stage=7,
        name="stage_07_actions_train",
        tier="WORKS",
        evidence="Isaac Lab loaded the train scenarios and emitted real multi-camera policy rollouts in the workflow GPU task.",
        artifacts={
            "prefix": destination,
            "rollout_count": len(payload.get("rollout_dirs") or []),
            "component_invocation": inv,
            "outer_iteration": args.outer_iteration,
            "inner_iteration": args.inner_iteration,
            "embodiment": embodiment,
        },
        require_gpu=True,
    )


def _stage8(args: argparse.Namespace) -> None:
    from npa.workflows.sim2real.stage8_cosmos3 import run

    run(args)


def _run_eval(
    args: argparse.Namespace,
    *,
    split: str,
    envs_uri: str,
    env_count: int,
    evidence: dict[str, Any],
    output_path: Path,
    tag: str,
) -> dict[str, Any]:
    from npa.workflows.sim2real import byo_isaac_eval

    evidence_path = output_path.parent / f"{split}-inner-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    env = _common_isaac_env(args, split_uri=f"{_root(args)}/envs/train/envs.jsonl")
    env.update(
        {
            "NPA_SIM2REAL_OUTPUT_JSON": output_path,
            "NPA_SIM2REAL_INNER_EVIDENCE_JSON": evidence_path,
            "NPA_SIM2REAL_HELDOUT_ENVS_URI": envs_uri,
            "NPA_SIM2REAL_HELDOUT_ENV_COUNT": env_count,
            "NPA_SIM2REAL_EVALUATION_SPLIT": split,
            "NPA_SIM2REAL_EVAL_TAG": tag,
            "NPA_BYO_ISAAC_SUCCESS_DIST_M": "0.05",
        }
    )
    _set_env(env)
    if byo_isaac_eval.main() != 0:
        raise RuntimeError(f"Isaac {split} evaluation failed")
    report = json.loads(output_path.read_text())
    if (
        report.get("component_invocation", {}).get("mode")
        != "npa_workflow_skypilot_task"
    ):
        raise RuntimeError(f"{split} evaluation escaped the workflow-owned task")
    return report


def _stage9(args: argparse.Namespace) -> None:
    from npa.workbench.cosmos.reason import cosmos_reason_family
    from npa.workflows.sim2real import byo_isaac_trainer
    from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint
    from npa.workflows.sim2real.temporal_credit import convert_evaluation

    root, work = _root(args), _work(9)
    lane_base = (
        f"{root}/vlm_eval/train/outer-{args.outer_iteration:02d}/"
        f"iter-{args.inner_iteration:02d}/"
    )
    cosmos3 = read_json(lane_base + "cosmos3.json", directory=work / "cosmos3")
    actions_uri = (
        f"{root}/actions/train/outer-{args.outer_iteration:02d}/"
        f"iter-{args.inner_iteration:02d}/"
    )
    stage7 = read_json(
        actions_uri + "rollouts-result.json", directory=work / "stage7-rollouts"
    )
    stage8_record = read_json(
        f"{root}/components/stage_08.json", directory=work / "stage8-record"
    )
    if (
        cosmos3.get("schema") not in {
            "npa.sim2real.cosmos3_evaluator.v1",
            "npa.sim2real.cosmos_reason_lane.v2",
        }
        or (cosmos3.get("evaluator") or cosmos3.get("lane")) != "cosmos3"
        or cosmos_reason_family(str(cosmos3.get("model") or "")) != "cosmos3"
        or cosmos3.get("backend") != "token_factory"
        or cosmos3.get("provider") != "nebius"
        or not cosmos3.get("provenance")
        or stage8_record.get("stage") != 8
        or stage8_record.get("name") != "stage_08_vlm_eval_train"
        or stage8_record.get("artifacts", {}).get("result")
        != lane_base + "cosmos3.json"
        or stage8_record.get("artifacts", {}).get("backend") != "token_factory"
        or int(stage8_record.get("artifacts", {}).get("outer_iteration") or 0)
        != args.outer_iteration
        or int(stage8_record.get("artifacts", {}).get("inner_iteration") or 0)
        != args.inner_iteration
    ):
        raise RuntimeError(
            "Stage 8 requires genuine hosted Cosmos3 identity, model family, and provenance"
        )
    right = _validate_stage7_cosmos3_coverage(stage7, cosmos3)
    cosmos3_usage = dict(cosmos3.get("evaluator_usage") or {})
    request_fields = {
        "request_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_seconds",
        "retries",
        "cost_usd",
    }
    requests = [dict(item.get("request") or {}) for item in right.values()]
    if (
        int(cosmos3_usage.get("request_count") or 0) != len(right)
        or any(not request_fields.issubset(request) for request in requests)
        or int(cosmos3_usage.get("input_tokens") or 0)
        != sum(int(request.get("input_tokens") or 0) for request in requests)
        or int(cosmos3_usage.get("output_tokens") or 0)
        != sum(int(request.get("output_tokens") or 0) for request in requests)
        or int(cosmos3_usage.get("total_tokens") or 0)
        != sum(int(request.get("total_tokens") or 0) for request in requests)
        or len(cosmos3_usage.get("per_request_latency_seconds") or []) != len(right)
        or any(
            item.get("schema") != "npa.sim2real.vlm_eval.v3"
            or item.get("backend") != "token_factory"
            or item.get("provider") != "nebius"
            or item.get("model") != cosmos3.get("model")
            or not isinstance(item.get("request"), dict)
            or int(item.get("action_count") or 0) < 1
            or len(item.get("per_step") or []) != int(item.get("action_count") or 0)
            or len(
                {
                    int(step.get("step"))
                    for step in item.get("per_step") or []
                    if isinstance(step, dict) and step.get("step") is not None
                }
            )
            != int(item.get("action_count") or 0)
            for item in right.values()
        )
    ):
        raise RuntimeError("Stage 8 Cosmos3 evaluations do not exactly cover Stage 7 rollouts")
    evaluation_dir, signal_dir = work / "evaluations", work / "signals"
    evaluation_dir.mkdir()
    signal_dir.mkdir()
    evaluations: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    for rollout_id in sorted(right):
        evaluation = dict(right[rollout_id])
        evaluation["threshold"] = args.threshold
        signal = convert_evaluation(evaluation)
        (evaluation_dir / f"{rollout_id}.json").write_text(json.dumps(evaluation, indent=2))
        (signal_dir / f"{rollout_id}.json").write_text(json.dumps(signal, indent=2))
        evaluations.append(evaluation)
        signals.append(signal)
    evidence_uri = f"{root}/inner_loop/outer-{args.outer_iteration:02d}/evidence.json"
    evaluation_uri = lane_base + "evaluations/"
    signal_uri = lane_base + "signals/"
    prior: dict[str, Any] = {}
    if list_prefix(evidence_uri):
        prior = read_json(evidence_uri, directory=work / "prior-evidence")
    replay = _stage9_existing_replay(
        prior=prior,
        outer_iteration=args.outer_iteration,
        inner_iteration=args.inner_iteration,
        actions_uri=actions_uri,
        evaluation_uri=evaluation_uri,
        signal_uri=signal_uri,
        sample_vlm_eval=evaluations[0],
        sample_signal=signals[0],
    )
    if replay is not None:
        candidate, selection, update = replay
        write_loop_output(evidence_uri, prior, work / "evidence", args.outer_iteration)
        publish_component_record(
            root_uri=root,
            stage=9,
            name="stage_09_training_signal",
            tier="WORKS",
            evidence="Re-adopted exact durable same-iteration PPO and validation evidence after a standard-runtime Job retry.",
            artifacts={
                "evidence": evidence_uri,
                "checkpoint": selection["checkpoint_uri"],
                "checkpoint_sha256": selection.get("checkpoint_sha256", ""),
                "validation_report": candidate["validation_report_uri"],
                "ppo_iterations": args.ppo_iterations,
                "component_invocation": update.get("component_invocation"),
                "idempotent_replay": True,
            },
            require_gpu=True,
        )
        return
    signal_batch = work / "signal-batch.json"
    signal_batch.write_text(json.dumps({"signals": signals}, indent=2))
    training_output = work / "training-update.json"
    env = _common_isaac_env(args, split_uri=f"{root}/envs/train/envs.jsonl")
    env.update(
        {
            "NPA_SIM2REAL_SIGNAL_JSON": signal_batch,
            "NPA_SIM2REAL_OUTPUT_JSON": training_output,
            "NPA_SIM2REAL_TRAINER_TAG": (
                f"outer-{args.outer_iteration:02d}-iter-{args.inner_iteration:02d}"
            ),
            "NPA_BYO_ISAAC_NUM_ENVS": args.ppo_num_envs,
            "NPA_BYO_ISAAC_ITERATIONS": args.ppo_iterations,
            "NPA_BYO_ISAAC_STEPS_PER_ENV": args.ppo_steps_per_env,
            "NPA_BYO_ISAAC_VALIDATION_INTERVAL": args.ppo_iterations,
        }
    )
    _set_env(env)
    if byo_isaac_trainer.main() != 0:
        raise RuntimeError("Stage 9 genuine BYO Isaac PPO failed")
    update = json.loads(training_output.read_text())
    training_embodiment = _assert_embodiment_evidence(
        root=root, payload=update, stage="Stage 9 PPO"
    )
    if (
        update.get("component_invocation", {}).get("mode")
        != "npa_workflow_skypilot_task"
    ):
        raise RuntimeError("Stage 9 PPO escaped the workflow-owned task")
    checkpoint_uri = str(update.get("checkpoint_path") or "")
    if not checkpoint_uri.startswith("s3://"):
        raise RuntimeError("Stage 9 PPO did not publish a checkpoint")
    validation_path = work / "validation-report.json"
    validation_evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "selected_checkpoint_uri": checkpoint_uri,
        "final_checkpoint_uri": checkpoint_uri,
    }
    validation = _run_eval(
        args,
        split="validation",
        envs_uri=f"{root}/envs/validation/envs.jsonl",
        env_count=args.validation_count,
        evidence=validation_evidence,
        output_path=validation_path,
        tag=f"validation-o{args.outer_iteration:02d}-i{args.inner_iteration:02d}",
    )
    validation_embodiment = _assert_embodiment_evidence(
        root=root, payload=validation, stage="Stage 9 validation"
    )
    if training_embodiment != validation_embodiment:
        raise RuntimeError("Stage 9 train/validation embodiment evidence differs")
    validation_uri = (
        f"{root}/eval/validation/outer-{args.outer_iteration:02d}/"
        f"iter-{args.inner_iteration:02d}/report.json"
    )
    write_json(validation_uri, validation, directory=work / "validation-out")
    candidate = {
        "evaluation_split": "validation",
        "outer_iteration": args.outer_iteration,
        "inner_iteration": args.inner_iteration,
        "training_iteration": args.ppo_iterations,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": validation.get("policy_checkpoint_sha256", ""),
        "validation_report_uri": validation_uri,
        "validation_report": validation,
        "embodiment": training_embodiment,
    }
    prior_iterations = list(prior.get("iterations") or [])
    candidates = list(prior.get("checkpoint_candidates") or []) + [candidate]
    selection = select_best_checkpoint(candidates)
    selected_validation = dict(selection.get("validation_report") or {})
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": args.outer_iteration,
        "iterations": prior_iterations
        + [
            {
                "iteration": args.inner_iteration,
                "actions_uri": actions_uri,
                "vlm_eval_uri": evaluation_uri,
                "signal_uri": signal_uri,
                "trainer_component_invocation": update.get("component_invocation"),
                "update": update,
                "embodiment": training_embodiment,
                "sample_vlm_eval": evaluations[0],
                "sample_signal": signals[0],
                "evaluator_usage": cosmos3.get("evaluator_usage"),
            }
        ],
        "checkpoint_candidates": candidates,
        "selected_checkpoint_uri": selection["checkpoint_uri"],
        "final_checkpoint_uri": selection["checkpoint_uri"],
        "checkpoint_selection": selection,
        "selected_validation_report": selected_validation,
        "selected_validation_strict_success": float(
            selected_validation.get("success_rate") or 0.0
        ),
    }
    storage().upload_directory(str(evaluation_dir), evaluation_uri)
    storage().upload_directory(str(signal_dir), signal_uri)
    write_json(
        lane_base + "signal-batch.json", {"signals": signals}, directory=work / "batch"
    )
    write_json(lane_base + "training-update.json", update, directory=work / "training")
    write_loop_output(evidence_uri, evidence, work / "evidence", args.outer_iteration)
    publish_component_record(
        root_uri=root,
        stage=9,
        name="stage_09_training_signal",
        tier="WORKS",
        evidence="Genuine BYO Isaac RSL-RL PPO consumed bounded temporal Reason signals and published an exact checkpoint; validation alone selected it.",
        artifacts={
            "evidence": evidence_uri,
            "checkpoint": selection["checkpoint_uri"],
            "checkpoint_sha256": selection.get("checkpoint_sha256", ""),
            "validation_report": selection["validation_report_uri"],
            "ppo_iterations": args.ppo_iterations,
            "component_invocation": update.get("component_invocation"),
            "embodiment": training_embodiment,
        },
        require_gpu=True,
    )


def _stage10(args: argparse.Namespace) -> None:
    root, work = _root(args), _work(10)
    evidence_uri = f"{root}/inner_loop/outer-{args.outer_iteration:02d}/evidence.json"
    evidence = read_json(evidence_uri, directory=work / "input")
    report_path = work / "report.json"
    report = _run_eval(
        args,
        split="gold_heldout",
        envs_uri=f"{root}/envs/gold-heldout/envs.jsonl",
        env_count=args.gold_count,
        evidence=evidence,
        output_path=report_path,
        tag=f"gold-o{args.outer_iteration:02d}",
    )
    gold_embodiment = _assert_embodiment_evidence(
        root=root, payload=report, stage="Stage 10 gold evaluation"
    )
    if gold_embodiment:
        selected_uri = str(evidence.get("selected_checkpoint_uri") or "")
        selected_candidate = next(
            (
                dict(item)
                for item in evidence.get("checkpoint_candidates") or []
                if str(item.get("checkpoint_uri") or "") == selected_uri
            ),
            {},
        )
        selected_embodiment = dict(selected_candidate.get("embodiment") or {})
        if not selected_embodiment:
            raise RuntimeError(
                "Stage 10 selected checkpoint has no recorded training embodiment"
            )
        if gold_embodiment != selected_embodiment:
            raise RuntimeError(
                "Stage 10 checkpoint train/eval embodiment parity mismatch"
            )
    render_manifest = dict(report.get("render_manifest") or {})
    render_prefix = str(render_manifest.get("renders_s3_uri") or "")
    canonical_renders = f"eval/gold-heldout/outer-{args.outer_iteration:02d}/renders"
    if not render_prefix or not render_manifest.get("episodes"):
        raise RuntimeError("Stage 10 gold evaluation lacks explicit render lineage")
    render_local = work / canonical_renders
    storage().download_directory(render_prefix.rstrip("/") + "/", str(render_local))
    if not any(render_local.rglob("camera-*.png")):
        raise RuntimeError("Stage 10 gold render prefix contains no camera frames")
    canonical_render_uri = f"{root}/{canonical_renders}/"
    storage().upload_directory(str(render_local), canonical_render_uri)
    report["render_lineage"] = {
        "evaluation_split": "gold_heldout",
        "source_s3_uri": render_prefix,
        "canonical_s3_uri": canonical_render_uri,
        "local_relative_dir": canonical_renders,
    }
    report["local_renders_dir"] = canonical_renders
    report_uri = (
        f"{root}/eval/gold-heldout/outer-{args.outer_iteration:02d}/report.json"
    )
    write_loop_output(report_uri, report, work / "out", args.outer_iteration)
    publish_component_record(
        root_uri=root,
        stage=10,
        name="stage_10_eval_heldout",
        tier="WORKS",
        evidence="Isaac loaded the validation-selected checkpoint and evaluated only the untouched gold split with strict 5 cm stable placement.",
        artifacts={
            "report": report_uri,
            "evaluation_split": "gold_heldout",
            "checkpoint": evidence["selected_checkpoint_uri"],
            "checkpoint_sha256": report.get("policy_checkpoint_sha256", ""),
            "renders": canonical_render_uri,
            "render_lineage": report["render_lineage"],
            "component_invocation": report.get("component_invocation"),
            "embodiment": gold_embodiment,
        },
        require_gpu=True,
    )


def _stage11(args: argparse.Namespace) -> None:
    root, work = _root(args), _work(11)
    report_uri = (
        f"{root}/eval/gold-heldout/outer-{args.outer_iteration:02d}/report.json"
    )
    report = read_json(report_uri, directory=work)
    strict_rate = float(report.get("success_rate") or 0.0)
    promote = args.allow_early_exit and strict_rate >= args.threshold
    decision = {
        "schema": "npa.sim2real.threshold_decision.v1",
        "run_id": args.run_id,
        "outer_iteration": args.outer_iteration,
        "decision": "promote_checkpoint" if promote else "loop_back_to_inner_loop",
        "success_rate": strict_rate,
        "threshold": args.threshold,
        "strict_success_distance_m": 0.05,
        "placement_stability_required": True,
        "early_exit_enabled": args.allow_early_exit,
        "checkpoint_uri": report.get("policy_checkpoint_uri", "")
        or (report.get("policy_inference_provenance") or {}).get("checkpoint_uri", ""),
        "gold_report_uri": report_uri,
    }
    uri = f"{root}/outer_loop/decision.json"
    write_json(uri, decision, directory=work)
    publish_component_record(
        root_uri=root,
        stage=11,
        name="stage_11_outer_loop",
        tier="WORKS",
        evidence="Applied the unchanged strict 5 cm stable-placement threshold to the sealed gold report; policy quality is reported, not a pipeline gate.",
        artifacts={
            "decision": uri,
            "gold_report": report_uri,
            "success_rate": strict_rate,
        },
        next_action="CONTINUE" if promote else "LOOP_OR_COMPLETE_BUDGET",
    )


def _stage12(args: argparse.Namespace) -> None:
    root, work = _root(args), _work(12)
    payload = {
        "schema": "npa.sim2real.external_validation_seam.v1",
        "stage": 12,
        "status": "external_not_executed",
        "tier": "SEAM",
        "description": "Physical deployment validation belongs to the robot operator outside this cloud workflow.",
        "input": f"{root}/outer_loop/decision.json",
    }
    uri = f"{root}/external_validation/seam.json"
    write_json(uri, payload, directory=work)
    publish_component_record(
        root_uri=root,
        stage=12,
        name="stage_12_external_validation",
        tier="SEAM",
        evidence="Documented external physical-robot validation boundary; intentionally not executed or labeled WORKS.",
        artifacts={"seam": uri},
        next_action="EXTERNAL_OPERATOR_VALIDATION",
    )


def _stage13(args: argparse.Namespace) -> None:
    root, work = _root(args), _work(13)
    decision = read_json(f"{root}/outer_loop/decision.json", directory=work)
    payload = {
        "schema": "npa.sim2real.retrigger_record.v1",
        "run_id": args.run_id,
        "outer_iteration": args.outer_iteration,
        "decision": decision["decision"],
        "next_run_trigger_uri": args.trigger_uri,
        "automatic_external_deployment": False,
        "status": "recorded",
    }
    uri = f"{root}/retrigger/record.json"
    write_json(uri, payload, directory=work)
    publish_component_record(
        root_uri=root,
        stage=13,
        name="stage_13_retrigger",
        tier="WORKS",
        evidence="Recorded the bounded-loop decision and explicit future trigger without fabricating physical validation.",
        artifacts={"record": uri, "decision": f"{root}/outer_loop/decision.json"},
    )


def _stage14(args: argparse.Namespace) -> None:
    root = _root(args)
    with tempfile.TemporaryDirectory(prefix="npa-s2r-stage-14-") as directory:
        _stage14_in_work(args, root=root, work=Path(directory))


_STAGES = {
    1: _stage1,
    2: _stage2,
    3: _stage3,
    4: _stage4,
    5: _stage5,
    6: _stage6,
    7: _stage7,
    8: _stage8,
    9: _stage9,
    10: _stage10,
    11: _stage11,
    12: _stage12,
    13: _stage13,
    14: _stage14,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 15), required=True)
    parser.add_argument("--root-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trigger-uri", default="")
    parser.add_argument("--seed-manifest-uri", default="")
    parser.add_argument("--dataset-id", default="npa/isaac-lift-cube-franka-seed-v1")
    parser.add_argument("--task-id", default="Isaac-Lift-Cube-Franka-v0")
    parser.add_argument("--assets-uri", default="")
    parser.add_argument("--scene-spec-uri", default="")
    parser.add_argument("--robot-spec-uri", default="")
    parser.add_argument("--env-count", type=int, default=12)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--envgen-image", default="")
    parser.add_argument("--outer-iteration", type=int, default=1)
    parser.add_argument("--inner-iteration", type=int, default=1)
    parser.add_argument("--rollout-count", type=int, default=1)
    parser.add_argument("--steps-per-rollout", type=int, default=32)
    parser.add_argument("--reason-model", default="nvidia/Cosmos3-Super-Reasoner")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ppo-num-envs", type=int, default=64)
    parser.add_argument("--ppo-iterations", type=int, default=10)
    parser.add_argument("--ppo-steps-per-env", type=int, default=24)
    parser.add_argument("--validation-count", type=int, default=3)
    parser.add_argument("--gold-count", type=int, default=3)
    parser.add_argument("--capture-fps", default="10")
    parser.add_argument("--capture-width", default="640")
    parser.add_argument("--capture-height", default="480")
    parser.add_argument("--png-compress-level", default="2")
    parser.add_argument("--allow-early-exit", type=_parse_bool, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _STAGES[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
