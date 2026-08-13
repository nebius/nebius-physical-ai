"""Legacy Stage 7–9 execution for pre-standard-runtime compatibility.

Only ``legacy_components`` imports this module. It supports archived callers and
artifact replay through the finite engine compatibility window (target removal:
0.5.0, no earlier than 2027-02-01); canonical workflow stages use
``workflow_stage`` and must not import it.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.constants import SCHEMA_RL_SIGNAL
from npa.workflows.sim2real.models import (
    Sim2RealLoopConfig,
    Sim2RealLoopError,
)
from npa.workflows.sim2real.utils import _write_json_artifact


def run_inner_loop(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    initial_quality: float,
    outer_iteration: int = 1,
    resume_checkpoint_uri: str = "",
) -> dict[str, Any]:
    """Run action generation, VLM eval, signal conversion, and policy update.

    ``resume_checkpoint_uri`` (from the prior outer iteration) lets a BYO trainer
    CONTINUE the same policy rather than restart from scratch, so the outer loop's
    "send back for more RL" (stage 11B) actually compounds. The checkpoint advances
    through both the inner iterations and across outer iterations.
    """

    # Resolve engine seams per invocation so focused tests and operators can
    # replace a component without maintaining a second global registry here.
    import npa.workflows.sim2real.engine as engine_api
    from npa.workflows.sim2real_stages import run_policy_rollouts

    _bool_value = engine_api._bool_value
    _convert_eval_to_signal = engine_api._convert_eval_to_signal
    _effective_k8s_parallelism = engine_api._effective_k8s_parallelism
    _run_trainer_via_command = engine_api._run_trainer_via_command
    _signal_diversity_report = engine_api._signal_diversity_report
    _signal_mean_reward = engine_api._signal_mean_reward
    _signal_training_imports = engine_api._signal_training_imports
    evaluate_rollout_with_vlm = engine_api.evaluate_rollout_with_vlm
    run_heldout_eval = engine_api.run_heldout_eval

    iteration_records: list[dict[str, Any]] = []
    reward_trend: list[float] = []
    loss_trend: list[dict[str, float]] = []
    policy_deltas: list[float] = []
    all_signals: list[dict[str, Any]] = []
    calibration_trend: list[dict[str, Any]] = []
    checkpoint_candidates: list[dict[str, Any]] = []
    quality = float(initial_quality)
    reward_head = 0.0
    action_bias = 0.0
    current_checkpoint_uri = str(resume_checkpoint_uri or "").strip()
    prior_selection_path = (
        local_dir
        / "checkpoints"
        / "validation-selection"
        / f"outer-{outer_iteration - 1:02d}.json"
    )
    if outer_iteration > 1 and prior_selection_path.is_file():
        prior_candidate = json.loads(prior_selection_path.read_text(encoding="utf-8"))
        if (
            prior_candidate.get("evaluation_split") == "validation"
            and prior_candidate.get("checkpoint_uri") == current_checkpoint_uri
        ):
            checkpoint_candidates.append(prior_candidate)
    from npa.workflows.sim2real.resume_state import DurableStateStore, canonical_digest

    durable = DurableStateStore(config, local_dir)
    for iteration in range(1, config.inner_iterations + 1):
        starting_checkpoint_uri = current_checkpoint_uri
        iteration_input = {
            "outer_iteration": outer_iteration,
            "inner_iteration": iteration,
            "starting_checkpoint_uri": starting_checkpoint_uri,
            "train_envs_uri": config.train_envs_uri,
            "task_contract_digest": getattr(config, "task_contract_digest", ""),
            "rollout_count": config.rollout_count,
            "steps_per_rollout": config.steps_per_rollout,
            "vlm_models": [config.vlm_reason2_model, config.vlm_reason3_model],
            "runtime_images": {
                "policy": config.policy_image,
                "trainer": config.trainer_image,
                "vlm_reason2": config.vlm_reason2_image or config.vlm_image,
                "vlm_reason3": config.vlm_reason3_image or config.vlm_image,
                "evaluation": config.eval_image,
                "isaac": config.isaac_image,
            },
        }
        completed_iteration = durable.load_unit(
            f"inner-o{outer_iteration:02d}-i{iteration:02d}-complete",
            iteration_input,
        )
        if completed_iteration is not None:
            record = dict(completed_iteration["iteration_record"])
            iteration_records.append(record)
            restored_signals = list(completed_iteration.get("signals") or [])
            all_signals.extend(restored_signals)
            calibration = dict(record.get("signal_calibration") or {})
            calibration_trend.append(calibration)
            reward_trend.append(float(record["mean_reward"]))
            loss_trend.append(dict(completed_iteration["loss_trend_entry"]))
            policy_deltas.append(float(record["policy_delta_vs_control"]))
            checkpoint_candidates.extend(
                list(completed_iteration.get("checkpoint_candidates") or [])
            )
            quality = float(completed_iteration.get("quality") or quality)
            reward_head = float(completed_iteration["reward_head"])
            action_bias = float(completed_iteration["action_bias"])
            current_checkpoint_uri = str(
                completed_iteration.get("current_checkpoint_uri") or ""
            )
            continue
        actions_dir = (
            local_dir
            / "actions"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        eval_dir = (
            local_dir
            / "vlm_eval"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        signal_dir = (
            local_dir
            / "training_signal"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        signal_converter_source = (
            "byo_command" if config.byo_signal_converter.strip() else "reference"
        )
        stage08 = durable.load_unit(
            f"inner-o{outer_iteration:02d}-i{iteration:02d}-stage08-vlm",
            iteration_input,
        )
        if stage08 is None:
            rollouts = run_policy_rollouts(
                config,
                local_dir=local_dir,
                actions_dir=actions_dir,
                outer_iteration=outer_iteration,
                iteration=iteration,
                checkpoint_uri=current_checkpoint_uri,
            )
            evals: list[dict[str, Any]] = []
            vlm_k8s_parallel = not config.byo_vlm_command.strip() and bool(
                config.s3_bucket.strip()
            )
            jobs_per_rollout = 2 if vlm_k8s_parallel and config.vlm_dual_reason else 1
            if vlm_k8s_parallel and len(rollouts) > 1:
                max_workers = min(
                    len(rollouts),
                    max(1, _effective_k8s_parallelism(config) // jobs_per_rollout),
                )
                evaluations: list[dict[str, Any] | None] = [None] * len(rollouts)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(
                            evaluate_rollout_with_vlm,
                            rollout,
                            output_dir=eval_dir,
                            config=config,
                        ): index
                        for index, rollout in enumerate(rollouts)
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        evaluations[index] = future.result()
                evals = [item for item in evaluations if item is not None]
                if len(evals) != len(rollouts):
                    raise Sim2RealLoopError(
                        "parallel VLM eval did not return all rollouts"
                    )
            else:
                for rollout in rollouts:
                    evals.append(
                        evaluate_rollout_with_vlm(
                            rollout,
                            output_dir=eval_dir,
                            config=config,
                        )
                    )
            stage08 = durable.commit_unit(
                f"inner-o{outer_iteration:02d}-i{iteration:02d}-stage08-vlm",
                iteration_input,
                {
                    "evaluations": evals,
                    "rollout_count": len(rollouts),
                    "starting_checkpoint_uri": starting_checkpoint_uri,
                },
            )
        else:
            evals = list(stage08.get("evaluations") or [])
        if not evals:
            raise Sim2RealLoopError("Stage 8 reconciliation produced no evaluations")

        stage09_input = {
            **iteration_input,
            "stage08_payload_digest": canonical_digest(stage08),
            "signal_converter": config.byo_signal_converter,
        }
        stage09 = durable.load_unit(
            f"inner-o{outer_iteration:02d}-i{iteration:02d}-stage09-signal",
            stage09_input,
        )
        if stage09 is None:
            signals = []
            for evaluation in evals:
                signal = _convert_eval_to_signal(
                    evaluation,
                    config=config,
                    output_dir=signal_dir,
                )
                _write_json_artifact(
                    signal_dir / f"{signal['rollout_id']}.json", signal
                )
                signals.append(signal)
            stage09 = durable.commit_unit(
                f"inner-o{outer_iteration:02d}-i{iteration:02d}-stage09-signal",
                stage09_input,
                {"signals": signals},
            )
        else:
            signals = list(stage09.get("signals") or [])
        if not signals:
            raise Sim2RealLoopError("Stage 9 reconciliation produced no signals")
        all_signals.extend(signals)
        candidate_count_before = len(checkpoint_candidates)
        signal_batch_path = (
            local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / f"signals-iter-{iteration:02d}.json"
        )
        calibration = {
            "rollout_count": len(signals),
            "step_count": sum(
                int((signal.get("calibration") or {}).get("step_count") or 0)
                for signal in signals
            ),
            "simulator_grounded_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("simulator_grounded_steps")
                    or 0
                )
                for signal in signals
            ),
            "nonzero_advantage_count": sum(
                int(
                    (signal.get("calibration") or {}).get("nonzero_advantage_count")
                    or 0
                )
                for signal in signals
            ),
            "model_disagreement_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("model_disagreement_steps")
                    or 0
                )
                for signal in signals
            ),
            "vlm_calibrated_steps": sum(
                int((signal.get("calibration") or {}).get("vlm_calibrated_steps") or 0)
                for signal in signals
            ),
            "vlm_accepted_steps": sum(
                int((signal.get("calibration") or {}).get("vlm_accepted_steps") or 0)
                for signal in signals
            ),
            "vlm_rejected_or_downweighted_steps": sum(
                int(
                    (signal.get("calibration") or {}).get(
                        "vlm_rejected_or_downweighted_steps"
                    )
                    or 0
                )
                for signal in signals
            ),
            "vlm_missing_or_malformed_steps": sum(
                int(
                    (signal.get("calibration") or {}).get(
                        "vlm_missing_or_malformed_steps"
                    )
                    or 0
                )
                for signal in signals
            ),
            "vlm_low_confidence_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("vlm_low_confidence_steps")
                    or 0
                )
                for signal in signals
            ),
            "vlm_contradictory_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("vlm_contradictory_steps")
                    or 0
                )
                for signal in signals
            ),
            "vlm_summary_broadcast_steps": sum(
                int(
                    (signal.get("calibration") or {}).get("vlm_summary_broadcast_steps")
                    or 0
                )
                for signal in signals
            ),
            "mean_reward_variance": round(
                sum(
                    float(
                        (signal.get("calibration") or {}).get("reward_variance") or 0.0
                    )
                    for signal in signals
                )
                / max(1, len(signals)),
                10,
            ),
            "simulator_fallback_rollouts": sum(
                bool(
                    (signal.get("calibration") or {}).get(
                        "degenerate_simulator_fallback_used"
                    )
                )
                for signal in signals
            ),
            "degenerate_rollouts": sum(
                bool((signal.get("calibration") or {}).get("degenerate"))
                for signal in signals
            ),
        }
        if (
            config.byo_trainer_command.strip()
            and calibration["simulator_grounded_steps"] > 0
            and calibration["nonzero_advantage_count"] == 0
        ):
            raise Sim2RealLoopError(
                "simulator-grounded temporal credit is degenerate: every "
                "advantage is zero; refusing to train on useless feedback"
            )
        calibration_trend.append(calibration)
        _write_json_artifact(
            signal_batch_path,
            {
                "schema": SCHEMA_RL_SIGNAL,
                "signals": signals,
                "calibration": calibration,
            },
        )
        parse_vlm_signal_batch, run_vlm_signal_training_step = (
            _signal_training_imports()
        )
        parsed_signals = parse_vlm_signal_batch({"signals": signals})
        trainer_dir = (
            local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / "trainer"
            / f"iter-{iteration:02d}"
        )
        training_input = {
            **stage09_input,
            "stage09_payload_digest": canonical_digest(stage09),
            "initial_reward_head": reward_head,
            "initial_action_bias": action_bias,
            "resume_checkpoint_uri": current_checkpoint_uri,
            "trainer_command": config.byo_trainer_command,
            "trainer_image": config.trainer_image,
            "isaac_image": config.isaac_image,
            "learning_rate": config.learning_rate,
            "signal_loss_weight": config.signal_loss_weight,
        }
        training_unit = durable.load_unit(
            f"inner-o{outer_iteration:02d}-i{iteration:02d}-stage09-ppo",
            training_input,
        )
        if training_unit is not None:
            from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

            update = VlmSignalUpdateResult.from_dict(training_unit["update"])
            control = VlmSignalUpdateResult.from_dict(training_unit["control"])
            trainer_source = str(training_unit["trainer_source"])
            trainer_component_invocation = dict(
                training_unit.get("trainer_component_invocation") or {}
            )
            current_checkpoint_uri = str(
                training_unit.get("produced_checkpoint_uri") or current_checkpoint_uri
            )
        else:
            if config.byo_trainer_command.strip():
                update = _run_trainer_via_command(
                    signal_batch_path,
                    config=config,
                    output_dir=trainer_dir,
                    initial_reward_head=reward_head,
                    initial_action_bias=action_bias,
                    train_envs_dir=local_dir / "envs" / "train",
                    resume_checkpoint_uri=current_checkpoint_uri,
                    outer_iteration=outer_iteration,
                    iteration=iteration,
                )
                if str(getattr(update, "checkpoint_path", "") or "").strip():
                    current_checkpoint_uri = update.checkpoint_path.strip()
                trainer_source = "byo_command"
                trainer_provenance_path = (
                    trainer_dir / "byo-trainer-gpu-provenance.json"
                )
                trainer_component_invocation = (
                    json.loads(trainer_provenance_path.read_text(encoding="utf-8"))
                    if trainer_provenance_path.is_file()
                    else {}
                )
            else:
                update = run_vlm_signal_training_step(
                    parsed_signals,
                    output_dir=trainer_dir,
                    learning_rate=config.learning_rate,
                    signal_loss_weight=config.signal_loss_weight,
                    initial_reward_head=reward_head,
                    initial_action_bias=action_bias,
                )
                trainer_source = "reference"
                trainer_component_invocation = {}
            # The no-signal control is durable with the real update so a restart
            # cannot reapply either side of the attribution comparison.
            control = run_vlm_signal_training_step(
                parsed_signals,
                output_dir=local_dir
                / "inner_loop"
                / f"outer-{outer_iteration:02d}"
                / "control"
                / f"iter-{iteration:02d}",
                learning_rate=config.learning_rate,
                signal_loss_weight=config.signal_loss_weight,
                initial_reward_head=reward_head,
                initial_action_bias=action_bias,
                control=True,
            )
            durable.commit_unit(
                f"inner-o{outer_iteration:02d}-i{iteration:02d}-stage09-ppo",
                training_input,
                {
                    "update": update.to_dict(),
                    "control": control.to_dict(),
                    "trainer_source": trainer_source,
                    "trainer_component_invocation": trainer_component_invocation,
                    "produced_checkpoint_uri": current_checkpoint_uri,
                },
            )

        validation_report = None
        validation_reports: list[dict[str, Any]] = []
        if config.byo_trainer_command.strip() and config.s3_bucket.strip():
            periodic = list(update.periodic_checkpoints) or [
                {
                    "training_iteration": int(
                        (update.ppo or {}).get("iterations") or 0
                    ),
                    "checkpoint_uri": current_checkpoint_uri,
                }
            ]
            for periodic_checkpoint in periodic:
                candidate_uri = str(periodic_checkpoint.get("checkpoint_uri") or "")
                training_iteration = int(
                    periodic_checkpoint.get("training_iteration") or 0
                )
                checkpoint_evidence = {
                    "schema": "npa.sim2real.checkpoint_evidence.v1",
                    "iterations": [{"update": update.to_dict()}],
                    "selected_checkpoint_uri": candidate_uri,
                    "final_checkpoint_uri": candidate_uri,
                }
                validation_input = {
                    "outer_iteration": outer_iteration,
                    "inner_iteration": iteration,
                    "training_iteration": training_iteration,
                    "checkpoint_uri": candidate_uri,
                    "checkpoint_evidence_digest": canonical_digest(checkpoint_evidence),
                    "validation_envs_uri": config.validation_envs_uri,
                    "eval_image": config.eval_image,
                    "isaac_image": config.isaac_image,
                }
                validation_unit = durable.load_unit(
                    f"inner-o{outer_iteration:02d}-i{iteration:02d}-validation-{training_iteration:04d}",
                    validation_input,
                )
                if validation_unit is not None:
                    report = dict(validation_unit["report"])
                else:
                    report = run_heldout_eval(
                        config,
                        local_dir=local_dir,
                        inner_evidence=checkpoint_evidence,
                        outer_iteration=outer_iteration,
                        evaluation_split="validation",
                        inner_iteration=iteration,
                        checkpoint_iteration=training_iteration,
                    )
                    durable.commit_unit(
                        f"inner-o{outer_iteration:02d}-i{iteration:02d}-validation-{training_iteration:04d}",
                        validation_input,
                        {"report": report},
                    )
                validation_reports.append(report)
                checkpoint_candidates.append(
                    {
                        "evaluation_split": "validation",
                        "outer_iteration": outer_iteration,
                        "inner_iteration": iteration,
                        "training_iteration": training_iteration,
                        "checkpoint_uri": candidate_uri,
                        "checkpoint_sha256": report.get("policy_checkpoint_sha256", ""),
                        "validation_report_uri": report["report_uri"],
                        "validation_report": report,
                    }
                )
            from npa.workflows.sim2real.checkpoint_selection import (
                select_best_checkpoint,
            )

            interim_selection = select_best_checkpoint(checkpoint_candidates)
            current_checkpoint_uri = str(interim_selection["checkpoint_uri"])
            validation_report = dict(interim_selection.get("validation_report") or {})
        reward_head = update.reward_head_after
        action_bias = (
            update.policy_output_after[0] if update.policy_output_after else action_bias
        )
        mean_reward = round(
            sum(_signal_mean_reward(signal) for signal in signals)
            / float(len(signals)),
            6,
        )
        reward_trend.append(mean_reward)
        loss_trend.append(
            {
                "before": round(float(update.loss_before), 8),
                "after": round(float(update.loss_after), 8),
            }
        )
        delta_vs_control = max(0.0, update.policy_delta_l2 - control.policy_delta_l2)
        policy_deltas.append(round(delta_vs_control, 8))
        iteration_record = {
            "iteration": iteration,
            "actions_dir": str(actions_dir),
            "vlm_eval_dir": str(eval_dir),
            "signal_dir": str(signal_dir),
            "signal_batch": str(signal_batch_path),
            "mean_reward": mean_reward,
            "effective_learning_rate": config.learning_rate,
            "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
            "trainer_source": trainer_source,
            "trainer_component_invocation": trainer_component_invocation,
            "signal_converter_source": signal_converter_source,
            "update": update.to_dict(),
            "no_signal_control": control.to_dict(),
            "policy_delta_vs_control": round(delta_vs_control, 8),
            "validation_report": validation_report,
            "periodic_validation_reports": validation_reports,
            "sample_vlm_eval": evals[0],
            "sample_signal": signals[0],
            "signal_calibration": calibration,
        }
        iteration_records.append(iteration_record)
        durable.commit_unit(
            f"inner-o{outer_iteration:02d}-i{iteration:02d}-complete",
            iteration_input,
            {
                "iteration_record": iteration_record,
                "signals": signals,
                "checkpoint_candidates": checkpoint_candidates[candidate_count_before:],
                "loss_trend_entry": loss_trend[-1],
                "quality": quality,
                "reward_head": reward_head,
                "action_bias": action_bias,
                "current_checkpoint_uri": current_checkpoint_uri,
            },
        )

    signal_diversity = _signal_diversity_report(all_signals)
    if signal_diversity["degenerate"] and _bool_value(
        os.environ.get("NPA_SIM2REAL_REQUIRE_SIGNAL_DIVERSITY", "0")
    ):
        raise Sim2RealLoopError(
            "VLM->RL signal is degenerate: "
            f"{signal_diversity['distinct_scores']} distinct score(s) and "
            f"{signal_diversity['distinct_mean_rewards']} distinct mean-reward(s) "
            f"across {signal_diversity['total_rollouts']} rollout(s) "
            f"(scores={signal_diversity['score_values']}). "
            "Unset NPA_SIM2REAL_REQUIRE_SIGNAL_DIVERSITY to downgrade this gate to a "
            "diagnostic."
        )
    checkpoint_selection: dict[str, Any] = {}
    selected_checkpoint_uri = current_checkpoint_uri
    selected_validation_report: dict[str, Any] | None = None
    if checkpoint_candidates:
        from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint

        checkpoint_selection = select_best_checkpoint(checkpoint_candidates)
        selected_checkpoint_uri = str(checkpoint_selection["checkpoint_uri"])
        selected_validation_report = dict(
            checkpoint_selection.get("validation_report") or {}
        )
        quality = float(selected_validation_report.get("success_rate") or 0.0)
        selection_path = (
            local_dir
            / "checkpoints"
            / "validation-selection"
            / f"outer-{outer_iteration:02d}.json"
        )
        _write_json_artifact(selection_path, checkpoint_selection)
        checkpoint_selection["selection_report_uri"] = str(selection_path)
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": outer_iteration,
        "status": "closed",
        "trainer_source": (
            "byo_command" if config.byo_trainer_command.strip() else "reference"
        ),
        "signal_converter_source": (
            "byo_command" if config.byo_signal_converter.strip() else "reference"
        ),
        "effective_learning_rate": config.learning_rate,
        "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
        "reward_trend": reward_trend,
        "loss_trend": loss_trend,
        "signal_diversity": signal_diversity,
        "signal_calibration": calibration_trend,
        "policy_delta_vs_no_signal_control": policy_deltas,
        "attribution": (
            "The reference update and no-signal control share initial adapter state. "
            "Only the VLM-derived rewards, advantages, and corrective targets produce the policy-output delta."
        ),
        "iterations": iteration_records,
        "selected_validation_strict_success": round(quality, 6),
        "efficacy_metric_definition": (
            "strict stable-placement success rate on the fixed validation split; "
            "never a synthetic training-progress uplift"
            if checkpoint_candidates
            else "reference-only compatibility metric; not real task efficacy"
        ),
        "final_quality": round(quality, 6),
        "latest_checkpoint_uri": current_checkpoint_uri,
        "selected_checkpoint_uri": selected_checkpoint_uri,
        "final_checkpoint_uri": selected_checkpoint_uri,
        "checkpoint_selection": checkpoint_selection,
        "selected_validation_report": selected_validation_report,
        "resumed_from_checkpoint_uri": str(resume_checkpoint_uri or "").strip(),
    }
    evidence_path = (
        local_dir / "inner_loop" / f"outer-{outer_iteration:02d}" / "evidence.json"
    )
    _write_json_artifact(evidence_path, evidence)
    return {**evidence, "evidence_uri": str(evidence_path)}
