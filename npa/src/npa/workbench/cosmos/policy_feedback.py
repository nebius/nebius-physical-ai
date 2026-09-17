"""Turn measured LIBERO failures into a qualification decision and video targets.

Video candidates carry no action-label validity claim and never enter policy SFT.
"""

from __future__ import annotations

import math
from typing import Any

from npa.workbench.cosmos.policy_contract import EvalSettings, validate_summary
from npa.workbench.cosmos.policy_eval import EVAL_SCHEMA
from npa.workbench.dataset.storage import read_json_uri, uri_join, write_json_uri

FEEDBACK_SCHEMA = "npa.cosmos3.policy-feedback.v1"


def policy_feedback(*, input_path: str, output_path: str, minimum_success_rate: float = 0.9) -> dict[str, Any]:
    """Qualify a measured checkpoint and derive targets from actual failed tasks.

    Args:
        input_path: Completed native evaluation manifest.
        output_path: Local/S3 prefix for feedback.json.
        minimum_success_rate: Operator's absolute qualification threshold.
    Returns:
        Decision and failed-task prompts linked to evaluation evidence.
    Raises:
        ValueError: Invalid threshold, incomplete evaluation, or wrong schema.
    """
    if not math.isfinite(minimum_success_rate) or not 0 <= minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must be between zero and one")
    report = read_json_uri(input_path)
    if report.get("schema") != EVAL_SCHEMA or report.get("status") != "succeeded":
        raise ValueError("feedback requires completed native evaluation")
    settings = EvalSettings.model_validate(report["settings"])
    summary = report["summary"]
    validate_summary(summary, settings)
    complete_benchmark = sorted(settings.task_ids) == list(range(10)) and settings.trials_per_task == 50
    qualified = complete_benchmark and summary["overall_success_rate"] >= minimum_success_rate
    result = {"schema": FEEDBACK_SCHEMA, "status": "succeeded", "evaluation_manifest": input_path,
              "training_manifest": report["training_manifest"], "qualified": qualified,
              "decision": "qualified" if qualified else "needs_work",
              "minimum_success_rate": minimum_success_rate, "benchmark_complete": complete_benchmark,
              "success_rate": summary["overall_success_rate"], "improvement_proven": False,
              "selected_checkpoint": report["training_manifest"] if qualified else None,
              "targets": [_failure_target(task) for task in summary["task_results"]
                          if task["successes"] < task["episodes"]],
              "generated_video_training_eligible": False,
              "next_round_requires": "validated action-labeled demonstrations; video appearance is not an action label"}
    write_json_uri(uri_join(output_path, "feedback.json"), result)
    return result


def _failure_target(task: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task["task_id"], "task_description": task["task_description"],
            "successes": task["successes"], "episodes": task["episodes"],
            "failed_episodes": [result["episode"] for result in task["episode_results"] if not result["success"]],
            "prompt": "A fixed camera observes a robot arm performing this manipulation task: "
                      + task["task_description"] + ". Show the objects, contact, and complete intended motion clearly. "
                      "This is a visual candidate for review, with no validated action labels."}


def generate_failure_candidates(*, input_path: str, output_path: str, seed: int = 0) -> dict[str, Any]:
    """Generate guarded Cosmos videos targeted by measured policy task failures.

    Args:
        input_path: Completed policy feedback manifest.
        output_path: Local/S3 prefix for candidate MP4s and candidates.json.
        seed: Base seed, offset by task ID.
    Returns:
        Native generation manifests or an explicit empty result when no task failed.
    Raises:
        ValueError: Input is not valid policy feedback.
        Cosmos3GenerateError: Real inference or guardrails failed.
    """
    from npa.workbench.cosmos.generate import generate_and_publish

    feedback = read_json_uri(input_path)
    if feedback.get("schema") != FEEDBACK_SCHEMA or feedback.get("status") != "succeeded":
        raise ValueError("candidate generation requires completed policy feedback")
    generated = []
    for target in feedback["targets"]:
        task_id = int(target["task_id"])
        result = generate_and_publish(mode="text2video", prompt=target["prompt"],
                    name=f"failed-task-{task_id}", checkpoint="Cosmos3-Nano", input_path="",
                    output_path=uri_join(output_path, f"task-{task_id}"), negative_prompt="",
                    seed=seed + task_id, num_steps=0, guidance=0.0, no_guardrails=False,
                    parallelism_preset="latency", run_id="", dry_run=False)
        generated.append({"task_id": task_id, "generation": result, "action_labels_validated": False})
    report = {"schema": "npa.cosmos3.failure-candidates.v1", "status": "succeeded",
              "feedback_manifest": input_path, "candidates": generated,
              "generation_needed": bool(feedback["targets"]), "training_eligible": False}
    write_json_uri(uri_join(output_path, "candidates.json"), report)
    return report
