"""Validation-only stable-placement gate used before a canonical full run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real import byo_isaac_eval as isaac_eval
from npa.workflows.sim2real.utils import _write_json_artifact


CANARY_SCHEMA = "npa.sim2real.placement_canary.v1"
STRICT_DISTANCE_M = 0.05


def assess_placement_report(
    report: dict[str, Any],
    *,
    checkpoint_uri: str,
    expected_scenarios: int,
) -> dict[str, Any]:
    """Fail closed unless validation shows at least one strict stable placement."""

    if report.get("evaluation_split") != "validation":
        raise ValueError("placement canary may consume only the validation split")
    if report.get("policy_checkpoint") != checkpoint_uri:
        raise ValueError("placement canary checkpoint lineage mismatch")
    inference = dict(report.get("policy_inference_provenance") or {})
    if (
        inference.get("checkpoint_uri") != checkpoint_uri
        or not inference.get("loaded_for_inference")
        or not inference.get("checkpoint_sha256")
    ):
        raise ValueError("placement canary lacks loaded checkpoint byte provenance")
    if (
        inference.get("policy_composition") != "learned_actor_only"
        or inference.get("actor_is_learned") is not True
        or inference.get("scripted_post_actor_controller") is not False
        or inference.get("post_actor_controller") is not None
    ):
        raise ValueError("placement canary requires learned-actor-only provenance")
    rows = list(report.get("per_env") or [])
    if len(rows) != expected_scenarios:
        raise ValueError(
            f"placement canary expected {expected_scenarios} scenarios, got {len(rows)}"
        )
    env_ids = [str(row.get("env_id") or "") for row in rows]
    digests = [
        str((row.get("details") or {}).get("scenario_config_digest") or "")
        for row in rows
    ]
    if not all(env_ids) or len(set(env_ids)) != len(rows):
        raise ValueError(
            "placement canary environment IDs must be non-empty and unique"
        )
    if not all(digests) or len(set(digests)) != len(rows):
        raise ValueError(
            "placement canary scenario digests must be non-empty and unique"
        )
    strict_rows = []
    for row in rows:
        details = dict(row.get("details") or {})
        distance = float(details.get("object_goal_distance_m", float("inf")))
        stable = bool(details.get("placement_stable", details.get("place", False)))
        if bool(row.get("success")) != (stable and distance < STRICT_DISTANCE_M):
            raise ValueError(
                "placement canary strict success semantics are inconsistent"
            )
        if stable and distance < STRICT_DISTANCE_M:
            strict_rows.append(row)
    stages = {
        name: sum(bool((row.get("details") or {}).get(name)) for row in rows)
        for name in ("reach", "contact", "stable_grasp", "lift", "place")
    }
    invocation = dict(report.get("component_invocation") or {})
    provenance = dict(invocation.get("gpu_provenance") or {})
    if not provenance.get("image_digests"):
        raise ValueError("placement canary lacks immutable runtime image provenance")
    scenario_input = dict(report.get("scenario_input_provenance") or {})
    scenario_digest = str(scenario_input.get("sha256") or "")
    if (
        scenario_input.get("transport") != "s3_sha256"
        or not scenario_input.get("content_addressed")
        or int(scenario_input.get("scenario_count") or 0) != expected_scenarios
        or int(scenario_input.get("size_bytes") or 0) <= 0
        or not str(scenario_input.get("uri") or "").endswith(
            f"/{scenario_digest}.jsonl"
        )
        or len(scenario_digest) != 64
    ):
        raise ValueError(
            "placement canary lacks content-addressed scenario input provenance"
        )
    return {
        "schema": CANARY_SCHEMA,
        "evaluation_split": "validation",
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": inference["checkpoint_sha256"],
        "policy_inference_provenance": inference,
        "scenario_count": len(rows),
        "scenario_config_digests": digests,
        "strict_distance_m": STRICT_DISTANCE_M,
        "strict_stable_placements": len(strict_rows),
        "strict_success_rate": len(strict_rows) / len(rows),
        "decomposed_success_counts": stages,
        "credible_placement_signal": bool(strict_rows),
        "gpu_provenance": provenance,
        "scenario_input_provenance": scenario_input,
    }


def _validation_rows(
    *, validation_envs_uri: str, gold_envs_uri: str, count: int, endpoint: str
) -> list[dict[str, Any]]:
    if not validation_envs_uri.startswith("s3://"):
        raise ValueError("validation canary requires an exact s3:// validation object")
    if not gold_envs_uri.startswith("s3://"):
        raise ValueError("gold URI is required to prove split separation")
    if validation_envs_uri.rstrip("/") == gold_envs_uri.rstrip("/"):
        raise ValueError("validation and gold scenario objects must be distinct")
    local = Path(
        os.environ.get(
            "NPA_SIM2REAL_CANARY_INPUT", "/tmp/npa-placement-canary-envs.jsonl"
        )
    )
    StorageClient.from_environment(endpoint_url=endpoint).download_path(
        validation_envs_uri, str(local)
    )
    rows = [
        json.loads(line)
        for line in local.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return isaac_eval.select_stratified_eval_envs(rows, count=count, split="validation")


def run_validation_canary(
    *,
    run_id: str,
    checkpoint_uri: str,
    validation_envs_uri: str,
    gold_envs_uri: str,
    output_json: Path,
    scenario_count: int,
    output_uri: str = "",
) -> dict[str, Any]:
    """Run the ordinary digest-pinned Isaac evaluator on validation only."""

    if not checkpoint_uri.startswith("s3://"):
        raise ValueError("placement canary requires an exact s3:// checkpoint")
    if scenario_count < 1:
        raise ValueError("placement canary scenario count must be positive")
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "")
    rows = _validation_rows(
        validation_envs_uri=validation_envs_uri,
        gold_envs_uri=gold_envs_uri,
        count=scenario_count,
        endpoint=endpoint,
    )
    os.environ["NPA_SIM2REAL_EVAL_TAG"] = "placement-canary-validation"
    os.environ["NPA_SIM2REAL_EVALUATION_SPLIT"] = "validation"
    isaac_eval._LAST_GPU_PROVENANCE = {}
    isaac_eval._CHECKPOINT_PROVENANCE = {}
    isaac_eval._APPLIED_SCENARIO_AUDIT = {}
    isaac_eval._RENDER_MANIFEST = {}
    isaac_eval._SCENARIO_INPUT_PROVENANCE = {}
    isaac_eval._RENDERS_LOCAL_DIR = str(output_json.parent / "renders")
    per_env = isaac_eval.run_isaac_eval_job(
        run_id,
        checkpoint_uri=checkpoint_uri,
        num_envs=scenario_count,
        generated_envs=rows,
    )
    report = isaac_eval.build_heldout_report(
        per_env,
        isaac_task=os.environ.get(
            "NPA_SIM2REAL_ISAAC_TASK", isaac_eval.DEFAULT_ISAAC_TASK
        ),
        checkpoint_uri=checkpoint_uri,
        source="validation_placement_canary",
    )
    report.update(
        {
            "evaluation_split": "validation",
            "gold_heldout_untouched": True,
            "validation_envs_uri": validation_envs_uri,
            "gold_envs_uri": gold_envs_uri,
            "generated_envs_tested": len(rows),
            "generated_env_ids": [row["env_id"] for row in rows],
            "component_invocation": {
                "mode": "kubernetes_job",
                "gpu_provenance": isaac_eval._LAST_GPU_PROVENANCE,
            },
            "policy_checkpoint_sha256": isaac_eval._CHECKPOINT_PROVENANCE.get(
                "sha256", ""
            ),
            "policy_checkpoint_size_bytes": isaac_eval._CHECKPOINT_PROVENANCE.get(
                "size_bytes", 0
            ),
            "policy_inference_provenance": (
                isaac_eval.policy_inference_provenance(
                    checkpoint_uri=checkpoint_uri,
                    checkpoint=isaac_eval._CHECKPOINT_PROVENANCE,
                )
            ),
            "applied_scenario_proof": isaac_eval._APPLIED_SCENARIO_AUDIT,
            "scenario_input_provenance": isaac_eval._SCENARIO_INPUT_PROVENANCE,
        }
    )
    if isaac_eval._RENDER_MANIFEST.get("episodes"):
        report["render_manifest"] = isaac_eval._RENDER_MANIFEST
    assessment = assess_placement_report(
        report,
        checkpoint_uri=checkpoint_uri,
        expected_scenarios=scenario_count,
    )
    payload = {"report": report, "assessment": assessment}
    _write_json_artifact(output_json, payload)
    if output_uri:
        StorageClient.from_environment(endpoint_url=endpoint).upload_file(
            str(output_json), output_uri
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--validation-envs-uri", required=True)
    parser.add_argument("--gold-envs-uri", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scenario-count", type=int, default=16)
    parser.add_argument("--output-uri", default="")
    args = parser.parse_args(argv)
    payload = run_validation_canary(
        run_id=args.run_id,
        checkpoint_uri=args.checkpoint_uri,
        validation_envs_uri=args.validation_envs_uri,
        gold_envs_uri=args.gold_envs_uri,
        output_json=args.output_json,
        scenario_count=args.scenario_count,
        output_uri=args.output_uri,
    )
    print(json.dumps(payload["assessment"], indent=2, sort_keys=True))
    return 0 if payload["assessment"]["credible_placement_signal"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
