"""Independent, review-visible policy ratchet for the agent task evaluation."""

from __future__ import annotations

from typing import Any, Mapping

POLICY_SCHEMA = "npa.agent_eval.policy.v1"
SCENARIO_COUNT = 10
SCENARIO_IDS = (
    "grounded_status",
    "grounded_tools_catalog",
    "grounded_cosmos_caps",
    "workflow_vlm_rl_draft",
    "action_status_then_answer",
    "action_gpu_needs_confirmation",
    "sim2real_promote",
    "sim2real_needs_confirmation",
    "semantic_watch_paraphrase",
    "retrieval_genesis_doc",
)
SCENARIO_SHA256 = "a9d3f7ee6b2763d21e855763192486409dd9a81c42ca67af6698966a00103e55"

# These limits are deliberately independent of the generated scorecard artifact.
# Changing the artifact alone can never lower the defended quality/efficiency bar.
MIN_SUCCESS_RATE = 1.0
MAX_AVG_STEPS = 1.1
MAX_AVG_TOKENS = 1.8


def scorecard_policy_violations(
    scorecard: Mapping[str, Any], *, role: str
) -> list[str]:
    """Return policy violations for a current or committed-baseline scorecard."""
    violations: list[str] = []
    prefix = f"{role} "
    if int(scorecard.get("total", -1)) != SCENARIO_COUNT:
        violations.append(
            f"{prefix}total={scorecard.get('total')} does not equal policy={SCENARIO_COUNT}"
        )
    if int(scorecard.get("scenario_count", -1)) != SCENARIO_COUNT:
        violations.append(
            f"{prefix}scenario_count={scorecard.get('scenario_count')} does not equal "
            f"policy={SCENARIO_COUNT}"
        )
    if tuple(scorecard.get("scenario_ids") or ()) != SCENARIO_IDS:
        violations.append(f"{prefix}scenario_ids do not match policy")
    if str(scorecard.get("scenario_sha256") or "") != SCENARIO_SHA256:
        violations.append(f"{prefix}scenario_sha256 does not match policy")
    if float(scorecard.get("success_rate", -1.0)) < MIN_SUCCESS_RATE:
        violations.append(
            f"{prefix}success_rate={scorecard.get('success_rate')} is below policy={MIN_SUCCESS_RATE}"
        )
    if float(scorecard.get("avg_steps", float("inf"))) > MAX_AVG_STEPS:
        violations.append(
            f"{prefix}avg_steps={scorecard.get('avg_steps')} is above policy={MAX_AVG_STEPS}"
        )
    if float(scorecard.get("avg_tokens", float("inf"))) > MAX_AVG_TOKENS:
        violations.append(
            f"{prefix}avg_tokens={scorecard.get('avg_tokens')} is above policy={MAX_AVG_TOKENS}"
        )
    return violations
