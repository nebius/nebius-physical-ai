"""Report enrichment helpers for Sim2Real operational progress."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json_link(local_dir: Path, value: Any) -> dict[str, Any]:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = local_dir / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_progress_metrics(
    local_dir: Path, outer_history: list[dict[str, Any]]
) -> dict[str, Any]:
    """Embed reward/loss/evaluation series from every persisted outer pass."""

    outer_metrics: list[dict[str, Any]] = []
    for position, history in enumerate(outer_history, start=1):
        evidence = _load_json_link(local_dir, history.get("inner_loop"))
        decision = dict(history.get("decision") or {})
        iterations: list[dict[str, Any]] = []
        for iteration in evidence.get("iterations") or []:
            if not isinstance(iteration, dict):
                continue
            update = dict(iteration.get("update") or {})
            iterations.append(
                {
                    "iteration": iteration.get("iteration"),
                    "mean_reward": iteration.get("mean_reward"),
                    "loss_before": update.get("loss_before"),
                    "loss_after": update.get("loss_after"),
                    "training_proxy_score": iteration.get(
                        "training_proxy_score",
                        iteration.get("next_rollout_training_proxy"),
                    ),
                    "checkpoint_uri": update.get("checkpoint_path", ""),
                    "effective_learning_rate": iteration.get("effective_learning_rate"),
                    "learning_rate_scope": iteration.get("learning_rate_scope", ""),
                }
            )
        loss_trend = list(evidence.get("loss_trend") or [])
        if not loss_trend:
            loss_trend = [
                {"before": item["loss_before"], "after": item["loss_after"]}
                for item in iterations
                if item.get("loss_before") is not None
                and item.get("loss_after") is not None
            ]
        outer_metrics.append(
            {
                "outer_iteration": int(history.get("outer_iteration") or position),
                "inner_iteration_count": len(iterations),
                "iterations": iterations,
                "reward_trend": list(evidence.get("reward_trend") or []),
                "loss_trend": loss_trend,
                "selected_validation_strict_success": evidence.get(
                    "selected_validation_strict_success",
                    evidence.get("final_quality"),
                ),
                "efficacy_metric_definition": evidence.get(
                    "efficacy_metric_definition",
                    "legacy evidence; inspect component_source before treating as efficacy",
                ),
                "evaluation_success_rate": decision.get("success_rate"),
                "evaluation_threshold": decision.get("threshold"),
                "decision": decision.get("decision", ""),
                "checkpoint_uri": history.get("checkpoint_uri", ""),
                "inner_evidence": history.get("inner_loop", ""),
            }
        )
    return {
        "outer_iteration_count": len(outer_metrics),
        "outer_iterations": outer_metrics,
        "stage_12_external_stub": {
            "tier": "SEAM",
            "status": "designed_external_byo_gate_not_dispatched",
        },
    }
