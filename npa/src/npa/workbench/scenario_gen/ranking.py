"""Rank generated adversarial scenarios by failure severity and diversity."""

from __future__ import annotations

from typing import Any

from .generation import compute_manifest_sha256, make_run_id
from .schemas import RANKED_SET_SCHEMA, RankRequest, RankResponse
from .storage import read_json_uri, uri_join, write_json_uri


class ScenarioRankError(RuntimeError):
    """Raised when ranking an adversarial scenario set fails."""


def ranked_manifest_uri(output_uri: str, run_id: str) -> str:
    return uri_join(output_uri, run_id, "ranked.json")


def rank_scenarios(request: RankRequest, *, run_id: str | None = None) -> RankResponse:
    """Score/rank scenarios by weighted failure severity + diversity."""
    manifest = compute_manifest_sha256("rank", request.model_dump(mode="json"))
    resolved_run_id = run_id or make_run_id("scenario-rank", manifest)

    try:
        source = read_json_uri(request.input_uri)
    except FileNotFoundError as exc:
        raise ScenarioRankError(f"adversarial set not found: {request.input_uri}") from exc
    except Exception as exc:
        raise ScenarioRankError(f"cannot read adversarial set {request.input_uri}: {exc}") from exc

    scenarios = source.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ScenarioRankError("adversarial set manifest has no scenarios")

    weight_sum = request.severity_weight + request.diversity_weight
    if weight_sum <= 0.0:
        raise ScenarioRankError("severity_weight + diversity_weight must be > 0")

    scored: list[dict[str, Any]] = []
    for scenario in scenarios:
        severity = float(scenario.get("severity", scenario.get("failure_score", 0.0)))
        diversity = float(scenario.get("diversity", 0.0))
        combined = (request.severity_weight * severity + request.diversity_weight * diversity) / weight_sum
        scored.append(
            {
                "scenario_id": str(scenario.get("scenario_id", "")),
                "severity": round(severity, 4),
                "diversity": round(diversity, 4),
                "rank_score": round(combined, 4),
                "config_uri": str(scenario.get("config_uri", "")),
            }
        )

    scored.sort(key=lambda item: (item["rank_score"], item["severity"]), reverse=True)
    top = scored[: request.top_k]

    target_uri = ranked_manifest_uri(request.output_uri, resolved_run_id)
    ranked_payload = {
        "schema": RANKED_SET_SCHEMA,
        "run_id": resolved_run_id,
        "manifest_sha256": manifest,
        "source_uri": request.input_uri,
        "source_schema": source.get("schema", ""),
        "lineage": {
            "workflow_run": request.workflow_run,
            "input_uris": [request.input_uri],
            "produced_by": "workbench.scenario_gen.rank",
        },
        "severity_weight": request.severity_weight,
        "diversity_weight": request.diversity_weight,
        "ranked_count": len(top),
        "scenarios": top,
    }
    write_json_uri(target_uri, ranked_payload)

    return RankResponse(
        run_id=resolved_run_id,
        status="completed",
        ranked_manifest_uri=target_uri,
        ranked_count=len(top),
        top_scenarios=top,
        manifest_sha256=manifest,
    )
