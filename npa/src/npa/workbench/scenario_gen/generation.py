"""Adversarial scenario generation on top of the workbench RL backend.

The adversary is an RL agent whose reward is the *failure* of a
policy-under-test: it perturbs the environment / other-agent behavior to drive
the policy into violations, surfacing hard scenarios for regression and
hardening. The real training pass runs on the Isaac Lab RL backend; the default
backend here is a deterministic simulated search so the tool is usable and
testable without a GPU. Swap the backend via ``adversary_backend`` to plug in a
live Isaac Lab adversary.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from typing import Any, Callable

from .schemas import (
    ADVERSARIAL_SET_SCHEMA,
    GenerateRequest,
    GenerateResponse,
    Lineage,
    ScenarioRecord,
)
from .storage import uri_join, write_json_uri

# Perturbation axes an adversary can drive to break a policy-under-test.
PERTURBATION_AXES: tuple[str, ...] = (
    "friction",
    "mass_scale",
    "actuator_noise",
    "obstacle_density",
    "adversary_aggression",
    "sensor_latency",
)

AdversaryBackend = Callable[["GenerateRequest", int], list[dict[str, Any]]]


class ScenarioGenError(RuntimeError):
    """Raised when adversarial scenario generation fails."""


def compute_manifest_sha256(kind: str, payload: dict[str, Any]) -> str:
    """Compute a deterministic manifest hash for inputs and hyperparameters."""
    digest = hashlib.sha256()
    digest.update(kind.encode("utf-8"))
    digest.update(b"\n")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    digest.update(b"\n")
    return digest.hexdigest()


def make_run_id(prefix: str, manifest_sha256: str) -> str:
    """Create a reproducible-looking but unique run id."""
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{manifest_sha256[:12]}"


def manifest_uri(output_uri: str, run_id: str) -> str:
    return uri_join(output_uri, run_id, "manifest.json")


def scenario_config_uri(output_uri: str, run_id: str, scenario_id: str) -> str:
    return uri_join(output_uri, run_id, "scenarios", f"{scenario_id}.json")


def simulate_adversary(request: GenerateRequest, seed: int) -> list[dict[str, Any]]:
    """Deterministic stand-in for an Isaac Lab adversarial RL rollout.

    Samples perturbation vectors and scores each by a heuristic failure model
    that rewards larger, correlated perturbations (an adversary maximizing the
    policy-under-test's violation rate). This keeps generation dependency-light
    and reproducible; a live backend replaces it via ``adversary_backend``.
    """

    rng = random.Random(seed)
    # More adversary training budget lets it find slightly harder scenarios.
    budget_gain = min(0.25, math.log10(max(request.adversary_steps, 10)) / 40.0)
    candidates: list[dict[str, Any]] = []
    for index in range(request.num_scenarios):
        perturbation = {axis: round(rng.uniform(0.0, 1.0), 4) for axis in PERTURBATION_AXES}
        magnitude = sum(perturbation.values()) / len(perturbation)
        # Heuristic: bigger correlated perturbations => higher predicted failure.
        failure_score = min(1.0, round(magnitude * (0.75 + budget_gain) + rng.uniform(0.0, 0.1), 4))
        candidates.append(
            {
                "scenario_id": f"adv-{index:04d}",
                "seed": seed + index,
                "perturbation": perturbation,
                "failure_score": failure_score,
                "metrics": {
                    "predicted_violation_rate": failure_score,
                    "predicted_collision_rate": round(min(1.0, failure_score * 0.6), 4),
                },
            }
        )
    return candidates


def _diversity_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    """Mean pairwise perturbation distance, normalized to [0, 1]."""
    axes = sorted({axis for record in records for axis in record.get("perturbation", {})})
    norm = math.sqrt(len(axes)) or 1.0
    diversity: dict[str, float] = {}
    for record in records:
        vector = record.get("perturbation", {})
        distances = []
        for other in records:
            if other is record:
                continue
            other_vector = other.get("perturbation", {})
            distance = math.sqrt(
                sum((vector.get(axis, 0.0) - other_vector.get(axis, 0.0)) ** 2 for axis in axes)
            )
            distances.append(distance)
        raw = sum(distances) / len(distances) if distances else 0.0
        # Max distance across the unit hypercube is sqrt(len(axes)).
        diversity[record["scenario_id"]] = round(min(1.0, raw / norm), 4)
    return diversity


def generate_scenarios(
    request: GenerateRequest,
    *,
    run_id: str | None = None,
    adversary_backend: AdversaryBackend | None = None,
) -> GenerateResponse:
    """Train an adversary, mine ranked adversarial scenarios, emit a manifest."""
    manifest = compute_manifest_sha256("generate", request.model_dump(mode="json"))
    resolved_run_id = run_id or make_run_id("scenario-gen", manifest)
    backend = adversary_backend or simulate_adversary

    try:
        raw = backend(request, request.seed)
    except ScenarioGenError:
        raise
    except Exception as exc:  # pragma: no cover - defensive backend boundary.
        raise ScenarioGenError(f"adversary backend failed: {exc}") from exc
    if not raw:
        raise ScenarioGenError("adversary backend produced no scenarios")

    diversity = _diversity_scores(raw)
    records: list[ScenarioRecord] = []
    for candidate in raw:
        scenario_id = str(candidate["scenario_id"])
        failure_score = float(candidate.get("failure_score", 0.0))
        record = ScenarioRecord(
            scenario_id=scenario_id,
            seed=int(candidate.get("seed", 0)),
            perturbation={k: float(v) for k, v in candidate.get("perturbation", {}).items()},
            failure_score=failure_score,
            severity=failure_score,
            diversity=diversity.get(scenario_id, 0.0),
            metrics={k: float(v) for k, v in candidate.get("metrics", {}).items()},
            config_uri=scenario_config_uri(request.output_uri, resolved_run_id, scenario_id),
        )
        records.append(record)

    records.sort(key=lambda item: item.severity, reverse=True)

    lineage = Lineage(
        workflow_run=request.workflow_run,
        input_uris=[request.policy_uri, request.base_config_uri],
        policy_uri=request.policy_uri,
        base_config_uri=request.base_config_uri,
        task_name=request.task_name,
    )
    target_manifest_uri = manifest_uri(request.output_uri, resolved_run_id)
    manifest_payload = {
        "schema": ADVERSARIAL_SET_SCHEMA,
        "run_id": resolved_run_id,
        "manifest_sha256": manifest,
        "task_name": request.task_name,
        "adversary_steps": request.adversary_steps,
        "scenario_count": len(records),
        "lineage": lineage.model_dump(mode="json"),
        "scenarios": [record.model_dump(mode="json") for record in records],
    }

    _emit_artifacts(request, resolved_run_id, records, manifest_payload, target_manifest_uri)

    top_severity = records[0].severity if records else 0.0
    return GenerateResponse(
        run_id=resolved_run_id,
        status="completed",
        manifest_uri=target_manifest_uri,
        scenario_count=len(records),
        top_severity=top_severity,
        manifest_sha256=manifest,
        lineage=lineage,
    )


def _emit_artifacts(
    request: GenerateRequest,
    run_id: str,
    records: list[ScenarioRecord],
    manifest_payload: dict[str, Any],
    target_manifest_uri: str,
) -> None:
    for record in records:
        write_json_uri(
            record.config_uri,
            {
                "schema": ADVERSARIAL_SET_SCHEMA,
                "run_id": run_id,
                "task_name": request.task_name,
                "scenario_id": record.scenario_id,
                "seed": record.seed,
                "perturbation": record.perturbation,
                "metrics": record.metrics,
            },
        )
    write_json_uri(target_manifest_uri, manifest_payload)
