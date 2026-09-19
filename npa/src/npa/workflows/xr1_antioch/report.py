"""Compare every paired held-out robot rollout without substituting training loss for task success."""

from __future__ import annotations

import math


def _cohort(rows: list[dict], seeds: list[int], checkpoint: str) -> dict:
    if len(rows) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation must contain every sealed held-out seed exactly once")
    indexed = {}
    for row in rows:
        if row["seed"] not in seeds or row["seed"] in indexed:
            raise ValueError("Evaluation contains an unexpected or duplicate seed")
        if row["checkpoint_sha256"] != checkpoint or row.get("expert_fallback") is not False:
            raise ValueError("Evaluation used a different checkpoint or an expert fallback")
        if row.get("grasp_mechanism") != "finger_contact":
            raise ValueError("Evaluation did not use the physical contact task")
        if row.get("control_hz") != 20 or row.get("replan_every_frames") != 6:
            raise ValueError("Evaluation changed the sealed policy control contract")
        if row.get("horizon_seconds") != 25 or not isinstance(row.get("success"), bool):
            raise ValueError("Evaluation lacks its complete task outcome")
        indexed[row["seed"]] = row
    return indexed


def _wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - margin), min(1., center + margin)]


def _contract(baseline: list[dict], candidate: list[dict]) -> dict:
    contract = {}
    for key in ("statistics_sha256", "simulation_sha256"):
        identities = {row.get(key) for row in [*baseline, *candidate]}
        if len(identities) != 1:
            raise ValueError("Baseline and candidate must share verified normalization and simulation bytes")
        identity = next(iter(identities))
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError("Normalization and simulation identities must be SHA-256 digests")
        contract[key] = identity
    return contract


def compare_rollouts(baseline: list[dict], candidate: list[dict], seeds: list[int],
                     baseline_sha256: str, candidate_sha256: str) -> dict:
    """Classify paired task outcomes with exact disagreement statistics.

    Args:
        baseline: All baseline rollouts, including unsuccessful episodes.
        candidate: All candidate rollouts on identical held-out seeds.
        seeds: Sealed test seeds unused in training and checkpoint selection.
        baseline_sha256: Verified base checkpoint identity.
        candidate_sha256: Verified trained checkpoint identity.
    Returns:
        Per-seed outcomes, success rates, Wilson intervals, and exact paired test.
    Raises:
        ValueError: A cohort is incomplete, leaks identity, or changes the evaluation contract.
    """
    if not seeds or baseline_sha256 == candidate_sha256:
        raise ValueError("Comparison requires held-out trials and distinct checkpoint identities")
    before = _cohort(baseline, seeds, baseline_sha256)
    after = _cohort(candidate, seeds, candidate_sha256)
    contract = _contract(baseline, candidate)
    pairs = [{"seed": seed, "baseline": before[seed]["success"], "candidate": after[seed]["success"]}
             for seed in seeds]
    wins = sum(row["candidate"] and not row["baseline"] for row in pairs)
    losses = sum(row["baseline"] and not row["candidate"] for row in pairs)
    disagreements = wins + losses
    probability = min(1., 2 * sum(math.comb(disagreements, k) for k in range(min(wins, losses) + 1))
                      / 2 ** disagreements) if disagreements else 1.
    rates = {}
    for name in ("baseline", "candidate"):
        count = sum(row[name] for row in pairs)
        rates[name] = {"successes": count, "trials": len(seeds), "success_rate": count / len(seeds),
                       "wilson_95_interval": _wilson(count, len(seeds))}
    return {"schema": "npa.xr1-antioch.comparison.v1", "pairs": pairs, **rates, **contract,
            "baseline_sha256": baseline_sha256, "candidate_sha256": candidate_sha256,
            "paired_wins": wins, "paired_losses": losses, "mcnemar_exact_two_sided_p": probability,
            "observed_improvement": wins > losses, "supported_improvement": wins > losses and probability < .05,
            "scope": "dual-Franka simulated pick-and-place; no real-robot validation"}
