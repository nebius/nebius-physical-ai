"""Gate each candidate on finite task-success measurements for its own holdout."""

from __future__ import annotations

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from .contracts import checkpoint, digest, probability, _validate_engine


def gate(
    evaluation_uri,
    candidate_uri,
    split_uri,
    policy_uri,
    phase,
    output_uri,
    *,
    _reference=False,
):
    """Require all configured evaluation systems to pass their thresholds.

    Args:
        evaluation_uri: Completed batch evaluation result.
        candidate_uri: Candidate training result.
        split_uri: Immutable partition index.
        policy_uri: Required systems and numeric thresholds by phase.
        phase: Pretrain or finetune.
        output_uri: Measured workflow decision location.
        _reference: Internal teaching-demo validation, unavailable in production CLI.
    Returns:
        None.
    Raises:
        ValueError: Measurements or provenance are missing or inconsistent.
    """
    evaluation = read_json_uri(evaluation_uri)
    trained = read_json_uri(candidate_uri)
    _validate_engine(trained, reference=_reference)
    _validate_engine(evaluation, reference=_reference)
    candidate = checkpoint(trained)
    partition = {"pretrain": "holdout_1", "finetune": "holdout_2"}[phase]
    expected = read_json_uri(split_uri)["partitions"][partition]
    _validate_evaluation(evaluation, candidate, expected, phase, partition)
    thresholds = read_json_uri(policy_uri)[phase]
    rates = _rates(evaluation["systems"], thresholds)
    _write_gate(evaluation, candidate, expected, phase, thresholds, rates, output_uri)


def _validate_evaluation(evaluation, candidate, expected, phase, partition):
    request = evaluation["request"]
    if evaluation.get("status") != "completed" or checkpoint(evaluation) != candidate:
        raise ValueError("evaluation is incomplete or used a different candidate")
    if (
        request.get("stage") != f"evaluate-{phase}"
        or request.get("partition") != partition
    ):
        raise ValueError("evaluation used the wrong stage or holdout")
    if request.get("checkpoint") != candidate or request.get("dataset") != expected:
        raise ValueError("evaluation provenance does not match the candidate and split")
    if evaluation.get("request_sha256") != digest(request):
        raise ValueError("evaluation request digest mismatch")


def _write_gate(evaluation, candidate, expected, phase, thresholds, rates, output_uri):
    passed = all(rate >= probability(thresholds[name]) for name, rate in rates.items())
    write_json_uri(
        output_uri,
        {
            "schema": "npa.policy.gate.v1",
            "engine": evaluation["engine"],
            "phase": phase,
            "decision": "promote_checkpoint" if passed else "loop_back",
            "checkpoint": candidate,
            "partition": expected,
            "success_rates": rates,
            "thresholds": thresholds,
            "evaluation_sha256": digest(evaluation),
        },
    )


def _rates(systems, thresholds):
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("gate requires explicit system thresholds")
    rates = {}
    for name in thresholds:
        probability(thresholds[name])
        report = systems[name]
        successes, trials = report["successes"], report["trials"]
        if any(
            isinstance(n, bool) or not isinstance(n, int) for n in (successes, trials)
        ):
            raise ValueError("success and trial counts must be integers")
        if trials <= 0 or not 0 <= successes <= trials:
            raise ValueError("task-success counts are invalid")
        rates[name] = successes / trials
    return rates
