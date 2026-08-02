"""Where a detection-training eval result lands, and how a checkpoint is discovered.

Both behaviours were implemented in the retired `bdd100k-pipeline.yaml`'s eval task as bash
plus `jq`, which means no `npa.workflow` spec could reach them:

* the task GET ``/runs``, picked the **last completed** training run for its view, and
  substituted ``{epoch}`` in that run's ``checkpoint_uri_pattern`` — so eval scored the
  checkpoint training actually produced, rather than a directory guessed by the spec;
* with ``WRITE_CANONICAL_EVAL_METRICS=1`` it uploaded the eval response to
  ``<output_uri>/metrics.json``. The BDD100K spec *declares* that artifact, so without this
  the stage would succeed and the declared file would never exist.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: The canonical metrics object an eval stage publishes, and the name the specs declare.
EVAL_METRICS_FILENAME = "metrics.json"

#: Fields the service must return as numbers. The template asserted exactly these with
#: `jq -e`, because a service can answer 200 with a null or a string and a stage would
#: otherwise report success on an unusable report.
REQUIRED_EVAL_METRICS = ("mAP", "mAP_50", "mAP_75")

#: Placeholder the service leaves in `checkpoint_uri_pattern` for the epoch number.
EPOCH_PLACEHOLDER = "{epoch}"

#: The status `/runs` reports for a finished run.
COMPLETED_STATUS = "completed"


class DetectionTrainingArtifactError(ValueError):
    """Raised when a run cannot be resolved or an eval response is unusable."""


def eval_result_uri_for(output_uri: str) -> str:
    """Return the canonical metrics URI for an eval output prefix."""

    if output_uri.endswith(".json"):
        return output_uri
    return output_uri.rstrip("/") + f"/{EVAL_METRICS_FILENAME}"


def discover_checkpoint_uri(runs: Iterable[Mapping[str, Any]], *, output_uri: str) -> str:
    """Resolve the checkpoint the last completed run under ``output_uri`` produced.

    ``runs`` is the ``runs`` list from ``GET /runs``. The retired template matched on
    ``checkpoint_uri_pattern | contains("/training/<view-slug>/")``; matching the training
    **output prefix** instead is the same intent and strictly narrower, because that prefix
    is exactly what the spec handed ``/train`` as ``output_uri``.

    ``last`` wins, as in the template: re-running training for a view supersedes the earlier
    checkpoint.
    """

    prefix = output_uri.rstrip("/") + "/"
    matches = [
        run
        for run in runs
        if str(run.get("status") or "").strip().lower() == COMPLETED_STATUS
        and str(run.get("checkpoint_uri_pattern") or "").startswith(prefix)
    ]
    if not matches:
        raise DetectionTrainingArtifactError(
            f"no completed training run found under {output_uri!r}. Run training first, or "
            "pass --checkpoint-uri explicitly."
        )
    run = matches[-1]
    pattern = str(run.get("checkpoint_uri_pattern") or "")
    epochs = run.get("total_epochs")
    if epochs in (None, ""):
        raise DetectionTrainingArtifactError(
            f"completed run {run.get('run_id')!r} reports no total_epochs, so the epoch in "
            f"{pattern!r} cannot be resolved"
        )
    if EPOCH_PLACEHOLDER not in pattern:
        # Already concrete; nothing to substitute.
        return pattern
    return pattern.replace(EPOCH_PLACEHOLDER, str(epochs))


def assert_eval_metrics(payload: Mapping[str, Any]) -> None:
    """Fail unless the eval response carries numeric mAP fields."""

    bad: list[str] = []
    for field in REQUIRED_EVAL_METRICS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            bad.append(f"{field}={value!r}")
    if bad:
        raise DetectionTrainingArtifactError(
            "eval response did not return numeric metrics: " + ", ".join(bad)
        )


__all__ = [
    "COMPLETED_STATUS",
    "EPOCH_PLACEHOLDER",
    "EVAL_METRICS_FILENAME",
    "REQUIRED_EVAL_METRICS",
    "DetectionTrainingArtifactError",
    "assert_eval_metrics",
    "discover_checkpoint_uri",
    "eval_result_uri_for",
]
