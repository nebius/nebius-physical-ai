"""Bind task-specific training subsets and weighted corpora to batch requests."""

from __future__ import annotations

import math

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from .contracts import digest


def select_training_data(request: dict, settings: dict) -> None:
    """Select training-only episodes without changing either evaluation holdout.

    Args:
        request: Mutable batch request whose training dataset will be narrowed.
        settings: Private batch settings with optional phase-specific selection.
    Returns:
        None.
    Raises:
        ValueError: A selection is invalid, empty, or has an uncovered corpus.
    """
    phase = request["stage"]
    selections = settings.get("training_selection", {})
    if phase not in {"pretrain", "finetune"} or not selections:
        return
    if request["partition"] != "train":
        raise ValueError("training selection cannot consume a holdout")
    policy = selections[phase]
    tasks = policy.get("task_ids", [])
    if not isinstance(tasks, list) or any(
        not isinstance(t, str) or not t for t in tasks
    ):
        raise ValueError("task_ids must be a list of nonempty strings")
    if phase == "finetune" and not tasks:
        raise ValueError("task-specific fine-tuning requires explicit task_ids")
    source = read_json_uri(request["dataset"]["uri"])
    if digest(source) != request["dataset"]["sha256"]:
        raise ValueError("training partition changed before selection")
    if set(tasks) - {e.get("task_id") for e in source["episodes"]}:
        raise ValueError("requested tasks are absent from the training partition")
    episodes = [e for e in source["episodes"] if not tasks or e.get("task_id") in tasks]
    if not episodes:
        raise ValueError("training selection contains no episodes")
    payload = _mixture(episodes, policy)
    uri = (
        request["result_uri"].rsplit("/", 1)[0]
        + f"/training-episodes-{digest(payload)}.json"
    )
    write_json_uri(uri, payload)
    request["dataset"] = {"uri": uri, "sha256": digest(payload)}
    request["training_selection"] = policy


def _mixture(episodes, policy):
    weights = policy.get("dataset_weights", {})
    sources = {e["dataset_uri"] for e in episodes}
    if set(weights) != sources:
        raise ValueError("dataset_weights must cover exactly the selected corpora")
    for weight in weights.values():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("mixture weights must be positive finite numbers")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("mixture weights must be positive finite numbers")
    total = sum(weights.values())
    if not math.isfinite(total):
        raise ValueError("mixture weight sum must be finite")
    return {
        "schema": "npa.policy.episodes.v1",
        "episodes": episodes,
        "mixture": [
            {"dataset_uri": uri, "probability": weight / total}
            for uri, weight in sorted(weights.items())
        ],
        "sampling_unit": "dataset-then-episode",
        "selection_sha256": digest(policy),
    }
