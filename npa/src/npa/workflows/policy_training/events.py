"""Materialize training entry events without forcing a fresh pretraining run."""

from __future__ import annotations

from copy import deepcopy


_PRETRAIN = {"pretrain-loop", "pretrain", "evaluate-pretrain", "gate-pretrain"}
_FINETUNE = {
    "finetune-loop",
    "finetune",
    "evaluate-finetune",
    "gate-finetune",
    "test-policy",
}


def event_workflow(template: dict, event: dict) -> dict:
    """Bind one explicit event to the existing measured training workflow.

    Args:
        template: Parsed policy-training-slurm workflow specification.
        event: Event kind and private immutable input references.
    Returns:
        A separate workflow specification with the selected entry path.
    Raises:
        ValueError: The event kind or required input reference is invalid.
    """
    kind = event.get("kind")
    if kind not in {"corpus-ready", "dataset-arrived", "code-changed"}:
        raise ValueError("unsupported training event kind")
    result = deepcopy(template)
    result["metadata"]["name"] = "policy-training-" + kind
    if kind == "corpus-ready":
        result["config"]["episodes_uri"] = _reference(event, "episodes_uri")
        return result
    states = result["states"]
    for name in _PRETRAIN:
        states.pop(name)
    if kind == "dataset-arrived":
        result["config"]["episodes_uri"] = _reference(event, "episodes_uri")
        if not event.get("benchmark", False):
            return _curation_only(result)
        states["split"]["next"] = "finetune-loop"
    else:
        result["initial"] = "finetune-loop"
        result["config"]["split_uri"] = _reference(event, "split_uri")
        for name in ("curate", "split"):
            states.pop(name)
    _bind_pretrained(states, _reference(event, "approved_checkpoint_uri"))
    return result


def _reference(event, name):
    value = event.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"event requires {name}")
    return value


def _curation_only(spec):
    states = spec["states"]
    for name in _FINETUNE | {"split"}:
        states.pop(name)
    states["curate"].pop("next")
    states["curate"]["terminal"] = True
    return spec


def _bind_pretrained(states, uri):
    state = states["finetune"]
    state["params"]["batch_input_uri"] = uri
    state["inputs"] = [{"uri": uri, "schema": "npa.policy.decision.v1"}]
