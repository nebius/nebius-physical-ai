"""Trigger config must reach the real watcher after submit --var overrides."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import _make_context, wait_for_trigger
from npa.orchestration.npa_workflow.runtime import _workflow_identity, s3_trigger_waiter
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import (
    load_spec_for_submit,
    merge_config_overrides,
)

SHIPPED = (
    Path(__file__).resolve().parents[4]
    / "workflows/testing/token-factory-trigger-watch.yaml"
)


def test_threshold_override_waits_for_requested_objects():
    spec = load_spec_for_submit(SHIPPED, config_overrides={"inbox_min_objects": "3"})
    listing_sizes = iter([1, 3])
    sleeps = []
    result = wait_for_trigger(
        spec.states["caption-inbox"],
        _make_context(spec, run_id="trigger-test"),
        waiter=s3_trigger_waiter(
            lister=lambda bucket, prefix: [
                f"frame-{index}" for index in range(next(listing_sizes))
            ],
            sleeper=sleeps.append,
        ),
    )
    assert result["objects"] == 3, (
        "must not launch captioning with only one of three frames"
    )
    assert result["polls"] == 2
    assert sleeps == [15]


def test_poll_override_reaches_actual_sleep():
    spec = load_spec_for_submit(SHIPPED, config_overrides={"inbox_poll_seconds": "2"})
    listing_sizes = iter([0, 1])
    sleeps = []
    wait_for_trigger(
        spec.states["caption-inbox"],
        _make_context(spec, run_id="trigger-test"),
        waiter=s3_trigger_waiter(
            lister=lambda bucket, prefix: ["frame"] * next(listing_sizes),
            sleeper=sleeps.append,
        ),
    )
    assert sleeps == [2]


def test_max_polls_override_stops_at_requested_count():
    spec = load_spec_for_submit(SHIPPED, config_overrides={"inbox_max_polls": "2"})
    sleeps = []
    with pytest.raises(NpaWorkflowError, match=r"after 2 poll\(s\)"):
        wait_for_trigger(
            spec.states["caption-inbox"],
            _make_context(spec, run_id="trigger-test"),
            waiter=s3_trigger_waiter(
                lister=lambda bucket, prefix: [], sleeper=sleeps.append
            ),
        )
    assert sleeps == [15]


def test_repeated_merges_preserve_original_and_rebind_expression():
    original = load_spec(SHIPPED)
    first = merge_config_overrides(original, {"inbox_min_objects": "3"})
    second = merge_config_overrides(first, {"inbox_min_objects": "5"})
    assert original.states["caption-inbox"].trigger.min_objects == 1
    assert first.states["caption-inbox"].trigger.min_objects == 3
    assert second.states["caption-inbox"].trigger.min_objects == 5


@pytest.mark.parametrize(
    "key,value",
    [("inbox_poll_seconds", 0), ("inbox_min_objects", 0), ("inbox_max_polls", -1)],
)
def test_invalid_templated_trigger_values_rejected(tmp_path, key, value):
    document = yaml.safe_load(SHIPPED.read_text())
    document["config"][key] = value
    path = tmp_path / "invalid-trigger.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(NpaWorkflowError, match=r"trigger\..*must be >="):
        load_spec(path)


def test_unchanged_workflow_identity_keeps_legacy_value():
    spec = load_spec(SHIPPED)
    state = spec.states["caption-inbox"]
    assert state.inputs and state.outputs and state.trigger.config_expressions
    legacy_payload = asdict(spec)
    for state in legacy_payload["states"].values():
        # Both fields were absent when the original workflow identity was stored.
        for artifact in [*state["inputs"], *state["outputs"]]:
            assert artifact.pop("kind") == ""
        if state.get("trigger"):
            state["trigger"].pop("config_expressions", None)
    legacy = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _workflow_identity(spec) == legacy


def test_workflow_identity_ignores_trigger_parse_provenance_without_mutating_spec():
    spec = load_spec(SHIPPED)
    original = asdict(spec)
    identity = _workflow_identity(spec)
    assert asdict(spec) == original

    spec.states["caption-inbox"].trigger.config_expressions.clear()

    assert _workflow_identity(spec) == identity


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("inputs", "kind", "directory"),
        ("outputs", "kind", "file"),
        ("trigger", "min_objects", 3),
        ("trigger", "poll_seconds", 2),
        ("trigger", "max_polls", 2),
    ],
)
def test_workflow_identity_keeps_artifact_roles_and_resolved_trigger_values(
    target, field, value
):
    spec = load_spec(SHIPPED)
    identity = _workflow_identity(spec)
    component = getattr(spec.states["caption-inbox"], target)
    if isinstance(component, list):
        component = component[0]
    setattr(component, field, value)

    assert _workflow_identity(spec) != identity
