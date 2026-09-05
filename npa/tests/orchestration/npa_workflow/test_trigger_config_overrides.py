"""Config overrides must control the watcher without invalidating resume identity."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import _make_context, wait_for_trigger
from npa.orchestration.npa_workflow.runtime import _workflow_identity, s3_trigger_waiter
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import load_spec_for_submit, merge_config_overrides

REFERENCE = (
    Path(__file__).resolve().parents[3]
    / "workflows/workbench/npa-workflows/token-factory-trigger-watch.yaml"
)
STATE = "caption-inbox"
FIELDS = [
    ("pollSeconds", "poll_seconds", "inbox_poll_seconds", 0),
    ("maxPolls", "max_polls", "inbox_max_polls", -1),
    ("minObjects", "min_objects", "inbox_min_objects", 0),
]


def _watch(spec, sizes):
    listings = iter(sizes)
    sleeps = []
    result = wait_for_trigger(
        spec.states[STATE],
        _make_context(spec, run_id="trigger-test"),
        waiter=s3_trigger_waiter(
            lister=lambda bucket, prefix: [f"frame-{i}" for i in range(next(listings))],
            sleeper=sleeps.append,
        ),
    )
    return result, sleeps


def _write_spec(tmp_path, *, snake_case=False, literal=False, invalid=None):
    document = yaml.safe_load(REFERENCE.read_text())
    trigger = document["states"][STATE]["trigger"]
    for camel, snake, config_key, _ in FIELDS:
        value = trigger.pop(camel)
        trigger[snake if snake_case else camel] = (
            document["config"][config_key] if literal else value
        )
    if invalid:
        key, value = invalid
        document["config"][key] = value
    path = tmp_path / "trigger.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def test_overrides_reach_object_threshold_and_poll_interval():
    spec = load_spec_for_submit(
        REFERENCE,
        config_overrides={"inbox_min_objects": "3", "inbox_poll_seconds": "2"},
    )
    watermark, sleeps = _watch(spec, [1, 2, 3])
    assert watermark["objects"] == 3
    assert watermark["polls"] == 3
    assert sleeps == [2, 2]


def test_max_polls_override_stops_watcher():
    spec = load_spec_for_submit(REFERENCE, config_overrides={"inbox_max_polls": "2"})
    sleeps = []
    with pytest.raises(NpaWorkflowError, match=r"after 2 poll\(s\)"):
        wait_for_trigger(
            spec.states[STATE],
            _make_context(spec, run_id="trigger-test"),
            waiter=s3_trigger_waiter(lister=lambda bucket, prefix: [], sleeper=sleeps.append),
        )
    assert sleeps == [15]


def test_zero_max_polls_keeps_watcher_unbounded():
    spec = load_spec_for_submit(REFERENCE, config_overrides={"inbox_max_polls": "0"})
    watermark, sleeps = _watch(spec, [0] * 41 + [1])
    assert watermark["polls"] == 42
    assert sleeps == [15] * 41


@pytest.mark.parametrize("snake_case", [False, True])
def test_repeated_overrides_rebind_without_mutating_prior_specs(tmp_path, snake_case):
    original = load_spec(_write_spec(tmp_path, snake_case=snake_case))
    first = merge_config_overrides(original, {"inbox_min_objects": "3"})
    second = merge_config_overrides(first, {"inbox_min_objects": "5"})
    versions = (original, first, second)
    assert [spec.states[STATE].trigger.min_objects for spec in versions] == [1, 3, 5]
    assert [spec.config["inbox_min_objects"] for spec in versions] == [1, "3", "5"]


@pytest.mark.parametrize("camel,snake,key,value", FIELDS)
@pytest.mark.parametrize("override", [False, True])
def test_invalid_templated_values_are_rejected(tmp_path, camel, snake, key, value, override):
    with pytest.raises(NpaWorkflowError, match=rf"trigger\.{camel} must be >="):
        if override:
            load_spec_for_submit(REFERENCE, config_overrides={key: str(value)})
        else:
            load_spec(_write_spec(tmp_path, invalid=(key, value)))


def test_literal_trigger_fields_do_not_follow_unrelated_config(tmp_path):
    original = load_spec(_write_spec(tmp_path, literal=True))
    changed = merge_config_overrides(
        original,
        {"inbox_min_objects": "9", "inbox_poll_seconds": "2", "inbox_max_polls": "4"},
    )
    assert changed.states[STATE].trigger == original.states[STATE].trigger
    assert _watch(changed, [1])[0]["objects"] == 1


def test_retaining_expressions_preserves_legacy_resume_identity(tmp_path):
    templated = load_spec(REFERENCE)
    literal = load_spec(_write_spec(tmp_path, literal=True))
    legacy_payload = asdict(literal)
    # Persisted workflows predate expression provenance. Their identity includes
    # the effective numbers, so equivalent literal and templated specs must match.
    for state in legacy_payload["states"].values():
        if state["trigger"] is not None:
            state["trigger"].pop("config_expressions", None)
    legacy_identity = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _workflow_identity(templated) == legacy_identity
    assert _workflow_identity(literal) == legacy_identity
    changed = merge_config_overrides(templated, {"inbox_min_objects": "3"})
    assert _workflow_identity(changed) != legacy_identity


@pytest.mark.parametrize("field", ["poll_seconds", "max_polls", "min_objects"])
def test_resume_identity_still_includes_effective_trigger_values(field):
    spec = load_spec(REFERENCE)
    original_identity = _workflow_identity(spec)
    state = spec.states[STATE]
    spec.states[STATE] = replace(
        state, trigger=replace(state.trigger, **{field: getattr(state.trigger, field) + 1})
    )
    assert _workflow_identity(spec) != original_identity
