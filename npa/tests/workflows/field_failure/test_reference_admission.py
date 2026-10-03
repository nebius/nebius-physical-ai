"""Reject fabricated, incomplete, changed or held-out-derived failure admissions."""

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from admission_fixture import build_observation, reseal, set_outcomes
from npa.workflows.field_failure import reference_demo_admission as admission
from npa.workflows.field_failure.artifacts import _digest, _encode
from npa.workflows.field_failure.reference_demo_failure_evidence import (
    verify_observation,
)
from npa.workflows.navigation.artifacts import _files, write_json


@pytest.fixture
def observed(tmp_path, monkeypatch):
    return build_observation(tmp_path, monkeypatch)


def verify(fixture):
    return verify_observation(
        fixture.root, fixture.plan, fixture.recipe.initial_checkpoint.sha256
    )


def test_complete_native_timeouts_are_real_failures_and_all_success_is_not(observed):
    rows, files = verify(observed)
    assert len(rows) == 2 and all(row["success"] is False for row in rows)
    assert all(row["steps"] == 3 and row["collision_steps"] == 0 for row in rows)
    assert files["native-process.json"] and files["trajectory.json"]
    set_outcomes(observed, success=True)
    rows, _ = verify(observed)
    assert all(row["success"] is True for row in rows)


@pytest.mark.parametrize(
    "changed", ["count", "order", "seed", "reset", "rule", "budget"]
)
def test_observation_must_match_entire_ordered_prelearning_training_freeze(
    observed, changed
):
    plan = observed.plan
    office = plan["cohorts"]["regions"]["training"]["office"]
    if changed == "count":
        office["count"] = 1
    elif changed == "order":
        office["case_ids"].reverse()
    elif changed == "seed":
        office["seeds"][0] += 1
    elif changed == "reset":
        office["sha256"] = "b" * 64
    elif changed == "rule":
        plan["failure_observation"]["rule"] = {"predicate": "invent failure"}
    else:
        plan["failure_observation"]["recipe_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        verify(observed)


@pytest.mark.parametrize(
    "changed", ["summary", "short", "mask", "nan", "goal", "reset", "incomplete"]
)
def test_outcomes_recompute_from_complete_finite_native_trajectories(observed, changed):
    path = observed.root / "trajectory.json"
    rows = json.loads(path.read_text())
    if changed == "summary":
        observed.report["episodes"][0]["success"] = True
        write_json(observed.root / "evaluation.json", observed.report)
    elif changed == "short":
        rows.pop()
    elif changed == "mask":
        rows[1]["active"][0] = False
    elif changed == "nan":
        rows[1]["position_m"][0][0] = float("nan")
    elif changed == "goal":
        rows[1]["goal_m"][0][0] += 1
    elif changed == "reset":
        rows[0]["position_m"][0][0] += 1
    else:
        write_json(observed.root / "failure.json", {"status": "failed"})
    path.write_text(json.dumps(rows))
    reseal(observed.root)
    with pytest.raises(ValueError):
        verify(observed)


@pytest.mark.parametrize(
    "changed", ["crash", "control", "after_load", "checkpoint", "claim", "raw_control"]
)
def test_failed_or_substituted_native_execution_never_admits_capture(observed, changed):
    if changed == "crash":
        path = observed.root / "process.json"
        data = json.loads(path.read_text())
        data["returncode"] = 1
    elif changed == "control":
        path = observed.root / "controls/solo/process.json"
        data = json.loads(path.read_text())
        data["interruption"] = "KeyboardInterrupt"
    elif changed in {"after_load", "claim"}:
        path = observed.root / "native-process.json"
        data = json.loads(path.read_text())
        if changed == "after_load":
            data["provenance"]["initialization"]["checkpoint_sha256"] = "b" * 64
        else:
            data["attempt_id"] = "b" * 32
    elif changed == "checkpoint":
        (observed.root / "baseline.pt").write_bytes(b"different checkpoint")
        reseal(observed.root)
        with pytest.raises(ValueError):
            verify(observed)
        return
    else:
        path = observed.root / "controls/solo/control.json"
        data = json.loads(path.read_text())
        data["files"]["probe-solo.npz"] = "b" * 64
    write_json(path, data)
    reseal(observed.root)
    with pytest.raises(ValueError):
        verify(observed)


class _Records:
    def __init__(self):
        self.objects, self.reads = {}, []

    def write(self, uri, value):
        self.objects[uri] = _encode(value)
        return {"uri": uri, "sha256": _digest(self.objects[uri])}

    def read(self, uri, expected=None):
        assert "/final" not in uri and "/development" not in uri
        self.reads.append(uri)
        data = self.objects[uri]
        if expected is not None and _digest(data) != expected:
            raise ValueError("artifact SHA-256 mismatch")
        return json.loads(data), _digest(data)

    def complete(self, prefix, files):
        value = {
            "schema": "npa.navigation.publication.v1",
            "attempt": "attempts/" + "a" * 32,
            "files": files,
            "manifest_sha256": _digest(_encode(files)),
        }
        for name in ("claim", "completion"):
            self.write(prefix + "/" + name + ".json", value)


@pytest.fixture
def stored(observed, monkeypatch):
    root = "s3://example/demo/run"
    records = _Records()
    records.args = SimpleNamespace(output_root=root, run_id="run")
    records.observed = observed
    records.write(root + "/reference-plan.json", observed.plan)
    records.complete(
        root + "/baseline-training",
        {"policy.pt": observed.recipe.initial_checkpoint.sha256},
    )
    records.complete(root + "/failure-observation", _files(observed.root))
    monkeypatch.setattr(admission, "_read", records.read)
    monkeypatch.setattr(
        admission,
        "materialize",
        lambda _uri, target: Path(shutil.copytree(observed.root, target)),
    )
    monkeypatch.setattr(admission, "publish_record", records.write)
    return records


def test_admission_binds_actual_failures_into_exact_bundle_reference(stored):
    record = admission.admit_capture(stored.args)
    bundle = {
        "baseline": {"checkpoint": {"sha256": record["checkpoint_sha256"]}},
        "captures": [
            {"scenario_id": "office-with-warehouse-replay", "asset": record["capture"]}
        ],
    }
    actual, descriptor = admission.verify_admission(stored.args, bundle=bundle)
    assert actual == record and record["observed_failures"] == record["episodes"] == 2
    assert record["admitted"] and not record["final_cohort_consumed"]
    assert (
        record["raw_outcomes_sha256"] == _files(stored.observed.root)["trajectory.json"]
    )
    assert (
        admission.verify_admission(stored.args, descriptor=descriptor, bundle=bundle)[0]
        == record
    )
    bundle["captures"][0]["asset"] = {
        "uri": "s3://example/other.tar",
        "sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="sealed capture or baseline"):
        admission.verify_admission(stored.args, descriptor=descriptor, bundle=bundle)


def test_same_capture_cannot_swap_failure_record_after_seal(stored):
    record = admission.admit_capture(stored.args)
    _, descriptor = admission.verify_admission(stored.args)
    changed = copy.deepcopy(record)
    changed["failures"] = changed["failures"][:1]
    changed["observed_failures"] = 1
    stored.write(descriptor["uri"], changed)
    with pytest.raises(ValueError, match="SHA-256"):
        admission.verify_admission(stored.args, descriptor=descriptor)
    with pytest.raises(ValueError, match="actual observed"):
        admission.verify_admission(stored.args)


def test_zero_failure_stops_before_continuation_and_final(stored, monkeypatch):
    from npa.workflows.field_failure import reference_demo_admission_report as report

    set_outcomes(stored.observed, success=True)
    stored.complete(
        stored.args.output_root + "/failure-observation", _files(stored.observed.root)
    )
    published = []
    monkeypatch.setattr(
        report, "no_failures_report", lambda *args: published.append(args[2])
    )
    with pytest.raises(RuntimeError, match="no simulation failures observed"):
        admission.admit_capture(stored.args)
    assert published[0]["observed_failures"] == 0 and not published[0]["admitted"]
    with pytest.raises(ValueError, match="actual observed"):
        admission.verify_admission(stored.args)


def test_zero_failure_report_is_standalone_factual_and_contains_no_storage_routes(
    stored, monkeypatch
):
    from npa.workflows.field_failure import reference_demo_attribution as attribution
    from npa.workflows.field_failure import reference_demo_admission_report as report
    from npa.workflows.navigation import preview

    set_outcomes(stored.observed, success=True)
    stored.complete(
        stored.args.output_root + "/failure-observation", _files(stored.observed.root)
    )
    monkeypatch.setattr(attribution, "sample_credit", lambda *_: None)
    monkeypatch.setattr(
        preview, "scored_rollout_group", lambda *_args, **_kwargs: _preview_group()
    )
    published = {}
    monkeypatch.setattr(
        report, "publish_record", lambda uri, value: published.update({uri: value})
    )
    monkeypatch.setattr(
        report, "publish_html", lambda uri, value: published.update({uri: value})
    )
    with pytest.raises(RuntimeError, match="no simulation failures"):
        admission.admit_capture(stored.args)
    result = published[stored.args.output_root + "/reports/result.json"]
    html = published[stored.args.output_root + "/reports/index.html"]
    assert result["status"] == "no_observed_failures"
    assert (
        result["observation_completed"] is True and result["runtime_completed"] is False
    )
    assert result["continuation_started"] is result["final_evaluated"] is False
    assert result["quality_passed"] is result["deployment_authorized"] is False
    assert "No capture was admitted" in html and "physical field logs" in html
    assert "data:image/jpeg;base64," in html and "s3://" not in html


def _preview_group():
    from PIL import Image
    from npa.workflows.preview_html import image_preview

    return {
        "title": "Synthetic preview fixture",
        "note": "Unit fixture, not GPU evidence",
        "frames": [
            {
                "label": "Unit frame",
                "images": [
                    {
                        "label": "Unit image",
                        "data": image_preview(Image.new("RGB", (2, 2))),
                    },
                ],
            }
        ],
    }


@pytest.mark.parametrize("stage", ["reconstruct", "train"])
def test_public_stage_cannot_bypass_bound_admission(monkeypatch, stage):
    from npa.workflows.field_failure import reference_demo as demo
    from npa.workflows.field_failure import stages

    reference = {
        "uri": "s3://example/bundle.json",
        "sha256": "b" * 64,
        "admission": {"uri": "s3://example/observed-failures.json", "sha256": "a" * 64},
    }
    seen = []
    monkeypatch.setattr(
        demo,
        "_read",
        lambda uri, *_: (
            reference if uri.endswith("bundle-reference.json") else {},
            "a" * 64,
        ),
    )
    monkeypatch.setattr(
        stages, "run_stage", lambda *args: seen.append("native executed")
    )

    def reject(args, *, descriptor, bundle):
        assert descriptor == reference["admission"]
        raise ValueError("changed admitted raw outcomes")

    monkeypatch.setattr(admission, "verify_admission", reject)
    with pytest.raises(ValueError, match="changed admitted"):
        demo._loop_stage(
            SimpleNamespace(output_root="s3://example", run_id="unit", stage=stage)
        )
    assert not seen
