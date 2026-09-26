"""Exercise genuine validation and publication against synthetic artifact bytes."""

import copy

import pytest
from botocore.exceptions import ClientError

from npa.workflows.field_failure import artifacts
from npa.workflows.field_failure.contracts import _Bundle, _object_uri
from npa.workflows.field_failure.stages import run_stage


def _change_report(case, name, edit, *, sync_measurements=False):
    uri = case.root + "/" + name + ".json"
    report = case.store.read(uri)
    edit(report)
    if sync_measurements:
        for episode in report["episodes"]:
            trace = case.store.read(episode["evidence"]["uri"])
            trace.update({k: v for k, v in episode.items() if k != "evidence"})
            episode["evidence"] = case.store.asset(episode["evidence"]["uri"], trace)
    case.store.asset(uri, report)


def test_complete_chain_recommends_without_deployment(evaluated):
    evaluated.run("compare")
    result = evaluated.store.read(evaluated.root + "/decision.json")
    assert result["recommendation"] == "promote"
    assert result["promote_checkpoint"] is True
    assert result["deployment_authorized"] is False
    assert result["episodes_per_policy"] == 2
    assert result["metrics"]["progress"]["improvement"] == pytest.approx(0.3)
    assert result["metrics"]["collisions"]["improvement"] == 1.0
    assert len(result["record_sha256"]) == 4
    for stage, request in evaluated.requests:
        if stage in {"train", "reconstruct"}:
            assert "held_out" not in request
        assert "baseline-evaluation.json" not in str(request)


@pytest.mark.parametrize(
    "values,promote",
    [
        ([0.5, 0.5], False),
        ([0.4, 0.4], False),
        ([0.55, 0.55], False),
        ([0.9, 0.4], False),
        ([0.7, 0.7], True),
    ],
)
def test_primary_improvement_and_each_episode_regression(evaluated, values, promote):
    def edit(report):
        for episode, value in zip(report["episodes"], values):
            episode["metrics"]["progress"] = value

    _change_report(evaluated, "candidate-evaluation", edit, sync_measurements=True)
    evaluated.run("compare")
    result = evaluated.store.read(evaluated.root + "/decision.json")
    assert result["promote_checkpoint"] is promote


def test_secondary_regression_blocks_promotion(evaluated):
    _change_report(
        evaluated,
        "candidate-evaluation",
        lambda r: r["episodes"][0]["metrics"].update(collisions=2.0),
        sync_measurements=True,
    )
    evaluated.run("compare")
    result = evaluated.store.read(evaluated.root + "/decision.json")
    assert result["recommendation"] == "retain_baseline"
    assert result["regressions"][0]["metric"] == "collisions"


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["episodes"].pop(),
        lambda r: r["episodes"].append(copy.deepcopy(r["episodes"][0])),
        lambda r: r["episodes"][0].update(seed=9),
        lambda r: r["episodes"][0].update(seed=True),
        lambda r: r["episodes"][0].update(status="running"),
        lambda r: r["episodes"][0].update(steps=0),
        lambda r: r["episodes"][0].update(scene_sha256="b" * 64),
        lambda r: r["episodes"][0]["metrics"].pop("collisions"),
        lambda r: r["episodes"][0]["metrics"].update(progress=True),
        lambda r: r["episodes"][0]["metrics"].update(progress=2.0),
        lambda r: r["episodes"][0]["metrics"].update(extra=1.0),
        lambda r: r.update(protocol_sha256="a" * 64),
        lambda r: r.update(bundle_sha256="a" * 64),
        lambda r: r["adapter"].update(source_sha256="b" * 64),
        lambda r: r["adapter"].update(
            entrypoint="synthetic_navigation_fixture:substitute"
        ),
        lambda r: r["adapter"].update(
            runtime_image="registry.example.invalid/navigation@sha256:" + "b" * 64
        ),
        lambda r: r["policy"]["checkpoint"].update(sha256="b" * 64),
        lambda r: r["episodes"][0]["evidence"].update(
            uri="s3://fixture/outside/episode.json"
        ),
    ],
)
def test_incomplete_tampered_or_mismatched_evaluation_fails_closed(evaluated, edit):
    _change_report(evaluated, "candidate-evaluation", edit)
    with pytest.raises(ValueError):
        evaluated.run("compare")
    result = evaluated.store.read(evaluated.root + "/decision.json")
    assert result["status"] == "invalid_evidence"
    assert result["promote_checkpoint"] is False


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_metric_rejected_before_decision(evaluated, number):
    uri = evaluated.root + "/candidate-evaluation.json"
    evaluated.store.objects[uri] = evaluated.store.objects[uri].replace(
        b'"progress":0.8', b'"progress":' + number.encode()
    )
    with pytest.raises(ValueError):
        evaluated.run("compare")
    assert (
        evaluated.store.read(evaluated.root + "/decision.json")["status"]
        == "invalid_evidence"
    )


@pytest.mark.parametrize(
    "kind", ["checkpoint", "rollout", "capture", "held_scene", "protocol"]
)
def test_modified_or_missing_artifact_bytes_are_rejected(evaluated, kind):
    lookup = {
        "checkpoint": evaluated.store.read(evaluated.root + "/training.json")[
            "candidate"
        ]["checkpoint"]["uri"],
        "rollout": evaluated.store.read(evaluated.root + "/candidate-evaluation.json")[
            "episodes"
        ][0]["evidence"]["uri"],
        "capture": "s3://fixture/input/capture.mcap",
        "held_scene": "s3://fixture/input/held.usd",
        "protocol": "s3://fixture/input/protocol.json",
    }
    evaluated.store.objects[lookup[kind]] = b"tampered"
    with pytest.raises(ValueError, match="SHA-256"):
        evaluated.run("compare")


def test_missing_evaluation_emits_durable_rejection(evaluated):
    del evaluated.store.objects[evaluated.root + "/candidate-evaluation.json"]
    with pytest.raises(ClientError, match="NoSuchKey"):
        evaluated.run("compare")
    assert not evaluated.store.read(evaluated.root + "/decision.json")[
        "promote_checkpoint"
    ]


@pytest.mark.parametrize(
    "edit",
    [
        lambda b: b.update(task="manipulation"),
        lambda b: b["captures"][0].update(group_id="held-site"),
        lambda b: b["captures"][0].update(scenario_id="unseen"),
        lambda b: b["captures"][0].update(asset=b["held_out"][0]["asset"]),
        lambda b: b.update(baseline_training_groups=["held-site"]),
        lambda b: b["held_out"][0].update(seeds=[7, 7]),
        lambda b: b.update(primary_metric="unknown"),
        lambda b: b["metrics"][0].update(maximum_regression=-1.0),
        lambda b: b["metrics"][0].update(direction="unknown"),
        lambda b: b["metrics"].append(b["metrics"][0]),
        lambda b: b.update(captures=[]),
        lambda b: b.update(held_out=[]),
        lambda b: b["adapters"].pop("evaluate"),
    ],
)
def test_bundle_refuses_leakage_ambiguity_and_missing_contract(navigation, edit):
    edit(navigation.bundle)
    with pytest.raises(ValueError):
        _Bundle.model_validate(navigation.bundle)


@pytest.mark.parametrize(
    "uri",
    [
        "https://host/scan",
        "s3://fixture/../x",
        "s3://fixture/a//b",
        "s3://fixture/a/%2e%2e/b",
        "s3://fixture/a?secret=x",
        "s3://user:pass@fixture/a",
        "s3://fixture/a#fragment",
        "s3://fixture/a;touch",
        "s3://fixture/a\\b",
        "s3://fixture/a\nb",
    ],
)
def test_hostile_paths_rejected(uri):
    with pytest.raises(ValueError):
        _object_uri(uri)


def test_local_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        _object_uri(str(tmp_path / "scan"))


def test_pinned_bundle_tampering_rejected(navigation):
    navigation.store.objects[navigation.bundle_uri] += b" "
    with pytest.raises(ValueError, match="SHA-256"):
        navigation.run("validate")
    assert navigation.root + "/validated.json" not in navigation.store.objects


def test_duplicate_json_keys_rejected(navigation):
    uri = "s3://fixture/input/duplicate.json"
    navigation.store.objects[uri] = b'{"x":1,"x":2}'
    with pytest.raises(ValueError, match="duplicate JSON"):
        artifacts._read(uri)


def test_validation_retry_reuses_matching_record(navigation):
    navigation.run("validate")
    original = navigation.store.objects[navigation.root + "/validated.json"]
    navigation.run("validate")
    assert navigation.store.objects[navigation.root + "/validated.json"] == original


@pytest.mark.parametrize("adapter", ["", "$(touch /tmp/pwn)", "a:b;exit", "a:b:c"])
def test_missing_or_hostile_adapter_fails_closed(navigation, adapter):
    navigation.run("validate")
    with pytest.raises(
        ValueError, match="operator adapter required|configured callable"
    ):
        navigation.run("reconstruct", adapter)
    assert navigation.root + "/reconstruction.json" not in navigation.store.objects


def test_changed_adapter_code_fails_before_invocation(navigation):
    navigation.run("validate")
    navigation.source.write_text('raise RuntimeError("must never execute")\n')
    with pytest.raises(ValueError, match="adapter source SHA-256"):
        navigation.run("reconstruct")
    assert not navigation.requests


def test_run_scoping_is_enforced(navigation):
    with pytest.raises(ValueError, match="exact run ID"):
        run_stage(
            "validate",
            navigation.bundle_uri,
            navigation.pin,
            navigation.root,
            "different",
        )


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r.update(initial_checkpoint_sha256="a" * 64),
        lambda r: r.update(reconstruction_sha256="a" * 64),
        lambda r: r.update(training_scenario_ids=["unseen"]),
        lambda r: r.update(training_group_ids=["held-site"]),
        lambda r: r.update(training_scene_sha256=[]),
        lambda r: r["candidate"].update(policy_id="existing"),
    ],
)
def test_training_provenance_is_rechecked(evaluated, edit):
    _change_report(evaluated, "training", edit)
    with pytest.raises(ValueError):
        evaluated.run("compare")


def test_summary_metric_tampering_cannot_override_measured_trace(evaluated):
    _change_report(
        evaluated,
        "candidate-evaluation",
        lambda r: r["episodes"][0]["metrics"].update(progress=0.99),
    )
    with pytest.raises(ValueError, match="differs from episode measurements"):
        evaluated.run("compare")
    assert (
        evaluated.store.read(evaluated.root + "/decision.json")["status"]
        == "invalid_evidence"
    )


def test_finite_inputs_with_overflowing_mean_fail_explicitly():
    from npa.workflows.field_failure.comparison import _mean

    with pytest.raises(ValueError, match="aggregation overflowed"):
        _mean([1e308, 1e308])


def _copy_trajectory(case, arm, index, data):
    uri = case.root + f"/{arm}-evaluation.json"
    report = case.store.read(uri)
    episode = report["episodes"][index]
    trace = case.store.read(episode["evidence"]["uri"])
    trace["trajectory"] = case.store.asset(trace["trajectory"]["uri"], data)
    episode["evidence"] = case.store.asset(episode["evidence"]["uri"], trace)
    case.store.asset(uri, report)


def _trajectory_bytes(case, arm, index):
    report = case.store.read(case.root + f"/{arm}-evaluation.json")
    trace = case.store.read(report["episodes"][index]["evidence"]["uri"])
    return case.store.objects[trace["trajectory"]["uri"]]


@pytest.mark.parametrize(
    "reuse", ["within-baseline", "within-candidate", "across-arms", "all-four"]
)
def test_distinct_wrappers_cannot_hide_reused_trajectory_bytes(evaluated, reuse):
    first = _trajectory_bytes(evaluated, "baseline", 0)
    if reuse.startswith("within-"):
        arm = reuse.removeprefix("within-")
        _copy_trajectory(evaluated, arm, 1, _trajectory_bytes(evaluated, arm, 0))
    elif reuse == "across-arms":
        for index in range(2):
            _copy_trajectory(
                evaluated,
                "candidate",
                index,
                _trajectory_bytes(evaluated, "baseline", index),
            )
    else:
        for arm in ["baseline", "candidate"]:
            for index in range(2):
                _copy_trajectory(evaluated, arm, index, first)
    wrappers = [
        e["evidence"]["sha256"]
        for arm in ["baseline", "candidate"]
        for e in evaluated.store.read(evaluated.root + f"/{arm}-evaluation.json")[
            "episodes"
        ]
    ]
    assert len(set(wrappers)) == 4
    with pytest.raises(ValueError, match="underlying trajectory bytes were reused"):
        evaluated.run("compare")
    decision = evaluated.store.read(evaluated.root + "/decision.json")
    assert decision["status"] == "invalid_evidence"
    assert decision["promote_checkpoint"] is False
