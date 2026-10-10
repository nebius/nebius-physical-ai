"""Completed evidence survives a later refusal without becoming a partial pass.

Native episodes/checkpoint loading are synthetic; spawn, operation, matching,
containment, parent snapshot, HTTP status and immutable upload are real paths.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from npa.workbench.robocasa import capabilities as cap
from npa.workbench.robocasa import pair_diagnostics as diag
from npa.workbench.robocasa import service
from npa.workbench.robocasa.schemas import RoboCasaRunRequest


@pytest.fixture
def body():
    return RoboCasaRunRequest(
        capability="kitchen_policy_eval",
        output_uri="s3://example/prefix",
        checkpoint_uri="s3://example/checkpoint",
        download_assets=False,
        train_env_ids="robocasa/Train",
        heldout_env_ids="robocasa/Heldout,robocasa/Other",
        seed=42,
        num_envs=3,
        iterations=100,
        timeout_seconds=30,
    )


def _native_outcome(video_path, pixels):
    digest = cap._sha256_array(pixels)
    return dict(
        initial_state_sha256="c" * 64,
        initial_workspace_sha256=digest,
        initial_simulator_state_sha256=None,
        raw_action_sha256="d" * 64,
        action_sha256="e" * 64,
        steps=100,
        reward_sum=0.0,
        max_reward=0.0,
        success=False,
        success_sources=["binary_reward"],
        terminated=False,
        truncated=False,
        state_keys=list(cap.ROBOCASA_STATE_KEYS),
        state_dim=16,
        terminal_state_sha256="f" * 64,
        terminal_workspace_sha256=digest,
        action_count=100,
        action_clipping_applied=False,
        action_out_of_bounds_steps=0,
        max_action_bound_violation=0.0,
        video_sha256=cap._sha256_file(video_path),
        video_bytes=video_path.stat().st_size,
        video_frames=101,
    )


def _install_native_fixture(monkeypatch, *, fail_after=1):
    monkeypatch.setattr(cap, "_download_s3_tree", lambda _uri, root: root)
    monkeypatch.setattr(
        cap, "_checkpoint_identity", lambda root: (root, "a" * 64, "b" * 64)
    )
    monkeypatch.setattr(cap, "_verify_training_provenance", lambda *_: {})
    monkeypatch.setattr(
        cap, "_load_act_policy", lambda _: (SimpleNamespace(reset=lambda: None),)
    )
    monkeypatch.setattr(cap, "_act_action_selector", lambda _: object())

    def episode(_task, *, seed, video_path, **_kwargs):
        index = int(video_path.parent.name.removeprefix("episode_"))
        assert index <= fail_after, "unattempted pair was called"
        worker = video_path.parent.parent
        with (worker.parent / "episode-calls.jsonl").open("a") as calls:
            calls.write(json.dumps([index, seed, video_path.stem]) + "\n")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"synthetic unit video " + video_path.name.encode())
        mismatch = index == fail_after and video_path.stem == "random_baseline"
        pixels = np.full((8, 8, 3), 255 if mismatch else 0, dtype=np.uint8)
        Image.fromarray(pixels).save(video_path.with_suffix(".png"))
        (worker / "private-scratch.txt").write_text("never upload")
        return _native_outcome(video_path, pixels)

    monkeypatch.setattr(cap, "_run_eval_episode", episode)


def _spawned_native_fixture(sender, payload):
    # Install the native boundary fixture inside spawn, not through fork state.
    with pytest.MonkeyPatch.context() as patch:
        _install_native_fixture(patch)
        service._capability_worker_entry(sender, payload)


def _produce(tmp_path, monkeypatch, body, *, fail_after=1):
    _install_native_fixture(monkeypatch, fail_after=fail_after)
    worker = tmp_path / "worker"
    with pytest.raises(cap.RoboCasaMatchError, match="initial workspace"):
        cap.run_capability_with_output(body, output_dir=worker, upload=False)
    directory = worker / diag.DIRECTORY
    return directory, diag.validate_directory(directory, requested_pairs=body.num_envs)


@pytest.mark.parametrize("upload_failure", [False, True])
def test_spawn_http_retains_completed_and_failed_media_then_cleans(
    tmp_path, monkeypatch, body, upload_failure
):
    from test_robocasa import _TransactionalFakeS3

    storage = _TransactionalFakeS3()
    monkeypatch.setattr("boto3.client", lambda *_a, **_k: storage)
    monkeypatch.setattr(service, "verify_runtime_identity", lambda *_: None)
    monkeypatch.setattr(service, "_assets_root", lambda: tmp_path / "assets")
    execute = service._execute_capability_in_worker
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda request, **kw: execute(
            request, worker_target=_spawned_native_fixture, output_root=tmp_path, **kw
        ),
    )
    published = []

    def upload(directory, uri, report):
        assert not list(tmp_path.glob("robocasa-worker-*"))
        checked = diag.validate_directory(directory, requested_pairs=body.num_envs)
        service._validate_failure_request_binding(body, checked)
        published.append((directory, deepcopy(checked)))
        if upload_failure:
            raise OSError("synthetic upload refusal")
        cap.upload_output(directory, uri, report)
        report["completed_pairs"][0]["policy"]["success"] = True

    monkeypatch.setattr(service, "upload_output", upload)
    client = TestClient(
        service.create_app(
            auth_mode="none",
            runs=service.RunRegistry(),
            execution_lock=service.GpuExecutionGate(),
        )
    )
    response = client.post("/run", json=body.model_dump(mode="json"))
    assert response.status_code == 200
    status = client.get("/status", params={"run_id": response.json()["run_id"]}).json()
    assert status["status"] == "failed" and "initial workspace" in status["error"]
    assert len(published) == 1 and not published[0][0].exists()
    report = published[0][1]
    assert report["requested_pairs"] == 3 and report["unattempted_pairs"] == 1
    assert (
        report["episode_index"] == 1
        and report["completed_pairs"][0]["episode_index"] == 0
    )
    assert (
        "success_rate" not in report
        and not report["completed_pairs"][0]["policy"]["success"]
    )
    assert len(report["files"]) == 8
    assert [
        json.loads(line)
        for line in (tmp_path / "episode-calls.jsonl").read_text().splitlines()
    ] == [
        [0, 42, "rollout"],
        [0, 42, "random_baseline"],
        [1, 43, "rollout"],
        [1, 43, "random_baseline"],
    ]
    if upload_failure:
        assert status["result"] is None and "publication failed" in status["error"]
        assert not storage.objects
    else:
        assert status["result"] == report
        _assert_uploaded_population(storage, report)
    assert client.get("/health").json()["execution_available"] is True


def _assert_uploaded_population(storage, report):
    prefix = "prefix/failure-diagnostics/"
    keys = {key.removeprefix(prefix) for key in storage.objects}
    assert keys == {
        diag.REPORT,
        *diag.media_names(report),
        "_NPA_CLAIM.json",
        "_NPA_COMPLETE.json",
    }
    assert json.loads(storage.objects[prefix + diag.REPORT][0]) == report
    for name, expected in report["files"].items():
        import hashlib

        data = storage.objects[prefix + name][0]
        assert {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        } == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_index", 1),
        ("seed", 99),
        ("env_id", "robocasa/Other"),
        ("checkpoint_sha256", "9" * 64),
        ("artifact_tree_sha256", "9" * 64),
        ("request_manifest_sha256", "9" * 64),
        ("image_source_sha", "9" * 40),
        ("image_manifest_digest", "sha256:" + "9" * 64),
    ],
)
def test_parent_refuses_wrong_completed_pair_binding(
    tmp_path, monkeypatch, body, field, value
):
    directory, report = _produce(tmp_path, monkeypatch, body)
    report["completed_pairs"][0][field] = value
    (directory / diag.REPORT).write_text(json.dumps(report))
    outcome = service._retain_failure_diagnostics(
        body, directory.parent, service._WorkerOutcome(error="match refusal")
    )
    assert (
        outcome.stopped
        and not outcome.failure_diagnostics
        and outcome.output_dir is None
    )
    assert not directory.parent.exists() and "match refusal" in outcome.error


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "missing",
        "symlink",
        "hardlink",
        "pixels",
        "video",
        "private_field",
        "nan",
        "counts",
        "guard",
        "horizon",
        "state_layout",
        "duplicate",
        "gap",
        "unattempted",
        "huge_failed_index",
    ],
)
def test_refuses_hostile_prefix_without_publishing_scratch(
    tmp_path, monkeypatch, body, damage
):
    directory, report = _produce(tmp_path, monkeypatch, body)
    _damage_prefix(directory, report, damage)
    (directory / diag.REPORT).write_text(json.dumps(report))
    outcome = service._retain_failure_diagnostics(
        body, directory.parent, service._WorkerOutcome(error="match refusal")
    )
    assert (
        outcome.stopped
        and not outcome.failure_diagnostics
        and outcome.output_dir is None
    )
    assert not directory.parent.exists()


def _damage_prefix(directory, report, damage):
    media = directory / "completed-0000-policy.png"
    pair = report["completed_pairs"][0]
    arm = pair["policy"]
    if damage == "extra":
        (directory / "private.txt").write_text("not allowed")
    elif damage == "missing":
        media.unlink()
    elif damage in {"symlink", "hardlink"}:
        outside = directory.parent / "outside.png"
        media.rename(outside)
        media.symlink_to(outside) if damage == "symlink" else media.hardlink_to(outside)
    elif damage == "pixels":
        Image.fromarray(np.full((8, 8, 3), 17, dtype=np.uint8)).save(media)
        report["files"][media.name] = diag.file_evidence(media)
    elif damage == "video":
        arm["video_sha256"] = "0" * 64
    elif damage == "private_field":
        arm["environment"] = "forbidden"
    elif damage == "nan":
        arm["reward_sum"] = float("nan")
    elif damage == "counts":
        arm["action_count"] = 99
    elif damage == "guard":
        arm["initial_state_sha256"] = "0" * 64
    elif damage == "horizon":
        arm.update(steps=101, action_count=101, video_frames=102)
    elif damage == "state_layout":
        arm["state_dim"] = 7
    elif damage == "duplicate":
        report["completed_pairs"].append(deepcopy(pair))
    elif damage == "gap":
        report["completed_pairs"] = []
    elif damage == "huge_failed_index":
        report["episode_index"] = 10**30
    else:
        report["unattempted_pairs"] = 0


def test_admitted_long_prefix_exceeds_old_envelope_and_survives_snapshot(
    tmp_path, monkeypatch, body
):
    body = body.model_copy(update={"num_envs": 42})
    directory, report = _produce(tmp_path, monkeypatch, body, fail_after=40)
    assert (directory / diag.REPORT).stat().st_size > 64 * 1024
    retained = service._retain_failure_diagnostics(
        body, directory.parent, service._WorkerOutcome(error="match refusal")
    )
    try:
        assert retained.stopped and retained.failure_diagnostics
        assert not directory.parent.exists()
        assert (
            diag.validate_directory(retained.output_dir, requested_pairs=42) == report
        )
        assert len(report["completed_pairs"]) == 40 and len(report["files"]) == 164
        assert report["unattempted_pairs"] == 1
    finally:
        if retained.output_dir:
            shutil.rmtree(retained.output_dir)


def test_report_envelope_uses_admitted_not_claimed_population(
    tmp_path, monkeypatch, body
):
    directory, report = _produce(tmp_path, monkeypatch, body)
    report["requested_pairs"] = 100000
    report["unattempted_pairs"] = 99998
    (directory / diag.REPORT).write_text(
        json.dumps(report) + " " * (64 * 1024 * body.num_envs)
    )
    with pytest.raises(ValueError, match="schema envelope"):
        diag.validate_directory(directory, requested_pairs=body.num_envs)
    (directory / diag.REPORT).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="admitted request"):
        diag.validate_directory(directory, requested_pairs=body.num_envs)
    with pytest.raises(ValueError, match="admitted request"):
        diag.validate_directory(directory)


def test_runtime_image_binding_checked_even_if_all_pair_claims_agree(
    tmp_path, monkeypatch, body
):
    body = body.model_copy(update={"expected_image_source_sha": "8" * 40})
    directory, report = _produce(tmp_path, monkeypatch, body)
    with pytest.raises(ValueError, match="runtime image binding"):
        service._validate_failure_request_binding(body, report)


def test_legacy_failure_only_record_remains_readable(tmp_path, monkeypatch, body):
    directory, report = _produce(tmp_path, monkeypatch, body, fail_after=0)
    for field in ("requested_pairs", "unattempted_pairs", "completed_pairs"):
        report.pop(field)
    report["schema_version"] = "npa.robocasa.pair-failure.v1"
    (directory / diag.REPORT).write_text(json.dumps(report))
    assert diag.validate_directory(directory) == report


def test_completed_native_success_is_preserved_without_aggregate(
    tmp_path, monkeypatch, body
):
    native = _native_outcome

    def successful_policy(video_path, pixels):
        outcome = native(video_path, pixels)
        if video_path.stem == "rollout":
            outcome.update(success=True, reward_sum=1.0, max_reward=1.0)
        return outcome

    monkeypatch.setattr(
        __import__(__name__, fromlist=["_native_outcome"]),
        "_native_outcome",
        successful_policy,
    )
    directory, report = _produce(tmp_path, monkeypatch, body)
    assert report["completed_pairs"][0]["policy"]["success"] is True
    assert report["completed_pairs"][0]["random_baseline"]["success"] is False
    assert report["status"] == "failed" and "success_rate" not in report
    assert not (directory.parent / "metrics.json").exists()
    assert not (directory.parent / "eval.json").exists()
