"""Refused pairs retain only typed, contained evidence and remain failed."""

from __future__ import annotations

import json
import multiprocessing
import shutil
from pathlib import Path
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
def request_body():
    return RoboCasaRunRequest(
        capability="kitchen_policy_eval",
        output_uri="s3://example/run",
        checkpoint_uri="s3://example/checkpoint",
        download_assets=False,
        train_env_ids="robocasa/Train",
        heldout_env_ids="robocasa/Heldout",
        seed=42,
        num_envs=6,
        iterations=100,
        timeout_seconds=30,
    )


@pytest.fixture
def refused_pair(monkeypatch):
    monkeypatch.setattr(cap, "_download_s3_tree", lambda _uri, root: root)
    monkeypatch.setattr(
        cap, "_checkpoint_identity", lambda root: (root, "a" * 64, "b" * 64)
    )
    monkeypatch.setattr(cap, "_verify_training_provenance", lambda *_: {})
    monkeypatch.setattr(
        cap, "_load_act_policy", lambda _: (SimpleNamespace(reset=lambda: None),)
    )
    monkeypatch.setattr(cap, "_act_action_selector", lambda _: object())

    def episode(_task, *, video_path, **_kwargs):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"synthetic unit video: " + video_path.name.encode())
        pixels = np.full(
            (8, 8, 3), 0 if video_path.stem == "rollout" else 255, dtype=np.uint8
        )
        Image.fromarray(pixels).save(video_path.with_suffix(".png"))
        # An unrelated private file must never survive parent cleanup/publication.
        (video_path.parent.parent / "private-scratch.txt").write_text(
            "not for publication"
        )
        return {
            "initial_state_sha256": "c" * 64,
            "initial_workspace_sha256": cap._sha256_array(pixels),
            "initial_simulator_state_sha256": None,
            "raw_action_sha256": "d" * 64,
            "action_sha256": "e" * 64,
            "steps": 100,
        }

    monkeypatch.setattr(cap, "_run_eval_episode", episode)


def _failed_directory(tmp_path, request_body):
    with pytest.raises(cap.RoboCasaMatchError, match="initial workspace"):
        cap.run_capability_with_output(request_body, output_dir=tmp_path, upload=False)
    return tmp_path / diag.DIRECTORY


def test_operation_retains_exact_failed_pair_not_matched_result(
    tmp_path,
    request_body,
    refused_pair,
):
    directory = _failed_directory(tmp_path, request_body)
    report = diag.validate_directory(directory, requested_pairs=request_body.num_envs)
    assert report["status"] == "failed"
    assert (report["episode_index"], report["seed"], report["env_id"]) == (
        0,
        42,
        "robocasa/Heldout",
    )
    assert report["reason"] == "initial_workspace_mismatch"
    assert (
        report["policy"]["initial_state_sha256"]
        == report["random_baseline"]["initial_state_sha256"]
    )
    assert (
        report["policy"]["initial_workspace_sha256"]
        != report["random_baseline"]["initial_workspace_sha256"]
    )
    assert report["policy"]["initial_simulator_state_sha256"] is None
    assert report["checkpoint_sha256"] == "a" * 64
    assert not (tmp_path / "eval.json").exists()
    assert not (tmp_path / "result.json").exists()
    assert {p.name for p in directory.iterdir()} == {diag.REPORT, *diag.MEDIA}


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "symlink",
        "hardlink",
        "mutation",
        "json_extra",
        "json_status",
        "root_link",
    ],
)
def test_refuses_untrusted_diagnostic_population(
    tmp_path,
    request_body,
    refused_pair,
    damage,
):
    directory = _failed_directory(tmp_path, request_body)
    media = directory / "policy.png"
    if damage == "extra":
        (directory / "environment.txt").write_text("must not publish")
    elif damage in {"symlink", "hardlink"}:
        outside = tmp_path / "outside"
        media.rename(outside)
        media.symlink_to(outside) if damage == "symlink" else media.hardlink_to(outside)
    elif damage == "mutation":
        media.write_bytes(b"changed")
    elif damage.startswith("json_"):
        payload = json.loads((directory / diag.REPORT).read_text())
        payload["secret" if damage == "json_extra" else "status"] = "forbidden"
        (directory / diag.REPORT).write_text(json.dumps(payload))
    else:
        other = tmp_path / "other"
        directory.rename(other)
        directory.symlink_to(other, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        diag.validate_directory(directory, requested_pairs=request_body.num_envs)


def test_stopped_worker_retains_only_failure_snapshot(
    tmp_path,
    monkeypatch,
    request_body,
    refused_pair,
):
    monkeypatch.setattr(service, "_assets_root", lambda: tmp_path / "assets")
    outcome = service._execute_capability_in_worker(
        request_body,
        process_context=multiprocessing.get_context("fork"),
        output_root=tmp_path,
        retain_output=True,
    )
    try:
        assert outcome.stopped and outcome.failure_diagnostics
        assert outcome.result is None and "initial workspace" in outcome.error
        assert not list(tmp_path.glob("robocasa-worker-*"))
        assert {p.name for p in outcome.output_dir.iterdir()} == {
            diag.REPORT,
            *diag.MEDIA,
        }
        assert (
            diag.validate_directory(
                outcome.output_dir, requested_pairs=request_body.num_envs
            )["seed"]
            == 42
        )
    finally:
        if outcome.output_dir:
            shutil.rmtree(outcome.output_dir)


@pytest.mark.parametrize("upload_failure", [False, True])
def test_real_http_run_status_remains_failed_after_diagnostic_publication(
    tmp_path,
    monkeypatch,
    request_body,
    refused_pair,
    upload_failure,
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
        lambda body, **kwargs: execute(
            body,
            process_context=multiprocessing.get_context("fork"),
            output_root=tmp_path,
            **kwargs,
        ),
    )
    published = []

    def upload(directory, uri, report):
        assert not list(tmp_path.glob("robocasa-worker-*"))
        assert uri == "s3://example/run/failure-diagnostics"
        assert (
            diag.validate_directory(directory, requested_pairs=request_body.num_envs)[
                "status"
            ]
            == "failed"
        )
        published.append(directory)
        if upload_failure:
            raise OSError("private provider details must not be echoed")
        cap.upload_output(directory, uri, report)
        assert report["output_uri"] == uri
        assert report["output_commit"]["schema"] == "npa.robocasa.output-commit.v1"
        # Nested isolation also holds if publication later gains annotations.
        report["policy"]["steps"] = 999

    monkeypatch.setattr(service, "upload_output", upload)
    client = TestClient(
        service.create_app(
            auth_mode="none",
            runs=service.RunRegistry(),
            execution_lock=service.GpuExecutionGate(),
        )
    )
    response = client.post("/run", json=request_body.model_dump(mode="json"))
    assert response.status_code == 200
    status = client.get("/status", params={"run_id": response.json()["run_id"]}).json()
    assert status["status"] == "failed"
    assert "initial workspace" in status["error"]
    assert len(published) == 1 and not published[0].exists()
    assert "private provider" not in status["error"]
    if upload_failure:
        assert "publication failed (OSError)" in status["error"]
        assert status["result"] is None
    else:
        assert status["result"]["status"] == "failed"
        assert status["result"]["seed"] == 42
        assert status["result"]["policy"]["steps"] == 100
        diag.PairFailureWithPrefix.model_validate(status["result"])
        retained_report = json.loads(
            storage.objects["run/failure-diagnostics/failure.json"][0]
        )
        assert retained_report == status["result"]
    assert client.get("/health").json()["execution_available"] is True


def test_direct_operation_uploads_only_diagnostics_and_raises(
    tmp_path,
    monkeypatch,
    request_body,
    refused_pair,
):
    published = []
    monkeypatch.setattr(
        cap,
        "upload_output",
        lambda root, uri, report: published.append((root, uri, report)),
    )
    with pytest.raises(cap.RoboCasaMatchError):
        cap.run_capability_with_output(request_body, output_dir=tmp_path)
    assert len(published) == 1
    assert published[0][0] == tmp_path / diag.DIRECTORY
    assert published[0][1] == "s3://example/run/failure-diagnostics"
    assert published[0][2]["status"] == "failed"


def test_unstopped_failure_never_publishes(monkeypatch, request_body):
    gate = service.GpuExecutionGate()
    runs = service.RunRegistry()
    monkeypatch.setattr(service, "verify_runtime_identity", lambda *_: None)
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda *_a, **_k: service._WorkerOutcome(
            error="initial workspace mismatch",
            stopped=False,
            failure_diagnostics=True,
        ),
    )
    monkeypatch.setattr(
        service, "upload_output", lambda *_: pytest.fail("uncontained publication")
    )
    client = TestClient(
        service.create_app(auth_mode="none", runs=runs, execution_lock=gate)
    )
    response = client.post("/run", json=request_body.model_dump(mode="json"))
    status = client.get("/status", params={"run_id": response.json()["run_id"]}).json()
    assert status["status"] == "failed" and status["result"] is None
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("native_location", ["nested", "unwrapped", "outer"])
def test_initial_frame_is_saved_before_first_action(
    tmp_path, monkeypatch, native_location
):
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {"video.robot0_agentview_left": pixels}
    state = np.array([0.0, 1.0])
    native = SimpleNamespace(sim=SimpleNamespace(get_state=lambda: state))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(env=native),
        action_space=object(),
        reset=lambda **_: (observation, {}),
        close=lambda: None,
    )
    if native_location == "unwrapped":
        env.unwrapped = native
    elif native_location == "outer":
        env.unwrapped = object()
        env.sim = native.sim
    monkeypatch.setattr(cap, "_make_env", lambda *_, **__: env)
    monkeypatch.setattr(cap, "_seed_action_space", lambda *_: None)
    video = tmp_path / "rollout.mp4"

    def rollout(*_, **__):
        with Image.open(video.with_suffix(".png")) as image:
            assert np.array_equal(np.asarray(image), pixels)
        pixels[:] = 255
        return [pixels], {}

    def write_video(_frames, path):
        path.write_bytes(b"synthetic unit video")
        return path

    monkeypatch.setattr(cap, "_rollout_eval_episode", rollout)
    monkeypatch.setattr(cap, "_write_required_video", write_video)
    result = cap._run_eval_episode(
        "robocasa/Heldout",
        seed=42,
        iterations=1,
        selector=object(),
        video_path=video,
        download_assets=False,
    )
    with Image.open(video.with_suffix(".png")) as image:
        assert not np.asarray(image).any()
    assert result["initial_simulator_state_sha256"] == cap._sha256_array(state)


def test_changed_pixels_refused_even_with_updated_file_hash(
    tmp_path,
    request_body,
    refused_pair,
):
    directory = _failed_directory(tmp_path, request_body)
    Image.fromarray(np.full((8, 8, 3), 123, dtype=np.uint8)).save(
        directory / "policy.png"
    )
    payload = json.loads((directory / diag.REPORT).read_text())
    payload["files"]["policy.png"] = diag.file_evidence(directory / "policy.png")
    (directory / diag.REPORT).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="initial pixels differ"):
        diag.validate_directory(directory, requested_pairs=request_body.num_envs)


def test_failure_ancestor_symlink_refused(tmp_path, request_body, refused_pair):
    worker = tmp_path / "worker"
    directory = _failed_directory(worker, request_body)
    worker.rename(tmp_path / "relocated")
    worker.symlink_to(tmp_path / "relocated", target_is_directory=True)
    with pytest.raises(OSError):
        diag.validate_directory(directory, requested_pairs=request_body.num_envs)


def test_failure_request_binding_mismatch_cleans_all_scratch(
    tmp_path,
    request_body,
    refused_pair,
):
    directory = _failed_directory(tmp_path / "worker", request_body)
    outcome = service._retain_failure_diagnostics(
        request_body.model_copy(update={"seed": 99}),
        directory.parent,
        service._WorkerOutcome(error="original match refusal"),
    )
    assert outcome.stopped and outcome.output_dir is None
    assert not outcome.failure_diagnostics and not directory.parent.exists()
    assert "original match refusal" in outcome.error
    assert "diagnostics unavailable" in outcome.error


def test_robot_state_refusal_has_distinct_reason():
    arm = dict.fromkeys(diag.ArmEvidence.model_fields, "a" * 64)
    arm.update(initial_simulator_state_sha256=None, steps=1)
    other = {**arm, "initial_state_sha256": "b" * 64}
    with pytest.raises(cap.RoboCasaError, match="initial robot states"):
        cap._require_matched_initial_state(arm, other)
    evidence = cap._pair_failure_evidence(
        {"episode_index": 3, "seed": 45, "env_id": "robocasa/Heldout"},
        arm,
        other,
    )
    assert evidence["reason"] == "initial_robot_state_mismatch"
    assert evidence["episode_index"] == 3


@pytest.mark.parametrize("value", [None, object(), [1.0], np.array([object()])])
def test_unavailable_simulator_accessor_returns_null(value):
    env = SimpleNamespace(sim=SimpleNamespace(get_state=lambda: value))
    assert cap._initial_simulator_state_hash(env) is None


@pytest.mark.parametrize("error", [AttributeError, TypeError])
def test_optional_simulator_accessor_error_is_contained(error):
    def unavailable():
        raise error("unsupported native accessor")

    env = SimpleNamespace(sim=SimpleNamespace(get_state=unavailable))
    assert cap._initial_simulator_state_hash(env) is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_observed_nonfinite_native_state_still_fails(value):
    env = SimpleNamespace(sim=SimpleNamespace(get_state=lambda: np.array([value])))
    with pytest.raises(cap.RoboCasaError, match="non-finite initial simulator"):
        cap._initial_simulator_state_hash(env)


@pytest.mark.parametrize(
    "changes",
    [{"episode_index": 6}, {"seed": 43}, {"env_id": "robocasa/Other"}],
)
def test_parent_refuses_pair_identity_not_in_request(
    tmp_path, request_body, refused_pair, changes
):
    directory = _failed_directory(tmp_path / "worker", request_body)
    report_path = directory / diag.REPORT
    report = json.loads(report_path.read_text())
    report.update(changes)
    report_path.write_text(json.dumps(report))
    outcome = service._retain_failure_diagnostics(
        request_body, directory.parent, service._WorkerOutcome(error="match refusal")
    )
    assert outcome.stopped and outcome.output_dir is None
    assert not outcome.failure_diagnostics and not directory.parent.exists()
    assert "diagnostics unavailable" in outcome.error


def test_parent_binds_exact_round_robin_environment_and_default_seed(
    tmp_path, request_body, refused_pair
):
    body = request_body.model_copy(
        update={"seed": None, "heldout_env_ids": "robocasa/Heldout,robocasa/Other"}
    )
    directory = _failed_directory(tmp_path / "worker", body)
    report = diag.validate_directory(directory, requested_pairs=body.num_envs)
    service._validate_failure_request_binding(body, report)
    report["env_id"] = "robocasa/Other"
    with pytest.raises(ValueError, match="pair binding"):
        service._validate_failure_request_binding(body, report)


@pytest.mark.parametrize("retained_cleanup_fails", [False, True])
def test_worker_cleanup_failure_does_not_lose_retained_ownership(
    tmp_path, monkeypatch, request_body, refused_pair, retained_cleanup_fails
):
    directory = _failed_directory(tmp_path / "worker", request_body)
    retained = tmp_path / "retained"
    retained.mkdir()
    monkeypatch.setattr(service.tempfile, "mkdtemp", lambda **_: str(retained))
    remove = shutil.rmtree

    def cleanup(path, *args, **kwargs):
        if Path(path) == directory.parent or retained_cleanup_fails:
            raise OSError("injected owned-directory cleanup refusal")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(service.shutil, "rmtree", cleanup)
    outcome = service._retain_failure_diagnostics(
        request_body, directory.parent, service._WorkerOutcome(error="match refusal")
    )
    assert not outcome.stopped and not outcome.failure_diagnostics
    assert outcome.output_dir == (retained if retained_cleanup_fails else None)
    assert retained.exists() is retained_cleanup_fails
    assert "match refusal" in outcome.error


def test_success_provenance_binds_initial_png_bytes(tmp_path, request_body):
    for stem in ("rollout", "random_baseline"):
        directory = tmp_path / "episode_0000"
        directory.mkdir(exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            directory / f"{stem}.png"
        )
    provenance = cap._execution_provenance(request_body, tmp_path, {})
    assert provenance["png_artifacts"] == [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": cap._sha256_file(path),
        }
        for path in sorted(tmp_path.rglob("*.png"))
    ]
    assert len(provenance["png_artifacts"]) == 2
