"""Reject invalid robot-learning data and verify S3 transfers using real streaming bodies."""

from copy import deepcopy
from io import BytesIO
import hashlib
import sys
from types import SimpleNamespace

from botocore.response import StreamingBody
import numpy as np
import pytest

from npa.workflows.xr1_antioch.collection import _outcome
from npa.workflows.xr1_antioch.dataset import (
    ACTION_WIDTHS,
    CAMERAS,
    STATE_WIDTHS,
    native_annotation,
    validate_episode,
    validate_splits,
)
from npa.workflows.xr1_antioch.transport import _destination, _readback


@pytest.mark.parametrize(
    "interrupted, remote_exit, expected", [(False, 0, 0), (False, 7, 7), (True, 0, 130)]
)
def test_attached_evaluation_has_no_deadline_and_preserves_exit(
    monkeypatch, interrupted, remote_exit, expected
):
    from npa.workflows.xr1_antioch import attached_exec

    monkeypatch.setattr(attached_exec, "version", lambda name: "0.4.236")
    calls = []

    def execute(session, service, command, **options):
        calls.append((session, service, command, options))
        return interrupted, remote_exit

    monkeypatch.setitem(
        sys.modules,
        "antioch.cli.commands.service",
        SimpleNamespace(
            run_door_session=lambda profiles: "owned-session",
            default_service=lambda session, service, **options: "simulator",
            run_service_command=execute,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "antioch.cli.options",
        SimpleNamespace(StreamMode=SimpleNamespace(OFF="off")),
    )
    argv = ["python", "-c", "print('literal $(not-a-shell)')"]
    assert attached_exec.run(argv) == expected
    assert calls == [
        (
            "owned-session",
            "simulator",
            argv,
            {"stream": "off", "timeout_s": None, "tty": False},
        )
    ]


def test_attached_evaluation_rejects_empty_command_and_unverified_sdk(monkeypatch):
    from npa.workflows.xr1_antioch import attached_exec

    with pytest.raises(ValueError, match="command"):
        attached_exec.run([])
    monkeypatch.setattr(attached_exec, "version", lambda name: "different-version")
    with pytest.raises(ValueError, match="0.4.236"):
        attached_exec.run(["python", "-c", "print(1)"])


@pytest.fixture
def episode():
    def arrays(widths):
        result = {
            name: np.zeros((31, width)).tolist() for name, width in widths.items()
        }
        for name in ("left_ee_rotm", "right_ee_rotm"):
            result[name] = np.tile(np.eye(3).reshape(1, 9), (31, 1)).tolist()
        return result

    return {
        "episode_id": "train-1",
        "seed": 1,
        "success": True,
        "num_frames": 31,
        "control_hz": 20,
        "timestamps": (np.arange(31) / 20).tolist(),
        "grasp_mechanism": "finger_contact",
        "proprios": arrays(STATE_WIDTHS),
        "actions": arrays(ACTION_WIDTHS),
        "videos": {name: name + ".mp4" for name in CAMERAS},
    }


@pytest.mark.parametrize(
    "fault", ["nan", "shape", "missing", "reflection", "clock", "attachment"]
)
def test_reject_invalid_robot_recordings(episode, fault):
    if fault == "nan":
        episode["actions"]["left_ee_pos"][0][0] = float("nan")
    elif fault == "shape":
        episode["proprios"]["right_arm_joint"][0].pop()
    elif fault == "missing":
        del episode["actions"]["base_vel"]
    elif fault == "reflection":
        episode["actions"]["left_ee_rotm"][0][0] = -1
    elif fault == "clock":
        episode["timestamps"][12] += 0.05
    else:
        episode["grasp_mechanism"] = "fixed_joint"
    with pytest.raises(ValueError):
        validate_episode(episode)


def test_native_annotation_preserves_camera_order_and_issued_actions(episode, tmp_path):
    for name in CAMERAS:
        (tmp_path / f"{name}.mp4").touch()
    episode["actions"]["left_ee_pos"][0] = [0.4, 0.3, 0.2]
    result = native_annotation(episode, tmp_path)
    prompt = result["instruction"]["general"][0]
    assert prompt["images"] == [f"observations.{name}" for name in CAMERAS]
    assert prompt["conversations"][0]["value"].count("<image>") == 3
    assert result["actions"]["left_ee_pos"][0] == [0.4, 0.3, 0.2]
    assert result["proprios"]["left_ee_pos"][0] == [0.0, 0.0, 0.0]


def test_failed_demonstrations_cannot_enter_behavior_cloning(episode, tmp_path):
    episode["success"] = False
    with pytest.raises(ValueError, match="Failed demonstrations"):
        native_annotation(episode, tmp_path)


def test_video_cannot_escape_episode_directory(episode, tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.touch()
    inside = tmp_path / "episode"
    inside.mkdir()
    episode["videos"]["ego"] = "../outside.mp4"
    with pytest.raises(ValueError, match="out-of-root"):
        native_annotation(episode, inside)


def test_split_leakage_rejected_even_when_episode_names_differ():
    manifest = {
        "train": [{"episode_id": "train-1", "seed": 1}],
        "validation": [{"episode_id": "val-1", "seed": 2}],
        "test": [{"episode_id": "test-1", "seed": 3}],
    }
    validate_splits(manifest)
    manifest["test"][0]["seed"] = 1
    with pytest.raises(ValueError, match="disjoint"):
        validate_splits(manifest)


def test_physics_gate_requires_lift_placement_release_and_stability():
    row = {
        "positions": [[0.4, 0.2, 0.02], [0.4, -0.2, 0.02]],
        "targets": [[0.4, 0.2, 0.025], [0.4, -0.2, 0.025]],
        "velocities": [[0, 0, 0.046], [0, 0, 0.046]],
        "gripper_apertures": [0.08, 0.08],
    }
    samples = [deepcopy(row) for _ in range(40)]
    assert not _outcome(samples)["success"]
    samples[0]["positions"][0][2] = 0.2
    samples[0]["positions"][1][2] = 0.2
    assert _outcome(samples)["success"]
    samples[-5]["positions"][0][0] += 0.02
    assert not _outcome(samples)["success"]
    samples[-5] = deepcopy(row)
    samples[-1]["gripper_apertures"][0] = 0.04
    assert not _outcome(samples)["success"]


def test_full_readback_works_with_botocore_streaming_body():
    data = b"physical simulation artifact"

    class Client:
        def get_object(self, **kwargs):
            assert kwargs == {"Bucket": "example-bucket", "Key": "run/video.mp4"}
            return {"Body": StreamingBody(BytesIO(data), len(data))}

    manifest = {
        "video.mp4": {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    }
    assert _readback(Client(), "example-bucket", "run", manifest) == manifest
    manifest["video.mp4"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs"):
        _readback(Client(), "example-bucket", "run", manifest)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://example-bucket",
        "s3://example-bucket/run/../other",
        "https://example.test/run",
    ],
)
def test_s3_destination_requires_a_bounded_prefix(uri):
    with pytest.raises(ValueError):
        _destination(uri)


def _rollout(seed, checkpoint, success):
    return {
        "seed": seed,
        "checkpoint_sha256": checkpoint,
        "success": success,
        "expert_fallback": False,
        "grasp_mechanism": "finger_contact",
        "control_hz": 20,
        "replan_every_frames": 6,
        "horizon_seconds": 25,
        "statistics_sha256": "a" * 64,
        "simulation_sha256": "b" * 64,
    }


def test_paired_report_distinguishes_one_win_from_supported_improvement():
    from npa.workflows.xr1_antioch.report import compare_rollouts

    seeds = list(range(32))
    baseline = [_rollout(seed, "0" * 64, False) for seed in seeds]
    candidate = [_rollout(seed, "1" * 64, seed == 0) for seed in seeds]
    result = compare_rollouts(baseline, candidate, seeds, "0" * 64, "1" * 64)
    assert result["observed_improvement"] and not result["supported_improvement"]
    assert result["candidate"]["success_rate"] == 1 / 32
    assert result["mcnemar_exact_two_sided_p"] == 1
    for row in candidate[:6]:
        row["success"] = True
    result = compare_rollouts(baseline, candidate, seeds, "0" * 64, "1" * 64)
    assert result["supported_improvement"]
    assert result["mcnemar_exact_two_sided_p"] == 0.03125


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "expert", "normalization", "simulation", "horizon"],
)
def test_paired_report_rejects_incomplete_or_unfair_comparisons(fault):
    from npa.workflows.xr1_antioch.report import compare_rollouts

    baseline = [_rollout(seed, "0" * 64, False) for seed in range(2)]
    candidate = [_rollout(seed, "1" * 64, True) for seed in range(2)]
    if fault == "missing":
        candidate.pop()
    elif fault == "duplicate":
        candidate[1]["seed"] = 0
    elif fault == "expert":
        candidate[0]["expert_fallback"] = True
    elif fault == "normalization":
        candidate[0]["statistics_sha256"] = "c" * 64
    elif fault == "simulation":
        candidate[0]["simulation_sha256"] = "c" * 64
    else:
        candidate[0]["horizon_seconds"] = 100
    with pytest.raises(ValueError):
        compare_rollouts(baseline, candidate, [0, 1], "0" * 64, "1" * 64)


@pytest.mark.parametrize(
    "name",
    [
        "/outside/model.pt",
        "../model.pt",
        "sub/../../model.pt",
        "sub\\model.pt",
        "",
        ".",
    ],
)
def test_artifact_paths_rejected_before_local_materialization(name):
    from npa.workflows.xr1_antioch.storage import _relative

    with pytest.raises(ValueError):
        _relative(name)


def test_source_archive_cannot_write_outside_fresh_source(tmp_path):
    import zipfile
    from npa.workflows.xr1_antioch.bootstrap_worker import _extract

    with zipfile.ZipFile(tmp_path / "runtime.zip", "w") as archive:
        archive.writestr("../outside.py", "untrusted")
    with pytest.raises(ValueError):
        _extract(tmp_path, "runtime.zip")
    assert not (tmp_path / "outside.py").exists()


def test_policy_actuation_rejects_out_of_workspace_and_reflected_targets(episode):
    from npa.workflows.xr1_antioch.rollout import _target

    chunk = deepcopy(episode["actions"])
    chunk["left_ee_pos"][0] = [0.4, 0.4, 0.2]
    chunk["right_ee_pos"][0] = [0.4, -0.4, 0.2]
    _target(chunk, 0)
    chunk["left_ee_pos"][0][0] = 9
    with pytest.raises(ValueError, match="workspace"):
        _target(chunk, 0)
    chunk["left_ee_pos"][0][0] = 0.4
    chunk["left_ee_rotm"][0][0] = -1
    with pytest.raises(ValueError, match="reflection"):
        _target(chunk, 0)


def test_upstream_scientific_notation_has_native_training_numeric_types(tmp_path):
    from npa.workflows.xr1_antioch.training_data import _configuration

    model = tmp_path / "source/configs/model/posttrain.yaml"
    trainer = tmp_path / "source/configs/trainer/deepspeed.yaml"
    model.parent.mkdir(parents=True)
    trainer.parent.mkdir(parents=True)
    model.write_text("model: {params: {model: {type: xr1}}}")
    trainer.write_text("""trainer:
  max_steps: 10000
  strategy:
    params: {allgather_bucket_size: 5e8, reduce_bucket_size: 5e8}
  scheduler:
    params: {warmup_lr_start: 5e-7, max_lr: 0.00002, min_lr: 0.000005}
""")
    result = _configuration(tmp_path / "source", tmp_path, {})["trainer"]
    for value in result["strategy"]["params"].values():
        assert type(value) is int and value == 500_000_000
    rates = result["scheduler"]["params"]
    assert rates["warmup_lr_start"] == 0.0000005
    assert rates["num_training_steps"] == 10000


def test_collection_resume_repairs_interrupted_receipt_publication(tmp_path):
    import json
    from types import SimpleNamespace
    from botocore.exceptions import ClientError
    from npa.workflows.xr1_antioch.operator_data import _collect_one

    data = b"verified native robot recording"
    entry = {"episode_id": "train-1", "seed": 1}
    receipt = {
        **entry,
        "split": "train",
        "transfer": {
            "files": {
                "episode.json": {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            }
        },
    }
    local = tmp_path / "train-1"
    local.mkdir()
    (local / "receipt.json").write_text(json.dumps(receipt))
    objects = {"run/episodes/train-1/episode.json": data}

    class Client:
        def get_object(self, *, Bucket, Key):
            return {"Body": StreamingBody(BytesIO(objects[Key]), len(objects[Key]))}

        def put_object(self, *, Bucket, Key, Body, IfNoneMatch):
            if Key in objects:
                raise ClientError(
                    {
                        "Error": {"Code": "PreconditionFailed"},
                        "ResponseMetadata": {"HTTPStatusCode": 412},
                    },
                    "PutObject",
                )
            objects[Key] = Body

    args, storage = (
        SimpleNamespace(remote_root="/recordings", output_path=tmp_path),
        SimpleNamespace(s3=Client()),
    )
    for _ in range(2):
        assert (
            _collect_one(args, entry, "train", storage, "example-bucket", "run")
            == receipt
        )
        assert json.loads(objects["run/receipts/train-1.json"]) == receipt
    objects["run/receipts/train-1.json"] = b"{}"
    with pytest.raises(ValueError, match="receipt differs"):
        _collect_one(args, entry, "train", storage, "example-bucket", "run")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/object",
        "file:///example",
        "https://user:pass@example.test/object",
    ],
)
def test_signed_workers_reject_non_https_or_embedded_credentials(tmp_path, url):
    from npa.workflows.xr1_antioch.artifact_worker import _upload
    from npa.workflows.xr1_antioch.bootstrap_worker import _download

    payload = b"robot observation"
    (tmp_path / "episode.json").write_bytes(payload)
    entry = {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    with pytest.raises(ValueError, match="require HTTPS"):
        _download(tmp_path, "download.json", entry)
    with pytest.raises(ValueError, match="require HTTPS"):
        _upload(tmp_path, {"episode.json": entry})


def test_signed_download_rejects_redirect_without_following_it(tmp_path, monkeypatch):
    from npa.workflows.xr1_antioch import bootstrap_worker

    class Redirect:
        status = 307

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Connection:
        requests = []

        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, target):
            self.requests.append((method, target))

        def getresponse(self):
            return Redirect()

        def close(self):
            pass

    monkeypatch.setattr(bootstrap_worker, "HTTPSConnection", Connection)
    with pytest.raises(RuntimeError, match="did not return success"):
        bootstrap_worker._download(
            tmp_path,
            "episode.json",
            {"url": "https://example.test/object?signature=example"},
        )
    assert Connection.requests == [("GET", "/object?signature=example")]
    assert not (tmp_path / "episode.json").exists()
