"""Fail-closed contracts for the GR00T LIBERO-X closed-loop workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from npa.workflows import groot_libero_x as workflow
from npa.workflows import groot_libero_x_native as native
from npa.workflows.groot_visualization import _split_s3


class _Body:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}

    def seed(
        self, uri: str, body: bytes, content_type: str = "application/json"
    ) -> None:
        ref = _split_s3(uri)
        self.objects[(ref.bucket, ref.key)] = (body, content_type)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": _Body(self.objects[(Bucket, Key)][0])}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str = "application/json",
    ) -> None:
        self.objects[(Bucket, Key)] = (bytes(Body), ContentType)

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body, content_type = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(body),
            "ContentType": content_type,
            "ETag": '"fixture"',
        }

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, **_kwargs: Any
    ) -> dict[str, Any]:
        return {
            "Contents": [
                {"Key": key, "Size": len(body), "ETag": '"fixture"'}
                for (bucket, key), (body, _content_type) in sorted(self.objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)][0])


def _put_json(client: FakeS3, uri: str, payload: dict[str, Any]) -> None:
    client.seed(uri, json.dumps(payload, sort_keys=True).encode())


def _native_rollout_mp4(tmp_path: Path) -> bytes:
    """Encode a minimal, independently decodable native-rollout fixture."""

    av = pytest.importorskip("av")
    path = tmp_path / "rollout.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for level in (0, 96):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), level, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes()


def _task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "env_name": f"libero_x/{task_id}",
        "libero_x_bddl_path": (
            "libero/libero_x/bddl/LEVEL1/"
            "EXTENSION_KITCHEN_SCENE1_LEVEL1__T001_place_the_green_bowl_on_the_plate.bddl"
        ),
        "libero_x_bddl_sha256": "a" * 64,
        "language": "place the green bowl on the plate",
        "trajectory_ids": [0],
    }


def _prepare(client: FakeS3, run_id: str = "groot-libero-x-fixture") -> dict[str, Any]:
    training_uri = "s3://bucket/run/inputs/training.json"
    evaluation_uri = "s3://bucket/run/inputs/evaluation.json"
    dataset_uri = "s3://bucket/run/inputs/dataset.json"
    output_uri = "s3://bucket/run/prepared/protocol.json"
    training = {
        "schema": workflow.TRAINING_TASKS_SCHEMA,
        "tasks": [{"task_id": f"training-{index:02d}"} for index in range(60)],
    }
    evaluation = {
        "schema": workflow.EVALUATION_TASKS_SCHEMA,
        "tasks": [_task("heldout-0")],
    }
    _put_json(client, training_uri, training)
    _put_json(client, evaluation_uri, evaluation)
    client.seed(
        "s3://bucket/run/evaluation-data/meta/info.json", b"real-format-metadata"
    )
    inventory = workflow._object_inventory(client, "s3://bucket/run/evaluation-data/")
    _put_json(
        client,
        dataset_uri,
        {
            "schema": workflow.DATASET_SCHEMA,
            "source": {
                "repo": workflow.LIBERO_X_REPO,
                "revision": workflow.LIBERO_X_REVISION,
                "license": "CC-BY-4.0",
            },
            "evaluation_task_manifest_sha256": workflow._json_hash(evaluation),
            "materialized": {
                "uri": "s3://bucket/run/evaluation-data/",
                "object_inventory_sha256": inventory["sha256"],
            },
        },
    )
    return workflow.prepare_evaluation(
        training_uri, evaluation_uri, dataset_uri, output_uri, run_id, s3_client=client
    )


def _rollout(
    run_id: str,
    protocol: dict[str, Any],
    policy: str,
    video_uri: str,
    video_sha256: str = "a" * 64,
) -> dict[str, Any]:
    return {
        "schema": workflow.ROLLOUT_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "policy": policy,
        "protocol_sha256": protocol["protocol_sha256"],
        "rollout_protocol": protocol["rollout_protocol"],
        "execution_input_sha256": workflow._json_hash(
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "rollout_protocol": protocol["rollout_protocol"],
            }
        ),
        "evaluation_mode": protocol["evaluation_mode"],
        "claims": protocol["claims"],
        "closed_loop_verified": True,
        "open_loop_action_error": {
            "mse": 0.1 if policy == "baseline" else 0.05,
            "mae": 0.2 if policy == "baseline" else 0.1,
            "samples": 8,
            "forward_calls": 1,
        },
        "closed_loop": {
            "completed_episodes": 2,
            "successful_episodes": 1 if policy == "baseline" else 2,
            "success_rate": 0.5 if policy == "baseline" else 1.0,
            "tasks": [
                {
                    "task_id": "heldout-0",
                    "success_rate": 0.5 if policy == "baseline" else 1.0,
                    "videos": [{"uri": video_uri, "bytes": 9, "sha256": video_sha256}],
                }
            ],
        },
    }


def test_prepare_requires_exact_60_training_tasks_and_disjoint_evaluation() -> None:
    client = FakeS3()
    protocol = _prepare(client)

    assert protocol["task_disjoint"] is True
    assert protocol["claims"]["labels"]["held_out"] is True


def test_observed_paired_protocol_refuses_held_out_or_generalization_labels() -> None:
    client = FakeS3()
    observed_uri = "s3://bucket/run/inputs/observed.json"
    dataset_uri = "s3://bucket/run/inputs/dataset.json"
    output_uri = "s3://bucket/run/prepared/observed-protocol.json"
    observed = {"schema": workflow.OBSERVED_TASKS_SCHEMA, "tasks": [_task("seen-0")]}
    _put_json(client, observed_uri, observed)
    client.seed(
        "s3://bucket/run/evaluation-data/meta/info.json", b"real-format-metadata"
    )
    inventory = workflow._object_inventory(client, "s3://bucket/run/evaluation-data/")
    _put_json(
        client,
        dataset_uri,
        {
            "schema": workflow.DATASET_SCHEMA,
            "source": {
                "repo": workflow.LIBERO_X_REPO,
                "revision": workflow.LIBERO_X_REVISION,
                "license": "CC-BY-4.0",
            },
            "evaluation_task_manifest_sha256": workflow._json_hash(observed),
            "materialized": {
                "uri": "s3://bucket/run/evaluation-data/",
                "object_inventory_sha256": inventory["sha256"],
            },
        },
    )
    protocol = workflow.prepare_observed_paired_evaluation(
        observed_uri, dataset_uri, output_uri, "observed-run", s3_client=client
    )

    assert protocol["evaluation_mode"] == workflow.OBSERVED_PAIRED_MODE
    assert protocol["claims"]["training_coverage"] == "unknown"
    assert protocol["claims"]["task_disjointness"] == "unverified"
    assert protocol["claims"]["labels"] == {"held_out": False, "generalization": False}
    assert "training_task_manifest" not in protocol
    assert protocol["rollout_protocol"]["seed"] == 20261002
    assert protocol["rollout_protocol"]["initial_state_protocol"].startswith(
        "Isaac-GR00T reset seed batches"
    )


def test_observed_paired_tasks_require_direct_bddl_provenance() -> None:
    observed = {
        "schema": workflow.OBSERVED_TASKS_SCHEMA,
        "tasks": [
            {key: value for key, value in _task("seen-0").items() if key != "language"}
        ],
    }

    with pytest.raises(
        workflow.GrootVisualizationError, match="missing required field"
    ):
        workflow._task_rows(
            observed, training=False, schema=workflow.OBSERVED_TASKS_SCHEMA
        )


def test_observed_paired_comparison_rejects_mislabeled_rollout_claims() -> None:
    client = FakeS3()
    protocol = _prepare(client)
    protocol["evaluation_mode"] = workflow.OBSERVED_PAIRED_MODE
    protocol.pop("task_disjoint")
    protocol["claims"] = workflow._protocol_claims(workflow.OBSERVED_PAIRED_MODE)
    protocol["protocol_sha256"] = workflow._json_hash(
        {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    )
    _put_json(client, "s3://bucket/run/prepared/protocol.json", protocol)
    baseline = _rollout(
        "groot-libero-x-fixture", protocol, "baseline", "s3://bucket/a.mp4"
    )
    derivative = _rollout(
        "groot-libero-x-fixture", protocol, "derivative", "s3://bucket/b.mp4"
    )
    derivative["claims"] = workflow._protocol_claims(workflow.STRICT_HELD_OUT_MODE)
    _put_json(client, "s3://bucket/run/rollouts/baseline/report.json", baseline)
    _put_json(client, "s3://bucket/run/rollouts/derivative/report.json", derivative)

    with pytest.raises(workflow.GrootVisualizationError, match="claim classification"):
        workflow.compare_closed_loop(
            "s3://bucket/run/prepared/protocol.json",
            "s3://bucket/run/rollouts/baseline/report.json",
            "s3://bucket/run/rollouts/derivative/report.json",
            "s3://bucket/run/reports/comparison.json",
            "groot-libero-x-fixture",
            s3_client=client,
        )
    assert protocol["training_task_count"] == 60
    assert protocol["evaluation_task_count"] == 1
    assert protocol["dataset"]["source"]["repo"] == workflow.LIBERO_X_REPO

    bad_training = {
        "schema": workflow.TRAINING_TASKS_SCHEMA,
        "tasks": [{"task_id": "heldout-0"}] * 60,
    }
    _put_json(client, "s3://bucket/run/inputs/training.json", bad_training)
    with pytest.raises(workflow.GrootVisualizationError, match="duplicate task_id"):
        workflow.prepare_evaluation(
            "s3://bucket/run/inputs/training.json",
            "s3://bucket/run/inputs/evaluation.json",
            "s3://bucket/run/inputs/dataset.json",
            "s3://bucket/run/prepared/protocol-two.json",
            "groot-libero-x-fixture",
            s3_client=client,
        )


def test_prepare_rejects_task_identifiers_that_could_escape_artifact_paths() -> None:
    client = FakeS3()
    training_uri = "s3://bucket/run/inputs/training.json"
    evaluation_uri = "s3://bucket/run/inputs/evaluation.json"
    dataset_uri = "s3://bucket/run/inputs/dataset.json"
    training = {
        "schema": workflow.TRAINING_TASKS_SCHEMA,
        "tasks": [{"task_id": f"training-{index:02d}"} for index in range(60)],
    }
    evaluation = {
        "schema": workflow.EVALUATION_TASKS_SCHEMA,
        "tasks": [{**_task("heldout-0"), "task_id": "../../outside"}],
    }
    _put_json(client, training_uri, training)
    _put_json(client, evaluation_uri, evaluation)
    _put_json(
        client,
        dataset_uri,
        {
            "schema": workflow.DATASET_SCHEMA,
            "source": {
                "repo": workflow.LIBERO_X_REPO,
                "revision": workflow.LIBERO_X_REVISION,
                "license": "CC-BY-4.0",
            },
            "evaluation_task_manifest_sha256": workflow._json_hash(evaluation),
            "materialized": {
                "uri": "s3://bucket/run/evaluation-data/",
                "object_inventory_sha256": "0" * 64,
            },
        },
    )

    with pytest.raises(
        workflow.GrootVisualizationError, match="safe logical identifier"
    ):
        workflow.prepare_evaluation(
            training_uri,
            evaluation_uri,
            dataset_uri,
            "s3://bucket/run/prepared/protocol.json",
            "groot-libero-x-fixture",
            s3_client=client,
        )


@pytest.mark.parametrize(
    "path",
    [
        "../libero/libero_x/bddl/LEVEL1/task.bddl",
        "libero/libero_x/bddl/LEVEL5/task.bddl",
        "libero/libero_x/bddl/LEVEL1/../task.bddl",
        "libero/libero_x/bddl/LEVEL1/task.txt",
    ],
)
def test_prepare_rejects_unsafe_or_unsupported_libero_x_bddl_paths(path: str) -> None:
    client = FakeS3()
    training = {
        "schema": workflow.TRAINING_TASKS_SCHEMA,
        "tasks": [{"task_id": f"training-{index:02d}"} for index in range(60)],
    }
    evaluation = {
        "schema": workflow.EVALUATION_TASKS_SCHEMA,
        "tasks": [{**_task("heldout-0"), "libero_x_bddl_path": path}],
    }
    _put_json(client, "s3://bucket/run/inputs/training.json", training)
    _put_json(client, "s3://bucket/run/inputs/evaluation.json", evaluation)
    with pytest.raises(workflow.GrootVisualizationError, match="unsafe LIBERO-X BDDL"):
        workflow._task_rows(evaluation, training=False)


def test_native_libero_x_task_contract_is_hash_bound_and_source_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = _task("heldout-0")
    assert native._validate_libero_x_task(task) == workflow._native_libero_x_env_name(
        "heldout-0"
    )
    with pytest.raises(RuntimeError, match="unsafe"):
        native._validate_libero_x_task(
            {**task, "libero_x_bddl_path": "libero/libero_x/bddl/LEVEL5/task.bddl"}
        )

    source = tmp_path / "LIBERO-X"
    (source / "libero/libero_x/bddl").mkdir(parents=True)
    monkeypatch.setattr(native.sys, "path", list(native.sys.path))
    assert native._configure_libero_x_source({"libero_x_source": str(source)}) == source
    assert str(source / "libero") == native.sys.path[0]


def test_native_rollout_uses_fixed_seeded_episode_waves(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_rollout(**kwargs: Any) -> tuple[str, list[bool], dict[str, Any]]:
        calls.append(kwargs)
        return (
            kwargs["env_name"],
            [True] * kwargs["n_episodes"],
            {
                "episode_lengths": [4] * kwargs["n_episodes"],
                "episode_rewards": [1.0] * kwargs["n_episodes"],
            },
        )

    result = native._run_seeded_episode_waves(
        native_env_name="libero_sim/npa_groot_libero_x_0123456789abcdef0123456789abcdef",
        rollout_runner=fake_rollout,
        policy_host="127.0.0.1",
        policy_port=1,
        video_dir=tmp_path / "videos",
        task_index=2,
        episodes_per_task=10,
        n_envs=5,
        max_episode_steps=720,
        n_action_steps=8,
        seed=17,
    )

    assert [call["seed"] for call in calls] == [37, 42]
    assert [call["n_episodes"] for call in calls] == [5, 5]
    assert result["initial_reset_seed_batches"] == [
        [37, 38, 39, 40, 41],
        [42, 43, 44, 45, 46],
    ]
    assert result["completed_episodes"] == 10


def test_native_evaluator_writes_worker_registration_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "LIBERO-X"
    source.mkdir()
    captured: dict[str, Any] = {}

    def fake_runtime_command(
        command: list[str], *, cwd: Path, env: dict[str, str]
    ) -> None:
        captured["environment"] = env
        config = json.loads(Path(command[3]).read_text())
        captured["config"] = config
        Path(command[-1]).write_text(json.dumps({"tasks": []}))

    monkeypatch.setattr(workflow, "_run_runtime_command", fake_runtime_command)
    result = workflow._run_native_evaluator(
        runtime={
            "server_python": tmp_path / "server-python",
            "source": tmp_path / "Isaac-GR00T",
            "sim_python": tmp_path / "sim-python",
            "home": tmp_path / "home",
            "libero_x_source": source,
        },
        root=tmp_path,
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset",
        tasks=[_task("heldout-0")],
        episodes_per_task=1,
        n_envs=1,
        max_episode_steps=1,
        n_action_steps=1,
        seed=7,
    )

    assert result == {"tasks": []}
    registration = json.loads(
        Path(
            captured["environment"]["NPA_GROOT_LIBERO_X_REGISTRATION_MANIFEST"]
        ).read_text()
    )
    assert registration["tasks"] == [
        {
            "task_id": "heldout-0",
            "native_env_name": workflow._native_libero_x_env_name("heldout-0"),
            "libero_x_bddl_path": _task("heldout-0")["libero_x_bddl_path"],
            "libero_x_bddl_sha256": _task("heldout-0")["libero_x_bddl_sha256"],
            "language": _task("heldout-0")["language"],
        }
    ]
    assert captured["config"]["libero_x_source"] == str(source)


def test_runtime_overlay_and_git_fetch_contract_keep_lfs_bytes_out_of_cache(
    tmp_path: Path,
) -> None:
    command = workflow._git_without_lfs("git", "checkout", "--detach", "revision")
    assert command[:7] == [
        "git",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
    ]
    target = tmp_path / "Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py"
    target.parent.mkdir(parents=True)
    target.write_text("def register_libero_envs():\n    pass\n")
    overlay = workflow._overlay_libero_env(tmp_path / "Isaac-GR00T")
    assert (
        overlay["sha256"]
        == hashlib.sha256(workflow.LIBERO_X_RUNTIME_OVERLAY.encode()).hexdigest()
    )
    assert target.read_text().endswith(workflow.LIBERO_X_RUNTIME_OVERLAY + "\n")


def test_compare_and_rrd_evidence_decode_matched_native_rollout_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rerun")
    client = FakeS3()
    run_id = "groot-libero-x-fixture"
    protocol = _prepare(client, run_id)
    baseline_video = "s3://bucket/run/rollouts/baseline/mp4/episode.mp4"
    derivative_video = "s3://bucket/run/rollouts/derivative/mp4/episode.mp4"
    video_bytes = _native_rollout_mp4(tmp_path)
    client.seed(baseline_video, video_bytes, "video/mp4")
    client.seed(derivative_video, video_bytes, "video/mp4")
    baseline_uri = "s3://bucket/run/rollouts/baseline/report.json"
    derivative_uri = "s3://bucket/run/rollouts/derivative/report.json"
    video_sha256 = hashlib.sha256(video_bytes).hexdigest()
    _put_json(
        client,
        baseline_uri,
        _rollout(run_id, protocol, "baseline", baseline_video, video_sha256),
    )
    _put_json(
        client,
        derivative_uri,
        _rollout(run_id, protocol, "derivative", derivative_video, video_sha256),
    )
    comparison_uri = "s3://bucket/run/reports/comparison.json"

    comparison = workflow.compare_closed_loop(
        "s3://bucket/run/prepared/protocol.json",
        baseline_uri,
        derivative_uri,
        comparison_uri,
        run_id,
        s3_client=client,
    )
    assert comparison["closed_loop_success"]["derivative_minus_baseline"] == 0.5
    assert (
        comparison["open_loop_action_error"]["derivative_minus_baseline"]["mse"]
        == -0.05
    )

    evidence = workflow.emit_evidence(
        comparison_uri,
        baseline_uri,
        derivative_uri,
        "s3://bucket/run/reports/evidence.rrd",
        "s3://bucket/run/reports/evidence.json",
        run_id,
        s3_client=client,
    )
    assert evidence["rrd"]["bytes"] > 0
    assert evidence["mp4_count"] == 2
    assert all(
        item["frame_count"] == 2 for item in evidence["native_rollout_mp4_inspection"]
    )


def test_comparison_rejects_a_rollout_without_closed_loop_evidence() -> None:
    client = FakeS3()
    protocol = _prepare(client)
    baseline_uri = "s3://bucket/run/rollouts/baseline/report.json"
    derivative_uri = "s3://bucket/run/rollouts/derivative/report.json"
    report = _rollout(
        "groot-libero-x-fixture", protocol, "baseline", "s3://bucket/a.mp4"
    )
    report["closed_loop_verified"] = False
    _put_json(client, baseline_uri, report)
    _put_json(
        client,
        derivative_uri,
        _rollout("groot-libero-x-fixture", protocol, "derivative", "s3://bucket/b.mp4"),
    )

    with pytest.raises(workflow.GrootVisualizationError, match="closed-loop"):
        workflow.compare_closed_loop(
            "s3://bucket/run/prepared/protocol.json",
            baseline_uri,
            derivative_uri,
            "s3://bucket/run/reports/comparison.json",
            "groot-libero-x-fixture",
            s3_client=client,
        )


def test_evidence_rejects_an_undecodable_rollout_mp4() -> None:
    pytest.importorskip("rerun")
    client = FakeS3()
    run_id = "groot-libero-x-fixture"
    protocol = _prepare(client, run_id)
    baseline_uri = "s3://bucket/run/rollouts/baseline/report.json"
    derivative_uri = "s3://bucket/run/rollouts/derivative/report.json"
    baseline_video = "s3://bucket/run/rollouts/baseline/mp4/episode.mp4"
    derivative_video = "s3://bucket/run/rollouts/derivative/mp4/episode.mp4"
    bad_video = b"not-a-decodable-mp4"
    video_sha256 = hashlib.sha256(bad_video).hexdigest()
    client.seed(baseline_video, bad_video, "video/mp4")
    client.seed(derivative_video, bad_video, "video/mp4")
    _put_json(
        client,
        baseline_uri,
        _rollout(run_id, protocol, "baseline", baseline_video, video_sha256),
    )
    _put_json(
        client,
        derivative_uri,
        _rollout(run_id, protocol, "derivative", derivative_video, video_sha256),
    )
    comparison_uri = "s3://bucket/run/reports/comparison.json"
    workflow.compare_closed_loop(
        "s3://bucket/run/prepared/protocol.json",
        baseline_uri,
        derivative_uri,
        comparison_uri,
        run_id,
        s3_client=client,
    )

    with pytest.raises(workflow.GrootVisualizationError, match="MP4 is unreadable"):
        workflow.emit_evidence(
            comparison_uri,
            baseline_uri,
            derivative_uri,
            "s3://bucket/run/reports/evidence.rrd",
            "s3://bucket/run/reports/evidence.json",
            run_id,
            s3_client=client,
        )


def test_policy_stage_requires_native_results_and_uploads_actual_mp4s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeS3()
    run_id = "groot-libero-x-fixture"
    _prepare(client, run_id)
    model = tmp_path / "model"
    model.mkdir()

    monkeypatch.setattr(
        workflow,
        "_snapshot_model",
        lambda *_args: (
            model,
            {"repo": workflow.BASELINE_REPO, "revision": workflow.BASELINE_REVISION},
        ),
    )
    monkeypatch.setattr(workflow, "_download_prefix", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_materialize_native_runtime",
        lambda _root: {
            "provenance": {
                "delivery": "runtime-fetch",
                "isaac_groot_revision": workflow.IMAGE_GROOT_REF,
            }
        },
    )

    def native_result(**kwargs: Any) -> dict[str, Any]:
        video_dir = kwargs["root"] / "videos" / "heldout-0"
        video_dir.mkdir(parents=True)
        (video_dir / "rollout.mp4").write_bytes(b"actual-native-rollout")
        return {
            "open_loop_action_error": {
                "mse": 0.12,
                "mae": 0.34,
                "sample_count": 4,
                "forward_calls": 1,
            },
            "tasks": [
                {
                    "task_id": "heldout-0",
                    "env_name": _task("heldout-0")["env_name"],
                    "trajectory_ids": [0],
                    "native_env_name": _task("heldout-0")["env_name"],
                    "successes": [True],
                    "episode_lengths": [12],
                    "episode_rewards": [1.0],
                    "completed_episodes": 1,
                    "success_rate": 1.0,
                    "video_dir": str(video_dir),
                }
            ],
        }

    monkeypatch.setattr(workflow, "_run_native_evaluator", native_result)
    report = workflow.run_policy(
        "s3://bucket/run/prepared/protocol.json",
        "s3://bucket/run/rollouts/baseline/report.json",
        "s3://bucket/run/rollouts/baseline/mp4",
        run_id,
        policy_name="baseline",
        model_repo=workflow.BASELINE_REPO,
        model_revision=workflow.BASELINE_REVISION,
        model_subdir="libero_10",
        episodes_per_task=10,
        n_envs=5,
        max_episode_steps=720,
        n_action_steps=8,
        seed=20261002,
        s3_client=client,
    )

    assert report["closed_loop_verified"] is True
    assert report["runtime"]["native_runtime"]["delivery"] == "runtime-fetch"
    assert report["closed_loop"]["success_rate"] == 1.0
    video = report["closed_loop"]["tasks"][0]["videos"][0]
    assert video["bytes"] == len(b"actual-native-rollout")
    assert client.head_object(
        Bucket="bucket", Key="run/rollouts/baseline/mp4/heldout-0/rollout.mp4"
    )


def test_numeric_metrics_rejects_empty_or_nonfinite_native_measurements() -> None:
    with pytest.raises(workflow.GrootVisualizationError, match="at least one sample"):
        workflow._numeric_metrics(
            {"mse": 0.0, "mae": 0.0, "sample_count": 0, "forward_calls": 0}
        )
    with pytest.raises(workflow.GrootVisualizationError, match="finite"):
        workflow._numeric_metrics(
            {"mse": float("nan"), "mae": 0.0, "sample_count": 1, "forward_calls": 1}
        )
