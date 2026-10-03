"""Fail-closed contracts for the GR00T LIBERO-X closed-loop workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from npa.workflows import groot_libero_x as workflow
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


def _task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "env_name": "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
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
    run_id: str, protocol: dict[str, Any], policy: str, video_uri: str
) -> dict[str, Any]:
    return {
        "schema": workflow.ROLLOUT_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "policy": policy,
        "protocol_sha256": protocol["protocol_sha256"],
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
                    "videos": [{"uri": video_uri, "bytes": 9, "sha256": "a" * 64}],
                }
            ],
        },
    }


def test_prepare_requires_exact_60_training_tasks_and_disjoint_evaluation() -> None:
    client = FakeS3()
    protocol = _prepare(client)

    assert protocol["task_disjoint"] is True
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


def test_compare_and_rrd_evidence_use_matched_real_rollout_artifacts() -> None:
    pytest.importorskip("rerun")
    client = FakeS3()
    run_id = "groot-libero-x-fixture"
    protocol = _prepare(client, run_id)
    baseline_video = "s3://bucket/run/rollouts/baseline/mp4/episode.mp4"
    derivative_video = "s3://bucket/run/rollouts/derivative/mp4/episode.mp4"
    client.seed(baseline_video, b"baseline-mp4", "video/mp4")
    client.seed(derivative_video, b"derivative-mp4", "video/mp4")
    baseline_uri = "s3://bucket/run/rollouts/baseline/report.json"
    derivative_uri = "s3://bucket/run/rollouts/derivative/report.json"
    _put_json(
        client, baseline_uri, _rollout(run_id, protocol, "baseline", baseline_video)
    )
    _put_json(
        client,
        derivative_uri,
        _rollout(run_id, protocol, "derivative", derivative_video),
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
        episodes_per_task=1,
        n_envs=1,
        max_episode_steps=12,
        n_action_steps=4,
        seed=7,
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
