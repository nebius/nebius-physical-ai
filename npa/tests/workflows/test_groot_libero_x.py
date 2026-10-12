"""Fail-closed contracts for the GR00T LIBERO-X closed-loop workflow."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from npa.workflows import groot_libero_x as workflow
from npa.workflows import groot_libero_x_native as native
from npa.workflows import groot_libero_x_server as server_evidence
from npa.workflows.groot_visualization import _split_s3


class _Body:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body


class _StreamingResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.status = 200

    def __enter__(self) -> "_StreamingResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            return self.body
        result, self.body = self.body[:size], self.body[size:]
        return result


def _patch_lfs_https_transport(
    monkeypatch: pytest.MonkeyPatch, responses: list[object]
) -> tuple[
    list[tuple[str, float]],
    list[tuple[str, str, str, bytes | None, dict[str, str]]],
]:
    connections: list[tuple[str, float]] = []
    requests: list[tuple[str, str, str, bytes | None, dict[str, str]]] = []

    class Connection:
        def __init__(
            self, host: str, *, port: int, timeout: float, context: Any
        ) -> None:
            assert port == 443
            assert context.check_hostname is True
            self.host = host
            connections.append((host, timeout))

        def request(
            self,
            method: str,
            target: str,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            requests.append((self.host, method, target, body, headers or {}))

        def getresponse(self) -> _StreamingResponse:
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            assert isinstance(response, _StreamingResponse)
            return response

        def close(self) -> None:
            return None

    monkeypatch.setattr(workflow.http.client, "HTTPSConnection", Connection)
    return connections, requests


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


def _episode_parquet(episode_index: int, *, rows: int = 2) -> bytes:
    """Create a real v2-style parquet identity column for a production seam."""

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    output = pa.BufferOutputStream()
    pq.write_table(pa.table({"episode_index": [episode_index] * rows}), output)
    return output.getvalue().to_pybytes()


def _prepare_mapped_observed_pair(
    client: FakeS3,
    *,
    task_source_episode: int = 24,
    parquet_episode: int = 24,
) -> dict[str, Any]:
    """Prepare a v2 observed pair with explicit workflow/source identities."""

    observed_uri = "s3://bucket/run/inputs/observed.json"
    dataset_uri = "s3://bucket/run/inputs/dataset.json"
    output_uri = "s3://bucket/run/prepared/observed-protocol.json"
    task = _task("observed-0")
    task["reference_observation"] = {"source_episode_index": task_source_episode}
    observed = {"schema": workflow.OBSERVED_TASKS_SCHEMA, "tasks": [task]}
    _put_json(client, observed_uri, observed)
    _put_json(
        client,
        "s3://bucket/run/evaluation-data/meta/info.json",
        {"chunks_size": 1_000},
    )
    client.seed(
        "s3://bucket/run/evaluation-data/data/chunk-000/episode_000024.parquet",
        _episode_parquet(parquet_episode),
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
                "episode_index_mapping": {
                    "workflow_trajectory_id": 0,
                    "source_episode_index": 24,
                },
            },
        },
    )
    return workflow.prepare_observed_paired_evaluation(
        observed_uri, dataset_uri, output_uri, "observed-run", s3_client=client
    )


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
    _put_json(
        client,
        "s3://bucket/run/evaluation-data/meta/info.json",
        {"chunks_size": 1_000},
    )
    client.seed(
        "s3://bucket/run/evaluation-data/data/chunk-000/episode_000000.parquet",
        b"real-format-parquet",
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


def _execute_baseline_policy(
    client: FakeS3, *, phase: dict[str, str]
) -> dict[str, Any]:
    """Exercise the real S3 download boundary before model loading."""

    return workflow._execute_policy(
        client,
        "s3://bucket/run/prepared/protocol.json",
        "s3://bucket/run/rollouts/baseline/report.json",
        "s3://bucket/run/rollouts/baseline/mp4",
        "groot-libero-x-fixture",
        policy_name="baseline",
        model_repo=workflow.BASELINE_REPO,
        model_revision=workflow.BASELINE_REVISION,
        model_subdir="libero_10",
        episodes_per_task=10,
        n_envs=5,
        max_episode_steps=720,
        n_action_steps=8,
        seed=20261002,
        phase=phase,
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


def _native_worker_provenance() -> dict[str, Any]:
    """Return a complete synthetic worker record for parent-contract tests only."""

    return {
        "schema": "npa.groot_libero_x.native_worker_provenance.v1",
        "executed_native_entrypoint": {
            "module": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
            "path": "/private/runtime/gr00t/eval/rollout_policy.py",
            "sha256": "d" * 64,
        },
        "input_provenance": {
            "protocol_sha256": "a" * 64,
            "evaluation_tasks_sha256": "b" * 64,
            "dataset_object_inventory_sha256": "c" * 64,
            "dataset_content_inventory_sha256": "d" * 64,
            "rollout_protocol": {"initial_state_protocol": "deterministic reset seeds"},
        },
        "policy_server": {
            "model_load_completed": True,
            "model": {
                "repo": workflow.BASELINE_REPO,
                "revision": workflow.BASELINE_REVISION,
                "checkpoint_files": [
                    {"path": "model.safetensors", "sha256": "e" * 64, "bytes": 1}
                ],
            },
            "device": {
                "cuda_available": True,
                "cuda_runtime": "test-cuda",
                "torch_version": "test-torch",
                "device_count": 1,
                "selected_device_index": 0,
                "selected_device_name": "mocked-test-gpu",
                "selected_device_capability": [12, 0],
                "selected_model_tensor_device": "cuda:0",
            },
            "imported_modules": [
                {
                    "module": "gr00t.policy.gr00t_policy",
                    "path": "/private/runtime/gr00t/policy/gr00t_policy.py",
                    "sha256": "f" * 64,
                }
            ],
        },
    }


def test_prepare_requires_exact_60_training_tasks_and_disjoint_evaluation() -> None:
    client = FakeS3()
    protocol = _prepare(client)

    assert protocol["task_disjoint"] is True
    assert protocol["claims"]["labels"]["held_out"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "changed-bytes",
        "missing-object",
        "additional-object",
    ),
)
def test_policy_refuses_downloaded_dataset_mutations_before_model_execution(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """The actual download seam must bind every consumed byte before snapshots."""

    client = FakeS3()
    _prepare(client)
    parquet_key = "run/evaluation-data/data/chunk-000/episode_000000.parquet"
    if mutation == "changed-bytes":
        original, content_type = client.objects[("bucket", parquet_key)]
        client.objects[("bucket", parquet_key)] = (b"x" * len(original), content_type)
    elif mutation == "missing-object":
        del client.objects[("bucket", parquet_key)]
    else:
        client.seed("s3://bucket/run/evaluation-data/data/unexpected.bin", b"extra")

    snapshots: list[tuple[Any, ...]] = []

    def snapshot(*args: Any) -> tuple[Path, dict[str, Any]]:
        snapshots.append(args)
        pytest.fail("model snapshot must not execute before dataset content validation")

    monkeypatch.setattr(workflow, "_snapshot_model", snapshot)
    phase: dict[str, str] = {}
    with pytest.raises(
        workflow.GrootVisualizationError,
        match="downloaded evaluation dataset content differs from prepared inventory",
    ):
        _execute_baseline_policy(client, phase=phase)

    assert snapshots == []
    assert phase == {"name": "validate_downloaded_evaluation_dataset"}


def test_policy_accepts_exact_downloaded_dataset_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unchanged FakeS3 download reaches model loading only after validation."""

    client = FakeS3()
    protocol = _prepare(client)
    snapshots: list[tuple[Any, ...]] = []

    def snapshot(*args: Any) -> tuple[Path, dict[str, Any]]:
        snapshots.append(args)
        raise RuntimeError("model snapshot reached after exact dataset validation")

    monkeypatch.setattr(workflow, "_snapshot_model", snapshot)
    phase: dict[str, str] = {}
    with pytest.raises(RuntimeError, match="after exact dataset validation"):
        _execute_baseline_policy(client, phase=phase)

    assert snapshots == [
        (workflow.BASELINE_REPO, workflow.BASELINE_REVISION, "libero_10")
    ]
    assert phase == {"name": "snapshot_model"}
    assert protocol["dataset"]["materialized"]["content"]["sha256"] == (
        workflow._json_hash(
            {"objects": protocol["dataset"]["materialized"]["content"]["objects"]}
        )
    )


def test_observed_paired_protocol_refuses_held_out_or_generalization_labels() -> None:
    client = FakeS3()
    observed_uri = "s3://bucket/run/inputs/observed.json"
    dataset_uri = "s3://bucket/run/inputs/dataset.json"
    output_uri = "s3://bucket/run/prepared/observed-protocol.json"
    observed = {"schema": workflow.OBSERVED_TASKS_SCHEMA, "tasks": [_task("seen-0")]}
    _put_json(client, observed_uri, observed)
    _put_json(
        client,
        "s3://bucket/run/evaluation-data/meta/info.json",
        {"chunks_size": 1_000},
    )
    client.seed(
        "s3://bucket/run/evaluation-data/data/chunk-000/episode_000000.parquet",
        b"real-format-parquet",
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


def test_observed_pair_binds_explicit_v2_source_episode_identity() -> None:
    """A logical trajectory may use only its explicit source-episode mapping."""

    client = FakeS3()

    protocol = _prepare_mapped_observed_pair(client)

    assert protocol["evaluation_tasks"][0]["trajectory_ids"] == [0]
    assert protocol["dataset"]["materialized"]["episode_index_mapping"] == {
        "workflow_trajectory_id": 0,
        "source_episode_index": 24,
    }
    assert native._source_episode_indices(
        protocol["evaluation_tasks"][0], workflow._native_input_provenance(protocol)
    ) == [24]


def test_observed_pair_refuses_conflicting_mapped_parquet_identity() -> None:
    """A filename cannot substitute for the actual v2 parquet episode value."""

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="mapped materialized parquet episode identity conflicts",
    ):
        _prepare_mapped_observed_pair(FakeS3(), parquet_episode=23)


def test_observed_pair_refuses_mapped_parquet_changed_between_inventory_and_identity_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The identity column must come from the exact bytes already inventory-bound."""

    client = FakeS3()
    original_read = workflow._read_s3_bytes
    parquet_uri = (
        "s3://bucket/run/evaluation-data/data/chunk-000/episode_000024.parquet"
    )
    reads = 0

    def changed_second_read(s3_client: Any, uri: str) -> bytes:
        nonlocal reads
        if uri == parquet_uri:
            reads += 1
            if reads == 2:
                return _episode_parquet(24, rows=3)
        return original_read(s3_client, uri)

    monkeypatch.setattr(workflow, "_read_s3_bytes", changed_second_read)
    with pytest.raises(
        workflow.GrootVisualizationError,
        match="mapped materialized parquet differs from prepared content inventory",
    ):
        _prepare_mapped_observed_pair(client)

    assert reads == 2


def test_native_loader_uses_explicit_source_episode_for_mapped_logical_trajectory() -> (
    None
):
    """The adapter passes source episode 24, not logical trajectory 0, to GR00T."""

    task = _task("observed-0")
    task["reference_observation"] = {"source_episode_index": 24}

    assert native._source_episode_indices(
        task,
        {
            "episode_index_mapping": {
                "workflow_trajectory_id": 0,
                "source_episode_index": 24,
            }
        },
    ) == [24]


def test_native_loader_refuses_mapped_logical_trajectory_without_source_provenance() -> (
    None
):
    """A consumer mapping cannot substitute for a task's actual reference identity."""

    with pytest.raises(RuntimeError, match="mapping conflicts with task provenance"):
        native._source_episode_indices(
            _task("observed-0"),
            {
                "episode_index_mapping": {
                    "workflow_trajectory_id": 0,
                    "source_episode_index": 24,
                }
            },
        )


def test_observed_pair_refuses_mapping_without_matching_task_provenance() -> None:
    """Dataset mappings cannot silently replace the task's stated source episode."""

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="materialized episode identity mapping conflicts with task provenance",
    ):
        _prepare_mapped_observed_pair(FakeS3(), task_source_episode=23)


def test_observed_paired_preparation_refuses_unmaterialized_trajectory() -> None:
    client = FakeS3()
    observed_uri = "s3://bucket/run/inputs/observed.json"
    dataset_uri = "s3://bucket/run/inputs/dataset.json"
    output_uri = "s3://bucket/run/prepared/observed-protocol.json"
    observed = {"schema": workflow.OBSERVED_TASKS_SCHEMA, "tasks": [_task("seen-0")]}
    _put_json(client, observed_uri, observed)
    _put_json(
        client,
        "s3://bucket/run/evaluation-data/meta/info.json",
        {"chunks_size": 1_000},
    )
    client.seed(
        "s3://bucket/run/evaluation-data/data/chunk-000/episode_000024.parquet",
        b"different-selected-episode",
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

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="omits required trajectory parquet files",
    ):
        workflow.prepare_observed_paired_evaluation(
            observed_uri, dataset_uri, output_uri, "observed-run", s3_client=client
        )

    assert ("bucket", "run/prepared/observed-protocol.json") not in client.objects


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
    assert str(source) == native.sys.path[0]


def test_native_rollout_uses_fixed_seeded_episode_waves(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_rollout(**kwargs: Any) -> tuple[str, list[bool], dict[str, Any]]:
        calls.append(kwargs)
        wave_dir = Path(kwargs["video_dir"])
        for environment_index in range(kwargs["n_episodes"]):
            video = wave_dir / f"env-{environment_index:03d}" / "rollout.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(f"episode-{environment_index}".encode())
        return (
            kwargs["env_name"],
            [True] * kwargs["n_episodes"],
            {
                "episode_lengths": [4] * kwargs["n_episodes"],
                "episode_rewards": [1.0] * kwargs["n_episodes"],
                "npa_environment_indices": list(range(kwargs["n_episodes"])),
                "npa_completion_ordinals": [0] * kwargs["n_episodes"],
            },
        )

    result = native._run_seeded_episode_waves(
        native_env_name="libero_sim/npa_groot_libero_x_0123456789abcdef0123456789abcdef",
        rollout_runner=fake_rollout,
        policy_host="127.0.0.1",
        policy_port=1,
        video_dir=tmp_path / "videos",
        task_id="heldout-0",
        trajectory_ids=[7],
        initial_state_protocol="deterministic reset seeds",
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
    assert [episode["reset_seed"] for episode in result["episodes"]] == list(
        range(37, 47)
    )
    assert all(episode["trajectory_ids"] == [7] for episode in result["episodes"])
    assert all(episode["completion_ordinal"] == 0 for episode in result["episodes"])


def test_native_completion_identity_overlay_joins_reversed_results_to_env_videos(
    tmp_path: Path,
) -> None:
    """Execute the generated upstream hook, then join its completion identities."""

    source = tmp_path / "Isaac-GR00T"
    target = source / "gr00t/eval/rollout_policy.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "from collections import defaultdict\n"
        "from pathlib import Path\n\n"
        "def create_eval_env(env_idx, wrapper_configs):\n"
        "    return dict(\n"
        f"        {workflow.NATIVE_VIDEO_LAYOUT_NEEDLE}\n"
        "    )\n\n"
        "def capture(completion_order):\n"
        "    n_envs = 2\n"
        "    current_successes = [False] * n_envs\n"
        "    current_successes[1] = True\n"
        "    episode_infos = defaultdict(list)\n"
        "    episode_successes = []\n"
        "    lengths = [101, 202]\n"
        "    rewards = [1.25, 2.5]\n"
        "    for env_idx in completion_order:\n"
        "        if True:\n"
        "            if True:\n"
        "                if True:\n"
        "                    episode_infos['episode_lengths'].append(lengths[env_idx])\n"
        "                    episode_infos['episode_rewards'].append(rewards[env_idx])\n"
        "                    episode_successes.append(current_successes[env_idx])\n"
        "    return 'native-env', episode_successes, dict(episode_infos)\n"
    )
    workflow._overlay_native_video_layout(source)
    workflow._overlay_native_completion_identity(source)

    module_name = "npa_groot_libero_x_completion_hook_test"
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        parsed = native._parse_rollout(module.capture([1, 0]))
    finally:
        sys.modules.pop(module_name, None)

    wave_dir = tmp_path / "videos/wave-000"
    for environment_index in range(2):
        video = wave_dir / f"env-{environment_index:03d}/rollout.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(f"env-{environment_index}".encode())
    records = native._episode_video_records(
        task_video_dir=tmp_path / "videos",
        wave_dir=wave_dir,
        task_id="observed-0",
        trajectory_ids=[7],
        initial_state_protocol="deterministic reset seeds",
        reset_seeds=[41, 42],
        outcomes=parsed["completed_episode_outcomes"],
        wave_index=0,
    )

    assert [
        outcome["environment_index"] for outcome in parsed["completed_episode_outcomes"]
    ] == [1, 0]
    assert [record["environment_index"] for record in records] == [0, 1]
    assert [record["reset_seed"] for record in records] == [41, 42]
    assert [record["success"] for record in records] == [False, True]
    assert [record["episode_length"] for record in records] == [101, 202]
    assert [record["episode_reward"] for record in records] == [1.25, 2.5]


@pytest.mark.parametrize(
    ("outcomes", "message"),
    (
        (
            [
                {"environment_index": 0, "completion_ordinal": 0},
                {"environment_index": 0, "completion_ordinal": 0},
            ],
            "duplicate seeded environment outcomes",
        ),
        (
            [{"environment_index": 0, "completion_ordinal": 0}],
            "does not have one outcome per seeded environment",
        ),
        (
            [
                {"environment_index": 0, "completion_ordinal": 1},
                {"environment_index": 1, "completion_ordinal": 0},
            ],
            "does not describe an initial seeded episode",
        ),
    ),
)
def test_episode_video_records_refuse_ambiguous_completion_identities(
    tmp_path: Path, outcomes: list[dict[str, int]], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        native._episode_video_records(
            task_video_dir=tmp_path / "videos",
            wave_dir=tmp_path / "videos/wave-000",
            task_id="observed-0",
            trajectory_ids=[7],
            initial_state_protocol="deterministic reset seeds",
            reset_seeds=[41, 42],
            outcomes=outcomes,
            wave_index=0,
        )


def test_native_rollout_refuses_missing_completion_identity_hook() -> None:
    with pytest.raises(RuntimeError, match="lacks completion identities"):
        native._parse_rollout(
            (
                "native-env",
                [True],
                {"episode_lengths": [1], "episode_rewards": [1.0]},
            )
        )


def test_policy_server_device_evidence_uses_mocked_hardware_only_in_unit_test() -> None:
    """Mocked CUDA objects test the capture contract; they are not device evidence."""

    class Tensor:
        device = "cuda:0"

    class Model:
        def parameters(self) -> list[Tensor]:
            return [Tensor()]

        def buffers(self) -> list[Tensor]:
            return []

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_properties(_index: int) -> Any:
            return types.SimpleNamespace(name="test-gpu", major=12, minor=0)

        @staticmethod
        def device_count() -> int:
            return 1

    class Torch:
        cuda = Cuda()
        version = types.SimpleNamespace(cuda="test-cuda")
        __version__ = "test-torch"

        @staticmethod
        def device(value: str) -> Any:
            return types.SimpleNamespace(
                type="cuda", index=int(value.rsplit(":", 1)[1])
            )

    observed = server_evidence._selected_tensor_device(Model(), Torch())

    assert observed["selected_model_tensor_device"] == "cuda:0"
    assert observed["selected_device_capability"] == [12, 0]


def test_policy_server_device_evidence_refuses_missing_cuda_or_model_tensors() -> None:
    """Native server evidence must not invent CUDA or model-device observations."""

    class NoCuda:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    class EmptyModel:
        def parameters(self) -> list[Any]:
            return []

        def buffers(self) -> list[Any]:
            return []

    with pytest.raises(RuntimeError, match="CUDA availability"):
        server_evidence._selected_tensor_device(EmptyModel(), NoCuda())


def test_policy_server_checkpoint_inventory_is_digest_bound(tmp_path: Path) -> None:
    """The server binds bytes in the selected model directory, not a label."""

    model = tmp_path / "model"
    checkpoint = model / "nested" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"actual-checkpoint-bytes")

    inventory = server_evidence._checkpoint_inventory(model)

    assert inventory == [
        {
            "path": "nested/model.safetensors",
            "sha256": hashlib.sha256(b"actual-checkpoint-bytes").hexdigest(),
            "bytes": len(b"actual-checkpoint-bytes"),
        }
    ]


def test_open_loop_policy_adapts_nested_observation_for_sim_server() -> None:
    class SimServerClient:
        def __init__(self) -> None:
            self.observation: dict[str, Any] | None = None
            self.options: dict[str, Any] | None = None

        def get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None
        ) -> tuple[dict[str, Any], dict[str, str]]:
            self.observation = observation
            self.options = options
            return {"action.joints": np.array([[1.0]], dtype=np.float32)}, {
                "server": "sim"
            }

    client = SimServerClient()
    policy = native._OpenLoopSimServerPolicy(client)
    image = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
    state = np.zeros((1, 1, 7), dtype=np.float32)

    actions, info = policy.get_action(
        {
            "video": {"image": image},
            "state": {"joints": state},
            "language": {"task": [["place the bowl"]]},
        },
        {"temperature": 0.0},
    )

    assert client.observation is not None
    assert client.observation["video.image"] is image
    assert client.observation["state.joints"] is state
    assert client.observation["task"] == ("place the bowl",)
    assert client.options == {"temperature": 0.0}
    assert set(actions) == {"joints"}
    assert np.array_equal(actions["joints"], np.array([[1.0]], dtype=np.float32))
    assert info == {"server": "sim"}


def test_open_loop_policy_refuses_non_simulator_action_keys() -> None:
    class InvalidServerClient:
        def get_action(
            self, _observation: dict[str, Any], _options: dict[str, Any] | None
        ) -> tuple[dict[str, Any], dict[str, str]]:
            return {"joints": np.array([[1.0]], dtype=np.float32)}, {}

    policy = native._OpenLoopSimServerPolicy(InvalidServerClient())
    with pytest.raises(RuntimeError, match="invalid action key"):
        policy.get_action(
            {
                "video": {"image": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)},
                "state": {"joints": np.zeros((1, 1, 7), dtype=np.float32)},
                "language": {"task": [["place the bowl"]]},
            }
        )


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
        model_identity={
            "repo": workflow.BASELINE_REPO,
            "revision": workflow.BASELINE_REVISION,
            "checkpoint_files": [
                {"path": "model.safetensors", "sha256": "a" * 64, "bytes": 1}
            ],
        },
        dataset_path=tmp_path / "dataset",
        tasks=[_task("heldout-0")],
        input_provenance={
            "rollout_protocol": {"initial_state_protocol": "deterministic reset seeds"}
        },
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
    assert captured["config"]["server_script"].endswith("groot_libero_x_server.py")
    assert captured["config"]["sanitized_server_argv"][1] == "[MODEL_PATH]"


def test_native_evaluator_exposes_libero_x_namespace_at_checkout_root(
    tmp_path: Path,
) -> None:
    """The selected evaluator must preserve its ``libero.libero`` layout."""

    source = tmp_path / "LIBERO-X"
    (source / "libero/libero/utils").mkdir(parents=True)
    (source / "libero/libero_x/bddl").mkdir(parents=True)
    (source / "libero/libero/__init__.py").write_text("")
    (source / "libero/libero/utils/__init__.py").write_text("")
    parser = source / "libero/libero/utils/parse_bddl.py"
    parser.write_text(
        "def parse_bddl_file(_path: str) -> dict[str, str]:\n    return {}\n"
    )

    original_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "libero" or name.startswith("libero.")
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        assert (
            native._configure_libero_x_source({"libero_x_source": str(source)})
            == source
        )
        assert sys.path[0] == str(source)
        module = importlib.import_module("libero.libero.utils.parse_bddl")
        assert Path(str(module.__file__)).resolve() == parser.resolve()
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "libero" or name.startswith("libero."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def test_runtime_overlay_and_git_fetch_contract_avoid_broad_lfs_smudging(
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


def test_runtime_overlay_registration_is_idempotent_only_for_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "libero-x"
    bddl = source / "libero/libero_x/bddl/LEVEL1/task.bddl"
    bddl.parent.mkdir(parents=True)
    bddl.write_text("(define (problem placeholder))\n")
    native_env_name = "libero_sim/npa_groot_libero_x_0123456789abcdef0123456789abcdef"
    manifest = tmp_path / "registration.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "npa.groot_libero_x.registration.v1",
                "tasks": [
                    {
                        "native_env_name": native_env_name,
                        "libero_x_bddl_path": "libero/libero_x/bddl/LEVEL1/task.bddl",
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("NPA_GROOT_LIBERO_X_SOURCE", str(source))
    monkeypatch.setenv("NPA_GROOT_LIBERO_X_REGISTRATION_MANIFEST", str(manifest))
    parser_module = types.ModuleType("libero.libero.utils.parse_bddl")
    parser_module.parse_bddl_file = lambda _path: {"language": "place the item"}
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "libero" or name.startswith("libero.")
    }
    for name in list(saved_modules):
        del sys.modules[name]
    package = types.ModuleType("libero")
    package.__path__ = []  # type: ignore[attr-defined]
    nested = types.ModuleType("libero.libero")
    nested.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("libero.libero.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "libero": package,
            "libero.libero": nested,
            "libero.libero.utils": utils,
            "libero.libero.utils.parse_bddl": parser_module,
        }
    )
    overlay_module_path = tmp_path / "libero_env.py"
    overlay_module_path.write_text(
        "import os\n"
        "from types import SimpleNamespace\n"
        "gym = SimpleNamespace(registry={})\n"
        "registrations = []\n"
        "def register(*, id, entry_point, kwargs):\n"
        "    registrations.append({'id': id, 'entry_point': entry_point, 'kwargs': kwargs})\n"
        "    gym.registry[id] = SimpleNamespace(entry_point=entry_point, kwargs=kwargs)\n"
        "def register_libero_envs():\n"
        "    pass\n" + workflow.LIBERO_X_RUNTIME_OVERLAY
    )
    module_name = "npa_groot_libero_x_overlay_test"
    spec = importlib.util.spec_from_file_location(module_name, overlay_module_path)
    assert spec is not None and spec.loader is not None
    overlay_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = overlay_module
    try:
        spec.loader.exec_module(overlay_module)
        overlay_module.register_libero_envs()
        overlay_module.register_libero_envs()
        assert len(overlay_module.registrations) == 1
        overlay_module.gym.registry[native_env_name] = types.SimpleNamespace(
            entry_point="unexpected:entry", kwargs={}
        )
        with pytest.raises(RuntimeError, match="conflicts with the selected manifest"):
            overlay_module.register_libero_envs()
    finally:
        sys.modules.pop(module_name, None)
        for name in list(sys.modules):
            if name == "libero" or name.startswith("libero."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def test_runtime_readiness_failure_names_missing_documented_runtime(
    tmp_path: Path,
) -> None:
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "runtime-ready.json").write_text(
        json.dumps(
            {
                "schema": workflow.RUNTIME_READY_SCHEMA,
                "isaac_groot_revision": workflow.IMAGE_GROOT_REF,
                "required_groot_lfs_objects": workflow._groot_lfs_provenance(),
                "libero_parquet_engine": {
                    "package": workflow.LIBERO_PARQUET_ENGINE_PACKAGE,
                    "version": workflow.LIBERO_PARQUET_ENGINE_VERSION,
                },
                "libero_torchcodec": {
                    "package": workflow.LIBERO_TORCHCODEC_PACKAGE,
                    "version": workflow.LIBERO_TORCHCODEC_VERSION,
                },
                "libero_x_evaluator": {
                    "repository": workflow.LIBERO_X_EVALUATOR_REPOSITORY,
                    "revision": workflow.LIBERO_X_EVALUATOR_REVISION,
                    "license": workflow.LIBERO_X_EVALUATOR_LICENSE,
                },
                "npa_libero_x_overlay": {
                    "sha256": hashlib.sha256(
                        workflow.LIBERO_X_RUNTIME_OVERLAY.encode()
                    ).hexdigest()
                },
                "npa_native_video_layout": {
                    "sha256": hashlib.sha256(
                        workflow.NATIVE_VIDEO_LAYOUT_REPLACEMENT.encode()
                    ).hexdigest()
                },
                "npa_native_completion_identity": (
                    workflow._native_completion_identity_overlay_record()
                ),
            }
        )
    )

    assert (
        workflow._runtime_readiness_failure(target)
        == "documented_python_runtime_missing"
    )
    assert workflow._runtime_ready(target) is None


def test_runtime_readiness_refuses_cache_marker_without_torchcodec(
    tmp_path: Path,
) -> None:
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "runtime-ready.json").write_text(
        json.dumps(
            {
                "schema": workflow.RUNTIME_READY_SCHEMA,
                "isaac_groot_revision": workflow.IMAGE_GROOT_REF,
                "required_groot_lfs_objects": workflow._groot_lfs_provenance(),
                "libero_parquet_engine": {
                    "package": workflow.LIBERO_PARQUET_ENGINE_PACKAGE,
                    "version": workflow.LIBERO_PARQUET_ENGINE_VERSION,
                },
                "libero_x_evaluator": {
                    "repository": workflow.LIBERO_X_EVALUATOR_REPOSITORY,
                    "revision": workflow.LIBERO_X_EVALUATOR_REVISION,
                    "license": workflow.LIBERO_X_EVALUATOR_LICENSE,
                },
                "npa_libero_x_overlay": {
                    "sha256": hashlib.sha256(
                        workflow.LIBERO_X_RUNTIME_OVERLAY.encode()
                    ).hexdigest()
                },
            }
        )
    )

    assert (
        workflow._runtime_readiness_failure(target)
        == "ready_marker_torchcodec_mismatch"
    )


def test_runtime_video_layout_overlay_binds_videos_to_vector_environments(
    tmp_path: Path,
) -> None:
    """The materialized upstream worker must emit one named directory per env."""

    source = tmp_path / "Isaac-GR00T"
    target = source / "gr00t/eval/rollout_policy.py"
    target.parent.mkdir(parents=True)
    target.write_text(f"before\n{workflow.NATIVE_VIDEO_LAYOUT_NEEDLE}\nafter\n")

    record = workflow._overlay_native_video_layout(source)

    assert record["target"] == "gr00t/eval/rollout_policy.py"
    assert workflow.NATIVE_VIDEO_LAYOUT_REPLACEMENT in target.read_text()
    with pytest.raises(workflow.GrootVisualizationError, match="unexpectedly contains"):
        workflow._overlay_native_video_layout(source)


def test_runtime_post_publish_failure_is_actionable_and_path_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workflow, "_runtime_ready", lambda _target: None)
    monkeypatch.setattr(
        workflow,
        "_runtime_readiness_failure",
        lambda _target: "libero_x_registration_overlay_invalid",
    )

    with pytest.raises(
        workflow.GrootVisualizationError,
        match=(
            "atomically materialized native runtime did not verify: "
            "libero_x_registration_overlay_invalid"
        ),
    ) as error:
        workflow._verified_runtime_or_raise(tmp_path)

    assert str(tmp_path) not in str(error.value)


def test_libero_parquet_probe_reads_with_selected_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pyarrow

    monkeypatch.setattr(workflow, "LIBERO_PARQUET_ENGINE_VERSION", pyarrow.__version__)

    assert workflow._verify_libero_parquet_engine(
        Path(sys.executable),
        cwd=tmp_path,
        env=os.environ,
    ) == {
        "package": workflow.LIBERO_PARQUET_ENGINE_PACKAGE,
        "version": pyarrow.__version__,
    }


def test_libero_torchcodec_probe_rejects_unpinned_selected_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def completed(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"package": "torchcodec", "version": "0.0.0"}),
            stderr="",
        )

    monkeypatch.setattr(workflow.subprocess, "run", completed)

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="TorchCodec differs from the pinned contract",
    ):
        workflow._verify_libero_torchcodec(
            Path(sys.executable), cwd=tmp_path, env=os.environ
        )


def test_materializer_installs_runtime_dependencies_before_publishing_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NPA_GROOT_LIBERO_X_RUNTIME_CACHE", raising=False)
    monkeypatch.delenv("GROOT_DATA_MOUNT", raising=False)
    source = _native_runtime_cache_target(tmp_path) / "Isaac-GR00T"
    evaluator = _native_runtime_cache_target(tmp_path) / "LIBERO-X"
    commands: list[list[str]] = []
    dependency_calls: list[tuple[str, Path]] = []

    def runtime_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        commands.append(command)
        if command[-2:] == [workflow.IMAGE_GROOT_REPOSITORY, str(source)]:
            target = source / "gr00t/eval/sim/LIBERO/libero_env.py"
            target.parent.mkdir(parents=True)
            target.write_text("def register_libero_envs():\n    pass\n")
            rollout = source / "gr00t/eval/rollout_policy.py"
            rollout.parent.mkdir(parents=True, exist_ok=True)
            rollout.write_text(
                "\n".join(
                    (
                        workflow.NATIVE_VIDEO_LAYOUT_NEEDLE,
                        workflow.NATIVE_COMPLETION_ORDINALS_NEEDLE,
                        workflow.NATIVE_COMPLETION_APPEND_NEEDLE,
                    )
                )
            )
        elif command[-2:] == [
            workflow.LIBERO_X_EVALUATOR_REPOSITORY,
            str(evaluator),
        ]:
            (evaluator / "libero/libero_x/bddl").mkdir(parents=True)
        elif command[1:3] == ["sync", "--python"]:
            python = cwd / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        elif command[0] == "bash":
            python = cwd / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            (cwd / "external_dependencies/LIBERO").mkdir(parents=True)

    def revision(
        _command: list[str], *, cwd: Path, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if cwd == source:
            stdout = workflow.IMAGE_GROOT_REF
        elif cwd == evaluator:
            stdout = workflow.LIBERO_X_EVALUATOR_REVISION
        else:
            assert cwd == source / "external_dependencies/LIBERO"
            stdout = "reviewed-libero-submodule"
        return subprocess.CompletedProcess([], 0, f"{stdout}\n", "")

    expected_engine = {
        "package": workflow.LIBERO_PARQUET_ENGINE_PACKAGE,
        "version": workflow.LIBERO_PARQUET_ENGINE_VERSION,
    }

    def verify_parquet(python: Path, **_kwargs: Any) -> dict[str, str]:
        dependency_calls.append(("parquet", python))
        return expected_engine

    expected_torchcodec = {
        "package": workflow.LIBERO_TORCHCODEC_PACKAGE,
        "version": workflow.LIBERO_TORCHCODEC_VERSION,
    }

    def verify_torchcodec(python: Path, **_kwargs: Any) -> dict[str, str]:
        dependency_calls.append(("torchcodec", python))
        return expected_torchcodec

    monkeypatch.setattr(workflow.shutil, "which", lambda _name: "/tool")
    monkeypatch.setattr(workflow, "_run_runtime_command", runtime_command)
    monkeypatch.setattr(
        workflow, "_hydrate_required_groot_lfs_objects", lambda _source: None
    )
    monkeypatch.setattr(
        workflow, "_required_groot_lfs_objects_verified", lambda _source: True
    )
    monkeypatch.setattr(workflow.subprocess, "run", revision)
    monkeypatch.setattr(workflow, "_verify_libero_parquet_engine", verify_parquet)
    monkeypatch.setattr(workflow, "_verify_libero_torchcodec", verify_torchcodec)

    runtime = workflow._materialize_native_runtime(tmp_path)

    libero_python = source / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    assert [
        "/tool",
        "pip",
        "install",
        "--no-cache",
        "--python",
        str(libero_python),
        workflow.LIBERO_PARQUET_ENGINE_REQUIREMENT,
        workflow.LIBERO_TORCHCODEC_REQUIREMENT,
    ] in commands
    assert dependency_calls == [
        ("parquet", libero_python),
        ("torchcodec", libero_python),
        ("parquet", libero_python),
        ("torchcodec", libero_python),
    ]
    assert runtime["provenance"]["libero_parquet_engine"] == expected_engine
    assert runtime["provenance"]["libero_torchcodec"] == expected_torchcodec
    assert runtime["provenance"]["npa_native_video_layout"]["version"] == "v1"
    assert runtime["provenance"]["npa_native_completion_identity"]["version"] == "v1"


def _native_runtime_cache_target(temporary_root: Path) -> Path:
    cache_root = workflow._runtime_cache_root(temporary_root)
    lfs_identity = hashlib.sha256(
        json.dumps(workflow._groot_lfs_provenance(), sort_keys=True).encode()
    ).hexdigest()[:12]
    cache_identity = (
        f"isaac-groot-{workflow.IMAGE_GROOT_REF}-lfs-{lfs_identity}-libero-x-"
        f"{workflow.LIBERO_X_EVALUATOR_REVISION}"
    )
    return cache_root / cache_identity


def test_materializer_clears_stale_unready_runtime_before_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NPA_GROOT_LIBERO_X_RUNTIME_CACHE", raising=False)
    monkeypatch.delenv("GROOT_DATA_MOUNT", raising=False)
    target = _native_runtime_cache_target(tmp_path)
    target.mkdir(parents=True)
    stale_file = target / "interrupted-build"
    stale_file.write_text("partial runtime")
    monkeypatch.setattr(workflow.shutil, "which", lambda _name: None)

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="runtime requires git and uv in the image",
    ):
        workflow._materialize_native_runtime(tmp_path)

    assert stale_file.exists() is False
    assert target.exists() is False


def test_materializer_refuses_to_clear_a_symlinked_unready_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("NPA_GROOT_LIBERO_X_RUNTIME_CACHE", raising=False)
    monkeypatch.delenv("GROOT_DATA_MOUNT", raising=False)
    target = _native_runtime_cache_target(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    sentinel = outside / "retain"
    sentinel.write_text("outside cache root")
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="cache path is not a removable directory",
    ):
        workflow._materialize_native_runtime(tmp_path)

    assert sentinel.read_text() == "outside cache root"


def test_runtime_hydrates_only_verified_required_lfs_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "Isaac-GR00T"
    relative_path = "scripts/deployment/dgpu/wheels/torchcodec.whl"
    payload = b"PK\x03\x04-reviewed-wheel"
    digest = hashlib.sha256(payload).hexdigest()
    target = source / relative_path
    target.parent.mkdir(parents=True)
    target.write_text(workflow._expected_lfs_pointer(digest, len(payload)))
    monkeypatch.setattr(
        workflow,
        "GROOT_RUNTIME_LFS_OBJECTS",
        ((relative_path, digest, len(payload)),),
    )
    calls: list[tuple[Path, str, int]] = []

    def download(
        destination: Path, *, expected_sha256: str, expected_size: int
    ) -> None:
        calls.append((destination, expected_sha256, expected_size))
        destination.write_bytes(payload)

    monkeypatch.setattr(workflow, "_download_github_lfs_object", download)

    workflow._hydrate_required_groot_lfs_objects(source)

    assert calls == [(target, digest, len(payload))]
    assert target.read_bytes() == payload
    assert workflow._required_groot_lfs_objects_verified(source) is True
    assert workflow._groot_lfs_provenance() == [
        {"path": relative_path, "sha256": digest, "size": len(payload)}
    ]


def test_runtime_refuses_changed_required_lfs_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "Isaac-GR00T"
    relative_path = "scripts/deployment/dgpu/wheels/torchcodec.whl"
    payload = b"PK\x03\x04-reviewed-wheel"
    digest = hashlib.sha256(payload).hexdigest()
    target = source / relative_path
    target.parent.mkdir(parents=True)
    target.write_text(workflow._expected_lfs_pointer("0" * 64, len(payload)))
    monkeypatch.setattr(
        workflow,
        "GROOT_RUNTIME_LFS_OBJECTS",
        ((relative_path, digest, len(payload)),),
    )

    def unexpected_download(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("changed pointer must fail before download")

    monkeypatch.setattr(workflow, "_download_github_lfs_object", unexpected_download)

    with pytest.raises(workflow.GrootVisualizationError, match="pointer changed"):
        workflow._hydrate_required_groot_lfs_objects(source)


def test_runtime_lfs_download_requires_matching_batch_and_payload_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"PK\x03\x04-reviewed-wheel"
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "torchcodec.whl"
    destination.write_text(workflow._expected_lfs_pointer(digest, len(payload)))
    download_url = "https://github-cloud.githubusercontent.com/opaque-signed-object"
    responses: list[object] = [
        _StreamingResponse(
            json.dumps(
                {
                    "objects": [
                        {
                            "oid": digest,
                            "size": len(payload),
                            "actions": {"download": {"href": download_url}},
                        }
                    ]
                }
            ).encode()
        ),
        _StreamingResponse(payload),
    ]
    connections, requests = _patch_lfs_https_transport(monkeypatch, responses)

    workflow._download_github_lfs_object(
        destination, expected_sha256=digest, expected_size=len(payload)
    )

    assert destination.read_bytes() == payload
    assert connections == [
        ("github.com", workflow.GROOT_LFS_TRANSPORT_TIMEOUT_SECONDS),
        (
            "github-cloud.githubusercontent.com",
            workflow.GROOT_LFS_TRANSPORT_TIMEOUT_SECONDS,
        ),
    ]
    assert requests[0][:3] == (
        "github.com",
        "POST",
        "/NVIDIA/Isaac-GR00T.git/info/lfs/objects/batch",
    )
    assert json.loads(requests[0][3]) == {
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": digest, "size": len(payload)}],
    }
    assert requests[0][4] == {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }
    assert requests[1] == (
        "github-cloud.githubusercontent.com",
        "GET",
        "/opaque-signed-object",
        None,
        {},
    )


def test_lfs_download_timeout_is_actionable_and_releases_runtime_cache_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"PK\x03\x04-reviewed-wheel"
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = "scripts/deployment/dgpu/wheels/torchcodec.whl"
    monkeypatch.setattr(
        workflow,
        "GROOT_RUNTIME_LFS_OBJECTS",
        ((relative_path, digest, len(payload)),),
    )
    monkeypatch.setattr(workflow.shutil, "which", lambda _name: "tool")
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{workflow.IMAGE_GROOT_REF}\n"
        ),
    )

    def runtime_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        if str(command[-2]) == workflow.IMAGE_GROOT_REPOSITORY:
            source = Path(command[-1])
            target = source / relative_path
            target.parent.mkdir(parents=True)
            target.write_text(workflow._expected_lfs_pointer(digest, len(payload)))

    monkeypatch.setattr(workflow, "_run_runtime_command", runtime_command)
    responses: list[object] = [
        _StreamingResponse(
            json.dumps(
                {
                    "objects": [
                        {
                            "oid": digest,
                            "size": len(payload),
                            "actions": {
                                "download": {
                                    "href": "https://github-cloud.githubusercontent.com/signed"
                                }
                            },
                        }
                    ]
                }
            ).encode()
        ),
        TimeoutError("stalled transport"),
    ]
    connections, _requests = _patch_lfs_https_transport(monkeypatch, responses)

    with pytest.raises(
        workflow.GrootVisualizationError,
        match=r"download the required upstream Git LFS object timed out after 60 seconds",
    ):
        workflow._materialize_native_runtime(tmp_path)

    assert [timeout for _host, timeout in connections] == [
        workflow.GROOT_LFS_TRANSPORT_TIMEOUT_SECONDS,
        workflow.GROOT_LFS_TRANSPORT_TIMEOUT_SECONDS,
    ]
    lock_path = next(workflow._runtime_cache_root(tmp_path).glob("*.lock"))
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_lfs_batch_transport_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sha256 = "a" * 64
    connections, _requests = _patch_lfs_https_transport(
        monkeypatch, [OSError("name resolution failed")]
    )

    with pytest.raises(
        workflow.GrootVisualizationError,
        match="could not resolve the required upstream Git LFS object",
    ):
        workflow._github_lfs_download_url(expected_sha256, 1)

    assert connections == [("github.com", workflow.GROOT_LFS_TRANSPORT_TIMEOUT_SECONDS)]


def test_runtime_lfs_download_rejects_wrong_payload_without_replacing_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"PK\x03\x04-reviewed-wheel"
    digest = hashlib.sha256(payload).hexdigest()
    pointer = workflow._expected_lfs_pointer(digest, len(payload))
    destination = tmp_path / "torchcodec.whl"
    destination.write_text(pointer)
    responses: list[object] = [
        _StreamingResponse(
            json.dumps(
                {
                    "objects": [
                        {
                            "oid": digest,
                            "size": len(payload),
                            "actions": {
                                "download": {
                                    "href": "https://github-cloud.githubusercontent.com/signed"
                                }
                            },
                        }
                    ]
                }
            ).encode()
        ),
        _StreamingResponse(b"not-the-reviewed-wheel"),
    ]
    _patch_lfs_https_transport(monkeypatch, responses)

    with pytest.raises(workflow.GrootVisualizationError, match="did not match"):
        workflow._download_github_lfs_object(
            destination, expected_sha256=digest, expected_size=len(payload)
        )

    assert destination.read_text() == pointer
    assert not list(tmp_path.glob(".torchcodec.whl.lfs-*"))


@pytest.mark.parametrize(
    "href",
    [
        "http://github-cloud.githubusercontent.com/signed",
        "https://unexpected.example/signed",
    ],
)
def test_lfs_batch_rejects_disallowed_download_urls(href: str) -> None:
    batch = {
        "objects": [
            {
                "oid": "a" * 64,
                "size": 1,
                "actions": {"download": {"href": href}},
            }
        ]
    }

    with pytest.raises(
        workflow.GrootVisualizationError, match="permitted download URL"
    ):
        workflow._lfs_download_url_from_batch(batch, "a" * 64, 1)


@pytest.mark.parametrize(
    "url",
    [
        "http://github-cloud.githubusercontent.com/signed",
        "https://unexpected.example/signed",
    ],
)
def test_lfs_transport_rejects_disallowed_url_before_connection(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    def unexpected_connection(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("disallowed URL must fail before connecting")

    monkeypatch.setattr(workflow.http.client, "HTTPSConnection", unexpected_connection)

    with pytest.raises(workflow.GrootVisualizationError, match="URL is not permitted"):
        with workflow._open_lfs_https_response(
            url,
            allowed_hosts=workflow._GITHUB_LFS_DOWNLOAD_HOSTS,
            method="GET",
        ):
            pytest.fail("disallowed URL must not yield a response")


def test_runtime_rejects_downloaded_lfs_object_with_wrong_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "Isaac-GR00T"
    relative_path = "scripts/deployment/dgpu/wheels/torchcodec.whl"
    payload = b"PK\x03\x04-reviewed-wheel"
    digest = hashlib.sha256(payload).hexdigest()
    target = source / relative_path
    target.parent.mkdir(parents=True)
    target.write_text(workflow._expected_lfs_pointer(digest, len(payload)))
    monkeypatch.setattr(
        workflow,
        "GROOT_RUNTIME_LFS_OBJECTS",
        ((relative_path, digest, len(payload)),),
    )
    monkeypatch.setattr(
        workflow,
        "_download_github_lfs_object",
        lambda destination, **_kwargs: destination.write_bytes(
            b"not-the-reviewed-wheel"
        ),
    )

    with pytest.raises(workflow.GrootVisualizationError, match="did not materialize"):
        workflow._hydrate_required_groot_lfs_objects(source)


def test_runtime_cache_uses_writable_data_cache_for_implicit_groot_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mount = tmp_path / "groot-data"
    (mount / "data_cache").mkdir(parents=True)
    monkeypatch.delenv("NPA_GROOT_LIBERO_X_RUNTIME_CACHE", raising=False)
    monkeypatch.setenv("GROOT_DATA_MOUNT", str(mount))

    assert workflow._runtime_cache_root(tmp_path / "temporary") == (
        mount / "data_cache" / "runtime-fetch"
    )


def test_runtime_cache_preserves_explicit_cache_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = tmp_path / "operator-cache"
    monkeypatch.setenv("NPA_GROOT_LIBERO_X_RUNTIME_CACHE", str(configured))
    monkeypatch.setenv("GROOT_DATA_MOUNT", str(tmp_path / "ignored-mount"))

    assert workflow._runtime_cache_root(tmp_path / "temporary") == (
        configured / "runtime-fetch"
    )


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
            {
                "repo": workflow.BASELINE_REPO,
                "revision": workflow.BASELINE_REVISION,
                "checkpoint_files": [
                    {"path": "model.safetensors", "sha256": "a" * 64, "bytes": 1}
                ],
            },
        ),
    )
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
        episode_video = video_dir / "wave-000/env-000/rollout.mp4"
        episode_video.parent.mkdir(parents=True)
        episode_video.write_bytes(b"actual-native-rollout")
        return {
            "worker_provenance": _native_worker_provenance(),
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
                    "task_input": {
                        "task_id": "heldout-0",
                        "env_name": _task("heldout-0")["env_name"],
                        "trajectory_ids": [0],
                        "libero_x_bddl_path": _task("heldout-0")["libero_x_bddl_path"],
                        "libero_x_bddl_sha256": _task("heldout-0")[
                            "libero_x_bddl_sha256"
                        ],
                        "language": _task("heldout-0")["language"],
                    },
                    "native_env_name": _task("heldout-0")["env_name"],
                    "successes": [True],
                    "episode_lengths": [12],
                    "episode_rewards": [1.0],
                    "completed_episodes": 1,
                    "success_rate": 1.0,
                    "episodes": [
                        {
                            "task_id": "heldout-0",
                            "wave_index": 0,
                            "environment_index": 0,
                            "reset_seed": 20261002,
                            "initial_state": {"protocol": "deterministic reset seeds"},
                            "trajectory_ids": [0],
                            "success": True,
                            "episode_length": 12,
                            "episode_reward": 1.0,
                            "video_relative_path": "wave-000/env-000/rollout.mp4",
                        }
                    ],
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
    assert report["runtime"]["native_worker_provenance"]["model_load_completed"] is True
    assert (
        "path"
        not in report["runtime"]["native_worker_provenance"][
            "executed_native_entrypoint"
        ]
    )
    private_provenance = workflow._read_s3_json(
        client,
        "s3://bucket/run/rollouts/baseline/native-worker-provenance.private.json",
    )
    assert private_provenance["policy_server"]["device"]["selected_device_name"] == (
        "mocked-test-gpu"
    )
    assert report["closed_loop"]["success_rate"] == 1.0
    video = report["closed_loop"]["tasks"][0]["videos"][0]
    assert video["bytes"] == len(b"actual-native-rollout")
    media_prefix = workflow._read_s3_json(
        client, "s3://bucket/run/rollouts/baseline/mp4"
    )
    assert media_prefix == {
        "schema": workflow.ROLLOUT_MEDIA_PREFIX_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "policy": "baseline",
        "videos": [video],
    }
    assert report["rollout_media"]["uri"] == "s3://bucket/run/rollouts/baseline/mp4"
    assert client.head_object(
        Bucket="bucket",
        Key="run/rollouts/baseline/mp4/heldout-0/wave-000/env-000/rollout.mp4",
    )
    episode = report["closed_loop"]["tasks"][0]["episodes"][0]
    assert episode["final_report_member"] is True
    assert (
        episode["video"]["sha256"]
        == hashlib.sha256(b"actual-native-rollout").hexdigest()
    )


def test_policy_stage_refuses_missing_native_worker_provenance() -> None:
    """A native result cannot infer device or checkpoint proof from its request."""

    with pytest.raises(workflow.GrootVisualizationError, match="worker provenance"):
        workflow._require_native_worker_provenance({})


def test_episode_media_refuses_conflicting_native_video_membership() -> None:
    """A report cannot bind an episode to a different uploaded MP4."""

    with pytest.raises(workflow.GrootVisualizationError, match="does not match"):
        workflow._bind_episode_media(
            {
                "episodes": [
                    {
                        "reset_seed": 1,
                        "initial_state": {"protocol": "seeded"},
                        "trajectory_ids": [0],
                        "video_relative_path": "different.mp4",
                    }
                ]
            },
            [
                {
                    "relative_path": "actual.mp4",
                    "uri": "s3://bucket/actual.mp4",
                    "sha256": "a" * 64,
                    "bytes": 1,
                }
            ],
        )


def test_policy_stage_persists_redacted_native_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeS3()
    run_id = "groot-libero-x-fixture"
    _prepare(client, run_id)
    model = tmp_path / "model"
    model.mkdir()
    secret = "operator-private-token-value"

    monkeypatch.setenv("NPA_TEST_TOKEN", secret)
    monkeypatch.setattr(
        workflow,
        "_snapshot_model",
        lambda *_args: (
            model,
            {"repo": workflow.BASELINE_REPO, "revision": workflow.BASELINE_REVISION},
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_materialize_native_runtime",
        lambda _root: {"provenance": {"delivery": "runtime-fetch"}},
    )

    def native_failure(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(f"native evaluator refused payload {secret}")

    monkeypatch.setattr(workflow, "_run_native_evaluator", native_failure)
    failure_uri = "s3://bucket/run/diagnostics/baseline-failure.json"

    with pytest.raises(RuntimeError, match="native evaluator refused payload"):
        workflow.run_policy(
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
            failure_uri=failure_uri,
            s3_client=client,
        )

    failure = workflow._read_s3_json(client, failure_uri)
    assert failure["schema"] == workflow.POLICY_FAILURE_SCHEMA
    assert failure["status"] == "failed"
    assert failure["policy"] == "baseline"
    assert failure["phase"] == "native_closed_loop_rollout"
    assert failure["error"]["type"] == "RuntimeError"
    assert failure["error"]["redacted"] is True
    assert secret not in failure["error"]["message"]


def test_policy_parser_accepts_failure_uri() -> None:
    values = workflow.build_parser().parse_args(
        [
            "run-policy",
            "--protocol-uri",
            "s3://bucket/protocol.json",
            "--output-uri",
            "s3://bucket/output.json",
            "--rollout-media-uri",
            "s3://bucket/media",
            "--failure-uri",
            "s3://bucket/diagnostics/failure.json",
            "--run-id",
            "run-id",
            "--policy-name",
            "baseline",
            "--model-repo",
            workflow.BASELINE_REPO,
            "--model-revision",
            workflow.BASELINE_REVISION,
            "--episodes-per-task",
            "1",
            "--n-envs",
            "1",
            "--max-episode-steps",
            "1",
            "--n-action-steps",
            "1",
            "--seed",
            "1",
        ]
    )

    assert values.failure_uri == "s3://bucket/diagnostics/failure.json"


def test_runtime_command_retains_redacted_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "operator-private-token-value"

    class FailedCommand:
        returncode = 17
        stderr = f"native evaluator rejected {secret}"

    monkeypatch.setenv("NPA_TEST_TOKEN", secret)
    monkeypatch.setattr(
        workflow.subprocess, "run", lambda *_args, **_kwargs: FailedCommand()
    )

    with pytest.raises(workflow.GrootVisualizationError) as error:
        workflow._run_runtime_command(
            ["native-evaluator", "--config", "config.json"],
            cwd=tmp_path,
            env={},
        )

    assert "exit 17" in str(error.value)
    assert "native evaluator rejected" in str(error.value)
    assert secret not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert "('native evaluator" not in str(error.value)


def test_numeric_metrics_rejects_empty_or_nonfinite_native_measurements() -> None:
    with pytest.raises(workflow.GrootVisualizationError, match="at least one sample"):
        workflow._numeric_metrics(
            {"mse": 0.0, "mae": 0.0, "sample_count": 0, "forward_calls": 0}
        )
    with pytest.raises(workflow.GrootVisualizationError, match="finite"):
        workflow._numeric_metrics(
            {"mse": float("nan"), "mae": 0.0, "sample_count": 1, "forward_calls": 1}
        )
