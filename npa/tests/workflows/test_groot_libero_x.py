"""Fail-closed contracts for the GR00T LIBERO-X closed-loop workflow."""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
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
        == "documented_python_runtime_missing"
    )
    assert workflow._runtime_ready(target) is None


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
    monkeypatch.setattr(workflow, "_download_prefix", lambda *_args: None)
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
    assert "(\'native evaluator" not in str(error.value)


def test_numeric_metrics_rejects_empty_or_nonfinite_native_measurements() -> None:
    with pytest.raises(workflow.GrootVisualizationError, match="at least one sample"):
        workflow._numeric_metrics(
            {"mse": 0.0, "mae": 0.0, "sample_count": 0, "forward_calls": 0}
        )
    with pytest.raises(workflow.GrootVisualizationError, match="finite"):
        workflow._numeric_metrics(
            {"mse": float("nan"), "mae": 0.0, "sample_count": 1, "forward_calls": 1}
        )
