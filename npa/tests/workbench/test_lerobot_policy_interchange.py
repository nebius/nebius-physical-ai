"""Contract tests for real LeRobot policy-container interchange wiring."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.lerobot.policy_container import (
    INTERCHANGE_MANIFEST_FILENAME,
    PolicyContainerError,
    _load_interchange_manifest,
    _processor_identity,
    _smolvla_batch,
    _validate_native_policy_features,
    build_parser,
    main,
    write_smolvla_interchange_manifest,
)


def _checkpoint(tmp_path: Path) -> tuple[Path, str]:
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    weights = checkpoint / "pytorch_model.bin"
    weights.write_bytes(b"real-checkpoint-binding-fixture")
    (checkpoint / "policy_preprocessor.json").write_text(
        '{"normalize":{"mean":[0.1],"std":[0.2]}}\n', encoding="utf-8"
    )
    (checkpoint / "policy_postprocessor.json").write_text(
        '{"unnormalize":{"mean":[0.1],"std":[0.2]}}\n', encoding="utf-8"
    )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    return checkpoint, digest


def _manifest(digest: str, normalization_identity: str = "c" * 64) -> dict:
    return {
        "schema": "npa.sim2real.lerobot_policy_adapter.v1",
        "backend": "lerobot_smolvla",
        "policy_identity": "smolvla-six-axis",
        "robot_identity": "six-axis-arm",
        "task_identity": "released-surface-v1",
        "normalization_identity": normalization_identity,
        "checkpoint_sha256": digest,
        "measured_state_names": [
            "j1",
            "j2",
            "j3",
            "j4",
            "j5",
            "j6",
            "left_finger",
            "right_finger",
        ],
        "state_feature_key": "observation.state",
        "action_feature_key": "action",
        "cameras": [
            {
                "key": "scene",
                "feature_key": "observation.images.scene",
                "layout": "HWC",
                "image_size_wh": [224, 224],
                "maximum_state_skew_s": 1.0 / 60.0,
            },
            {
                "key": "tool",
                "feature_key": "observation.images.tool",
                "layout": "HWC",
                "image_size_wh": [224, 224],
                "maximum_state_skew_s": 1.0 / 60.0,
            },
        ],
        "action": {
            "representation": "absolute_joint_position",
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
            "units": ["rad", "rad", "rad", "rad", "rad", "rad", "normalized_closed"],
            "gripper_semantics": "normalized_closed",
            "horizon": 50,
            "control_hz": 15.0,
            "execute_prefix": 1,
        },
    }


def _write_manifest(checkpoint: Path, payload: dict) -> None:
    (checkpoint / INTERCHANGE_MANIFEST_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _png() -> str:
    pillow = pytest.importorskip("PIL.Image")
    image = pillow.new("RGB", (224, 224), (12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _observation(manifest: dict) -> dict:
    return {
        "schema": "npa.sim2real.policy_observation.v1",
        "policy_identity": manifest["policy_identity"],
        "robot_identity": manifest["robot_identity"],
        "task_identity": manifest["task_identity"],
        "prompt": "place the object",
        "privileged_object_state": False,
        "measured_state_names": manifest["measured_state_names"],
        "measured_state": [0.0] * len(manifest["measured_state_names"]),
        "views": [
            {
                "key": camera["key"],
                "layout": camera["layout"],
                "mask": True,
                "capture_timestamp_s": 2.0,
                "state_timestamp_s": 2.0,
                "image_base64": _png(),
            }
            for camera in manifest["cameras"]
        ],
    }


def test_manifest_is_cryptographically_bound_to_real_weight_file(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, files = _processor_identity(checkpoint)
    payload = _manifest(digest, normalization)
    payload["normalization_files"] = files
    _write_manifest(checkpoint, payload)
    assert _load_interchange_manifest(checkpoint)["checkpoint_sha256"] == digest
    payload["checkpoint_sha256"] = "0" * 64
    _write_manifest(checkpoint, payload)
    with pytest.raises(PolicyContainerError, match="not bound"):
        _load_interchange_manifest(checkpoint)


def test_writer_binds_complete_adapter_to_weight_bytes(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, _ = _processor_identity(checkpoint)
    manifest = _manifest(digest, normalization)
    path = write_smolvla_interchange_manifest(
        checkpoint,
        policy_identity=manifest["policy_identity"],
        robot_identity=manifest["robot_identity"],
        task_identity=manifest["task_identity"],
        normalization_identity=manifest["normalization_identity"],
        measured_state_names=manifest["measured_state_names"],
        cameras=manifest["cameras"],
        action=manifest["action"],
    )
    payload = json.loads(path.read_text())
    assert payload["checkpoint_sha256"] == digest
    assert payload["normalization_identity"] == normalization
    assert payload["action"]["names"][-1] == "gripper"


def test_writer_derives_normalization_identity_from_native_processors(
    tmp_path: Path,
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    manifest = _manifest(digest)
    path = write_smolvla_interchange_manifest(
        checkpoint,
        policy_identity=manifest["policy_identity"],
        robot_identity=manifest["robot_identity"],
        task_identity=manifest["task_identity"],
        measured_state_names=manifest["measured_state_names"],
        cameras=manifest["cameras"],
        action=manifest["action"],
    )
    payload = json.loads(path.read_text())
    assert payload["normalization_identity"] == _processor_identity(checkpoint)[0]


def test_manifest_rejects_changed_normalization_processor(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, files = _processor_identity(checkpoint)
    payload = _manifest(digest, normalization)
    payload["normalization_files"] = files
    _write_manifest(checkpoint, payload)
    (checkpoint / "policy_preprocessor.json").write_text(
        '{"normalize":{"mean":[9.9],"std":[0.2]}}\n', encoding="utf-8"
    )
    with pytest.raises(PolicyContainerError, match="normalization"):
        _load_interchange_manifest(checkpoint)


def test_checkpoint_features_must_match_camera_state_and_action_dimensions() -> None:
    manifest = _manifest("a" * 64)
    feature = lambda shape: SimpleNamespace(shape=shape)  # noqa: E731
    policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.state": feature((8,)),
                "observation.images.scene": feature((3, 224, 224)),
                "observation.images.tool": feature((3, 224, 224)),
            },
            output_features={"action": feature((7,))},
        )
    )
    _validate_native_policy_features(policy, manifest)
    policy.config.output_features["action"] = feature((8,))
    with pytest.raises(PolicyContainerError, match="action feature"):
        _validate_native_policy_features(policy, manifest)


def test_real_image_decode_and_stale_camera_rejection() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    batch = _smolvla_batch(observation, manifest, "cpu")
    assert tuple(batch["observation.images.scene"].shape) == (3, 224, 224)
    assert tuple(batch["observation.state"].shape) == (8,)
    observation["views"][1]["capture_timestamp_s"] = 1.0
    with pytest.raises(PolicyContainerError, match="stale"):
        _smolvla_batch(observation, manifest, "cpu")


def test_typed_raw_rgb_wire_is_consumed_and_hash_bound() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    rgb8 = bytes([23]) * (224 * 224 * 3)
    for view in observation["views"]:
        view.pop("image_base64")
        view["image_rgb8_base64"] = base64.b64encode(rgb8).decode("ascii")
        view["image_sha256"] = hashlib.sha256(rgb8).hexdigest()
    batch = _smolvla_batch(observation, manifest, "cpu")
    assert tuple(batch["observation.images.tool"].shape) == (3, 224, 224)
    observation["views"][1]["image_sha256"] = "0" * 64
    with pytest.raises(PolicyContainerError, match="hash"):
        _smolvla_batch(observation, manifest, "cpu")


def test_six_axis_manifest_cannot_accept_reordered_or_padded_state() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    observation["measured_state_names"] = [
        observation["measured_state_names"][1],
        observation["measured_state_names"][0],
        *observation["measured_state_names"][2:],
    ]
    with pytest.raises(PolicyContainerError, match="names/order"):
        _smolvla_batch(observation, manifest, "cpu")
    observation = _observation(manifest)
    observation["measured_state"].insert(6, 0.0)
    with pytest.raises(PolicyContainerError, match="complete and finite"):
        _smolvla_batch(observation, manifest, "cpu")


def test_smolvla_rejects_structured_privileged_object_state() -> None:
    pytest.importorskip("torch")
    manifest = _manifest("a" * 64)
    observation = _observation(manifest)
    observation["privileged_object_state"] = {"object_xyz": [0.0, 0.0, 0.0]}
    with pytest.raises(PolicyContainerError, match="privileged"):
        _smolvla_batch(observation, manifest, "cpu")


def test_cartesian_manifest_requires_qualified_controller(tmp_path: Path) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    normalization, files = _processor_identity(checkpoint)
    payload = _manifest(digest, normalization)
    payload["normalization_files"] = files
    payload["action"]["representation"] = "ee_pose"
    _write_manifest(checkpoint, payload)
    with pytest.raises(PolicyContainerError, match="IK/controller"):
        _load_interchange_manifest(checkpoint)


def test_interchange_cli_requires_explicit_checkpoint_inputs_and_outputs() -> None:
    parser = build_parser()
    bound = parser.parse_args(
        [
            "bind-interchange",
            "--checkpoint-path",
            "/tmp/checkpoint",
            "--adapter-spec",
            "/tmp/adapter.json",
            "--output-json",
            "/tmp/bound.json",
        ]
    )
    assert bound.command == "bind-interchange"
    inferred = parser.parse_args(
        [
            "infer-interchange",
            "--checkpoint-path",
            "/tmp/checkpoint",
            "--observation-json",
            "/tmp/observation.json",
            "--output-json",
            "/tmp/output.json",
        ]
    )
    assert inferred.command == "infer-interchange"
    trained = parser.parse_args(
        [
            "train",
            "--dataset-repo-id",
            "operator/surface-placement",
            "--output-dir",
            "/tmp/output",
            "--steps",
            "100",
            "--interchange-adapter-relative-path",
            "meta/npa_policy_adapter.json",
        ]
    )
    assert (
        trained.interchange_adapter_relative_path
        == "meta/npa_policy_adapter.json"
    )


def test_health_exposes_loaded_interchange_identities_and_byte_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest_path = checkpoint / INTERCHANGE_MANIFEST_FILENAME
    manifest_path.write_text('{"sealed":true}\n', encoding="utf-8")
    manifest = _manifest("a" * 64, "b" * 64)
    runtime = SimpleNamespace(
        checkpoint_path=checkpoint,
        manifest=manifest,
        validation=SimpleNamespace(to_dict=lambda: {"status": "loadable"}),
    )
    monkeypatch.setenv("NPA_POLICY_CHECKPOINT", str(checkpoint))
    import npa.workbench.lerobot.policy_container as policy_container

    monkeypatch.setattr(policy_container, "LeRobotPolicyRuntime", lambda _: runtime)
    monkeypatch.setattr(
        policy_container,
        "assert_lerobot_importable",
        lambda: SimpleNamespace(to_dict=lambda: {"status": "ok"}),
    )
    response = fastapi_testclient.TestClient(policy_container.create_app()).get(
        "/health"
    )
    assert response.status_code == 200
    interchange = response.json()["interchange"]
    assert interchange["robot_identity"] == "six-axis-arm"
    assert interchange["checkpoint_sha256"] == "a" * 64
    assert interchange["normalization_identity"] == "b" * 64
    assert interchange["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()


def test_interchange_training_forbids_early_checkpoint_only_upload() -> None:
    with pytest.raises(PolicyContainerError, match="checkpoint-only upload"):
        main(
            [
                "train",
                "--dataset-repo-id",
                "operator/surface-placement",
                "--output-dir",
                "/tmp/output",
                "--steps",
                "100",
                "--interchange-adapter-relative-path",
                "meta/npa_policy_adapter.json",
                "--checkpoint-s3-uri",
                "s3://example/checkpoint/",
            ]
        )
