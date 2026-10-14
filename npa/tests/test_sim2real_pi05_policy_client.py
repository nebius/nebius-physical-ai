from __future__ import annotations

import hashlib
import numpy as np
import pytest
import sys
import types

from npa.workflows.sim2real.interchange_contract import (
    GripperMapping,
    PolicyObservation,
    TimestampedCameraObservation,
)
from npa.workflows.sim2real.pi05_policy_client import (
    Pi05PolicyClient,
    Pi05PolicyClientError,
    _pack_array,
    _sampler_noise_identity,
    _sampling_request,
    _unpack_array,
)


def _map_pack(value, default):
    if isinstance(value, dict):
        return {key: _map_pack(item, default) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_map_pack(item, default) for item in value]
    converted = default(value) if default else value
    return converted if converted is value else _map_pack(converted, default)


def _map_unpack(value, object_hook):
    if isinstance(value, dict):
        mapped = {key: _map_unpack(item, object_hook) for key, item in value.items()}
        return object_hook(mapped) if object_hook else mapped
    if isinstance(value, list):
        return [_map_unpack(item, object_hook) for item in value]
    return value


def _fake_msgpack():
    packed: dict[bytes, object] = {}

    def packb(value, default=None):
        token = f"message-{len(packed)}".encode()
        packed[token] = _map_pack(value, default)
        return token

    def unpackb(value, object_hook=None):
        return _map_unpack(packed[value], object_hook)

    return types.SimpleNamespace(
        packb=packb,
        unpackb=unpackb,
    )


class _Connection:
    def __init__(self, msgpack, sampling_noise: np.ndarray | None = None) -> None:
        metadata = {"server": "pinned-openpi"}
        response: dict[str, object] = {
            "actions": np.tile(
                np.asarray(
                    [0.0, -0.7, 0.0, -2.2, 0.0, 1.7, 0.7, 0.63],
                    dtype=np.float64,
                ),
                (15, 1),
            ),
            "server_timing": {"infer_ms": 12.5},
        }
        if sampling_noise is not None:
            metadata["npa_explicit_noise_protocol"] = {
                "schema": "npa.openpi.pi05-explicit-noise.v1",
                "shape": list(sampling_noise.shape),
                "dtype": "float32",
            }
            response.update(
                {
                    "npa_sampler_noise_sha256": hashlib.sha256(
                        sampling_noise.tobytes()
                    ).hexdigest(),
                    "npa_sampler_noise_shape": list(sampling_noise.shape),
                    "npa_sampling_sequence": "scene-9-decision-0",
                    "npa_wrist_transform_receipt": {
                        "schema": "npa.openpi.pi05-live-wrist-transform.v1",
                        "source_wrist_sha256": hashlib.sha256(
                            np.ones((224, 224, 3), dtype=np.uint8).tobytes()
                        ).hexdigest(),
                        "left_wrist_0_rgb_sha256": "a" * 64,
                        "right_wrist_0_rgb_sha256": "b" * 64,
                        "left_wrist_mask": True,
                        "right_wrist_mask": False,
                    },
                }
            )
        self.responses = [
            msgpack.packb(metadata),
            msgpack.packb(response, default=_pack_array),
        ]
        self.sent: list[bytes] = []

    def recv(self):
        return self.responses.pop(0)

    def send(self, value: bytes) -> None:
        self.sent.append(value)


def test_client_preserves_nonbinary_droid_gripper_prediction(monkeypatch) -> None:
    msgpack = _fake_msgpack()
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    connection = _Connection(msgpack)
    connect_options = {}

    def connect(*_args, **kwargs):
        connect_options.update(kwargs)
        return connection

    monkeypatch.setattr("websockets.sync.client.connect", connect)
    client = Pi05PolicyClient("private-service", auth_token="test-only-token")
    wrist = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    wrist_hash = hashlib.sha256(wrist.tobytes()).hexdigest()
    actions, evidence = client.infer(
        exterior=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist=wrist,
        joint_position=np.zeros(7, dtype=np.float32),
        gripper_position=np.asarray([0.0], dtype=np.float32),
        prompt="place the object on the target surface",
        capture_age_s=0.01,
        source_wrist_sha256=wrist_hash,
    )
    assert actions.shape == (15, 8)
    assert actions[0, 7] == 0.63
    assert evidence["dtype"] == "float64"
    sent = msgpack.unpackb(connection.sent[0], object_hook=_unpack_array)
    assert np.array_equal(sent["observation/wrist_image_left"], wrist)
    assert evidence["wire_input"]["wrist_rgb_sha256"] == wrist_hash
    assert evidence["wire_input"]["wrist_source_hash_matched"] is True
    assert connect_options["additional_headers"] == {
        "Authorization": "Bearer test-only-token"
    }
    assert evidence["wire_input"]["capture_age_s"] == 0.01
    assert evidence["sampler_noise_sha256"] is None


def test_client_refuses_missing_auth_token(monkeypatch) -> None:
    msgpack = _fake_msgpack()
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    monkeypatch.delenv("NPA_OPENPI_SERVICE_AUTH_TOKEN", raising=False)
    with pytest.raises(Pi05PolicyClientError, match="auth token is required"):
        Pi05PolicyClient("private-service")


def test_sampler_noise_identity_is_strict_and_optional() -> None:
    assert _sampler_noise_identity({}) is None
    assert _sampler_noise_identity({"npa_sampler_noise_sha256": "a" * 64}) == "a" * 64
    with pytest.raises(Pi05PolicyClientError, match="malformed"):
        _sampler_noise_identity({"npa_sampler_noise_sha256": "not-a-digest"})


def test_explicit_noise_uses_server_model_shape_not_action_output_shape() -> None:
    metadata = {
        "npa_explicit_noise_protocol": {
            "schema": "npa.openpi.pi05-explicit-noise.v1",
            "shape": [15, 32],
            "dtype": "float32",
        }
    }
    noise = np.ones((15, 32), dtype=np.float32)
    receipt = _sampling_request(metadata, noise, "scene-9-decision-0")
    assert receipt is not None
    assert receipt["shape"] == [15, 32]
    assert receipt["payload"]["npa/sampling_noise"].shape != (15, 8)
    with pytest.raises(Pi05PolicyClientError, match="shape/dtype"):
        _sampling_request(metadata, np.ones((15, 8), dtype=np.float32), "sequence")


def test_client_binds_real_response_to_explicit_noise(monkeypatch) -> None:
    msgpack = _fake_msgpack()
    noise = np.ones((15, 32), dtype=np.float32)
    connection = _Connection(msgpack, noise)
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    monkeypatch.setattr(
        "websockets.sync.client.connect", lambda *args, **kwargs: connection
    )
    client = Pi05PolicyClient("private-service", auth_token="test-only-token")
    _, evidence = client.infer(
        exterior=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist=np.ones((224, 224, 3), dtype=np.uint8),
        joint_position=np.zeros(7, dtype=np.float32),
        gripper_position=np.asarray([0.0], dtype=np.float32),
        prompt="place the object",
        sampling_noise=noise,
        sampling_sequence="scene-9-decision-0",
    )
    assert (
        evidence["sampler_noise_sha256"] == hashlib.sha256(noise.tobytes()).hexdigest()
    )
    assert evidence["sampling_sequence"] == "scene-9-decision-0"


@pytest.mark.parametrize(
    ("capture_age_s", "source_hash", "message"),
    [
        (0.2, "", "stale"),
        (0.0, "0" * 64, "changed before wire"),
    ],
)
def test_client_rejects_stale_or_hash_mismatched_wrist(
    monkeypatch, capture_age_s: float, source_hash: str, message: str
) -> None:
    msgpack = _fake_msgpack()
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    connection = _Connection(msgpack)
    monkeypatch.setattr(
        "websockets.sync.client.connect", lambda *args, **kwargs: connection
    )
    client = Pi05PolicyClient("private-service", auth_token="test-only-token")
    with pytest.raises(Pi05PolicyClientError, match=message):
        client.infer(
            exterior=np.zeros((224, 224, 3), dtype=np.uint8),
            wrist=np.ones((224, 224, 3), dtype=np.uint8),
            joint_position=np.zeros(7, dtype=np.float32),
            gripper_position=np.asarray([0.0], dtype=np.float32),
            prompt="place the object on the target surface",
            capture_age_s=capture_age_s,
            source_wrist_sha256=source_hash,
        )


def _typed_observation(*, arm_dof: int) -> tuple[PolicyObservation, object, object]:
    arm = tuple(f"joint_{index}" for index in range(arm_dof))
    fingers = ("finger_left", "finger_right")
    rgb = bytes(224 * 224 * 3)
    digest = hashlib.sha256(rgb).hexdigest()
    views = tuple(
        TimestampedCameraObservation(
            key=role,
            role=role,
            calibration_sha256="c" * 64,
            image_size=(224, 224),
            layout="HWC",
            mask=True,
            capture_timestamp_s=1.0,
            state_timestamp_s=1.0,
            rgb8=rgb,
            rgb_sha256=digest,
        )
        for role in ("exterior", "wrist")
    )
    identity = f"arm-{arm_dof}"
    policy_identity = f"pi05-{arm_dof}"
    observation = PolicyObservation(
        input_identity="input-1",
        robot_identity=identity,
        task_identity="surface-v1",
        policy_identity=policy_identity,
        prompt="place the object",
        views=views,
        measured_state_names=arm + fingers,
        measured_state_units=("rad",) * arm_dof + ("m", "m"),
        measured_state=(0.0,) * arm_dof + (0.04, 0.04),
    )
    robot = types.SimpleNamespace(
        identity=identity,
        arm_joint_names=arm,
        measured_joint_names=arm + fingers,
        gripper=GripperMapping(
            joint_names=fingers,
            joint_units=("m", "m"),
            finger_links=("left_pad", "right_pad"),
            open_joint_positions=(0.04, 0.04),
            closed_joint_positions=(0.0, 0.0),
            maximum_width_m=0.08,
            model_semantics="normalized_open",
            model_open=0.0,
            model_closed=1.0,
        ),
    )
    policy = types.SimpleNamespace(
        backend="openpi_pi05",
        identity=policy_identity,
        task_identity="surface-v1",
        checkpoint_sha256="d" * 64,
        action=types.SimpleNamespace(
            names=arm + ("gripper",),
            execute_prefix=1,
            horizon=15,
            to_dict=lambda: {
                "representation": "absolute_joint_position",
                "names": list(arm + ("gripper",)),
            },
        ),
    )
    return observation, robot, policy


def test_typed_openpi_adapter_rejects_six_dof_without_padding(monkeypatch) -> None:
    msgpack = _fake_msgpack()
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    connection = _Connection(msgpack)
    monkeypatch.setattr(
        "websockets.sync.client.connect", lambda *args, **kwargs: connection
    )
    client = Pi05PolicyClient("private-service", auth_token="test-only-token")
    observation, robot, policy = _typed_observation(arm_dof=6)
    with pytest.raises(Pi05PolicyClientError, match="seven arm joints"):
        client.infer_typed(observation=observation, robot=robot, policy=policy)
    assert connection.sent == []


def test_typed_openpi_adapter_preserves_wrist_bytes_and_gripper_mapping(
    monkeypatch,
) -> None:
    msgpack = _fake_msgpack()
    monkeypatch.setitem(sys.modules, "msgpack", msgpack)
    connection = _Connection(msgpack)
    monkeypatch.setattr(
        "websockets.sync.client.connect", lambda *args, **kwargs: connection
    )
    client = Pi05PolicyClient("private-service", auth_token="test-only-token")
    observation, robot, policy = _typed_observation(arm_dof=7)
    output, evidence = client.infer_typed(
        observation=observation, robot=robot, policy=policy
    )
    assert len(output.action_chunk) == 15
    assert len(output.action_chunk[0]) == 8
    assert output.checkpoint_sha256 == "d" * 64
    sent = msgpack.unpackb(connection.sent[0], object_hook=_unpack_array)
    assert sent["observation/joint_position"].shape == (7,)
    assert sent["observation/gripper_position"].tolist() == pytest.approx(
        [0.0], abs=1.0e-6
    )
    assert evidence["wire_input"]["wrist_rgb_sha256"] == hashlib.sha256(
        observation.views[1].rgb8
    ).hexdigest()
