from __future__ import annotations

import numpy as np
import pickle
import sys
import types

from npa.workflows.sim2real.pi05_policy_client import (
    Pi05PolicyClient,
    _pack_array,
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
    return types.SimpleNamespace(
        packb=lambda value, default=None: pickle.dumps(_map_pack(value, default)),
        unpackb=lambda value, object_hook=None: _map_unpack(
            pickle.loads(value), object_hook
        ),
    )


class _Connection:
    def __init__(self, msgpack) -> None:
        self.responses = [
            msgpack.packb({"server": "pinned-openpi"}),
            msgpack.packb(
                {
                    "actions": np.tile(
                        np.asarray(
                            [0.0, -0.7, 0.0, -2.2, 0.0, 1.7, 0.7, 0.63],
                            dtype=np.float64,
                        ),
                        (15, 1),
                    ),
                    "server_timing": {"infer_ms": 12.5},
                },
                default=_pack_array,
            ),
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
    monkeypatch.setattr(
        "websockets.sync.client.connect", lambda *args, **kwargs: connection
    )
    client = Pi05PolicyClient("private-service")
    actions, evidence = client.infer(
        exterior=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist=np.ones((224, 224, 3), dtype=np.uint8),
        joint_position=np.zeros(7, dtype=np.float32),
        gripper_position=np.asarray([0.0], dtype=np.float32),
        prompt="place the object on the target surface",
    )
    assert actions.shape == (15, 8)
    assert actions[0, 7] == 0.63
    assert evidence["dtype"] == "float64"
    sent = msgpack.unpackb(connection.sent[0], object_hook=_unpack_array)
    assert sent["observation/wrist_image_left"].shape == (224, 224, 3)
