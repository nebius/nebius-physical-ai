from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real.interchange_contract import PolicyAction
from npa.workflows.sim2real.smolvla_policy_client import (
    SmolVLAPolicyClient,
    SmolVLAPolicyClientError,
)


ACTION = PolicyAction(
    representation="absolute_joint_position",
    names=("j1", "j2", "j3", "j4", "j5", "j6", "gripper"),
    units=("rad", "rad", "rad", "rad", "rad", "rad", "normalized_closed"),
    frame="joint",
    gripper_semantics="normalized_closed",
    horizon=50,
    control_hz=15.0,
    execute_prefix=1,
)


@contextmanager
def _server(payload: dict, *, response_headers: dict[str, str] | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            self.server.request_body = self.rfile.read(length)  # type: ignore[attr-defined]
            self.server.authorization = self.headers.get("Authorization")  # type: ignore[attr-defined]
            self.server.sampling_seed = self.headers.get("X-NPA-Sampling-Seed")  # type: ignore[attr-defined]
            self.server.sampling_sequence = self.headers.get("X-NPA-Sampling-Sequence")  # type: ignore[attr-defined]
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in (response_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def _response() -> dict:
    return {
        "schema": "npa.sim2real.policy_output.v1",
        "observation_input_identity": "input-1",
        "policy_identity": "smolvla-ur10",
        "checkpoint_sha256": "a" * 64,
        "action": ACTION.to_dict(),
        "action_chunk": [[0.0] * 7],
        "generated_timestamp_s": 2.0,
    }


def test_client_requires_private_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_POLICY_AUTH_TOKEN", raising=False)
    with pytest.raises(SmolVLAPolicyClientError, match="token"):
        SmolVLAPolicyClient("policy.default.svc")


def test_client_preserves_typed_identities_and_auth() -> None:
    observation = SimpleNamespace(
        input_identity="input-1",
        views=(),
        to_wire_dict=lambda: {
            "schema": "npa.sim2real.policy_observation.v1",
            "input_identity": "input-1",
        },
    )
    policy = SimpleNamespace(
        backend="lerobot_smolvla",
        identity="smolvla-ur10",
        checkpoint_sha256="a" * 64,
        action=ACTION,
    )
    with _server(_response()) as server:
        client = SmolVLAPolicyClient(
            "127.0.0.1", server.server_port, auth_token="task-token"
        )
        output, evidence = client.infer_typed(observation=observation, policy=policy)
    assert output.policy_identity == policy.identity
    assert evidence["deterministic_noise_coupled"] is False
    assert evidence["repeated_paired_trials_required"] is True
    assert server.authorization == "Bearer task-token"  # type: ignore[attr-defined]


def test_client_rejects_server_action_contract_change() -> None:
    payload = _response()
    payload["action"]["units"][0] = "degree"
    observation = SimpleNamespace(
        input_identity="input-1",
        views=(),
        to_wire_dict=lambda: {"schema": "npa.sim2real.policy_observation.v1"},
    )
    policy = SimpleNamespace(
        backend="lerobot_smolvla",
        identity="smolvla-ur10",
        checkpoint_sha256="a" * 64,
        action=ACTION,
    )
    with _server(payload) as server:
        client = SmolVLAPolicyClient(
            "127.0.0.1", server.server_port, auth_token="task-token"
        )
        with pytest.raises(SmolVLAPolicyClientError, match="changed"):
            client.infer_typed(observation=observation, policy=policy)


def test_client_binds_explicit_paired_noise_and_fresh_observation() -> None:
    observation = SimpleNamespace(
        input_identity="input-1",
        views=(),
        to_wire_dict=lambda: {
            "schema": "npa.sim2real.policy_observation.v1",
            "input_identity": "input-1",
        },
    )
    policy = SimpleNamespace(
        backend="lerobot_smolvla",
        identity="smolvla-ur10",
        checkpoint_sha256="a" * 64,
        action=ACTION,
    )
    sequence = "scene-9-decision-3"
    headers = {
        "X-NPA-Fresh-Observation-Consumed": "true",
        "X-NPA-Sampler-Noise-SHA256": "b" * 64,
        "X-NPA-Sampler-Noise-Shape": "1,50,32",
        "X-NPA-Sampling-Sequence": sequence,
    }
    with _server(_response(), response_headers=headers) as server:
        client = SmolVLAPolicyClient(
            "127.0.0.1", server.server_port, auth_token="task-token"
        )
        _, evidence = client.infer_typed(
            observation=observation,
            policy=policy,
            sampling_seed=17,
            sampling_sequence=sequence,
        )
    assert evidence["fresh_observation_consumed"] is True
    assert evidence["deterministic_noise_coupled"] is True
    assert evidence["sampler_noise_sha256"] == "b" * 64
    assert evidence["sampler_noise_shape"] == [50, 32]
    assert server.sampling_seed == "17"  # type: ignore[attr-defined]
    assert server.sampling_sequence == sequence  # type: ignore[attr-defined]


def test_client_verifies_native_episode_reset_receipt() -> None:
    policy = SimpleNamespace(
        identity="smolvla-ur10",
        checkpoint_sha256="a" * 64,
    )
    payload = {
        "status": "reset",
        "policy_identity": policy.identity,
        "checkpoint_sha256": policy.checkpoint_sha256,
    }
    with _server(payload) as server:
        client = SmolVLAPolicyClient(
            "127.0.0.1", server.server_port, auth_token="task-token"
        )
        receipt = client.reset(policy=policy)
    assert receipt["status"] == "reset"
    assert len(receipt["receipt_sha256"]) == 64
    assert server.authorization == "Bearer task-token"  # type: ignore[attr-defined]
