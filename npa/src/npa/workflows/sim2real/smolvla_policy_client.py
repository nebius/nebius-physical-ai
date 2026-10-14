"""Authenticated HTTP client for the typed LeRobot SmolVLA policy seam."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from npa.workflows.sim2real.interchange_contract import (
    InterchangeContractError,
    policy_output_from_wire,
)


class SmolVLAPolicyClientError(RuntimeError):
    """Raised when transport or typed policy output violates the contract."""


class SmolVLAPolicyClient:
    """Send one hash-bound typed observation to a private native SmolVLA service."""

    def __init__(
        self,
        host: str,
        port: int = 8080,
        *,
        auth_token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        token = auth_token or os.environ.get("NPA_POLICY_AUTH_TOKEN", "")
        if not token:
            raise SmolVLAPolicyClientError("SmolVLA policy service token is required")
        if not host.strip() or not 0 < port < 65536 or timeout_s <= 0:
            raise SmolVLAPolicyClientError("invalid SmolVLA service endpoint")
        self._url = f"http://{host}:{port}/infer-typed"
        self._reset_url = f"http://{host}:{port}/reset"
        self._token = token
        self._timeout_s = timeout_s
        self.metadata = {
            "backend": "lerobot_smolvla",
            "deterministic_noise_protocol": None,
            "repeated_paired_trials_required": True,
        }

    def reset(self, *, policy: Any) -> dict[str, Any]:
        """Reset the native action queue and verify the loaded policy identity."""

        request = urllib.request.Request(
            self._reset_url,
            data=b"{}",
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self._timeout_s) as response:
                body = response.read()
            payload = json.loads(body)
        except (OSError, urllib.error.HTTPError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmolVLAPolicyClientError(
                f"SmolVLA reset request failed: {exc}"
            ) from exc
        if (
            payload.get("status") != "reset"
            or payload.get("policy_identity") != policy.identity
            or payload.get("checkpoint_sha256") != policy.checkpoint_sha256
        ):
            raise SmolVLAPolicyClientError(
                "SmolVLA reset receipt does not match the selected policy"
            )
        return {
            **payload,
            "receipt_sha256": hashlib.sha256(body).hexdigest(),
        }

    def infer_typed(self, *, observation: Any, policy: Any) -> tuple[Any, dict[str, Any]]:
        if policy.backend != "lerobot_smolvla":
            raise SmolVLAPolicyClientError(
                "typed SmolVLA client requires lerobot_smolvla"
            )
        request_payload = observation.to_wire_dict()
        body = json.dumps(
            request_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self._timeout_s) as response:
                response_body = response.read()
        except (OSError, urllib.error.HTTPError) as exc:
            raise SmolVLAPolicyClientError(
                f"SmolVLA inference request failed: {exc}"
            ) from exc
        try:
            payload = json.loads(response_body)
            output = policy_output_from_wire(payload, expected_action=policy.action)
        except (UnicodeDecodeError, json.JSONDecodeError, InterchangeContractError) as exc:
            raise SmolVLAPolicyClientError(
                f"SmolVLA typed response is invalid: {exc}"
            ) from exc
        if (
            output.observation_input_identity != observation.input_identity
            or output.policy_identity != policy.identity
            or output.checkpoint_sha256 != policy.checkpoint_sha256
        ):
            raise SmolVLAPolicyClientError(
                "SmolVLA response identities do not match the request/profile"
            )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return output, {
            "backend": "lerobot_smolvla",
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "response_sha256": hashlib.sha256(response_body).hexdigest(),
            "latency_ms": latency_ms,
            "round_trip_ms": latency_ms,
            "deterministic_noise_coupled": False,
            "repeated_paired_trials_required": True,
            "camera_sha256": {
                view.key: view.rgb_sha256 for view in observation.views
            },
        }
