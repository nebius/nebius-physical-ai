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
            "deterministic_noise_protocol": (
                "npa.lerobot.smolvla-explicit-flow-noise.v1"
            ),
            "repeated_paired_trials_required": False,
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
        except (
            OSError,
            urllib.error.HTTPError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
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

    def infer_typed(
        self,
        *,
        observation: Any,
        policy: Any,
        sampling_seed: int | None = None,
        sampling_sequence: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Infer one typed action and return transport provenance."""

        if policy.backend != "lerobot_smolvla":
            raise SmolVLAPolicyClientError(
                "typed SmolVLA client requires lerobot_smolvla"
            )
        request_payload = observation.to_wire_dict()
        body = json.dumps(
            request_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if (sampling_seed is None) != (sampling_sequence is None):
            raise SmolVLAPolicyClientError(
                "paired SmolVLA sampling requires both seed and sequence"
            )
        request = self._inference_request(
            body,
            sampling_seed=sampling_seed,
            sampling_sequence=sampling_sequence,
        )
        started = time.perf_counter()
        response_body, response_headers = self._send_inference(request)
        output = self._typed_output(response_body, observation, policy)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return output, self._inference_receipt(
            body,
            response_body,
            response_headers,
            latency_ms,
            observation,
            sampling_seed=sampling_seed,
            sampling_sequence=sampling_sequence,
        )

    def _inference_request(
        self,
        body: bytes,
        *,
        sampling_seed: int | None,
        sampling_sequence: str | None,
    ) -> urllib.request.Request:
        """Build one authenticated request without ambient proxy routing."""

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if sampling_seed is not None and sampling_sequence is not None:
            headers.update(
                {
                    "X-NPA-Sampling-Seed": str(sampling_seed),
                    "X-NPA-Sampling-Sequence": sampling_sequence,
                }
            )
        return urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers=headers,
        )

    def _send_inference(
        self, request: urllib.request.Request
    ) -> tuple[bytes, dict[str, str]]:
        """Send one request and normalize transport failures."""

        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=self._timeout_s) as response:
                return response.read(), {
                    key.lower(): value for key, value in response.headers.items()
                }
        except (OSError, urllib.error.HTTPError) as exc:
            raise SmolVLAPolicyClientError(
                f"SmolVLA inference request failed: {exc}"
            ) from exc

    @staticmethod
    def _typed_output(response_body: bytes, observation: Any, policy: Any) -> Any:
        """Decode and identity-bind one typed response."""

        try:
            payload = json.loads(response_body)
            output = policy_output_from_wire(payload, expected_action=policy.action)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            InterchangeContractError,
        ) as exc:
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
        return output

    @staticmethod
    def _inference_receipt(
        body: bytes,
        response: bytes,
        response_headers: dict[str, str],
        latency_ms: float,
        observation: Any,
        *,
        sampling_seed: int | None,
        sampling_sequence: str | None,
    ) -> dict[str, Any]:
        """Build transport and camera-byte provenance for one response."""

        noise_sha256 = response_headers.get("x-npa-sampler-noise-sha256")
        noise_shape_raw = response_headers.get("x-npa-sampler-noise-shape")
        returned_sequence = response_headers.get("x-npa-sampling-sequence")
        fresh_consumed = response_headers.get("x-npa-fresh-observation-consumed")
        paired = sampling_seed is not None
        try:
            noise_shape = (
                [int(value) for value in noise_shape_raw.split(",")]
                if noise_shape_raw is not None
                else None
            )
        except ValueError as exc:
            raise SmolVLAPolicyClientError(
                "SmolVLA response has an invalid sampler-noise shape"
            ) from exc
        if paired and (
            noise_sha256 is None
            or len(noise_sha256) != 64
            or any(character not in "0123456789abcdef" for character in noise_sha256)
            or returned_sequence != sampling_sequence
            or fresh_consumed != "true"
            or noise_shape is None
            or len(noise_shape) != 3
            or noise_shape[0] != 1
            or any(value < 1 for value in noise_shape)
        ):
            raise SmolVLAPolicyClientError(
                "SmolVLA response lacks bound fresh-observation sampling evidence"
            )
        return {
            "backend": "lerobot_smolvla",
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "response_sha256": hashlib.sha256(response).hexdigest(),
            "latency_ms": latency_ms,
            "round_trip_ms": latency_ms,
            "deterministic_noise_coupled": paired,
            "repeated_paired_trials_required": not paired,
            "fresh_observation_consumed": fresh_consumed == "true",
            "sampler_noise_sha256": noise_sha256,
            "sampler_noise_shape": noise_shape[1:] if noise_shape else None,
            "sampling_seed": sampling_seed,
            "sampling_sequence": returned_sequence,
            "camera_sha256": {view.key: view.rgb_sha256 for view in observation.views},
        }
