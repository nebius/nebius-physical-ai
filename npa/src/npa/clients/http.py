"""HTTP client for the npa-lerobot-server FastAPI endpoints."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx


class ServerError(Exception):
    pass


class HTTPClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0, retries: int = 3) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = retries

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        effective_timeout = timeout or self._timeout
        last_exc: Exception | None = None

        for attempt in range(self._retries):
            try:
                resp = httpx.request(
                    method, url, json=json, timeout=effective_timeout
                )
                if resp.status_code >= 500:
                    raise ServerError(f"Server error {resp.status_code}: {resp.text}")
                if resp.status_code >= 400:
                    raise ServerError(f"Client error {resp.status_code}: {resp.text}")
                return resp.json()
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self._retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self._retries - 1:
                    time.sleep(2 ** attempt)
                continue

        raise ServerError(
            f"Failed to reach {url} after {self._retries} attempts: {last_exc}\n"
            f"Check NPA_WORKBENCH_ENDPOINT or ~/.npa/config.yaml"
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def serve(
        self,
        checkpoint: str,
        env_type: str | None = None,
        env_task: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"checkpoint": checkpoint}
        if env_type:
            payload["env_type"] = env_type
        if env_task:
            payload["env_task"] = env_task
        return self._request("POST", "/serve", json=payload)

    def serve_model(self, model: str, *, timeout: float | None = None) -> dict[str, Any]:
        return self._request("POST", "/serve", json={"model": model}, timeout=timeout)

    def stop_serve(self) -> dict[str, Any]:
        return self._request("DELETE", "/serve")

    def infer(self, observation: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
        return self._request("POST", "/infer", json=observation, timeout=timeout)

    def job_status(self, job_id: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{quote(job_id, safe='')}", timeout=timeout)

    def wait_healthy(self, *, timeout: float = 60.0, interval: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.health()
                return True
            except ServerError:
                time.sleep(interval)
        return False
