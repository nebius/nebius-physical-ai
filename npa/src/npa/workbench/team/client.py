"""Provide one authenticated client for CLI, SDK, agents, and browser integrations."""

from __future__ import annotations

import re
from urllib.parse import quote, urlsplit

import httpx

from .errors import AuthenticationError, TeamError
from .models import SubmitRequest


class TeamClient:
    """Call team operations with an external identity token and no cloud credentials.

    Args:
        endpoint, token: HTTPS gateway and external bearer token.
        transport: Optional HTTP transport for deterministic client tests.
    Returns:
        A TeamClient instance.
    Raises:
        TeamError: Endpoint or bearer token is invalid.
    """

    def __init__(self, endpoint: str, token: str, *, transport=None):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            raise TeamError("team endpoint must use HTTPS without embedded credentials")
        if not token or "\n" in token or "\r" in token:
            raise AuthenticationError("an external identity bearer token is required")
        self.http = httpx.Client(
            base_url=endpoint.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=False,
            transport=transport,
        )

    def close(self):
        """Release client connections.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self.http.close()

    def submit(self, request: SubmitRequest):
        """Submit a canonical workflow with a caller-retained idempotency key.

        Args:
            request: Workflow, placement, and caller-retained retry key.
        Returns:
            Server-issued run identity and status.
        Raises:
            TeamError: Authentication, transport, or submission fails.
        """
        return self._request("POST", "v1/runs", json=request.model_dump(mode="json"))

    def whoami(self):
        """Read verified identity and current personal workspace permissions.

        Args:
            None.
        Returns:
            External identity, groups, and authorized workspaces.
        Raises:
            TeamError: Authentication or transport fails.
        """
        return self._request("GET", "v1/me")

    def list(self, workspace: str):
        """List the authenticated person's own runs in a workspace.

        Args:
            workspace: Authorized workspace name.
        Returns:
            The authenticated person's visible run records.
        Raises:
            TeamError: Access or transport fails.
        """
        return self._request("GET", "v1/runs", params={"workspace": workspace})

    def run(self, run_id: str, action: str = "status"):
        """Read, cancel, or reconcile one owned run using the same authorization path.

        Args:
            run_id, action: Owned run and supported operation.
        Returns:
            Operation response decoded from JSON.
        Raises:
            TeamError: Identity, operation, or transport is invalid.
        """
        if action not in {"status", "cancel", "resume", "logs", "artifacts"}:
            raise TeamError("unsupported run operation")
        path = _run_path(run_id) + ("" if action == "status" else "/" + action)
        return self._request("POST" if action in {"cancel", "resume"} else "GET", path)

    def download(self, run_id: str, relative: str, destination):
        """Stream one owned artifact into a caller-selected binary output stream.

        Args:
            run_id, relative: Owned run and relative artifact key.
            destination: Writable binary stream owned by the caller.
        Returns:
            None.
        Raises:
            TeamError: Scope or authorization fails.
            httpx.HTTPError, OSError: Transfer or destination write fails.
        """
        from .storage import object_key

        object_key("artifacts", relative)
        path = _run_path(run_id) + "/artifacts/" + quote(relative, safe="/")
        with self.http.stream("GET", path) as response:
            _check_response(response)
            for chunk in response.iter_bytes():
                destination.write(chunk)

    def _request(self, method, path, **kwargs):
        try:
            response = self.http.request(method, path, **kwargs)
            _check_response(response)
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TeamError(
                "team service request failed; retain the idempotency key"
            ) from exc


def _run_path(run_id):
    if not re.fullmatch(r"run-[0-9a-f]{32}", run_id):
        raise TeamError("invalid team run ID")
    return f"v1/runs/{run_id}"


def _check_response(response):
    if response.is_success:
        return
    if response.status_code == 401:
        raise AuthenticationError("identity token is invalid or expired")
    if response.status_code in {403, 404}:
        raise TeamError("operation is not authorized or run is unavailable")
    raise TeamError(
        f"team operation failed (HTTP {response.status_code}); reconcile before retrying"
    )
