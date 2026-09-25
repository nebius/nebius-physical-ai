"""Issue repository-scoped GitHub App tokens without exporting operator credentials."""

from __future__ import annotations

import base64
from datetime import datetime
import http.client
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time

_PERMISSIONS = {"administration": "write", "actions": "read", "variables": "write"}
_TOKEN_LOCK = threading.Lock()
_TOKENS: dict[tuple, tuple[str, float]] = {}


def _private_file(path: Path) -> Path:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise ValueError("GitHub App files must be regular private files (chmod 600)")
    if metadata.st_uid != os.getuid():
        raise ValueError("GitHub App files must belong to the controller user")
    return path


def _encoded(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _app_jwt(settings: dict) -> str:
    key = _private_file(Path(settings["private_key_file"]))
    now = int(time.time())
    claims = {"iat": now - 60, "exp": now + 540, "iss": str(settings["app_id"])}
    header = _encoded(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _encoded(json.dumps(claims).encode())
    message = f"{header}.{payload}"
    signed = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", str(key)],
        input=message.encode(),
        capture_output=True,
        check=False,
    )
    if signed.returncode:
        raise RuntimeError("GitHub App signing failed")
    return f"{message}.{_encoded(signed.stdout)}"


def _api(endpoint: str, *, token: str = "", payload=None):
    if not endpoint.startswith("/") or "?" in endpoint or "#" in endpoint:
        raise ValueError("Invalid GitHub App API path")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "npa-ci-controller",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else json.dumps(payload).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPSConnection("api.github.com")
    try:
        connection.request(
            "GET" if data is None else "POST", endpoint, body=data, headers=headers
        )
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"GitHub App request failed (HTTP {response.status})")
        return json.load(response)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"GitHub App request failed ({type(error).__name__})"
        ) from None
    finally:
        connection.close()


def _installation_token(settings: dict, repository: str) -> tuple[str, float]:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("Invalid GitHub App repository")
    installation = settings["installation_id"]
    if not isinstance(installation, int) or installation < 1:
        raise ValueError("Invalid GitHub App installation")
    response = _api(
        f"/app/installations/{installation}/access_tokens",
        token=_app_jwt(settings),
        payload={
            "repositories": [repository.split("/")[1]],
            "permissions": _PERMISSIONS,
        },
    )
    permissions = response["permissions"]
    if any(permissions.get(name) != value for name, value in _PERMISSIONS.items()):
        raise ValueError("GitHub App token lacks required runner permissions")
    if set(permissions) - {*_PERMISSIONS, "metadata"}:
        raise ValueError("GitHub App token has unexpected permissions")
    repositories = response.get("repositories", [])
    if [item["full_name"].lower() for item in repositories] != [repository.lower()]:
        raise ValueError("GitHub App token is not restricted to this repository")
    expires = datetime.fromisoformat(
        response["expires_at"].replace("Z", "+00:00")
    ).timestamp()
    if expires <= time.time() + 60 or not response.get("token"):
        raise ValueError("GitHub App returned an unusable token")
    return response["token"], expires


def _github_environment(root: Path, config: dict) -> dict | None:
    mode = config.get("github_auth")
    if mode not in {None, "app"}:
        raise ValueError("Unknown GitHub authentication mode")
    if mode is None:
        return None
    settings_path = _private_file(root / "github-app.json")
    settings = json.loads(settings_path.read_text())
    if settings["repository"].lower() != config["repository"].lower():
        raise ValueError("GitHub App configuration belongs to another repository")
    key = _private_file(Path(settings["private_key_file"]))
    identity = (
        str(settings_path),
        settings_path.stat().st_mtime_ns,
        key.stat().st_mtime_ns,
    )
    with _TOKEN_LOCK:
        token, expires = _TOKENS.get(identity, ("", 0))
        if expires <= time.time() + 60:
            token, expires = _installation_token(settings, config["repository"])
            _TOKENS[identity] = token, expires
    environment = dict(os.environ)
    environment.pop("GITHUB_TOKEN", None)
    environment.update(GH_TOKEN=token, GH_HOST="github.com", GH_PROMPT_DISABLED="1")
    return environment
