"""Verify controller credentials stay scoped, private, renewable, and fail closed."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def auth(monkeypatch):
    """Load isolated controller authentication.

    Args:
        monkeypatch: Isolated script imports.
    Returns:
        Authentication module.
    Raises:
        ImportError: The module cannot be loaded.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("ci_cpu_runner_auth")
    module._TOKENS.clear()
    return module


def _settings(root):
    key = root / "app.pem"
    key.write_text("synthetic signing key")
    key.chmod(0o600)
    settings = {
        "app_id": 1,
        "installation_id": 2,
        "repository": "example/workbench",
        "private_key_file": str(key),
    }
    path = root / "github-app.json"
    path.write_text(json.dumps(settings))
    path.chmod(0o600)
    return settings


def _openssl(*arguments, content=None):
    return subprocess.run(
        ["openssl", *map(str, arguments)],
        input=content,
        capture_output=True,
        check=True,
    )


def test_app_signatures_verify_with_the_corresponding_public_key(auth, tmp_path):
    """Verify a real RSA signature and bounded App JWT validity.

    Args:
        auth: Checked-out authentication module.
        tmp_path: Private ephemeral signing files.
    Returns:
        None.
    Raises:
        AssertionError: JWT identity, validity, or signature is incorrect.
    """
    settings = _settings(tmp_path)
    key = Path(settings["private_key_file"])
    _openssl("genrsa", "-out", key, "2048")
    header, claims, signature = auth._app_jwt(settings).split(".")
    payload = json.loads(base64.urlsafe_b64decode(claims + "=="))
    assert payload["iss"] == "1"
    assert payload["iat"] <= time.time() < payload["exp"]
    assert payload["exp"] - payload["iat"] == 600
    public = tmp_path / "public.pem"
    _openssl("rsa", "-in", key, "-pubout", "-out", public)
    signature_path = tmp_path / "signature"
    signature_path.write_bytes(base64.urlsafe_b64decode(signature + "=="))
    _openssl(
        "dgst",
        "-sha256",
        "-verify",
        public,
        "-signature",
        signature_path,
        content=f"{header}.{claims}".encode(),
    )


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o666])
def test_nonprivate_keys_are_rejected(auth, tmp_path, mode):
    """Reject readable signing keys.

    Args:
        auth, tmp_path, mode: Module, private files, and unsafe permissions.
    Returns:
        None.
    Raises:
        AssertionError: A nonprivate key is accepted.
    """
    settings = _settings(tmp_path)
    Path(settings["private_key_file"]).chmod(mode)
    with pytest.raises(ValueError, match="private"):
        auth._app_jwt(settings)


def test_app_token_is_cached_across_workers_and_refreshed(auth, tmp_path, monkeypatch):
    """Renew scoped tokens without leaking the operator identity.

    Args:
        auth, tmp_path, monkeypatch: Module, private files, and fake issuer.
    Returns:
        None.
    Raises:
        AssertionError: Tokens are duplicated, stale, or overprivileged.
    """
    _settings(tmp_path)
    config = {"github_auth": "app", "repository": "example/workbench"}
    issued = []

    def mint(*args):
        issued.append(1)
        return "installation-only", time.time() + 3600

    monkeypatch.setattr(auth, "_installation_token", mint)
    monkeypatch.setenv("GH_TOKEN", "operator-admin")
    monkeypatch.setenv("GITHUB_TOKEN", "other-admin")
    with ThreadPoolExecutor(max_workers=3) as threads:
        results = list(
            threads.map(lambda _: auth._github_environment(tmp_path, config), range(3))
        )
    assert issued == [1]
    assert all(result["GH_TOKEN"] == "installation-only" for result in results)
    assert all("GITHUB_TOKEN" not in result for result in results)
    identity = next(iter(auth._TOKENS))
    auth._TOKENS[identity] = ("expired", 0)
    assert auth._github_environment(tmp_path, config)["GH_TOKEN"] == "installation-only"
    assert issued == [1, 1]


def test_missing_app_never_falls_back_to_operator_auth(auth, tmp_path, monkeypatch):
    """Fail closed when dedicated App credentials are missing.

    Args:
        auth, tmp_path, monkeypatch: Module, absent files, and operator token.
    Returns:
        None.
    Raises:
        AssertionError: Missing App authentication permits operator fallback.
    """
    monkeypatch.setenv("GH_TOKEN", "operator-admin")
    with pytest.raises(FileNotFoundError):
        auth._github_environment(tmp_path, {"github_auth": "app"})
    with pytest.raises(ValueError, match="authentication mode"):
        auth._github_environment(tmp_path, {"github_auth": "misspelled-app"})


@pytest.mark.parametrize("failure", ["repository", "permission", "expiry"])
def test_installation_rejects_wrong_scope_or_expired_tokens(
    auth, tmp_path, monkeypatch, failure
):
    """Reject unusable or overly broad installation tokens.

    Args:
        auth, tmp_path, monkeypatch, failure: Module, files, issuer, failure type.
    Returns:
        None.
    Raises:
        AssertionError: An invalid installation token is accepted.
    """
    settings = _settings(tmp_path)
    response = {
        "token": "installation-only",
        "expires_at": "2099-01-01T00:00:00Z",
        "permissions": dict(auth._PERMISSIONS),
        "repositories": [{"full_name": "example/workbench"}],
    }
    if failure == "repository":
        response["repositories"].append({"full_name": "example/other"})
    elif failure == "permission":
        response["permissions"]["contents"] = "write"
    else:
        response["expires_at"] = "2000-01-01T00:00:00Z"
    monkeypatch.setattr(auth, "_app_jwt", lambda _: "jwt")
    monkeypatch.setattr(auth, "_api", lambda *a, **k: response)
    with pytest.raises(ValueError):
        auth._installation_token(settings, settings["repository"])


def test_manifest_requests_only_controller_permissions(auth, tmp_path):
    """Limit App privileges and reject a mismatched registration callback.

    Args:
        auth, tmp_path: Module and private setup state.
    Returns:
        None.
    Raises:
        AssertionError: Privileges expand or callback state is not checked.
    """
    setup_module = importlib.import_module("ci_cpu_runner_app_setup")
    setup = SimpleNamespace(
        root=tmp_path,
        repository="example/workbench",
        url="http://127.0.0.1:1234",
        state="synthetic-state",
    )
    manifest = setup_module._manifest(setup)
    assert manifest["default_permissions"] == auth._PERMISSIONS
    assert manifest["public"] is False
    assert manifest["hook_attributes"]["active"] is False
    assert manifest["default_events"] == []
    with pytest.raises(ValueError, match="state"):
        setup_module._record_app(setup, {"state": ["wrong"], "code": ["synthetic"]})


@pytest.mark.parametrize("scope", ["all", "extra-repository", "one-repository"])
def test_setup_rejects_installations_covering_other_repositories(
    auth, monkeypatch, scope
):
    """Require the App itself to cover only the intended repository.

    Args:
        auth, monkeypatch, scope: Module, API substitute, and installation scope.
    Returns:
        None.
    Raises:
        AssertionError: An organization-wide or multi-repository App is accepted.
    """
    setup = importlib.import_module("ci_cpu_runner_app_setup")
    settings = {"repository": "example/workbench"}
    installation = {"id": 2, "repository_selection": "selected"}
    names = ["example/workbench"]
    if scope == "all":
        installation["repository_selection"] = "all"
    elif scope == "extra-repository":
        names.append("example/other")
    monkeypatch.setattr(setup, "_app_jwt", lambda _: "jwt")

    def api(endpoint, **kwargs):
        if endpoint.endswith("/access_tokens"):
            assert kwargs["payload"] == {"permissions": {"metadata": "read"}}
            return {"token": "metadata-only"}
        assert kwargs["token"] == "metadata-only"
        return {
            "total_count": len(names),
            "repositories": [{"full_name": name} for name in names],
        }

    monkeypatch.setattr(setup, "_api", api)
    if scope == "one-repository":
        setup._verify_installation_scope(settings, installation)
    else:
        with pytest.raises(ValueError, match="only"):
            setup._verify_installation_scope(settings, installation)
