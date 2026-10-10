"""Verify native JSON service-account cache creation without changing live auth."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from npa.orchestration.skypilot import local_api

pytestmark = pytest.mark.e2e


def _private_selection():
    config_name = os.environ.get("NPA_E2E_NEBIUS_JSON_SA_CONFIG", "")
    profile = os.environ.get("NPA_E2E_NEBIUS_JSON_SA_PROFILE", "")
    if not config_name or not profile:
        pytest.skip("An explicit private JSON service-account profile is required")
    config = Path(config_name)
    if not config.is_absolute() or not config.is_file():
        pytest.fail("The selected private CLI configuration is unavailable")
    if config.stat().st_mode & 0o077:
        pytest.fail("The selected private CLI configuration must be owner-only")
    selection = local_api._nebius_profile_selection(
        config, profile, {}, local_api._supported_service_account_profile
    )
    if not selection or "service-account-credentials-file-path" not in selection[3]:
        pytest.fail("The selected profile is not a supported JSON service account")
    credentials = Path(selection[3]["service-account-credentials-file-path"])
    if not credentials.is_file() or credentials.stat().st_mode & 0o077:
        pytest.fail("The selected private JSON credentials must be owner-only")
    try:
        local_api._read_service_account_subject(credentials)
    except local_api.IsolatedApiError:
        pytest.fail("The selected private JSON credentials cannot be verified")
    return config, profile, credentials


def _selected_kube_config(config, profile, cli):
    return {
        "current-context": "fixture-context",
        "contexts": [
            {
                "name": "fixture-context",
                "context": {"user": "fixture-user", "cluster": "fixture-cluster"},
            }
        ],
        "users": [
            {
                "name": "fixture-user",
                "user": {
                    "exec": {
                        "command": cli,
                        "args": ["--config", str(config), "--profile", profile],
                    }
                },
            }
        ],
        "clusters": [{"name": "fixture-cluster", "cluster": {}}],
    }


def _private_test_environment(home, config, profile, cli):
    home.mkdir(mode=0o700)
    kube = home / "kube.yaml"
    kube.write_text(yaml.safe_dump(_selected_kube_config(config, profile, cli)))
    kube.chmod(0o600)
    sky = home / "sky.yaml"
    sky.write_text("kubernetes:\n  allowed_contexts: [fixture-context]\n")
    sky.chmod(0o600)
    env = dict(os.environ)
    for name in (
        "NEBIUS_CONFIG_DIR",
        "NEBIUS_PROFILE",
        "NPA_NEBIUS_PROFILE",
        "NEBIUS_ENDPOINT",
        "NEBIUS_IAM_TOKEN",
        "NEBIUS_IAM_TOKEN_FILE",
        "NPA_NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_IAM_TOKEN_FILE",
        "NPA_NEBIUS_CREDENTIAL_SOURCE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        env.pop(name, None)
    env.update(
        HOME=str(home),
        NPA_CONFIG_DIR=str(home / ".npa"),
        KUBECONFIG=str(kube),
        SKYPILOT_GLOBAL_CONFIG=str(sky),
    )
    return env


def _native_cli_probe(cli, config, profile, env, command):
    result = subprocess.run(
        [
            cli,
            "--config",
            str(config),
            "--profile",
            profile,
            "--no-browser",
            "--no-check-update",
            "iam",
            command,
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode == 0, "Native private service-account auth probe failed"


def test_native_json_service_account_cache_creation_preserves_identity(tmp_path):
    """Mint into a fresh private cache while retaining the durable identity."""
    config, profile, credentials = _private_selection()
    cli = os.environ.get("NPA_E2E_NEBIUS_CLI") or shutil.which("nebius")
    if not cli or not Path(cli).is_absolute():
        pytest.fail("An absolute native Nebius CLI executable is required")
    version = subprocess.run(
        [cli, "version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    assert version.returncode == 0 and "0.12.254" in version.stdout, (
        "This auth contract requires Nebius CLI 0.12.254"
    )
    original_config = config.read_bytes()
    original_credentials = credentials.read_bytes()
    env = _private_test_environment(tmp_path / "private-cli-home", config, profile, cli)
    cache = Path(env["HOME"]) / ".nebius/credentials.yaml"
    before = local_api._identity_files(env)
    assert before[str(cache)].startswith("derived-nebius-sa-json-cache-v1:")
    assert before[str(credentials)] == hashlib.sha256(original_credentials).hexdigest()
    assert not cache.exists()
    try:
        _native_cli_probe(cli, config, profile, env, "whoami")
        _native_cli_probe(cli, config, profile, env, "get-access-token")
        assert cache.is_file()
        assert local_api._derived_service_account_cache(cache.read_bytes())
        assert local_api._identity_files(env) == before
    finally:
        cache.unlink(missing_ok=True)
        configuration_preserved = config.read_bytes() == original_config
        credentials_preserved = credentials.read_bytes() == original_credentials
        assert configuration_preserved, "The original private configuration changed"
        assert credentials_preserved, "The original private credentials changed"
