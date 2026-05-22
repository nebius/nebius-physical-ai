from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module


pytestmark = pytest.mark.byovm_live

runner = CliRunner()


def _live_env() -> dict[str, str]:
    required = {
        "host": "NPA_E2E_BYOVM_FIFTYONE_HOST",
        "ssh_key": "NPA_E2E_BYOVM_FIFTYONE_SSH_KEY",
        "input_path": "NPA_E2E_BYOVM_FIFTYONE_INPUT_PATH",
    }
    missing = [env for env in required.values() if not os.environ.get(env)]
    if os.environ.get("NPA_E2E_BYOVM_SELF_HEAL") != "1" or missing:
        pytest.skip(
            "Set NPA_E2E_BYOVM_SELF_HEAL=1 plus "
            + ", ".join(required.values())
            + " to run this live BYOVM test."
        )
    return {key: os.environ[env] for key, env in required.items()}


def test_pre_fix_fiftyone_byovm_alias_self_heals_status_and_load_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _live_env()
    cfg_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    for env_var in config_module.ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    port = int(os.environ.get("NPA_E2E_BYOVM_FIFTYONE_PORT", "5151"))
    project = os.environ.get("NPA_E2E_BYOVM_FIFTYONE_PROJECT", "e2e-byovm")
    name = os.environ.get("NPA_E2E_BYOVM_FIFTYONE_NAME", "pre-fix-fiftyone")
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump({
        "projects": {
            project: {
                "workbenches": {
                    name: {
                        "endpoint": f"http://{env['host']}:{port}",
                        "runtime": "byovm",
                        "app_port": port,
                        "ssh": {
                            "host": env["host"],
                            "user": os.environ.get("NPA_E2E_BYOVM_FIFTYONE_SSH_USER", "ubuntu"),
                            "key_path": env["ssh_key"],
                        },
                    },
                },
            },
        },
    }))

    status = runner.invoke(app, ["workbench", "fiftyone", "-p", project, "-n", name, "status"])
    assert status.exit_code == 0, status.output

    healed = yaml.safe_load(cfg_path.read_text())["projects"][project]["workbenches"][name]
    assert healed["endpoint_strategy"] == "ssh_fallback"
    assert healed["service_port"] == port

    dataset_name = os.environ.get("NPA_E2E_BYOVM_FIFTYONE_DATASET", "npa_e2e_endpoint_self_heal")
    loaded = runner.invoke(
        app,
        [
            "workbench",
            "fiftyone",
            "-p",
            project,
            "-n",
            name,
            "load-dataset",
            "--name",
            dataset_name,
            "--input-path",
            env["input_path"],
            "--format",
            "auto",
        ],
    )
    assert loaded.exit_code == 0, loaded.output
