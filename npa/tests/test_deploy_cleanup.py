from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.clients import config
from npa.deploy.cleanup import classify_alias_state


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg_path = tmp_path / ".npa" / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    return cfg_path


def _write_config(path: Path, workbench: dict, *, terraform_state: bool = True) -> None:
    project: dict = {
        "project_id": "project",
        "tenant_id": "tenant",
        "region": "eu-north1",
        "workbenches": {"alias": workbench},
    }
    if terraform_state:
        project["terraform_state"] = {
            "bucket": "state-bucket",
            "endpoint": "https://storage.example",
            "access_key": "access",
            "secret_key": "secret",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"projects": {"proj": project}}, sort_keys=False))


def test_classifier_returns_fresh_for_missing_entry(isolated_config: Path) -> None:
    _write_config(isolated_config, {"ssh": {"host": "10.0.0.1"}})

    assert classify_alias_state("proj", "missing") == "fresh"


def test_classifier_returns_byovm_for_byovm_runtime(isolated_config: Path) -> None:
    _write_config(isolated_config, {"runtime": "byovm", "ssh": {"host": "10.0.0.1"}})

    assert classify_alias_state("proj", "alias") == "byovm"


def test_classifier_returns_partial_for_tfstate_no_host(isolated_config: Path) -> None:
    _write_config(isolated_config, {"app_status": "provisioning"})

    assert classify_alias_state("proj", "alias") == "partial"


def test_classifier_returns_fully_deployed_for_tfstate_and_host(isolated_config: Path) -> None:
    _write_config(isolated_config, {"ssh": {"host": "10.0.0.1"}})

    assert classify_alias_state("proj", "alias") == "fully_deployed"


def test_classifier_returns_partial_for_entry_without_state_or_host(isolated_config: Path) -> None:
    _write_config(isolated_config, {"app_status": "provisioning"}, terraform_state=False)

    assert classify_alias_state("proj", "alias") == "partial"


def test_classifier_defaults_to_fully_deployed_when_ambiguous(isolated_config: Path) -> None:
    _write_config(isolated_config, {"ssh": {"host": "10.0.0.1"}}, terraform_state=False)

    assert classify_alias_state("proj", "alias") == "fully_deployed"
