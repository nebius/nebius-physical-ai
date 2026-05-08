from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.clients import config
from npa.clients import credentials


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg_path = tmp_path / ".npa" / "config.yaml"
    credentials_path = tmp_path / ".npa" / "credentials.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", credentials_path)
    for env_var in config.ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)
    return cfg_path


def _write_full_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "default_project": "proj-a",
                "default_workbench": "wb-a",
                "projects": {
                    "proj-a": {
                        "project_id": "project-1",
                        "tenant_id": "tenant-1",
                        "region": "eu-north1",
                        "container_registry": "registry.example/npa",
                        "terraform_state": {
                            "bucket": "state-bucket",
                            "endpoint": "https://state-storage.example",
                            "access_key": "state-key",
                            "secret_key": "state-secret",
                        },
                        "workbenches": {
                            "wb-a": {
                                "endpoint": "http://vm-a:8080",
                                "tf_instance_name": "tf-a",
                                "runtime": "container",
                                "gpu_platform": "NVIDIA H200",
                                "gpu_count": 2,
                                "detected_gpu_count": 4,
                                "cuda_visible_devices": "0,1",
                                "ssh": {
                                    "host": "vm-a",
                                    "user": "ubuntu",
                                    "key_path": "~/.ssh/a",
                                },
                                "storage": {
                                    "checkpoint_bucket": "s3://bucket/a/",
                                    "endpoint_url": "https://storage.example",
                                    "aws_access_key_id": "key-a",
                                    "aws_secret_access_key": "secret-a",
                                },
                            },
                            "wb-b": {
                                "endpoint": "http://vm-b:8080",
                                "ssh": {
                                    "host": "vm-b",
                                    "user": "robot",
                                    "key_path": "~/.ssh/b",
                                },
                                "storage": {
                                    "checkpoint_bucket": "s3://bucket/b/",
                                    "endpoint_url": "https://storage-b.example",
                                },
                            },
                        },
                    },
                    "proj-b": {
                        "project_id": "project-2",
                        "tenant_id": "tenant-2",
                        "region": "us-east1",
                        "workbenches": {},
                    },
                },
            },
            sort_keys=False,
        )
    )


def test_load_yaml_missing_file_returns_empty_dict(isolated_config: Path) -> None:
    assert config._load_yaml() == {}


def test_load_yaml_non_mapping_returns_empty_dict(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("- not\n- a\n- mapping\n")

    assert config._load_yaml() == {}


def test_load_yaml_malformed_yaml_raises(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text("projects: [unterminated\n")

    with pytest.raises(yaml.YAMLError):
        config._load_yaml()


def test_parse_bundled_sample_config_has_required_project_keys() -> None:
    sample = Path("src/npa/config/sample_config.yaml")
    data = yaml.safe_load(sample.read_text())

    assert data["default_project"] in data["projects"]
    project = data["projects"][data["default_project"]]
    assert {"project_id", "tenant_id", "region", "workbenches"} <= project.keys()
    assert data["default_workbench"] in project["workbenches"]


def test_list_projects_and_defaults(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    projects = config.list_projects()

    assert sorted(projects) == ["proj-a", "proj-b"]
    assert config.default_project_name() == "proj-a"
    assert config.default_workbench_name() == "wb-a"


def test_resolve_environment_uses_yaml_and_cli_overrides(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    env = config.resolve_environment("proj-a", region="override-region")

    assert env == config.EnvironmentConfig(
        project_id="project-1",
        tenant_id="tenant-1",
        region="override-region",
    )


def test_resolve_environment_returns_none_when_absent(isolated_config: Path) -> None:
    assert config.resolve_environment() is None


def test_resolve_terraform_state_reads_project_backend_credentials(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    resolved = config.resolve_terraform_state("proj-a")

    assert resolved == config.TerraformStateConfig(
        bucket="state-bucket",
        endpoint="https://state-storage.example",
        access_key="state-key",
        secret_key="state-secret",
    )


def test_resolve_terraform_state_missing_returns_empty(isolated_config: Path) -> None:
    assert config.resolve_terraform_state("missing") == config.TerraformStateConfig()


def test_resolve_container_registry_uses_project_override(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    assert config.resolve_container_registry("proj-a") == "registry.example/npa"
    assert config.resolve_container_registry("proj-b") == config.DEFAULT_CONTAINER_REGISTRY


def test_resolve_config_uses_default_project_and_workbench(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    resolved = config.resolve_config()

    assert resolved.endpoint == "http://vm-a:8080"
    assert resolved.ssh == config.SSHConfig(
        host="vm-a",
        user="ubuntu",
        key_path="~/.ssh/a",
    )
    assert resolved.storage.checkpoint_bucket == "s3://bucket/a/"
    assert resolved.storage.aws_access_key_id == "key-a"
    assert resolved.hf_token == ""
    assert resolved.tf_instance_name == "tf-a"
    assert resolved.runtime == "container"
    assert resolved.container_registry == "registry.example/npa"
    assert resolved.gpu_platform == "NVIDIA H200"
    assert resolved.gpu_count == 2
    assert resolved.detected_gpu_count == 4
    assert resolved.cuda_visible_devices == "0,1"


def test_resolve_config_env_overrides_yaml(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_full_config(isolated_config)
    monkeypatch.setenv("NPA_WORKBENCH_ENDPOINT", "http://env:8080")
    monkeypatch.setenv("NPA_SSH_HOST", "env-host")
    monkeypatch.setenv("NPA_SSH_USER", "env-user")
    monkeypatch.setenv("NPA_SSH_KEY", "/tmp/env-key")
    monkeypatch.setenv("NPA_CHECKPOINT_BUCKET", "s3://env/checkpoints/")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env-storage")
    monkeypatch.setenv("HF_TOKEN", "hf-env")

    resolved = config.resolve_config(project="proj-a", name="wb-a")

    assert resolved.endpoint == "http://env:8080"
    assert resolved.ssh.host == "env-host"
    assert resolved.ssh.user == "env-user"
    assert resolved.ssh.key_path == "/tmp/env-key"
    assert resolved.storage.checkpoint_bucket == "s3://env/checkpoints/"
    assert resolved.storage.endpoint_url == "https://env-storage"
    assert resolved.hf_token == "hf-env"


def test_resolve_config_uses_credentials_yaml_for_hf_token(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)
    credentials_path = credentials.CREDENTIALS_PATH
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(
        yaml.safe_dump({"tokens": {"HF_TOKEN": "hf-credentials"}})
    )

    resolved = config.resolve_config(project="proj-a", name="wb-a")

    assert resolved.hf_token == "hf-credentials"


def test_resolve_config_cli_overrides_env(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_full_config(isolated_config)
    monkeypatch.setenv("NPA_WORKBENCH_ENDPOINT", "http://env:8080")

    resolved = config.resolve_config(endpoint="http://cli:8080")

    assert resolved.endpoint == "http://cli:8080"


def test_resolve_config_missing_required_fields_raises(
    isolated_config: Path,
) -> None:
    with pytest.raises(config.ConfigError, match="Workbench endpoint"):
        config.resolve_config()


def test_resolve_ssh_config_does_not_require_endpoint(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "workbenches": {
                            "sim": {
                                "ssh": {
                                    "host": "sim-host",
                                    "user": "ubuntu",
                                    "key_path": "~/.ssh/sim",
                                }
                            }
                        }
                    }
                }
            }
        )
    )

    resolved = config.resolve_ssh_config(project="proj", name="sim")

    assert resolved.endpoint == ""
    assert resolved.ssh.host == "sim-host"


def test_unknown_project_and_workbench_errors(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    with pytest.raises(config.ConfigError, match="Project 'missing' not found"):
        config.resolve_config(project="missing")

    with pytest.raises(config.ConfigError, match="Workbench 'missing' not found"):
        config.resolve_config(project="proj-a", name="missing")


def test_write_config_deep_merges_existing_config(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    written = config.write_config(
        {
            "projects": {
                "proj-a": {
                    "workbenches": {
                        "wb-a": {"endpoint": "http://updated:8080"},
                    },
                },
            },
        }
    )

    data = yaml.safe_load(written.read_text())
    wb = data["projects"]["proj-a"]["workbenches"]["wb-a"]
    assert wb["endpoint"] == "http://updated:8080"
    assert wb["ssh"]["host"] == "vm-a"
    assert written.stat().st_mode & 0o777 == 0o600


def test_remove_workbench_config_updates_defaults(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    config.remove_workbench_config("proj-a", "wb-a")
    data = yaml.safe_load(isolated_config.read_text())

    assert "wb-a" not in data["projects"]["proj-a"]["workbenches"]
    assert data["default_project"] == "proj-a"

    config.remove_workbench_config("proj-a", "wb-b")
    data = yaml.safe_load(isolated_config.read_text())
    assert "proj-a" not in data["projects"]
    assert data["default_project"] == "proj-b"
