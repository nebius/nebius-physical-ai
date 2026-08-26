from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from npa.clients import config
from npa.clients import credentials


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


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
                                "endpoint_strategy": "ssh",
                                "service_port": 8080,
                                "tf_instance_name": "tf-a",
                                "instance_id": "computeinstance-a",
                                "project_id": "project-alias-a",
                                "security_group_id": "vpcsecuritygroup-a",
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
    sample = PACKAGE_ROOT / "src/npa/config/sample_config.yaml"
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


def test_resolve_terraform_state_reads_project_backend_credentials(
    isolated_config: Path,
) -> None:
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


def test_alias_has_terraform_state_for_saved_managed_alias(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    assert config.alias_has_terraform_state("proj-a", "wb-a") is True


def test_alias_has_terraform_state_false_for_missing_alias(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    assert config.alias_has_terraform_state("proj-a", "missing") is False


def test_alias_has_terraform_state_false_for_byovm_alias(isolated_config: Path) -> None:
    _write_full_config(isolated_config)
    data = yaml.safe_load(isolated_config.read_text())
    data["projects"]["proj-a"]["workbenches"]["wb-a"]["runtime"] = "byovm"
    isolated_config.write_text(yaml.safe_dump(data, sort_keys=False))

    assert config.alias_has_terraform_state("proj-a", "wb-a") is False
    assert config.workbench_is_byovm("proj-a", "wb-a") is True


# ── teardown helpers: clear_terraform_state_for_bucket / forget_project ───────


def test_clear_terraform_state_for_bucket_removes_matching_state(
    isolated_config: Path,
) -> None:
    _write_full_config(
        isolated_config
    )  # proj-a.terraform_state.bucket == "state-bucket"

    # A bucket URI must normalize to the bare name before comparing.
    cleared = config.clear_terraform_state_for_bucket("s3://state-bucket/")

    assert cleared == ["proj-a"]
    saved = yaml.safe_load(isolated_config.read_text())
    assert "terraform_state" not in saved["projects"]["proj-a"]
    # Everything else in the stanza survives.
    assert saved["projects"]["proj-a"]["project_id"] == "project-1"
    assert saved["projects"]["proj-a"]["workbenches"]["wb-a"]


def test_clear_terraform_state_for_bucket_no_match_leaves_config(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    assert config.clear_terraform_state_for_bucket("some-other-bucket") == []
    saved = yaml.safe_load(isolated_config.read_text())
    assert saved["projects"]["proj-a"]["terraform_state"]["access_key"] == "state-key"


def test_forget_project_removes_stanza_and_repoints_default(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)  # default_project == "proj-a"

    assert config.forget_project("proj-a") is True
    saved = yaml.safe_load(isolated_config.read_text())
    assert "proj-a" not in saved["projects"]
    assert "proj-b" in saved["projects"]
    assert saved["default_project"] == "proj-b"


def test_forget_project_missing_is_a_noop(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    assert config.forget_project("nope") is False
    saved = yaml.safe_load(isolated_config.read_text())
    assert set(saved["projects"]) == {"proj-a", "proj-b"}


def test_clear_skypilot_bin_removes_only_that_key(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(
        yaml.safe_dump({"skypilot": {"sky_bin": "/x/bin/sky"}, "default_project": "p"})
    )

    assert config.clear_skypilot_bin() is True
    saved = yaml.safe_load(isolated_config.read_text())
    assert "skypilot" not in saved  # section emptied and dropped
    assert saved["default_project"] == "p"
    # Idempotent: nothing left to clear.
    assert config.clear_skypilot_bin() is False


def test_resolve_project_storage_reads_object_storage(isolated_config: Path) -> None:
    isolated_config.parent.mkdir(parents=True)
    isolated_config.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "object-storage": {
                            "bucket": "s3://bucket/checkpoints/",
                            "endpoint": "https://storage.example",
                            "access_key": "access",
                            "secret_key": "secret",
                        }
                    }
                }
            }
        )
    )

    resolved = config.resolve_project_storage("proj")

    assert resolved == config.StorageConfig(
        checkpoint_bucket="s3://bucket/checkpoints/",
        endpoint_url="https://storage.example",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
    )


def test_resolve_project_storage_falls_back_to_terraform_state(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    resolved = config.resolve_project_storage("proj-a")

    assert resolved == config.StorageConfig(
        checkpoint_bucket="state-bucket",
        endpoint_url="https://state-storage.example",
        aws_access_key_id="state-key",
        aws_secret_access_key="state-secret",
    )


def test_resolve_project_storage_uses_env_fallback_when_project_storage_missing(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(yaml.safe_dump({"projects": {"proj": {}}}))
    monkeypatch.setenv("NPA_CHECKPOINT_BUCKET", "s3://env-bucket/checkpoints/")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env-storage.example")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")

    resolved = config.resolve_project_storage("proj")

    assert resolved == config.StorageConfig(
        checkpoint_bucket="s3://env-bucket/checkpoints/",
        endpoint_url="https://env-storage.example",
        aws_access_key_id="env-access",
        aws_secret_access_key="env-secret",
    )


def test_resolve_project_storage_uses_credentials_file_fallback(
    isolated_config: Path,
) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(yaml.safe_dump({"projects": {"proj": {}}}))
    credentials_path = credentials.CREDENTIALS_PATH
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "bucket": "s3://creds-bucket/checkpoints/",
                    "endpoint_url": "https://creds-storage.example",
                    "aws_access_key_id": "creds-access",
                    "aws_secret_access_key": "creds-secret",
                }
            }
        )
    )

    resolved = config.resolve_project_storage("proj")

    assert resolved == config.StorageConfig(
        checkpoint_bucket="s3://creds-bucket/checkpoints/",
        endpoint_url="https://creds-storage.example",
        aws_access_key_id="creds-access",
        aws_secret_access_key="creds-secret",
    )


def test_resolve_container_registry_uses_project_override(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    assert config.resolve_container_registry("proj-a") == "registry.example/npa"
    assert (
        config.resolve_container_registry("proj-b") == config.DEFAULT_CONTAINER_REGISTRY
    )


def test_resolve_config_uses_default_project_and_workbench(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    resolved = config.resolve_config()

    assert resolved.endpoint == "http://vm-a:8080"
    assert resolved.endpoint_strategy == "ssh_fallback"
    assert resolved.service_port == 8080
    assert resolved.endpoint_strategy_configured is True
    assert resolved.service_port_configured is True
    assert resolved.project == "proj-a"
    assert resolved.name == "wb-a"
    assert resolved.ssh == config.SSHConfig(
        host="vm-a",
        user="ubuntu",
        key_path="~/.ssh/a",
    )
    assert resolved.storage.checkpoint_bucket == "s3://bucket/a/"
    assert resolved.storage.aws_access_key_id == "key-a"
    assert resolved.hf_token == ""
    assert resolved.tf_instance_name == "tf-a"
    assert resolved.instance_id == "computeinstance-a"
    assert resolved.project_id == "project-alias-a"
    assert resolved.security_group_id == "vpcsecuritygroup-a"
    assert resolved.runtime == "container"
    assert resolved.container_registry == "registry.example/npa"
    assert resolved.gpu_platform == "NVIDIA H200"
    assert resolved.gpu_count == 2
    assert resolved.detected_gpu_count == 4
    assert resolved.cuda_visible_devices == "0,1"


def test_resolve_config_keeps_network_fields_optional(isolated_config: Path) -> None:
    _write_full_config(isolated_config)

    resolved = config.resolve_config(project="proj-a", name="wb-b")

    assert resolved.instance_id == ""
    assert resolved.project_id == ""
    assert resolved.security_group_id == ""


def test_resolve_config_parses_serverless_alias_without_ssh(
    isolated_config: Path,
) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "project_id": "project-1",
                        "region": "eu-north1",
                        "workbenches": {
                            "cosmos": {
                                "runtime": "serverless",
                                "endpoint": "https://cosmos.example",
                                "serverless": {
                                    "resource_type": "endpoint",
                                    "endpoint_id": "endpoint-1",
                                    "endpoint_name": "cosmos",
                                    "project_id": "project-1",
                                    "url": "https://cosmos.example",
                                    "image": "registry/cosmos:cuda12",
                                    "platform": "gpu-h200-sxm",
                                    "preset": "1gpu-16vcpu-200gb",
                                    "container_port": 8080,
                                    "auth": "none",
                                },
                            },
                        },
                    },
                },
            },
            sort_keys=False,
        )
    )

    resolved = config.resolve_config(project="proj", name="cosmos")

    assert resolved.runtime == "serverless"
    assert resolved.endpoint == "https://cosmos.example"
    assert resolved.ssh.host == ""
    assert resolved.serverless.endpoint_id == "endpoint-1"
    assert resolved.serverless.endpoint_name == "cosmos"
    assert resolved.serverless.project_id == "project-1"
    assert resolved.serverless.image == "registry/cosmos:cuda12"
    assert resolved.serverless.container_port == 8080
    assert resolved.serverless_job == config.ServerlessJobConfig()


def test_resolve_config_uses_serverless_url_as_endpoint(
    isolated_config: Path,
) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "workbenches": {
                            "cosmos": {
                                "runtime": "serverless",
                                "serverless": {
                                    "endpoint_id": "endpoint-1",
                                    "url": "https://cosmos.example",
                                },
                            },
                        },
                    },
                },
            },
            sort_keys=False,
        )
    )

    resolved = config.resolve_config(project="proj", name="cosmos")

    assert resolved.endpoint == "https://cosmos.example"


def test_update_workbench_serverless_endpoint_persists_metadata(
    isolated_config: Path,
) -> None:
    config.update_workbench_serverless_endpoint(
        "proj",
        "cosmos",
        endpoint_id="endpoint-1",
        endpoint_name="cosmos",
        project_id="project-1",
        url="https://cosmos.example",
        image="registry/cosmos:cuda12",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
        container_port=8080,
        auth="none",
    )

    saved = yaml.safe_load(isolated_config.read_text())
    wb = saved["projects"]["proj"]["workbenches"]["cosmos"]
    assert wb["runtime"] == "serverless"
    assert wb["endpoint"] == "https://cosmos.example"
    assert wb["serverless"]["endpoint_id"] == "endpoint-1"
    assert wb["serverless"]["project_id"] == "project-1"
    assert wb["serverless"]["container_port"] == 8080


def test_resolve_config_parses_serverless_job_alias_without_endpoint_or_ssh(
    isolated_config: Path,
) -> None:
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "project_id": "project-1",
                        "workbenches": {
                            "lerobot": {
                                "runtime": "serverless",
                                "serverless_job": {
                                    "resource_type": "job",
                                    "job_id": "job-1",
                                    "job_name": "train-1",
                                    "project_id": "project-1",
                                    "image": "registry/lerobot:0.5.1",
                                    "gpu_type": "gpu-h200-sxm",
                                    "gpu_count": 1,
                                    "subnet_id": "vpcsubnet-1",
                                    "output_path": "s3://bucket/lerobot/train-1/",
                                    "last_status": "succeeded",
                                    "last_submitted_at": "2026-05-13T00:00:00Z",
                                },
                            },
                        },
                    },
                },
            },
            sort_keys=False,
        )
    )

    resolved = config.resolve_config(project="proj", name="lerobot")

    assert resolved.runtime == "serverless"
    assert resolved.endpoint == ""
    assert resolved.ssh.host == ""
    assert resolved.serverless_job.job_id == "job-1"
    assert resolved.serverless_job.job_name == "train-1"
    assert resolved.serverless_job.gpu_type == "gpu-h200-sxm"
    assert resolved.serverless_job.gpu_count == 1
    assert resolved.serverless_job.output_path == "s3://bucket/lerobot/train-1/"


def test_serverless_job_config_accepts_alias_fields() -> None:
    parsed = config._serverless_job_config(
        {
            "serverless_job": {
                "id": "job-1",
                "name": "train-1",
                "platform": "gpu-h200-sxm",
                "gpus": "2",
                "subnet": "vpcsubnet-1",
                "output_uri": "s3://bucket/out/",
                "status": "running",
                "submitted_at": "2026-05-13T00:00:00Z",
            }
        }
    )

    assert parsed.job_id == "job-1"
    assert parsed.job_name == "train-1"
    assert parsed.gpu_type == "gpu-h200-sxm"
    assert parsed.gpu_count == 2
    assert parsed.subnet_id == "vpcsubnet-1"
    assert parsed.output_path == "s3://bucket/out/"
    assert parsed.last_status == "running"
    assert parsed.last_submitted_at == "2026-05-13T00:00:00Z"


def test_update_workbench_serverless_job_persists_metadata(
    isolated_config: Path,
) -> None:
    config.update_workbench_serverless_job(
        "proj",
        "lerobot",
        job_id="job-1",
        job_name="train-1",
        project_id="project-1",
        image="registry/lerobot:0.5.1",
        gpu_type="gpu-h200-sxm",
        gpu_count=1,
        subnet_id="vpcsubnet-1",
        output_path="s3://bucket/lerobot/train-1/",
        last_status="queued",
        last_submitted_at="2026-05-13T00:00:00Z",
    )

    saved = yaml.safe_load(isolated_config.read_text())
    wb = saved["projects"]["proj"]["workbenches"]["lerobot"]
    assert wb["runtime"] == "serverless"
    assert wb["app_status"] == config.APP_STATUS_PROVISIONED
    assert wb["serverless_job"]["resource_type"] == "job"
    assert wb["serverless_job"]["job_id"] == "job-1"
    assert wb["serverless_job"]["project_id"] == "project-1"
    assert wb["serverless_job"]["output_path"] == "s3://bucket/lerobot/train-1/"


def test_resolve_config_env_overrides_yaml(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_full_config(isolated_config)
    monkeypatch.setenv("NPA_WORKBENCH_ENDPOINT", "http://env:8080")
    monkeypatch.setenv("NPA_ENDPOINT_STRATEGY", "public")
    monkeypatch.setenv("NPA_SERVICE_PORT", "9090")
    monkeypatch.setenv("NPA_SSH_HOST", "env-host")
    monkeypatch.setenv("NPA_SSH_USER", "env-user")
    monkeypatch.setenv("NPA_SSH_KEY", "/tmp/env-key")
    monkeypatch.setenv("NPA_CHECKPOINT_BUCKET", "s3://env/checkpoints/")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env-storage")
    monkeypatch.setenv("HF_TOKEN", "hf-env")

    resolved = config.resolve_config(project="proj-a", name="wb-a")

    assert resolved.endpoint == "http://env:8080"
    assert resolved.endpoint_strategy == "public"
    assert resolved.service_port == 9090
    assert resolved.endpoint_strategy_configured is True
    assert resolved.service_port_configured is True
    assert resolved.ssh.host == "env-host"
    assert resolved.ssh.user == "env-user"
    assert resolved.ssh.key_path == "/tmp/env-key"
    assert resolved.storage.checkpoint_bucket == "s3://env/checkpoints/"
    assert resolved.storage.endpoint_url == "https://env-storage"
    assert resolved.hf_token == "hf-env"


def test_resolve_config_marks_missing_endpoint_strategy_metadata(
    isolated_config: Path,
) -> None:
    _write_full_config(isolated_config)

    resolved = config.resolve_config(project="proj-a", name="wb-b")

    assert resolved.endpoint_strategy == "public"
    assert resolved.service_port == 8080
    assert resolved.endpoint_strategy_configured is False
    assert resolved.service_port_configured is False
    assert resolved.project == "proj-a"
    assert resolved.name == "wb-b"


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


# ── workbench_type alias guard ───────────────────────────────────────────


def test_guard_workbench_type_allows_matching_type() -> None:
    # No raise when the alias type is in the tool's accepted set.
    config._guard_workbench_type({"workbench_type": "groot"}, "groot", name="wb")
    config._guard_workbench_type(
        {"workbench_type": "groot-container"}, "groot", name="wb"
    )


def test_guard_workbench_type_rejects_foreign_alias() -> None:
    with pytest.raises(config.ConfigError, match="is a 'groot' workbench, not cosmos"):
        config._guard_workbench_type({"workbench_type": "groot"}, "cosmos", name="wb")


def test_guard_workbench_type_is_legacy_safe() -> None:
    # Aliases written by older clients omit workbench_type -> never blocked.
    config._guard_workbench_type({}, "cosmos", name="wb")
    config._guard_workbench_type({"workbench_type": ""}, "cosmos", name="wb")
    # Unknown expected tool and None expected are both no-ops.
    config._guard_workbench_type({"workbench_type": "groot"}, "unknown-tool", name="wb")
    config._guard_workbench_type({"workbench_type": "groot"}, None, name="wb")


def _write_typed_workbench(path: Path, wtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "proj": {
                        "workbenches": {
                            "wb": {
                                "workbench_type": wtype,
                                "ssh": {
                                    "host": "h",
                                    "user": "ubuntu",
                                    "key_path": "~/.ssh/k",
                                },
                            }
                        }
                    }
                }
            }
        )
    )


def test_resolve_ssh_config_guard_rejects_wrong_tool(isolated_config: Path) -> None:
    _write_typed_workbench(isolated_config, "groot")
    with pytest.raises(config.ConfigError, match="not cosmos"):
        config.resolve_ssh_config(
            project="proj", name="wb", expected_workbench_type="cosmos"
        )


def test_resolve_ssh_config_guard_allows_matching_tool(isolated_config: Path) -> None:
    _write_typed_workbench(isolated_config, "groot")
    resolved = config.resolve_ssh_config(
        project="proj", name="wb", expected_workbench_type="groot"
    )
    assert resolved.workbench_type == "groot"


def test_resolve_container_registry_prefers_environment_override(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_full_config(isolated_config)
    monkeypatch.setenv("NPA_REGISTRY", "registry-env.example/npa")
    assert config.resolve_container_registry("proj-a") == "registry-env.example/npa"


def test_write_config_locks_down_file_and_directory(tmp_path, monkeypatch) -> None:
    """~/.npa holds S3 keys, kubeconfigs and agent auth secrets: owner-only."""
    import stat as stat_module

    from npa.clients import config as config_module

    config_path = tmp_path / ".npa" / "config.yaml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    config_module.write_config({"projects": {"p": {"project_id": "pid"}}})

    assert stat_module.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat_module.S_IMODE(config_path.parent.stat().st_mode) == 0o700


def test_config_writer_rejects_keys_strict_readers_would_reject(
    isolated_config: Path,
) -> None:
    with pytest.raises(config.ConfigError, match="typo_key.*Valid keys"):
        config.write_config({"skypilot": {"typo_key": True}})

    assert not isolated_config.exists()


def test_skypilot_cleanup_preserves_valid_controller_metadata(
    isolated_config: Path,
) -> None:
    owner = {
        "schema_version": "npa.controller-owner.v1",
        "project_alias": "demo",
        "project_id": "project-a",
        "cluster_id": "cluster-a",
        "context": "npa-cluster",
    }
    config.write_config(
        {"skypilot": {"sky_bin": "/tmp/sky", "controller_owner": owner}}
    )

    assert config.clear_skypilot_bin()
    saved = yaml.safe_load(isolated_config.read_text(encoding="utf-8"))
    assert saved["skypilot"] == {"controller_owner": owner}


def test_config_mutations_do_not_bypass_the_schema_validating_gateway() -> None:
    source_root = PACKAGE_ROOT / "src" / "npa"
    offenders: set[tuple[str, int, str]] = set()
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            target = ast.unparse(node.args[0]) if node.args else ""
            if name in {"update_private_yaml", "write_private_yaml"} and (
                "CONFIG_PATH" in target
            ):
                offenders.add((str(path.relative_to(PACKAGE_ROOT)), node.lineno, name))
            if (
                name in {"open", "write_text", "write_bytes"}
                and isinstance(node.func, ast.Attribute)
                and "CONFIG_PATH" in ast.unparse(node.func.value)
            ):
                offenders.add((str(path.relative_to(PACKAGE_ROOT)), node.lineno, name))

    assert offenders == set()


@pytest.mark.parametrize("fallback_succeeds", [True, False])
def test_forget_project_converges_when_cleanup_receipt_persistence_fails(
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fallback_succeeds: bool,
) -> None:
    from typer.testing import CliRunner

    from npa import teardown_receipts
    from npa.cli.main import app
    from npa.provisioning_journal import ProvisioningOperation

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        yaml.safe_dump(
            {
                "default_project": "target",
                "projects": {
                    "target": {
                        "project_id": "project-target",
                        "tenant_id": "tenant-target",
                        "region": "eu-test1",
                    },
                    "unrelated": {
                        "project_id": "project-unrelated",
                        "tenant_id": "tenant-unrelated",
                        "region": "eu-test2",
                    },
                },
                "skypilot": {
                    "controller_owner": {
                        "schema_version": "npa.controller-owner.v1",
                        "project_alias": "target",
                        "project_id": "project-target",
                        "cluster_id": "cluster-target",
                        "cluster_name": "npa-target",
                        "context": "npa-target",
                        "context_fingerprint": "fingerprint-target",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ProvisioningOperation.prepare(
        command="npa agent deploy",
        resume_command="",
        project_alias="target",
        project_id="project-target",
        tenant_id="tenant-target",
        region="eu-test1",
        resource_type="agent",
        requested_name="agent",
        backend={
            "bucket": "state-bucket",
            "endpoint": "https://storage.example.invalid",
            "state_key": "npa/target/agent.tfstate",
            "credential_source": "project_saved",
        },
    )
    real_record = teardown_receipts.record_teardown_event

    def flaky_record(**kwargs):  # noqa: ANN003, ANN202
        if not fallback_succeeds or (kwargs.get("identity") or {}).get("operations"):
            raise ValueError("simulated receipt validation/write failure")
        return real_record(**kwargs)

    monkeypatch.setattr(teardown_receipts, "record_teardown_event", flaky_record)

    result = CliRunner().invoke(app, ["configure", "--forget-project", "target"])

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(isolated_config.read_text(encoding="utf-8"))
    assert saved["default_project"] == "unrelated"
    assert set(saved["projects"]) == {"unrelated"}
    assert saved["projects"]["unrelated"]["project_id"] == "project-unrelated"
    assert "controller_owner" not in saved.get("skypilot", {})
    assert "Removed project 'target'" in result.output
    if fallback_succeeds:
        assert "wrote a minimal safe cleanup receipt" in result.output
        [receipt] = teardown_receipts.list_teardown_receipts(
            project_id="project-target", legacy="exclude"
        )
        assert receipt["identity"] == {
            "parent_id": "project-target",
            "project_alias": "target",
            "project_id": "project-target",
        }
    else:
        assert "degraded audit evidence" in result.output
        assert not (tmp_path / "receipts").exists()


def test_config_permissions_warning_flags_a_world_readable_file(
    tmp_path, monkeypatch
) -> None:
    """Terraform backend S3 keys live in config.yaml, so loose modes matter."""
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text("projects: {}\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    config_path.chmod(0o600)
    assert config_module.config_permissions_warning() == ""

    config_path.chmod(0o644)
    warning = config_module.config_permissions_warning()
    assert "terraform_state" in warning
    assert "chmod 600" in warning


def test_config_permissions_warning_is_quiet_without_a_file(
    tmp_path, monkeypatch
) -> None:
    from npa.clients import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "missing.yaml")

    assert config_module.config_permissions_warning() == ""
