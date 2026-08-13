from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.terraform_lock import (
    LOCK_PLATFORM_METADATA_NAME,
    SUPPORTED_OPERATOR_PLATFORMS,
    TerraformLockError,
    configure_plugin_cache,
    terraform_platform,
    validate_provider_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_MODULES = (
    REPO_ROOT / "deploy" / "cluster",
    REPO_ROOT / "npa" / "src" / "npa" / "deploy" / "terraform",
)


@pytest.mark.parametrize("terraform_dir", TERRAFORM_MODULES)
@pytest.mark.parametrize("target_platform", SUPPORTED_OPERATOR_PLATFORMS)
def test_tracked_provider_locks_cover_supported_operator_platforms(
    terraform_dir: Path,
    target_platform: str,
) -> None:
    assert (
        validate_provider_lock(terraform_dir, target_platform=target_platform)
        == target_platform
    )


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "linux_amd64"),
        ("Linux", "aarch64", "linux_arm64"),
        ("Darwin", "x86_64", "darwin_amd64"),
        ("Darwin", "arm64", "darwin_arm64"),
    ],
)
def test_terraform_platform_normalizes_operator_architectures(
    system: str,
    machine: str,
    expected: str,
) -> None:
    assert terraform_platform(system=system, machine=machine) == expected


def test_provider_lock_sidecar_rejects_runtime_lock_mutation(tmp_path: Path) -> None:
    source = TERRAFORM_MODULES[1]
    lock = tmp_path / ".terraform.lock.hcl"
    metadata = tmp_path / LOCK_PLATFORM_METADATA_NAME
    lock.write_bytes((source / lock.name).read_bytes())
    metadata.write_bytes((source / metadata.name).read_bytes())
    lock.write_text(lock.read_text() + "\n# runtime mutation\n")

    with pytest.raises(TerraformLockError, match="changed without a reviewed"):
        validate_provider_lock(tmp_path, target_platform="darwin_arm64")


def test_provider_lock_sidecar_rejects_unrecorded_platform(tmp_path: Path) -> None:
    source = TERRAFORM_MODULES[1]
    lock = tmp_path / ".terraform.lock.hcl"
    metadata = tmp_path / LOCK_PLATFORM_METADATA_NAME
    lock.write_bytes((source / lock.name).read_bytes())
    payload = json.loads((source / metadata.name).read_text())
    payload["platforms"] = ["linux_amd64"]
    metadata.write_text(json.dumps(payload))

    with pytest.raises(TerraformLockError, match="required operator platform set"):
        validate_provider_lock(tmp_path, target_platform="darwin_arm64")


def test_plugin_cache_is_platform_scoped_and_outside_source(tmp_path: Path) -> None:
    module_dir = tmp_path / "source" / "module"
    cache_root = tmp_path / "runtime-cache"
    module_dir.mkdir(parents=True)
    env: dict[str, str] = {}

    cache = configure_plugin_cache(
        env,
        module_dir,
        default_root=cache_root,
        target_platform="darwin_arm64",
    )

    assert cache == (cache_root / "darwin_arm64").resolve()
    assert env["TF_PLUGIN_CACHE_DIR"] == str(cache)
    assert cache.is_dir()
    assert module_dir not in cache.parents


def test_plugin_cache_rejects_source_checkout_path(tmp_path: Path) -> None:
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    env = {"TF_PLUGIN_CACHE_DIR": str(module_dir / ".terraform-providers")}

    with pytest.raises(TerraformLockError, match="inside Terraform source module"):
        configure_plugin_cache(
            env,
            module_dir,
            default_root=tmp_path / "unused",
            target_platform="linux_amd64",
        )
