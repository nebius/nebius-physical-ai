"""Hermetic Terraform provider-lock and runtime-cache guardrails."""

from __future__ import annotations

import hashlib
import json
import platform as host_platform
import re
from pathlib import Path
from typing import MutableMapping


LOCK_FILE_NAME = ".terraform.lock.hcl"
LOCK_PLATFORM_METADATA_NAME = ".terraform.lock.npa.json"
SUPPORTED_OPERATOR_PLATFORMS = (
    "linux_amd64",
    "linux_arm64",
    "darwin_amd64",
    "darwin_arm64",
)


class TerraformLockError(RuntimeError):
    """The tracked provider lock cannot verify this operator platform."""


def terraform_platform(*, system: str | None = None, machine: str | None = None) -> str:
    """Return Terraform's ``os_arch`` spelling for this operator host."""

    os_name = str(system or host_platform.system()).strip().lower()
    architecture = str(machine or host_platform.machine()).strip().lower()
    os_aliases = {"macos": "darwin"}
    arch_aliases = {
        "x86_64": "amd64",
        "x64": "amd64",
        "aarch64": "arm64",
    }
    return f"{os_aliases.get(os_name, os_name)}_{arch_aliases.get(architecture, architecture)}"


def _lock_digest(lock_file: Path) -> str:
    return hashlib.sha256(lock_file.read_bytes()).hexdigest()


def _provider_h1_counts(lock_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in re.finditer(
        r'^provider\s+"(?P<source>[^"]+)"\s*\{(?P<body>.*?)^\}',
        lock_text,
        re.MULTILINE | re.DOTALL,
    ):
        counts[match.group("source")] = len(
            re.findall(r'^\s*"h1:[^"]+",?\s*$', match.group("body"), re.MULTILINE)
        )
    return counts


def validate_provider_lock(
    terraform_dir: str | Path,
    *,
    target_platform: str | None = None,
) -> str:
    """Validate tracked lock provenance/coverage and return the target platform.

    Terraform lock files contain platform package hashes but do not label each
    ``h1`` line with its platform. NPA therefore tracks a small SHA-bound sidecar
    recording the exact ``terraform providers lock -platform=...`` invocation.
    Runtime never regenerates either file.
    """

    module_dir = Path(terraform_dir)
    lock_file = module_dir / LOCK_FILE_NAME
    metadata_file = module_dir / LOCK_PLATFORM_METADATA_NAME
    platform_name = str(target_platform or terraform_platform()).strip()
    remediation = (
        "Regenerate in a clean reviewed checkout with `terraform "
        f"-chdir={module_dir} providers lock "
        + " ".join(f"-platform={item}" for item in SUPPORTED_OPERATOR_PLATFORMS)
        + "`, update the SHA-bound .terraform.lock.npa.json metadata, and review "
        "the exact lock-file diff. Do not delete the lock or bypass checksums."
    )
    try:
        lock_text = lock_file.read_text(encoding="utf-8")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerraformLockError(
            f"Terraform provider-lock metadata is missing or malformed for {module_dir}. "
            f"{remediation}"
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise TerraformLockError(
            f"Terraform provider-lock metadata has an unsupported schema for {module_dir}. "
            f"{remediation}"
        )
    platforms = metadata.get("platforms")
    if platforms != list(SUPPORTED_OPERATOR_PLATFORMS):
        raise TerraformLockError(
            f"Terraform provider-lock metadata does not record the required operator "
            f"platform set for {module_dir}. {remediation}"
        )
    if metadata.get("lock_sha256") != _lock_digest(lock_file):
        raise TerraformLockError(
            f"Terraform provider-lock metadata does not match {lock_file}; the lock "
            f"changed without a reviewed platform-coverage update. {remediation}"
        )
    if platform_name not in platforms:
        raise TerraformLockError(
            f"Terraform provider lock for {module_dir} does not cover current platform "
            f"{platform_name}. {remediation}"
        )
    h1_counts = _provider_h1_counts(lock_text)
    if not h1_counts:
        raise TerraformLockError(
            f"Terraform provider lock {lock_file} contains no provider hashes. {remediation}"
        )
    required_count = len(platforms)
    incomplete = sorted(
        provider for provider, count in h1_counts.items() if count < required_count
    )
    if incomplete:
        raise TerraformLockError(
            "Terraform provider lock does not contain one signed package hash per "
            f"required platform for: {', '.join(incomplete)}. {remediation}"
        )
    return platform_name


def configure_plugin_cache(
    env: MutableMapping[str, str],
    terraform_dir: str | Path,
    *,
    default_root: str | Path,
    target_platform: str | None = None,
) -> Path:
    """Set a platform-scoped provider cache outside the Terraform source module."""

    module_dir = Path(terraform_dir).resolve()
    platform_name = str(target_platform or terraform_platform()).strip()
    configured_root = str(env.get("TF_PLUGIN_CACHE_DIR", "") or "").strip()
    cache_root = Path(configured_root).expanduser() if configured_root else Path(default_root)
    cache_dir = (cache_root / platform_name).resolve()
    try:
        cache_dir.relative_to(module_dir)
    except ValueError:
        pass
    else:
        raise TerraformLockError(
            f"TF_PLUGIN_CACHE_DIR resolves inside Terraform source module {module_dir}. "
            "Choose an external cache path (for example ~/.npa/terraform-plugin-cache); "
            "NPA will keep each operator platform in its own subdirectory."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["TF_PLUGIN_CACHE_DIR"] = str(cache_dir)
    return cache_dir


def provider_lock_metadata(lock_file: str | Path) -> dict[str, object]:
    """Build deterministic sidecar content after a reviewed providers-lock run."""

    path = Path(lock_file)
    return {
        "version": 1,
        "generated_by": "terraform providers lock",
        "platforms": list(SUPPORTED_OPERATOR_PLATFORMS),
        "lock_sha256": _lock_digest(path),
    }
