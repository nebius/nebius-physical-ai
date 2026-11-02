from __future__ import annotations

import re
from pathlib import Path
import tomllib

from npa.clients import nebius
from npa.deploy.images import SUPPORTED_TOOL_VERSIONS

ROOT = Path(__file__).resolve().parents[3]
RECOMMENDED = "0.12.298"
VM_PROVIDER = "0.5.293"


def test_runtime_packaging_and_documented_cli_version_do_not_drift() -> None:
    pyproject = (ROOT / "npa" / "pyproject.toml").read_text(encoding="utf-8")
    packaged = re.search(r'^nebius-cli\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert packaged and packaged.group(1) == RECOMMENDED
    assert SUPPORTED_TOOL_VERSIONS["nebius-cli"] == RECOMMENDED
    assert RECOMMENDED in nebius._TESTED_NEBIUS_CLI_VERSIONS

    install_fragment = f"NEBIUS_CLI_VERSION={RECOMMENDED} bash"
    for relative in ("README.md", "docs/install.md"):
        document = (ROOT / relative).read_text(encoding="utf-8")
        assert install_fragment in document, (
            f"{relative} does not install the tested CLI"
        )


def test_standalone_vm_spot_provider_pin_and_lock_do_not_drift() -> None:
    project = tomllib.loads((ROOT / "npa/pyproject.toml").read_text())
    constraint = f"~> {VM_PROVIDER}"
    assert project["tool"]["npa"]["supported-tools"]["terraform"] == constraint
    assert SUPPORTED_TOOL_VERSIONS["terraform"] == constraint
    module = ROOT / "npa/src/npa/deploy/terraform"
    assert f'version = "{constraint}"' in (module / "main.tf").read_text()
    lock = (module / ".terraform.lock.hcl").read_text()
    assert f'version     = "{VM_PROVIDER}"' in lock


def test_preemptible_vm_template_pairs_spot_price_without_retired_priority() -> None:
    template = (ROOT / "npa/src/npa/deploy/terraform/main.tf").read_text()
    assert re.search(
        r"(?m)^\s*follows_spot_price\s*=\s*var.enable_preemptible\s*\?\s*\{\}\s*:\s*null$",
        template,
    )
    preemptible = re.search(
        r"preemptible\s*=\s*var.enable_preemptible\s*\?\s*\{([^}]+)\}\s*:\s*null",
        template,
    )
    assert preemptible
    assert re.search(r'on_preemption\s*=\s*"STOP"', preemptible.group(1))
    assert not re.search(r"(?m)^\s*priority\s*=", preemptible.group(1))
    assert re.search(
        r"ignore_changes\s*=\s*\[[^\]]*\bpreemptible\.priority\b", template
    )


def test_documented_cluster_shape_uses_real_cli_flags() -> None:
    paidf = (
        ROOT / "docs" / "workbench" / "guides" / "physical-ai-data-factory-deploy.md"
    ).read_text(encoding="utf-8")
    required = (
        "--cpu-nodes 1",
        "--cpu-platform cpu-d3",
        "--cpu-preset 8vcpu-32gb",
        "--gpu-nodes 1",
        "--gpu-platform gpu-rtx6000",
        "--gpu-preset 1gpu-24vcpu-218gb",
        "--on-demand",
        "--preemptible",
    )
    for flag in required:
        assert flag in paidf
