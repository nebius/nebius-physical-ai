from __future__ import annotations

import re
from pathlib import Path

from npa.clients import nebius
from npa.deploy.images import SUPPORTED_TOOL_VERSIONS

ROOT = Path(__file__).resolve().parents[3]
RECOMMENDED = "0.12.254"


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


def test_documented_cluster_shape_uses_real_cli_flags() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
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
        assert flag in readme
        assert flag in paidf
    assert '"$RUN_ID" --runtime --var bucket=' in readme
    assert "--runtime --auto-load" not in readme
    assert "--resume-run" in readme
    assert 'npa provision-if-absent --project "$PROJECT" --skip-k8s' in readme


def test_documented_project_creation_uses_official_v2_cli_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'nebius iam v2 project create --parent-id "$TENANT_ID"' in readme
    assert '--name "$PROJECT_NAME" --region "$REGION" --format json' in readme
    assert 'nebius iam v2 project list --parent-id "$TENANT_ID" --all' in readme
    assert 'nebius iam v2 project get --id "$PROJECT_ID" --format json' in readme
    assert "tenant-level administrative permission" in readme
