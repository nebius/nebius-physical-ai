from __future__ import annotations

import re
from pathlib import Path

import pytest

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
    """Wherever the cluster shape is documented, its flags must be real.

    This read the README as well until #289 simplified it; the deploy guide is
    now the only document that describes the shape, and it is the one a reader
    copies from.
    """

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
    assert "--resume-run" in paidf
    # The README additionally pinned its own submit shape (`--var bucket=`, and
    # no `--auto-load`). Those were choices that document made, not facts about
    # the CLI, and the guide deliberately makes different ones — so they are not
    # carried over. What belongs here is the flag vocabulary itself.


def test_documented_project_creation_uses_official_v2_cli_contract() -> None:
    """If a doc tells a reader how to create a project, it must be the v2 contract.

    No document does, since #289 removed the README section that did. The check
    is kept and scoped rather than deleted: the reason it exists — a v1-shaped
    command that fails against the real CLI — applies to whichever document
    reintroduces it, and finding none is a fact worth stating rather than a
    reason to stop looking.
    """

    documented = [
        path
        for path in sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
        if "nebius iam v2 project create" in path.read_text(encoding="utf-8")
        or "nebius iam project create" in path.read_text(encoding="utf-8")
    ]
    if not documented:
        pytest.skip("no document currently describes project creation")

    for path in documented:
        text = path.read_text(encoding="utf-8")
        assert "nebius iam project create" not in text, (
            f"{path} uses the v1 project-create shape, which the real CLI rejects"
        )
        assert 'nebius iam v2 project create --parent-id "$TENANT_ID"' in text
        assert '--name "$PROJECT_NAME" --region "$REGION" --format json' in text
