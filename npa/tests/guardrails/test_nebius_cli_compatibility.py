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
