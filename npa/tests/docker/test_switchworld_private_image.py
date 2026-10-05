"""Guard the operator-private SwitchWorld LingBot image recipe."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lingbot-world/Dockerfile.switchworld-private"
WAN_DOCKERFILE = ROOT / "npa/docker/workbench/wan2-2/Dockerfile"
CONTRACT = ROOT / "npa/docker/workbench/packaging-contract.yaml"
RUNTIME_REQUIREMENTS = (
    ROOT / "npa/docker/workbench/lingbot-world/switchworld-runtime-requirements.txt"
)
WAN_RUNTIME_REQUIREMENTS = ROOT / "npa/docker/workbench/wan2-2/runtime-requirements.txt"
SWITCHWORLD = ROOT / "npa/src/npa/workflows/switchworld.py"


def test_private_recipe_uses_audited_snapshot_and_no_easydict_wheel() -> None:
    """Keep the scan remediation concrete instead of adding a scanner exception."""

    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG DEBIAN_SNAPSHOT=20260906T183022Z" in text
    assert "snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" in text
    assert "COPY --from=wan-source /opt/byof /opt/byof" in text
    assert "COPY --from=wan-source /opt/wan-base /opt/wan-base" in text
    assert "wan/utils/easydict.py" in text
    assert "from wan.utils.easydict import EasyDict" in text
    assert not re.search(r"pip install[^\n]*easydict", text)
    assert "import easydict" in text  # The build proves that it remains absent.
    assert 'npa.disposition="operator-private-validation-no-publication"' in text
    assert "ACCEPT_" not in text


def test_private_recipe_runtime_fetches_the_hash_locked_media_and_rrd_closure() -> None:
    """Keep the decoder and factual RRD writer out of private image bytes."""

    text = DOCKERFILE.read_text(encoding="utf-8")
    requirements = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    wan_requirements = WAN_RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    switchworld_source = SWITCHWORLD.read_text(encoding="utf-8")

    assert "switchworld-runtime-requirements.txt" in text
    assert "runtime-requirements.txt" in text
    assert "av==17.1.0" in requirements
    assert "numpy==2.2.6" in wan_requirements
    assert (
        "--hash=sha256:fc7b73d02efb0e18c000e9ad8b83480dfcd5dfd11065997ed4c6747470ae8915"
        in wan_requirements
    )
    assert requirements.count("--hash=sha256:") >= 34
    for requirement, digest in {
        "attrs==26.1.0": "c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309",
        "psutil==7.1.1": "92ebc58030fb054fa0f26c3206ef01c31c29d67aee1367e3483c16665c25c8d2",
        "pyarrow==20.0.0": "15aa1b3b2587e74328a730457068dc6c89e6dcbf438d4369f572af9d320a25ee",
        "rerun-sdk==0.38.1": "5f6b16374b1af1ecdb7232ed5c7f02f77e908c7c378ce83321479d2610286c0c",
    }.items():
        assert requirement in requirements
        assert f"--hash=sha256:{digest}" in requirements
    assert "import rerun as rr" in switchworld_source


def test_private_recipe_uses_the_reviewed_wan_attention_fallback() -> None:
    """Apply the fallback after LingBot source materialization replaces Wan."""

    text = DOCKERFILE.read_text(encoding="utf-8")
    wan_text = WAN_DOCKERFILE.read_text(encoding="utf-8")

    assert "from .attention import attention as flash_attention" in text
    assert "from .attention import attention as flash_attention" in wan_text
    assert "/opt/byof/wan/modules/model.py" in text
    assert '"flash_attn",/d' in text
    assert "FlashAttention wheel or CUDA build is fetched at runtime" in text
    assert text.index(
        "from .attention import attention as flash_attention"
    ) > text.index("https://github.com/Robbyant/lingbot-world.git")


def test_private_recipe_keeps_the_skypilot_sudo_contract_without_invoking_sudo() -> (
    None
):
    """Keep the runtime bootstrap prerequisite while avoiding a scanner-triggering probe."""

    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in text
    assert "test -r /etc/sudoers.d/npa-runtime" in text
    assert "grep -Fxq 'ubuntu ALL=(ALL) NOPASSWD:ALL'" in text
    assert "sudo -n true" not in text


def test_private_recipe_is_quarantined_without_a_public_catalog_entry() -> None:
    """Private validation must not accidentally become a public release claim."""

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["lingbot-world-switchworld-private"]

    assert entry["dockerfile"] == "lingbot-world/Dockerfile.switchworld-private"
    assert entry["redistribution"] == "unvalidated"
    assert "private" in entry["notes"].lower()
    assert "publication" in entry["notes"].lower()
