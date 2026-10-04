"""Guard the operator-private SwitchWorld LingBot image recipe."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lingbot-world/Dockerfile.switchworld-private"
CONTRACT = ROOT / "npa/docker/workbench/packaging-contract.yaml"


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


def test_private_recipe_is_quarantined_without_a_public_catalog_entry() -> None:
    """Private validation must not accidentally become a public release claim."""

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["lingbot-world-switchworld-private"]

    assert entry["dockerfile"] == "lingbot-world/Dockerfile.switchworld-private"
    assert entry["redistribution"] == "unvalidated"
    assert "private" in entry["notes"].lower()
    assert "publication" in entry["notes"].lower()
