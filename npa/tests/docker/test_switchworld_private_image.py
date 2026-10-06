"""Guard the operator-private SwitchWorld LingBot image recipe."""

from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lingbot-world/Dockerfile.switchworld-private"
WAN_DOCKERFILE = ROOT / "npa/docker/workbench/wan2-2/Dockerfile"
CONTRACT = ROOT / "npa/docker/workbench/packaging-contract.yaml"
WAN_MEDIA_REQUIREMENTS = ROOT / (
    "npa/docker/workbench/lingbot-world/switchworld-wan-runtime-requirements.txt"
)
VISUALIZE_REQUIREMENTS = ROOT / (
    "npa/docker/workbench/lingbot-world/switchworld-visualize-runtime-requirements.txt"
)
WAN_RUNTIME_REQUIREMENTS = ROOT / "npa/docker/workbench/wan2-2/runtime-requirements.txt"
SWITCHWORLD = ROOT / "npa/src/npa/workflows/switchworld.py"
VISUALIZE_RUNTIME = ROOT / (
    "npa/docker/workbench/lingbot-world/switchworld_visualize_runtime.sh"
)


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
    """Keep the model and NumPy-2 visualization closures separate at runtime."""

    text = DOCKERFILE.read_text(encoding="utf-8")
    wan_media_requirements = WAN_MEDIA_REQUIREMENTS.read_text(encoding="utf-8")
    visualize_requirements = VISUALIZE_REQUIREMENTS.read_text(encoding="utf-8")
    wan_requirements = WAN_RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    visualize_runtime = VISUALIZE_RUNTIME.read_text(encoding="utf-8")
    switchworld_source = SWITCHWORLD.read_text(encoding="utf-8")

    assert "switchworld-wan-runtime-requirements.txt" in text
    assert "switchworld-visualize-runtime-requirements.txt" in text
    assert "switchworld-visualize-runtime" in text
    assert "runtime-requirements.txt" in text
    assert "av==17.1.0" in wan_media_requirements
    assert "numpy==1.26.4" in wan_requirements
    assert "numpy==2.2.6" not in wan_requirements
    assert "numpy==2.2.6" in visualize_requirements
    assert visualize_requirements.count("--hash=sha256:") >= 35
    for requirement, digest in {
        "attrs==26.1.0": "c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309",
        "boto3==1.39.11": "af8f1dad35eceff7658fab43b39b0f55892b6e3dd12308733521cc24dd2c9a02",
        "botocore==1.39.17": "41db169e919f821b3ef684794c5e67dd7bb1f5ab905d33729b1d8c27fafe8004",
        "jmespath==1.1.0": "a5663118de4908c91729bea0acadca56526eb2698e83de10cd116ae0f4e97c64",
        "pillow==12.3.0": "f0606c8bf2cdefea14a43530f7657cbbb7ecf1c4222512492ef4a4434a9501ec",
        "psutil==7.1.1": "92ebc58030fb054fa0f26c3206ef01c31c29d67aee1367e3483c16665c25c8d2",
        "pyarrow==20.0.0": "15aa1b3b2587e74328a730457068dc6c89e6dcbf438d4369f572af9d320a25ee",
        "python-dateutil==2.9.0.post0": "a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427",
        "rerun-sdk==0.38.1": "5f6b16374b1af1ecdb7232ed5c7f02f77e908c7c378ce83321479d2610286c0c",
        "s3transfer==0.13.1": "a981aa7429be23fe6dfc13e80e4020057cbab622b08c0315288758d67cabc724",
        "six==1.17.0": "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
        "typing_extensions==4.16.0": "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8",
        "urllib3==2.7.0": "9fb4c81ebbb1ce9531cce37674bbc6f1360472bc18ca9a553ede278ef7276897",
    }.items():
        assert requirement in visualize_requirements
        assert f"--hash=sha256:{digest}" in visualize_requirements
    assert "NPA_SWITCHWORLD_VISUALIZE_RUNTIME_CACHE" in visualize_runtime
    assert "--require-hashes" in visualize_runtime
    assert "_npa_switchworld_image_site.pth" not in visualize_runtime
    assert 'int(md.version("numpy").split(".", 1)[0]) >= 2' in visualize_runtime
    assert 'md.version("boto3") == "1.39.11"' in visualize_runtime
    assert 'md.version("rerun-sdk") == "0.38.1"' in visualize_runtime
    assert "-m rerun rrd --help" in visualize_runtime
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
