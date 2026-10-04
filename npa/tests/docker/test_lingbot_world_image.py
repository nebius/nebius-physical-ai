"""Guard LingBot's source-only runtime packaging contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/docker/workbench/lingbot-world/Dockerfile"
NOTICE = ROOT / "npa/docker/workbench/lingbot-world/REDISTRIBUTION.md"


def test_lingbot_reuses_apache_easydict_compat_not_lgpl_distribution() -> None:
    """The child runtime must retain the parent compatibility implementation."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    notice = NOTICE.read_text(encoding="utf-8")

    assert "cp -a /opt/npa/wan2-2/source/wan/utils /opt/byof/wan/" in dockerfile
    assert "from wan.utils.easydict import EasyDict" in dockerfile
    assert "easydict==1.13" not in dockerfile
    assert "! grep -qx 'easydict' /opt/byof/requirements.txt" in dockerfile
    assert "no `easydict` distribution is shipped" in " ".join(notice.split())


def test_lingbot_defaults_to_immutable_public_parent_but_allows_private_requalification() -> None:
    """Private validation can use a scanned parent without retagging the public default."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "ARG WAN_BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/"
        "npa-wan2-2@sha256:5780959ca6c6e7eb77ee7ea7d005fcf0f56db50783ce798dddee2809185eb837"
        in dockerfile
    )
    assert "FROM ${WAN_BASE_IMAGE}" in dockerfile
