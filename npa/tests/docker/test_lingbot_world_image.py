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


def test_lingbot_defaults_to_immutable_public_parent_but_allows_private_requalification() -> (
    None
):
    """Private validation can use a scanned parent without retagging the public default."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "ARG WAN_BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/"
        "npa-wan2-2@sha256:5780959ca6c6e7eb77ee7ea7d005fcf0f56db50783ce798dddee2809185eb837"
        in dockerfile
    )
    assert "FROM ${WAN_BASE_IMAGE}" in dockerfile


def test_lingbot_model_runtime_includes_pyav_for_predecessor_video_validation() -> None:
    """The CUDA runtime executes both generation stages and decodes their handoff."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "/opt/wan-base/bin/python -m pip install --no-cache-dir --no-deps" in dockerfile
    )
    assert "protobuf==6.33.6 scipy==1.15.3 av==17.1.0" in dockerfile


def test_lingbot_upgrades_fixable_parent_perl_security_packages() -> None:
    """The private derivative must not retain the vulnerable parent revisions."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "apt-get install -y --no-install-recommends --only-upgrade" in dockerfile
    for package in (
        "perl=5.36.0-7+deb12u4",
        "perl-base=5.36.0-7+deb12u4",
        "libperl5.36=5.36.0-7+deb12u4",
        "perl-modules-5.36=5.36.0-7+deb12u4",
    ):
        assert package in dockerfile

    security_upgrade = dockerfile.index("perl=5.36.0-7+deb12u4")
    source_fetch = dockerfile.index(
        "COPY --chmod=0755 docker/workbench/common/model_source.sh"
    )
    assert dockerfile.index("USER root") < security_upgrade < source_fetch
    assert "rm -rf /var/lib/apt/lists/*" in dockerfile[security_upgrade:source_fetch]
