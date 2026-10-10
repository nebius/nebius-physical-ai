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


def test_lingbot_model_runtime_builds_pyav_against_audited_system_ffmpeg() -> None:
    """The runtime decodes both generation handoffs without wheel-bundled FFmpeg."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM ${WAN_BASE_IMAGE} AS pyav-wheel-builder" in dockerfile
    assert "--no-binary=av --wheel-dir /opt/npa-pyav-wheel av==17.1.0" in dockerfile
    for package in (
        "build-essential",
        "pkg-config",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavfilter-dev",
        "libavformat-dev",
        "libavutil-dev",
        "libswresample-dev",
        "libswscale-dev",
    ):
        assert package in dockerfile
    assert (
        "COPY --from=pyav-wheel-builder /opt/npa-pyav-wheel /opt/npa-pyav-wheel"
        in dockerfile
    )
    assert (
        "protobuf==6.33.6 scipy==1.15.3 /opt/npa-pyav-wheel/av-17.1.0-*.whl"
        in dockerfile
    )
    assert "protobuf==6.33.6 scipy==1.15.3 av==17.1.0" not in dockerfile


def test_lingbot_upgrades_fixable_parent_perl_security_packages() -> None:
    """The private derivative has a reproducible fixed parent Perl floor."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    fixed_version = "5.36.0-7+deb12u4"
    assert "ARG DEBIAN_SNAPSHOT=20261010T000000Z" in dockerfile
    assert f"ARG DEBIAN_PERL_SECURITY_VERSION={fixed_version}" in dockerfile
    assert (
        "https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}/" in dockerfile
    )
    assert (
        "https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}/"
        in dockerfile
    )
    assert "Suites: bookworm bookworm-updates" in dockerfile
    assert "Suites: bookworm-security" in dockerfile
    assert 'Acquire::Check-Valid-Until "false";' in dockerfile
    assert "apt-get install -y --no-install-recommends --only-upgrade" in dockerfile
    for package in (
        "perl",
        "perl-base",
        "libperl5.36",
        "perl-modules-5.36",
    ):
        assert f'"{package}=${{DEBIAN_PERL_SECURITY_VERSION}}"' in dockerfile

    snapshot_config = dockerfile.index("/etc/apt/apt.conf.d/99snapshot")
    security_upgrade = dockerfile.index('"perl=${DEBIAN_PERL_SECURITY_VERSION}"')
    source_fetch = dockerfile.index(
        "COPY --chmod=0755 docker/workbench/common/model_source.sh"
    )
    assert (
        dockerfile.index("USER root")
        < snapshot_config
        < security_upgrade
        < source_fetch
    )
    assert "rm -rf /var/lib/apt/lists/*" in dockerfile[security_upgrade:source_fetch]
