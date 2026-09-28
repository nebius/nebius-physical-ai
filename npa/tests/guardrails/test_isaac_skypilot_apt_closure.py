"""Keep Isaac images independent of runtime APT for SkyPilot's early tools."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

DOCKER_ROOT = Path(__file__).resolve().parents[2] / "docker" / "workbench"
EARLY_PACKAGES = ("curl", "fuse", "netcat-openbsd", "rsync", "wget")


def _executable_text(text: str) -> str:
    """Return source lines that can affect a build.

    Args:
        text: Shell or Dockerfile source.

    Returns:
        Source with comment-only lines removed.

    Raises:
        None.
    """

    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _assert_early_package_block(text: str, start: str, end: str) -> None:
    """Require installation and build-time verification of every early package."""

    instructions = _executable_text(text)
    block = instructions[instructions.index(start) : instructions.index(end)]
    install = block[: block.index("for package in")]
    for package in EARLY_PACKAGES:
        assert package in install.split(), f"APT install omits {package}"
    loop = _status_loop(block)
    assert "${db:Status-Abbrev}" in loop
    assert "= 'ii '" in loop


def _status_loop(text: str) -> str:
    """Extract the installed-status loop as executable shell."""

    start = text.index("for package in curl")
    end = text.index("done", start) + len("done")
    return text[start:end].replace("\\\n", "")


def _fake_dpkg_query(path: Path) -> None:
    """Create a deterministic dpkg-query replacement for shell validation."""

    path.write_text(
        "#!/bin/sh\n"
        'if [ "$3" != "$NPA_DPKG_BAD_PACKAGE" ]; then printf \'ii \'; exit 0; fi\n'
        'case "$NPA_DPKG_STATE" in\n'
        "  installed) printf 'ii ' ;;\n"
        "  removed) printf 'rc ' ;;\n"
        "  missing) exit 1 ;;\n"
        "esac\n"
    )
    path.chmod(0o755)


@pytest.fixture(name="isaac_sources")
def fixture_isaac_sources() -> tuple[tuple[str, str, str], ...]:
    """Load canonical and repair sources with their package-block boundaries."""

    common = (DOCKER_ROOT / "common" / "install_isaac_runtime_base.sh").read_text()
    repair = (DOCKER_ROOT / "isaac-lab" / "Dockerfile.k8s-prereqs").read_text()
    return (
        (common, 'if [ "$INSTALL_SKYPILOT_PREREQS" = "1" ]', "printf 'ubuntu"),
        (repair, "RUN apt-get update", "&& rm -rf /var/lib/apt/lists/*"),
    )


def test_isaac_sources_bake_skypilot_early_packages(
    isaac_sources: tuple[tuple[str, str, str], ...],
) -> None:
    """Require the canonical installer and derived repair to carry the closure."""

    for source, start, end in isaac_sources:
        _assert_early_package_block(source, start, end)


@pytest.mark.parametrize(
    ("state", "expected_returncode"),
    (("installed", 0), ("removed", 1), ("missing", 1)),
)
@pytest.mark.parametrize("bad_package", EARLY_PACKAGES)
def test_status_loop_requires_installed_packages(
    state: str,
    expected_returncode: int,
    bad_package: str,
    isaac_sources: tuple[tuple[str, str, str], ...],
    tmp_path: Path,
) -> None:
    """Execute both source assertions against installed and invalid dpkg states."""

    _fake_dpkg_query(tmp_path / "dpkg-query")
    environment = {
        **os.environ,
        "PATH": str(tmp_path),
        "NPA_DPKG_STATE": state,
        "NPA_DPKG_BAD_PACKAGE": bad_package,
    }
    for source, _start, _end in isaac_sources:
        result = subprocess.run(
            ["/bin/sh", "-c", "true && " + _status_loop(source) + " && true"],
            env=environment,
            check=False,
        )
        assert result.returncode == expected_returncode


@pytest.mark.parametrize("missing", EARLY_PACKAGES)
def test_package_omission_fails_closed(
    missing: str,
    isaac_sources: tuple[tuple[str, str, str], ...],
) -> None:
    """Prove that an install-list omission cannot hide behind the assertion."""

    source, start, end = isaac_sources[0]
    install = source.index(missing, source.index(start))
    mutated = source[:install] + f"omitted-{missing}" + source[install + len(missing) :]
    with pytest.raises(AssertionError, match=f"APT install omits {missing}"):
        _assert_early_package_block(mutated, start, end)
