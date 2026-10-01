"""Exercise CI media setup without network access or changes to the host."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/scripts/ci_install_ffmpeg.sh"


def _executable(path: Path, content: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + content)
    path.chmod(0o755)


@pytest.fixture
def media_setup(tmp_path):
    """Build isolated executable probes and a fake package manager.

    Args:
        tmp_path: Temporary directory supplied by pytest.
    Returns:
        Runner directory and isolated process environment.
    Raises:
        OSError: Fixture files cannot be created.
    """
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for command in ("sed", "mktemp", "chmod", "rm"):
        (binaries / command).symlink_to(shutil.which(command))
    _executable(binaries / "sudo", _apt_stub())
    (tmp_path / "ubuntu.sources").write_text(
        "Types: deb\nURIs: http://azure.archive.ubuntu.com/ubuntu/\n"
        "Suites: noble noble-updates noble-security\n"
        "Components: main universe\nArchitectures: amd64\n"
        "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
    )
    return tmp_path, {**os.environ, "PATH": str(binaries), "RUNNER_TEMP": str(tmp_path)}


def _apt_stub() -> str:
    return """
printf '%s\n' "$*" >> "$RUNNER_TEMP/apt-calls"
for argument in "$@"; do
  case "$argument" in
    Dir::Etc::sourcelist=*)
      /bin/cp "${argument#*=}" "$RUNNER_TEMP/selected.sources" ;;
  esac
done
case " $* " in
  *' update '*) exit "${UPDATE_STATUS:-0}" ;;
esac
if [ "${INSTALL_STATUS:-0}" != 0 ]; then exit "$INSTALL_STATUS"; fi
for tool in ffmpeg ffprobe; do
  if [ "$tool" = "${OMIT_TOOL:-}" ]; then continue; fi
  printf '#!/bin/sh\nexit 0\n' > "$RUNNER_TEMP/bin/$tool"
  chmod +x "$RUNNER_TEMP/bin/$tool"
done
"""


def _run(setup, **overrides):
    directory, environment = setup
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), str(directory / "ubuntu.sources")],
        env={**environment, **overrides},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("present", [(), ("ffmpeg",), ("ffprobe",)])
def test_missing_media_tools_use_signed_primary_archive(media_setup, present):
    """Install either missing tool while preserving the signed Ubuntu policy.

    Args:
        media_setup: Isolated package-manager fixture.
        present: Tools already available on the runner.
    Returns:
        None.
    Raises:
        AssertionError: Setup changes sources, skips installation or leaves files.
    """
    directory, _ = media_setup
    original = (directory / "ubuntu.sources").read_text()
    for tool in present:
        _executable(directory / "bin" / tool, "exit 0\n")
    result = _run(media_setup)
    assert result.returncode == 0, result.stderr
    assert (directory / "selected.sources").read_text() == original.replace(
        "http://azure.archive.ubuntu.com/ubuntu/", "https://archive.ubuntu.com/ubuntu"
    )
    assert (directory / "ubuntu.sources").read_text() == original
    calls = (directory / "apt-calls").read_text().splitlines()
    assert len(calls) == 2
    assert all("Dir::Etc::sourceparts=-" in call for call in calls)
    assert calls[0].endswith("update")
    assert calls[1].endswith("install -y --no-install-recommends ffmpeg")
    assert not list(directory.glob("npa-media-*.sources"))


@pytest.mark.parametrize("broken", [None, "ffmpeg", "ffprobe"])
def test_existing_tools_are_verified_without_package_downloads(media_setup, broken):
    """Avoid apt for installed tools, but reject a broken executable.

    Args:
        media_setup: Isolated package-manager fixture.
        broken: Tool whose version probe fails, if any.
    Returns:
        None.
    Raises:
        AssertionError: Setup downloads packages or accepts a failed probe.
    """
    directory, _ = media_setup
    (directory / "ubuntu.sources").unlink()
    for tool in ("ffmpeg", "ffprobe"):
        _executable(directory / "bin" / tool, f"exit {7 if broken == tool else 0}\n")
    assert _run(media_setup).returncode == (7 if broken else 0)
    assert not (directory / "apt-calls").exists()


@pytest.mark.parametrize(
    "overrides",
    [{"UPDATE_STATUS": "9"}, {"INSTALL_STATUS": "8"}, {"OMIT_TOOL": "ffprobe"}],
)
def test_failed_installation_blocks_validation(media_setup, overrides):
    """Keep package and executable failures blocking, with temporary cleanup.

    Args:
        media_setup: Isolated package-manager fixture.
        overrides: Simulated package or executable failure.
    Returns:
        None.
    Raises:
        AssertionError: Setup succeeds despite missing prerequisites.
    """
    assert _run(media_setup, **overrides).returncode != 0
    assert not list(media_setup[0].glob("npa-media-*.sources"))


def test_every_coverage_shard_requires_media_setup():
    """Keep executable media validation on every full-suite shard.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The workflow omits setup or permits silent media skips.
    """
    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
    job = workflow["jobs"]["test"]
    assert job["env"]["NPA_REQUIRE_FFMPEG"] == "1"
    setup = next(step for step in job["steps"] if step.get("name") == "Install ffmpeg")
    assert setup == {
        "name": "Install ffmpeg",
        "run": "bash npa/scripts/ci_install_ffmpeg.sh",
    }
