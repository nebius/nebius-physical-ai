from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).parents[2]
    / "docker"
    / "workbench"
    / "common"
    / "nvidia_graphics_runtime.sh"
)
DRIVER_VERSION = "580.95.05"
ARTIFACT = f"NVIDIA-Linux-x86_64-{DRIVER_VERSION}-no-compat32.run"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fixture(tmp_path: Path) -> tuple[dict[str, str], Path]:
    commands = tmp_path / "commands"
    commands.mkdir()
    origin = tmp_path / "origin"
    origin.mkdir()
    artifact = origin / ARTIFACT
    _write_executable(
        artifact,
        """#!/bin/sh
set -eu
target=''
while [ "$#" -gt 0 ]; do
  [ "$1" != --target ] || { shift; target="$1"; }
  shift
done
mkdir -p "$target/payload"
touch "$target/payload/libnvidia-glcore.so.580.95.05"
touch "$target/payload/libnvidia-eglcore.so.580.95.05"
touch "$target/payload/libGLX_nvidia.so.580.95.05"
touch "$target/payload/libEGL_nvidia.so.580.95.05"
touch "$target/payload/libGL.so.1.7.0"
touch "$target/payload/libGLdispatch.so.0"
touch "$target/payload/libGLESv1_CM.so.1.2.0"
touch "$target/payload/libGLESv2.so.2.1.0"
""",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (origin / f"{ARTIFACT}.sha256sum").write_text(
        f"{digest}  {ARTIFACT}\n", encoding="utf-8"
    )
    _write_executable(
        commands / "nvidia-smi",
        f"#!/bin/sh\nprintf '%s\\n' '{DRIVER_VERSION}'\n",
    )
    _write_executable(
        commands / "curl",
        """#!/bin/sh
set -eu
out=''
last=''
while [ "$#" -gt 0 ]; do
  [ "$1" != -o ] || { shift; out="$1"; }
  last="$1"
  shift
done
printf 'download\n' >> "$FAKE_CURL_COUNTER"
cp "$FAKE_ORIGIN/${last##*/}" "$out"
""",
    )
    cache = tmp_path / "cache"
    env = os.environ | {
        "PATH": f"{commands}:{os.environ['PATH']}",
        "ACCEPT_EULA": "Y",
        "NPA_NVIDIA_GRAPHICS_DIR": str(cache),
        "NPA_NVIDIA_DRIVER_ORIGIN": "https://vendor.example.invalid/driver",
        "FAKE_ORIGIN": str(origin),
        "FAKE_CURL_COUNTER": str(tmp_path / "downloads"),
    }
    return env, cache


def test_cold_population_is_verified_atomic_and_warm_reuse_skips_download(
    tmp_path: Path,
) -> None:
    env, cache = _fixture(tmp_path)
    first = subprocess.run([SCRIPT], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    current = (cache / "current").resolve()
    assert current.parent == cache
    assert (current / ".ready").read_text(encoding="utf-8").startswith(
        f"driver={DRIVER_VERSION}\nsha256="
    )
    assert (current / "libGLX_nvidia.so.0").is_symlink()
    assert (current / "libGL.so.1").resolve() == current / "libGL.so.1.7.0"
    assert (current / "libGLESv1_CM.so.1").is_symlink()
    assert (current / "libGLESv2.so.2").is_symlink()
    assert (cache / "runtime.env").is_file()
    assert (tmp_path / "downloads").read_text(encoding="utf-8").count("download") == 2

    second = subprocess.run([SCRIPT], env=env, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert (tmp_path / "downloads").read_text(encoding="utf-8").count("download") == 2


def test_checksum_mismatch_fails_closed_without_publishing_current(
    tmp_path: Path,
) -> None:
    env, cache = _fixture(tmp_path)
    checksum = Path(env["FAKE_ORIGIN"]) / f"{ARTIFACT}.sha256sum"
    checksum.write_text(f"{'0' * 64}  {ARTIFACT}\n", encoding="utf-8")
    result = subprocess.run([SCRIPT], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
    assert not (cache / "current").exists()
    assert not any(cache.glob("*/.ready"))
