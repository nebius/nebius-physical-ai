from __future__ import annotations

import os
import hashlib
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SCRIPT = ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "wan_runtime.sh"
RUNTIME_REQUIREMENTS = (
    ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "runtime-requirements.txt"
)


def test_runtime_requirements_are_hash_locked() -> None:
    logical_lines: list[str] = []
    current = ""
    for raw in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        current += line.removesuffix("\\").strip() + " "
        if not line.endswith("\\"):
            logical_lines.append(current.strip())
            current = ""
    assert not current
    requirements = [line for line in logical_lines if not line.startswith("--")]
    assert requirements
    assert all(" --hash=sha256:" in line for line in requirements)
    assert "--require-hashes" in RUNTIME_SCRIPT.read_text(encoding="utf-8")


def _offline_runtime_env(
    tmp_path: Path, *, current_stamp: str
) -> tuple[dict[str, str], Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1 --hash=sha256:" + "0" * 64 + "\n")
    fake_base = tmp_path / "fake-base-python"
    fake_base.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '3.10-linux_x86_64'\n",
        encoding="utf-8",
    )
    fake_base.chmod(0o755)
    tree = cache / current_stamp
    (tree / "venv" / "bin").mkdir(parents=True)
    fake_venv = tree / "venv" / "bin" / "python"
    fake_venv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_venv.chmod(0o755)
    (tree / ".complete").touch()
    (cache / "current").symlink_to(tree)
    return (
        {
            **os.environ,
            "NPA_WAN_RUNTIME_CACHE": str(cache),
            "NPA_WAN_BASE_PYTHON": str(fake_base),
            "NPA_WAN_RUNTIME_REQUIREMENTS": str(requirements),
            "NPA_WAN_RUNTIME_OFFLINE": "1",
            "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS": "YES",
        },
        requirements,
    )


def test_offline_runtime_refuses_a_complete_stale_stamp(tmp_path: Path) -> None:
    env, _requirements = _offline_runtime_env(tmp_path, current_stamp="stale-stamp")
    completed = subprocess.run(
        ["bash", str(RUNTIME_SCRIPT), "ensure"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 69
    assert "does not match the current requirements and Python ABI" in completed.stderr


def test_offline_runtime_reuses_only_the_exact_current_stamp(tmp_path: Path) -> None:
    requirements_payload = "example==1 --hash=sha256:" + "0" * 64 + "\n"
    requirements_sha = hashlib.sha256(requirements_payload.encode()).hexdigest()
    stamp = hashlib.sha256(
        f"{requirements_sha}|3.10-linux_x86_64".encode()
    ).hexdigest()[:16]
    env, _requirements = _offline_runtime_env(tmp_path, current_stamp=stamp)
    completed = subprocess.run(
        ["bash", str(RUNTIME_SCRIPT), "ensure"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _real_runtime_verification_env(
    tmp_path: Path, *, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    requirements_payload = "example==1 --hash=sha256:" + "0" * 64 + "\n"
    requirements_sha = hashlib.sha256(requirements_payload.encode()).hexdigest()
    stamp = hashlib.sha256(
        f"{requirements_sha}|3.10-linux_x86_64".encode()
    ).hexdigest()[:16]
    env, _requirements = _offline_runtime_env(tmp_path, current_stamp=stamp)
    tree = (tmp_path / "cache" / "current").resolve()
    fake_python = tree / "venv" / "bin" / "python"
    fake_python.unlink()
    fake_python.symlink_to(sys.executable)

    module_root = tmp_path / "runtime-modules"
    torch_module = module_root / "torch"
    torch_module.mkdir(parents=True)
    versions = {
        "torch": "2.7.1+cu128",
        "torchvision": "0.22.1+cu128",
        "torchaudio": "2.7.1+cu128",
        "triton": "3.3.1",
        "nvidia-cublas-cu12": "12.8.3.14",
        "nvidia-cuda-cupti-cu12": "12.8.57",
        "nvidia-cuda-nvrtc-cu12": "12.8.61",
        "nvidia-cuda-runtime-cu12": "12.8.57",
        "nvidia-cudnn-cu12": "9.7.1.26",
        "nvidia-cufft-cu12": "11.3.3.41",
        "nvidia-cufile-cu12": "1.13.0.11",
        "nvidia-curand-cu12": "10.3.9.55",
        "nvidia-cusolver-cu12": "11.7.2.55",
        "nvidia-cusparse-cu12": "12.5.7.53",
        "nvidia-cusparselt-cu12": "0.6.3",
        "nvidia-nccl-cu12": "2.27.7",
        "nvidia-nvjitlink-cu12": "12.8.61",
        "nvidia-nvtx-cu12": "12.8.55",
    }
    versions.update(overrides or {})
    torch_module.joinpath("__init__.py").write_text(
        f'__version__ = "{versions["torch"]}"\nclass version:\n    cuda = "12.8"\n',
        encoding="utf-8",
    )
    for name, version in versions.items():
        if name == "torch":
            continue
        metadata_dir = module_root / f"{name.replace('-', '_')}-{version}.dist-info"
        metadata_dir.mkdir()
        metadata_dir.joinpath("METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    env["PYTHONPATH"] = str(module_root)
    return env


def test_runtime_verification_accepts_only_intended_local_version_suffixes(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", str(RUNTIME_SCRIPT), "ensure"],
        env=_real_runtime_verification_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "overrides",
    [
        {"torch": "2.7.10"},
        {"nvidia-nccl-cu12": "2.27.70"},
    ],
)
def test_runtime_verification_rejects_prefix_extension_versions(
    tmp_path: Path, overrides: dict[str, str]
) -> None:
    completed = subprocess.run(
        ["bash", str(RUNTIME_SCRIPT), "ensure"],
        env=_real_runtime_verification_env(tmp_path, overrides=overrides),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0


def test_interrupted_install_removes_partial_runtime_tree(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".old-stamp.tmp.999").mkdir()
    (cache / ".current.999").symlink_to(cache / ".old-stamp.tmp.999")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "example==1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8"
    )
    marker = tmp_path / "pip-started"
    fake_base = tmp_path / "fake-base-python"
    fake_base.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ \"${1:-}\" == \"-c\" ]]; then
  if [[ \"${2:-}\" == *'get_paths()[\"purelib\"]'* ]]; then
    printf '%s\\n' \"${FAKE_BASE_SITE}\"
  else
    printf '%s\\n' '3.10-linux_x86_64'
  fi
  exit 0
fi
if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"venv\" ]]; then
  target=\"${4}\"
  mkdir -p \"${target}/bin\" \"${FAKE_CACHE_SITE}\"
  cp \"${FAKE_VENV_PYTHON}\" \"${target}/bin/python\"
  chmod +x \"${target}/bin/python\"
  exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_base.chmod(0o755)
    fake_venv = tmp_path / "fake-venv-python"
    fake_venv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ \"${1:-}\" == \"-c\" ]]; then
  printf '%s\\n' \"${FAKE_CACHE_SITE}\"
  exit 0
fi
if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"pip\" && \"${3:-}\" == \"install\" ]]; then
  : > \"${FAKE_PIP_MARKER}\"
  while true; do read -r -t 60 _ || true; done
fi
if [[ \"${1:-}\" == \"-\" ]]; then
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_venv.chmod(0o755)
    env = {
        **os.environ,
        "NPA_WAN_RUNTIME_CACHE": str(cache),
        "NPA_WAN_BASE_PYTHON": str(fake_base),
        "NPA_WAN_RUNTIME_REQUIREMENTS": str(requirements),
        "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS": "YES",
        "FAKE_BASE_SITE": str(tmp_path / "base-site"),
        "FAKE_CACHE_SITE": str(tmp_path / "cache-site"),
        "FAKE_VENV_PYTHON": str(fake_venv),
        "FAKE_PIP_MARKER": str(marker),
    }
    proc = subprocess.Popen(
        ["bash", str(RUNTIME_SCRIPT), "ensure"],
        env=env,
        start_new_session=True,
    )
    for _ in range(100):
        if marker.exists():
            break
        assert proc.poll() is None
        time.sleep(0.02)
    assert marker.exists()
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(timeout=10) != 0
    assert not list(cache.glob(".*.tmp.*"))
    assert not list(cache.glob(".current.*"))
