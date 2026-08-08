from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


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
