from __future__ import annotations

import os
import hashlib
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from npa.solutions.wan2_2.dependency_closure import (
    DependencyClosureError,
    DistributionMetadata,
    RUNTIME_ONLY_DISTRIBUTIONS,
    parse_runtime_requirements,
    validate_dependency_union,
)


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SCRIPT = ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "wan_runtime.sh"
RUNTIME_REQUIREMENTS = (
    ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "runtime-requirements.txt"
)
BAKED_CONSTRAINTS = (
    ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "baked-constraints.txt"
)
WAN_DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "Dockerfile"
WAN_SMOKE = ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "smoke.sh"
WAN_ENTRYPOINT = ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "entrypoint.sh"


def test_entrypoint_allows_skypilot_bootstrap_without_runtime_fetch() -> None:
    completed = subprocess.run(
        ["bash", str(WAN_ENTRYPOINT), "printf", "BOOTSTRAP_OK"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "BOOTSTRAP_OK"


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


def test_security_fixed_runtime_and_baked_image_are_fully_pinned() -> None:
    runtime = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    baked = BAKED_CONSTRAINTS.read_text(encoding="utf-8")
    dockerfile = WAN_DOCKERFILE.read_text(encoding="utf-8")

    assert "torch==2.13.0" in runtime
    assert "pillow==12.3.0" in runtime
    assert "nvidia-nccl-cu13==2.29.7" in runtime
    assert "torchaudio" not in runtime
    assert "pillow==12.3.0" in baked
    assert "diffusers==0.38.0" in baked
    assert "peft==0.20.0" in baked
    assert "transformers==5.5.0" in baked
    assert "sentencepiece==0.2.1" in baked
    assert "pip==26.1.2" in baked
    assert "setuptools==83.0.0" in baked
    assert "wheel==0.46.2" in baked
    assert not any(
        line.startswith(("torch==", "torchvision==", "torchaudio=="))
        for line in baked.splitlines()
    )
    assert "pip install --no-cache-dir --no-deps" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" not in dockerfile
    assert "^(torch|torchvision|torchaudio|nvidia-)" in dockerfile
    assert "dependency_closure.py validate" in dockerfile
    assert '"opencv-python-headless>=4.9.0.80"' in dockerfile
    assert "-e '/\"easydict\",/d'" in dockerfile
    assert "-e '/\"flash_attn\",/d'" in dockerfile
    assert "pip install --no-cache-dir --no-deps -e /opt/byof" in dockerfile
    assert "-m compileall -q --invalidation-mode checked-hash" in dockerfile
    assert "find /opt/wan-base -type d -name __pycache__ -prune" in dockerfile
    assert "find /opt/wan-base -type f -name '*.pyc'" in dockerfile
    assert '"$tree/venv/bin/python" -m pip check' in RUNTIME_SCRIPT.read_text(
        encoding="utf-8"
    )
    smoke = WAN_SMOKE.read_text(encoding="utf-8")
    assert 'find_spec("torch") is None' in smoke
    assert "dependency_closure.py verify-report" in smoke
    assert "resolve_wan_input_contract" in smoke
    assert "test -r /opt/byof/wan/textimage2video.py" not in smoke
    assert "wan-runtime ensure" in smoke


def _metadata(
    name: str, version: str, *requires_dist: str
) -> DistributionMetadata:
    return DistributionMetadata(
        name=name,
        version=version,
        requires_dist=tuple(requires_dist),
    )


def test_dependency_closure_allows_only_declared_runtime_only_family() -> None:
    report = validate_dependency_union(
        {
            "wan": _metadata("wan", "2.2", "torch>=2.13", "shared==1.0"),
            "shared": _metadata("shared", "1.0"),
        },
        {"torch": _metadata("torch", "2.13.0", "shared>=1")},
        runtime_only_allowlist=frozenset({"torch"}),
    )

    assert report["status"] == "validated"
    assert report["runtime_only_distributions"] == ["torch"]
    assert report["applicable_dependency_edges_checked"] == 3


def test_dependency_closure_rejects_unapproved_runtime_only_package() -> None:
    with pytest.raises(DependencyClosureError, match="runtime-only distribution set"):
        validate_dependency_union(
            {"wan": _metadata("wan", "2.2")},
            {"unreviewed": _metadata("unreviewed", "1.0")},
            runtime_only_allowlist=frozenset(),
        )


def test_dependency_closure_rejects_missing_transitive_requirement() -> None:
    with pytest.raises(DependencyClosureError, match="requires missing omitted"):
        validate_dependency_union(
            {"wan": _metadata("wan", "2.2", "omitted>=1")},
            {},
            runtime_only_allowlist=frozenset(),
        )


def test_dependency_closure_rejects_incompatible_transitive_requirement() -> None:
    with pytest.raises(
        DependencyClosureError, match=r"requires shared<2; effective shared==2\.0"
    ):
        validate_dependency_union(
            {
                "wan": _metadata("wan", "2.2", "shared<2"),
                "shared": _metadata("shared", "2.0"),
            },
            {},
            runtime_only_allowlist=frozenset(),
        )


def test_checked_in_runtime_only_set_is_exact() -> None:
    runtime = set(parse_runtime_requirements(RUNTIME_REQUIREMENTS))
    baked = {
        line.split("==", 1)[0].lower().replace("_", "-")
        for line in BAKED_CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }

    assert runtime.difference(baked) == RUNTIME_ONLY_DISTRIBUTIONS


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
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "check" ]]; then
  exit 0
fi
exec "${NPA_TEST_REAL_PYTHON}" "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    module_root = tmp_path / "runtime-modules"
    torch_module = module_root / "torch"
    torch_module.mkdir(parents=True)
    versions = {
        "torch": "2.13.0",
        "torchvision": "0.28.0",
        "cuda-toolkit": "13.0.3.0",
        "cuda-bindings": "13.3.1",
        "cuda-pathfinder": "1.6.0",
        "nvidia-cublas": "13.1.1.3",
        "nvidia-cuda-cupti": "13.0.85",
        "nvidia-cuda-nvrtc": "13.0.88",
        "nvidia-cuda-runtime": "13.0.96",
        "nvidia-cudnn-cu13": "9.20.0.48",
        "nvidia-cufft": "12.0.0.61",
        "nvidia-cufile": "1.15.1.6",
        "nvidia-curand": "10.4.0.35",
        "nvidia-cusolver": "12.0.4.66",
        "nvidia-cusparse": "12.6.3.3",
        "nvidia-cusparselt-cu13": "0.8.1",
        "nvidia-nccl-cu13": "2.29.7",
        "nvidia-nvjitlink": "13.3.33",
        "nvidia-nvshmem-cu13": "3.4.5",
        "nvidia-nvtx": "13.0.85",
        "triton": "3.7.1",
    }
    versions.update(overrides or {})
    torch_module.joinpath("__init__.py").write_text(
        f'__version__ = "{versions["torch"]}"\nclass version:\n    cuda = "13.0"\n',
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
    env["NPA_TEST_REAL_PYTHON"] = sys.executable
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
        {"torch": "2.13.00"},
        {"nvidia-nccl-cu13": "2.29.70"},
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
