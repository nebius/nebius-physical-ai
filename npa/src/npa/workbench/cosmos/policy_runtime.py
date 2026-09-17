"""Private runtime fetching for native Cosmos training and LIBERO simulation.

The inference image is only a CUDA/bootstrap base. Training gets its own pinned
upstream environment; the image's inference environment is never modified.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from npa.workbench.cosmos.policy_contract import FRAMEWORK_REVISION, LIBERO_REVISION


def run_native(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    """Run native code with argv boundaries and retain diagnostics privately.

    Args:
        argv: Executable and individual arguments; never shell text.
        cwd: Native source directory.
        env: Child environment, including operator-owned credentials.
        log: Private output file.
    Returns:
        None after successful process completion.
    Raises:
        subprocess.CalledProcessError: Native command failed.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=stream, check=True)


def _checkout(url: str, revision: str, target: Path, log: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for argv in (["git", "init", "."], ["git", "remote", "add", "origin", url],
                 ["git", "fetch", "--depth=1", "origin", revision],
                 ["git", "checkout", "--detach", "FETCH_HEAD"]):
        run_native(argv, cwd=target, env=dict(os.environ), log=log)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if actual != revision:
        raise ValueError("native source revision mismatch")


def prepare_training_runtime(root: Path, *, guardrails: bool = False) -> tuple[Path, dict[str, str]]:
    """Fetch an immutable framework and sync its real training dependencies.

    Args:
        root: Fresh, private runtime directory on a Linux CUDA worker.
        guardrails: Include the native policy server's guardrail dependencies.
    Returns:
        Framework checkout and environment enabling native training.
    Raises:
        OSError: Required git/uv executable is absent.
        subprocess.CalledProcessError: Checkout, dependency sync, or imports fail.
    """
    repo = root / "framework"
    log = root / "bootstrap.log"
    _checkout("https://github.com/NVIDIA/cosmos-framework.git", FRAMEWORK_REVISION, repo, log)
    env = dict(os.environ, COSMOS_TRAINING="1", PYTHONPATH=str(repo), WANDB_MODE="disabled")
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    sync = ["uv", "sync", "--frozen", "--extra", "train", "--group", "cu130-train"]
    if guardrails:
        sync.extend(["--extra", "guardrail"])
    run_native(sync,
               cwd=repo, env=env, log=log)
    run_native([str(repo / ".venv/bin/python"), "-c",
                "import torch, transformer_engine; "
                "assert torch.__version__.startswith('2.10.0'); "
                "assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"],
               cwd=repo, env=env, log=log)
    if guardrails:
        run_native([str(repo / ".venv/bin/python"), "-c",
                    "from cosmos_framework.auxiliary.guardrail.common import presets; "
                    "from cosmos_framework.auxiliary.guardrail.face_blur_filter.face_blur_filter "
                    "import RetinaFaceFilter; print('Native guardrail imports passed')"],
                   cwd=repo, env=env, log=log)
    run_native(["uv", "pip", "freeze", "--python", str(repo / ".venv/bin/python")],
               cwd=repo, env=env, log=root / "training-environment.log")
    return repo, env


def prepare_simulation_runtime(root: Path, repo: Path, env: dict[str, str]) -> Path:
    """Build the native recipe's separate Python 3.10 simulator environment.

    Args:
        root: Private runtime directory.
        repo: Cosmos framework checkout.
        env: Runtime environment; receives private LIBERO configuration path.
    Returns:
        Simulator Python executable.
    Raises:
        subprocess.CalledProcessError: Native dependency installation failed.
        RuntimeError: OSMesa libraries are absent outside a disposable container.
    """
    libero = root / "LIBERO"
    log = root / "simulation-bootstrap.log"
    _checkout("https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_REVISION, libero, log)
    _install_simulation_dependencies(root, env, log)
    python = root / "libenv/bin/python"
    run_native(["uv", "venv", "--python", "3.10", str(python.parent.parent)], cwd=root, env=env, log=log)
    constraints = root / "simulation-constraints.txt"
    constraints.write_text("torch==2.5.1\ntorchvision==0.20.1\n")
    install = ["uv", "pip", "install", "--python", str(python), "--constraint", str(constraints)]
    run_native([*install, "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cpu"],
               cwd=root, env=env, log=log)
    run_native([*install, "-e", str(libero), "-r", str(libero / "requirements.txt")],
               cwd=root, env=env, log=log)
    run_native([*install, "robosuite==1.4.1", "mujoco==2.3.7", "loguru==0.7.3",
                "requests==2.32.5", "scipy==1.13.1", "pillow==11.3.0"], cwd=root, env=env, log=log)
    config = root / "libero-config"
    config.mkdir()
    (config / "config.yaml").touch()
    env.update(LIBERO_CONFIG_PATH=str(config), MUJOCO_GL="osmesa",
               PYOPENGL_PLATFORM="osmesa", PYTHONPATH=f"{repo}{os.pathsep}{libero}")
    run_native([str(python), "-c", "from libero.libero import set_libero_default_path; "
                "set_libero_default_path()"], cwd=root, env=env, log=log)
    run_native([str(python), "-c", "import torch, torchvision; "
                "assert torch.__version__.split('+')[0] == '2.5.1'; "
                "assert torchvision.__version__.split('+')[0] == '0.20.1'; "
                "assert torch.version.cuda is None; "
                "print('Verified CPU-only simulator Torch/torchvision runtime')"],
               cwd=root, env=env, log=log)
    run_native(["uv", "pip", "freeze", "--python", str(python)],
               cwd=root, env=env, log=root / "simulation-environment.log")
    return python


def _install_simulation_dependencies(root: Path, env: dict[str, str], log: Path) -> None:
    from ctypes.util import find_library

    if find_library("OSMesa") and all(shutil.which(tool) for tool in ("cmake", "c++", "make")):
        return
    if not Path("/.dockerenv").exists() and not env.get("KUBERNETES_SERVICE_HOST"):
        raise RuntimeError("Install libosmesa6, cmake and build-essential in the worker image before policy evaluation")
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    run_native([*prefix, "apt-get", "update"], cwd=root, env=env, log=log)
    run_native([*prefix, "apt-get", "install", "-y", "libosmesa6", "libgl1", "libglib2.0-0",
                "cmake", "build-essential"],
               cwd=root, env=env, log=log)
