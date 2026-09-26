"""Create the pinned CPU LIBERO environment separately from the CUDA trainer."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def _run(command, root, env, log):
    subprocess.run(command, cwd=root, env=env, stdout=log, stderr=log, check=True)


def _checkout(root, env, log):
    revision = json.loads(Path(__file__).with_name("sources.json").read_text())[
        "libero"
    ]
    for command in (
        ["git", "init", "LIBERO"],
        [
            "git",
            "-C",
            "LIBERO",
            "remote",
            "add",
            "origin",
            "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        ],
        ["git", "-C", "LIBERO", "fetch", "--depth=1", "origin", revision],
        ["git", "-C", "LIBERO", "checkout", "--detach", "FETCH_HEAD"],
    ):
        _run(command, root, env, log)


def _install(root, env, log):
    python = str(root / "libenv/bin/python")
    _run(["uv", "venv", "--python", "3.10.21", "libenv"], root, env, log)
    install = ["uv", "pip", "install", "--python", python]
    _run(
        [
            *install,
            "torch==2.5.1+cpu",
            "torchvision==0.20.1+cpu",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        root,
        env,
        log,
    )
    lock = Path(__file__).with_name("simulation-requirements.txt").resolve()
    _run([*install, "-r", str(lock)], root, env, log)
    _run([*install, "--no-deps", "-e", "LIBERO"], root, env, log)


def _configure(root, env, log):
    config = root / "libero-config"
    config.mkdir()
    (config / "config.yaml").touch()
    env.update(
        LIBERO_CONFIG_PATH=str(config),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        PYTHONPATH=str(root / "LIBERO"),
    )
    code = """
import pathlib, subprocess, sys, torch, torchvision
assert torch.__version__ == '2.5.1+cpu'
assert torchvision.__version__ == '0.20.1+cpu'
assert torch.version.cuda is None
from libero.libero import set_libero_default_path
set_libero_default_path()
import robosuite
subprocess.run([sys.executable, str(pathlib.Path(robosuite.__file__).parent / 'scripts/setup_macros.py')], check=True)
from libero.libero import benchmark
assert benchmark.get_benchmark_dict()['libero_10']().n_tasks == 10
print('Pinned CPU simulator imports and all ten LIBERO tasks verified')
"""
    _run([str(root / "libenv/bin/python"), "-c", code], root, env, log)


def _main(args):
    root = (args.shared_root / "simulation").resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    env = dict(os.environ)
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH"):
        env.pop(key, None)
    with (root / "bootstrap.log").open("x") as log:
        _checkout(root, env, log)
        _install(root, env, log)
        _configure(root, env, log)
    with (root / "frozen-requirements.txt").open("x") as log:
        _run(
            ["uv", "pip", "freeze", "--python", str(root / "libenv/bin/python")],
            root,
            env,
            log,
        )
    print("Pinned LIBERO simulator ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    os.umask(0o077)
    _main(parser.parse_args())
