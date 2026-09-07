"""Build exact PyAV/NumPy/SciPy sources without bundled wheel runtimes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from native_build_support import FFMPEG, ROOT, WHEELS, record, run, unpack


def wheel(env: Path, name: str, version: str, lane: str) -> Path:
    source = unpack(name, version)
    target = WHEELS / lane
    target.mkdir(parents=True, exist_ok=True)
    command = [
        str(env / "bin/python"),
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-index",
        "--wheel-dir",
        str(target),
        str(source),
    ]
    if name in ("numpy", "scipy"):
        command += [
            f"-Csetup-args=-Dblas={'blas-netlib' if name == 'scipy' else 'blas'}",
            "-Csetup-args=-Dlapack=lapack",
            "-Csetup-args=--wrap-mode=nodownload",
        ]
    if name == "numpy":
        # Do not accidentally specialize the distributable to the build host.
        command += [
            "-Csetup-args=-Dcpu-baseline=min",
            "-Csetup-args=-Dcpu-dispatch=max",
        ]
    run(*command)
    (result,) = target.glob(f"{name}-{version}-*.whl")
    run(
        str(env / "bin/python"),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        str(result),
    )
    return result


def build_lane(lane: str) -> None:
    numpy_version, scipy_version = {
        "converter": ("1.26.4", "1.15.2"),
        "reader": ("2.5.3", "1.18.1"),
    }[lane]
    os.environ["PKG_CONFIG_PATH"] = str(FFMPEG / "lib/pkgconfig")
    os.environ["LD_LIBRARY_PATH"] = str(FFMPEG / "lib")
    env = ROOT / f"{lane}-build"
    os.environ["PATH"] = f"{env}/bin:" + os.environ["PATH"]
    run("python", "-m", "venv", str(env))
    run(
        str(env / "bin/python"),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--require-hashes",
        "-r",
        f"/build/native-{lane}-build.lock",
    )
    wheel(env, "numpy", numpy_version, lane)
    wheel(env, "scipy", scipy_version, lane)
    if lane == "reader":
        wheel(env, "av", "17.1.0", lane)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("converter", "reader"))
    component = parser.parse_args().component
    ROOT.mkdir(parents=True, exist_ok=True)
    build_lane(component)
    record(component)
