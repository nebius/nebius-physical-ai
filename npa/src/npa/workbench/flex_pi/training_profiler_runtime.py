"""Prepare a private CUDA 13 profiler while retaining the training compute stack."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from npa.workbench.flex_pi.training_artifacts import sha256_file

CUPTI_SHA256 = "e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e"


def prepare_profiler_environment(capability, mode, directory, environment):
    """Select native CUPTI 13 for a B300 diagnostic profile only.

    Args:
        capability: Verified four-rank placement and GPU receipts.
        mode: Requested training phase.
        directory: Fresh private profiler dependency directory.
        environment: Vendor-process environment, copied before modification.
    Returns:
        Vendor environment and optional immutable profiler provenance.
    Raises:
        RuntimeError: Dependency installation or binary verification fails.
    """
    env = environment.copy()
    env.pop("NPA_FLEX_PI_PROFILE_BACKEND", None)
    env.pop("NPA_FLEX_PI_CUPTI_LIBRARY", None)
    peers = capability.get("rank_placement", [])
    selected = mode in {"profile", "profile-resume"} and any(
        "B300" in peer["gpu"].upper() for peer in peers
    )
    if not selected:
        return env, None
    if len(peers) != 4 or not all("B300" in peer["gpu"].upper() for peer in peers):
        raise RuntimeError("B300 profiling requires four verified B300 ranks")
    directory = Path(directory)
    directory.mkdir(mode=0o700)
    requirements = Path(__file__).with_name("training-profiler-requirements.txt")
    with (directory / "install.log").open("wb") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                "--require-hashes",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--target",
                str(directory),
                "--requirement",
                str(requirements),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    library = directory / "nvidia/cu13/lib/libcupti.so.13"
    if completed.returncode or not library.is_file():
        raise RuntimeError(
            "private B300 profiler installation failed; inspect install.log"
        )
    if sha256_file(library) != CUPTI_SHA256:
        raise RuntimeError("private B300 CUPTI binary differs from its pinned hash")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(directory), env.get("PYTHONPATH")])
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(library.parent), env.get("LD_LIBRARY_PATH")])
    )
    env["NPA_FLEX_PI_PROFILE_BACKEND"] = "cupti13"
    env["NPA_FLEX_PI_CUPTI_LIBRARY"] = str(library)
    return env, {
        "backend": "cupti13",
        "cupti_version": "13.0.85",
        "bindings_version": "13.0.0",
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "cupti_sha256": CUPTI_SHA256,
        "runtime_fetch": True,
        "compute_stack_replaced": False,
        "licenses": [
            "NVIDIA CUDA Toolkit EULA",
            "NVIDIA CUPTI Python Software License Agreement",
            "NVIDIA CUDA Python Software License",
            "Apache-2.0 (cuda-pathfinder)",
        ],
        "package_license_files_retained": True,
    }
