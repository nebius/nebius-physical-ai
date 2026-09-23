"""Runtime-fetch the pinned CUDA assembler needed by B300 compiled inference."""

import hashlib
from pathlib import Path
import urllib.request
import zipfile

VERSION = "12.9.86"
WHEEL_SHA256 = "5d6a0d32fdc7ea39917c20065614ae93add6f577d840233237ff08e9a38f58f0"
PTXAS_SHA256 = "983b0e9283855979f42cebfd80d43f9b6e786eb84f03f7570bf941c4d3a3c461"
WHEEL_BYTES = 40_546_229
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/25/48/"
    "b54a06168a2190572a312bfe4ce443687773eb61367ced31e064953dd2f7/"
    "nvidia_cuda_nvcc_cu12-12.9.86-py3-none-"
    "manylinux2010_x86_64.manylinux_2_12_x86_64.whl"
)
PTXAS_MEMBER = "nvidia/cuda_nvcc/bin/ptxas"
LICENSE_MEMBER = "nvidia_cuda_nvcc_cu12-12.9.86.dist-info/licenses/License.txt"


def prepare_b300_compiler(directory: Path) -> tuple[dict[str, str], dict]:
    """Fetch and verify one assembler in the inference run's private directory.

    Args:
        directory: Fresh child of the caller's private temporary run directory.
    Returns:
        Vendor-process environment overrides and non-secret compiler provenance.
    Raises:
        ValueError: Downloaded bytes do not match either immutable hash.
        OSError: Downloading or private local staging fails.
    """
    directory.mkdir(mode=0o700)
    wheel = directory / "compiler.whl"
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(WHEEL_URL, timeout=60) as response:
        with wheel.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > WHEEL_BYTES:
                    raise ValueError("CUDA compiler wheel exceeds its pinned size")
                digest.update(chunk)
                stream.write(chunk)
    if size != WHEEL_BYTES or digest.hexdigest() != WHEEL_SHA256:
        raise ValueError("CUDA compiler wheel content verification failed")
    with zipfile.ZipFile(wheel) as archive:
        # Select exact members; never extract archive-provided filesystem paths.
        binary = archive.read(PTXAS_MEMBER)
        license_text = archive.read(LICENSE_MEMBER)
    if hashlib.sha256(binary).hexdigest() != PTXAS_SHA256:
        raise ValueError("CUDA assembler content verification failed")
    executable = directory / "ptxas"
    executable.write_bytes(binary)
    executable.chmod(0o700)
    (directory / "License.txt").write_bytes(license_text)
    wheel.unlink()
    return (
        # Triton 3.3 derives this name from the literal 'ptxas-blackwell'.
        {"TRITON_PTXAS-BLACKWELL_PATH": str(executable)},
        {
            "component": "nvidia-cuda-nvcc-cu12",
            "version": VERSION,
            "wheel_sha256": WHEEL_SHA256,
            "ptxas_sha256": PTXAS_SHA256,
            "license": "NVIDIA CUDA Toolkit EULA",
            "runtime_fetch": True,
        },
    )
