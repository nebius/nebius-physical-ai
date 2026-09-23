"""Runtime-fetch the pinned CUDA assembler needed by B300 compiled inference."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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
        {
            "TRITON_PTXAS-BLACKWELL_PATH": str(executable),
            "TRITON_PTXAS_PATH": str(executable),
        },
        {
            "component": "nvidia-cuda-nvcc-cu12",
            "version": VERSION,
            "wheel_sha256": WHEEL_SHA256,
            "ptxas_sha256": PTXAS_SHA256,
            "license": "NVIDIA CUDA Toolkit EULA",
            "runtime_fetch": True,
        },
    )


_COMPILED_KERNEL_PREFLIGHT = """
import json, torch, triton
from torch._inductor import metrics
assert torch.__version__ == '2.8.0+cu129', torch.__version__
assert triton.__version__ == '3.4.0', triton.__version__
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (10, 3)
def kernel(x):
    return (x.sin() * x).sum(dim=-1)
x = torch.linspace(-1, 1, 8192, device='cuda').reshape(8, 1024)
compiled = torch.compile(kernel, fullgraph=True)
torch.testing.assert_close(compiled(x), kernel(x))
torch.cuda.synchronize()
assert metrics.generated_kernel_count > 0
print(json.dumps({'torch': torch.__version__, 'triton': triton.__version__,
                  'cuda': torch.version.cuda, 'compute_capability': '10.3',
                  'compiled_kernel_preflight': 'passed'}))
"""


def prepare_b300_runtime(
    directory: Path, *, base_python: str
) -> tuple[str, dict[str, str], dict]:
    """Prepare a private matched runtime and prove compilation before model fetch.

    The released vendor environment remains intact. Its pure-Python Flex-Pi
    dependencies are inherited; the native PyTorch/CUDA/Triton stack is replaced
    together using direct, hash-pinned official wheels for CPython 3.11/x86_64.
    """
    directory.mkdir(mode=0o700)
    compiler_env, provenance = prepare_b300_compiler(directory / "assembler")
    compiler_env.update(
        TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor-cache"),
        TRITON_CACHE_DIR=str(directory / "triton-cache"),
    )
    requirements = Path(__file__).with_name("b300-runtime-requirements.txt")
    lock = requirements.read_bytes()
    vendor_env = {**os.environ, **compiler_env}
    vendor_env.pop("NPA_FLEX_PI_TOKEN", None)

    def run(argv: list[str]) -> str:
        completed = subprocess.run(
            argv,
            env=vendor_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            tail = re.sub(r"(?:s3|https?)://\S+", "<uri-ref>", completed.stdout[-4096:])
            tail = re.sub(r"(?i)Bearer\s+\S+", "Bearer <redacted>", tail)
            tail = re.sub(
                r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{12,}\b",
                "<redacted>",
                tail,
            )
            raise ValueError(f"B300 private runtime preparation failed: {tail}")
        return completed.stdout

    run(
        [
            base_python,
            "-c",
            "import sys,platform; assert sys.version_info[:2] == (3,11); assert platform.system() == 'Linux' and platform.machine() == 'x86_64'",
        ]
    )
    runtime = directory / "vendor"
    print(
        "B300: preparing hash-pinned PyTorch/CUDA/Triton runtime",
        file=sys.stderr,
        flush=True,
    )
    run([base_python, "-m", "venv", "--system-site-packages", str(runtime)])
    python = str(runtime / "bin/python")
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--ignore-installed",
            "--require-hashes",
            "--no-cache-dir",
            "--requirement",
            str(requirements),
        ]
    )
    print(
        "B300: checking native compiled reduction before model download",
        file=sys.stderr,
        flush=True,
    )
    probe = json.loads(run([python, "-c", _COMPILED_KERNEL_PREFLIGHT]).splitlines()[-1])
    if probe.get("compiled_kernel_preflight") != "passed":
        raise ValueError("B300 compiled-kernel preflight did not pass")
    provenance["vendor_runtime"] = {
        **probe,
        "requirements_sha256": hashlib.sha256(lock).hexdigest(),
        "runtime_fetch": True,
    }
    return python, compiler_env, provenance
