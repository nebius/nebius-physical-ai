"""Prepare Train dependencies separately from SkyPilot's management Ray."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def _create_runtime_root(root: Path) -> None:
    """Require a trusted parent and reject any previously populated runtime."""
    parent = root.parent.lstat()
    if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
            or parent.st_mode & 0o022):
        raise ValueError("Runtime parent must be an owned directory without shared write access")
    root.mkdir(mode=0o700)
    (root / "exports").mkdir(mode=0o700)


def _install_dependencies(environment: Path) -> str:
    """Keep the image's CUDA wheels while isolating application Ray dependencies."""
    subprocess.run([
        sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(environment),
    ], check=True)
    interpreter = str(environment / "bin/python")
    subprocess.run([
        sys.executable, "-m", "pip", "--python", interpreter, "install",
        "-r", str(Path(__file__).with_name("requirements.txt")),
    ], check=True)
    return interpreter


def _runtime_receipt(interpreter: str) -> dict:
    """Reject drift before a service can use the prepared environment."""
    inspection = subprocess.check_output([interpreter, "-c", """
import json, ray, torch, rerun, PIL
assert ray.__version__ == '2.58.0'
assert torch.__version__ == '2.13.0+cu130'
assert PIL.__version__ == '12.3.0'
assert torch.cuda.device_count() >= 1
print(json.dumps(dict(ray=ray.__version__, torch=torch.__version__,
    cuda=torch.version.cuda, rerun=rerun.__version__, pillow=PIL.__version__,
    gpus=torch.cuda.device_count())))
"""], text=True)
    receipt = json.loads(inspection)
    receipt["dependency_freeze"] = subprocess.check_output(
        [interpreter, "-m", "pip", "freeze"], text=True,
    ).splitlines()
    return receipt


def main() -> None:
    """Prepare a fresh private runtime using the image's CUDA wheels.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: The runtime parent permits untrusted writes.
        OSError: A runtime already exists or private files cannot be created.
        subprocess.CalledProcessError: Installation or runtime verification fails.
    """
    os.umask(0o077)
    root = Path("/opt/npa-ray-train")
    _create_runtime_root(root)
    interpreter = _install_dependencies(root / "env")
    receipt = _runtime_receipt(interpreter)
    (root / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
