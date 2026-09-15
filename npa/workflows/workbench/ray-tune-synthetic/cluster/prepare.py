"""Prepare an isolated Ray Tune environment on the SkyPilot host."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def _create_runtime(root: Path) -> None:
    """Create a fresh private runtime below a trusted host directory."""
    parent = root.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o022
    ):
        raise ValueError("runtime parent must not permit shared writes")
    root.mkdir(mode=0o700)
    (root / "exports").mkdir(mode=0o700)


def main() -> None:
    """Install and verify the application Ray environment.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: The runtime parent is unsafe.
        OSError: The fresh runtime cannot be created.
        subprocess.CalledProcessError: Installation or verification fails.
    """
    os.umask(0o077)
    root = Path.home() / ".npa-ray-tune"
    _create_runtime(root)
    environment = root / "env"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
    interpreter = str(environment / "bin/python")
    requirements = str(Path(__file__).with_name("requirements.txt"))
    subprocess.run([interpreter, "-m", "pip", "install", "-r", requirements], check=True)
    output = subprocess.check_output([interpreter, "-c", "import json, ray; print(json.dumps({'ray': ray.__version__}))"], text=True)
    receipt = json.loads(output)
    if receipt != {"ray": "2.58.0"}:
        raise RuntimeError("prepared environment did not retain Ray 2.58.0")
    (root / "preparation.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
