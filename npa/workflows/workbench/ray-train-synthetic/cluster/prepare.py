"""Prepare Train dependencies separately from SkyPilot's management Ray."""

import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Retain image CUDA wheels and record runtime pins outside submitted source."""
    environment = "/tmp/ray-train-env"
    subprocess.run([
        sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", environment,
    ], check=True)
    interpreter = environment + "/bin/python"
    subprocess.run([
        sys.executable, "-m", "pip", "--python", interpreter, "install",
        "-r", str(Path(__file__).with_name("requirements.txt")),
    ], check=True)
    inspection = subprocess.check_output([interpreter, "-c", """
import json, ray, torch, rerun
assert ray.__version__ == '2.58.0'
assert torch.__version__ == '2.12.1+cu130'
assert torch.cuda.device_count() >= 1
print(json.dumps(dict(ray=ray.__version__, torch=torch.__version__,
    cuda=torch.version.cuda, rerun=rerun.__version__, gpus=torch.cuda.device_count())))
"""], text=True)
    receipt = json.loads(inspection)
    receipt["dependency_freeze"] = subprocess.check_output(
        [interpreter, "-m", "pip", "freeze"], text=True,
    ).splitlines()
    Path("/tmp/ray-train-preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
