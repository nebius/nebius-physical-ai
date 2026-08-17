"""Build-time proof for the payload-free SONIC MuJoCo image."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
from importlib import metadata
from pathlib import Path


def main() -> int:
    lock = Path("/opt/npa/manifest/mujoco-requirements.lock")
    expected = os.environ.get("NPA_SONIC_MUJOCO_CLOSURE_SHA256", "")
    if hashlib.sha256(lock.read_bytes()).hexdigest() != expected:
        raise SystemExit("FATAL: SONIC MuJoCo closure checksum drift")
    for module in ("gear_sonic", "torch", "mujoco", "boto3", "yaml"):
        importlib.import_module(module)
    for package, wanted in {"torch": "2.9.0", "mujoco": "3.11.0"}.items():
        found = metadata.version(package)
        if found != wanted:
            raise SystemExit(f"FATAL: {package} expected {wanted}, found {found}")

    weight_suffixes = {".onnx", ".pt", ".pth", ".safetensors", ".ckpt"}
    for path in Path("/opt/sonic").rglob("*"):
        if (
            path.is_file()
            and path.stat().st_size > 1024 * 1024
            and path.suffix.lower() in weight_suffixes
        ):
            raise SystemExit(f"FATAL: weight payload baked into image: {path}")
    for name in ("isaacsim", "isaaclab"):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
        else:
            raise SystemExit(f"FATAL: restricted package baked into image: {name}")

    env = os.environ.copy()
    env["ACCEPT_EULA"] = ""
    env.pop("OMNI_KIT_ACCEPT_EULA", None)
    env.pop("ISAACSIM_ACCEPT_EULA", None)
    refusal = subprocess.run(
        ["/opt/npa/bin/isaac-bootstrap", "ensure"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if refusal.returncode != 78 or "ACCEPT_EULA=Y" not in refusal.stderr:
        raise SystemExit(
            "FATAL: Isaac runtime fetch did not refuse without caller acceptance "
            f"(rc={refusal.returncode})"
        )
    print("NPA_SONIC_MUJOCO_OSS_BUILD_OK no Isaac/no weights/no vendor container")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
