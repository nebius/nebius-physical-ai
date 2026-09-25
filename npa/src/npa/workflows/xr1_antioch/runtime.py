"""Build and prove the upstream XR1 CUDA runtime on the selected physical GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

SOURCE_REVISION = "0dd7aef8dc87296246aae812a1f59ccb708e5546"


def _system_dependencies() -> None:
    if (
        all(shutil.which(name) for name in ("git", "cc"))
        and Path("/usr/include/libaio.h").is_file()
    ):
        return
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    subprocess.run([*prefix, "apt-get", "update"], check=True)
    subprocess.run(
        [
            *prefix,
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "git",
            "build-essential",
            "libaio-dev",
        ],
        check=True,
    )


def _install(root: Path, wheel_directory: Path | None = None) -> Path:
    _system_dependencies()
    source = root / "source"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "https://github.com/XiaomiRobotics/Xiaomi-Robotics-1",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", SOURCE_REVISION], cwd=source, check=True
    )
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(
        pip
        + ["install", "packaging", "ninja", "psutil", "wheel", "boto3", "av==17.1.0"],
        check=True,
    )
    subprocess.run(pip + ["install", "-e", str(source / "xr1")], check=True)
    wheels = wheel_directory or root / "wheels"
    if wheel_directory is None:
        wheels.mkdir()
        env = dict(
            os.environ, FLASH_ATTN_CUDA_ARCHS="120", FLASH_ATTENTION_FORCE_BUILD="TRUE"
        )
        subprocess.run(
            pip
            + [
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "flash-attn==2.8.3",
                "-w",
                str(wheels),
            ],
            env=env,
            check=True,
        )
    subprocess.run(
        pip + ["install", "--no-deps", *map(str, wheels.glob("*.whl"))], check=True
    )
    return wheels


def _cuda_proof(*, require_optimizer: bool = True) -> dict:
    import torch
    from flash_attn import flash_attn_func

    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError(
            "This runtime artifact is compiled for the requested RTX PRO 6000 SM120"
        )
    torch.manual_seed(0)
    tensors = [
        torch.randn(
            2, 256, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    output = flash_attn_func(*tensors, causal=True)
    output.float().square().mean().backward()
    if not all(torch.isfinite(value.grad).all().item() for value in tensors):
        raise ValueError("FlashAttention backward produced non-finite gradients")
    if require_optimizer:
        _optimizer_proof()
    torch.cuda.synchronize()
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": [12, 0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attention_forward_backward": True,
        "fused_adam_update": True if require_optimizer else None,
    }


def _optimizer_proof() -> None:
    import torch
    from deepspeed.ops.adam import FusedAdam

    parameter = torch.nn.Parameter(torch.ones(32, device="cuda"))
    optimizer = FusedAdam([parameter], lr=1e-3)
    parameter.square().sum().backward()
    optimizer.step()
    if not torch.all(parameter.detach() < 1).item():
        raise ValueError(
            "The native FusedAdam optimizer did not update CUDA parameters"
        )


def _publish(root: Path, destination: str) -> None:
    import boto3

    location = urlsplit(destination)
    if location.scheme != "s3" or not location.netloc or not location.path.strip("/"):
        raise ValueError("Runtime artifacts require a run-scoped S3 destination")
    client = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    prefix = location.path.strip("/")
    paths = [
        root / "runtime-proof.json",
        root / "pip-freeze.txt",
        *(root / "wheels").glob("*.whl"),
    ]
    manifest = {}
    for path in paths:
        name = str(path.relative_to(root))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        client.upload_file(
            str(path),
            location.netloc,
            f"{prefix}/{name}",
            ExtraArgs={"Metadata": {"sha256": digest}},
        )
        response = client.get_object(Bucket=location.netloc, Key=f"{prefix}/{name}")
        verified = hashlib.sha256()
        with response["Body"] as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                verified.update(block)
        if verified.hexdigest() != digest:
            raise ValueError("Published runtime failed full S3 readback")
        manifest[name] = {"sha256": digest, "bytes": path.stat().st_size}
    client.put_object(
        Bucket=location.netloc,
        Key=f"{prefix}/manifest.json",
        Body=json.dumps(manifest).encode(),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--work-path", type=Path, required=True)
    args = parser.parse_args()
    args.work_path.mkdir(parents=True, exist_ok=False, mode=0o700)
    _install(args.work_path)
    proof = {"source_revision": SOURCE_REVISION, **_cuda_proof()}
    (args.work_path / "runtime-proof.json").write_text(json.dumps(proof, indent=2))
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.work_path / "pip-freeze.txt").write_text(freeze)
    _publish(args.work_path, args.output_path)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    _main()
