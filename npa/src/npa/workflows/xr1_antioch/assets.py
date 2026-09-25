"""Fetch immutable upstream XR1 source and verify the base policy checkpoint bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import requests

SOURCE_URL = "https://github.com/XiaomiRobotics/Xiaomi-Robotics-1"
SOURCE_REVISION = "0dd7aef8dc87296246aae812a1f59ccb708e5546"
MODEL_REPO = "XiaomiRobotics/Xiaomi-Robotics-1-5B"
MODEL_REVISION = "ee21d524b5c52ac961d941e1bc7d6d92836c3d5e"
MODEL_SHA256 = "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7"
PROCESSOR_REPO = "Qwen/Qwen3-VL-4B-Instruct"
PROCESSOR_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def prepare_processor_cache(root: Path) -> Path:
    """Prepare the exact offline processor snapshot required by upstream XR1.

    Args:
        root: Verified asset directory containing processor files and assets.json.
    Returns:
        HF_HOME directory for the native training or inference process.
    Raises:
        ValueError: Processor revision or bytes differ from the asset receipt.
    """
    metadata = json.loads((root / "assets.json").read_text())["processor"]
    if metadata["revision"] != PROCESSOR_REVISION:
        raise ValueError("Processor revision differs from the pinned upstream config")
    cache = root / "hf-cache" / "hub" / "models--Qwen--Qwen3-VL-4B-Instruct"
    snapshot = cache / "snapshots" / PROCESSOR_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, expected in metadata["files"].items():
        if Path(name).name != name:
            raise ValueError("Processor assets must have flat file names")
        source = root / "processor" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Processor file differs from its verified asset receipt")
        shutil.copyfile(source, snapshot / name)
    (cache / "refs").mkdir(exist_ok=True)
    (cache / "refs" / "main").write_text(PROCESSOR_REVISION)
    return root / "hf-cache"


def fetch_source(target: Path) -> Path:
    """Check out the reviewed upstream revision without changing another checkout.

    Args:
        target: New source checkout directory.
    Returns:
        Directory containing XR1's training package.
    Raises:
        FileExistsError: The destination already exists.
        subprocess.CalledProcessError: Git cannot retrieve the immutable revision.
        ValueError: Git reports a different source revision.
    """
    if target.exists():
        raise FileExistsError(target)
    subprocess.run(
        ["git", "clone", "--no-checkout", SOURCE_URL, str(target)], check=True
    )
    subprocess.run(
        ["git", "checkout", "--detach", SOURCE_REVISION], cwd=target, check=True
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=target, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("XR1 source differs from the reviewed revision")
    return target / "xr1"


def fetch_checkpoint(target: Path) -> dict:
    """Download and hash the exact upstream checkpoint before any torch deserialization.

    Args:
        target: New local checkpoint file.
    Returns:
        Immutable model identity and verified byte count.
    Raises:
        FileExistsError: A checkpoint or unfinished transfer already exists.
        requests.RequestException: Download failed.
        ValueError: Bytes differ from the publisher's immutable LFS digest.
    """
    if target.exists() or target.with_suffix(".partial").exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/model_states.pt"
    )
    temporary, digest, size = target.with_suffix(".partial"), hashlib.sha256(), 0
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("xb") as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError(
            "XR1 checkpoint SHA-256 disagrees with the pinned upstream release"
        )
    temporary.rename(target)
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "sha256": MODEL_SHA256,
        "bytes": size,
    }


def fetch_processor(target: Path) -> dict:
    """Fetch pinned processor/config assets; the XR1 checkpoint already contains VLM weights.

    Args:
        target: New processor destination directory.
    Returns:
        Processor identity and hashes of the files used by training and inference.
    Raises:
        FileExistsError: The processor directory exists.
        requests.RequestException: A public processor asset cannot be fetched.
    """
    target.mkdir(parents=True, exist_ok=False)
    api = f"https://huggingface.co/api/models/{PROCESSOR_REPO}/revision/{PROCESSOR_REVISION}"
    response = requests.get(api, timeout=30)
    response.raise_for_status()
    hashes = {}
    for entry in response.json()["siblings"]:
        name = entry["rfilename"]
        if "/" in name or not name.endswith((".json", ".jinja", ".txt")):
            continue
        url = f"https://huggingface.co/{PROCESSOR_REPO}/resolve/{PROCESSOR_REVISION}/{name}"
        asset = requests.get(url, timeout=120)
        asset.raise_for_status()
        (target / name).write_bytes(asset.content)
        hashes[name] = hashlib.sha256(asset.content).hexdigest()
    return {"repo": PROCESSOR_REPO, "revision": PROCESSOR_REVISION, "files": hashes}
