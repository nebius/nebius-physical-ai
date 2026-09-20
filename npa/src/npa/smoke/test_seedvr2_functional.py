"""Run real SeedVR2 restoration on a pinned physical-robot video excerpt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse

import httpx

from npa.workbench.seedvr2.runtime import _probe_video, restore
from npa.workbench.seedvr2.schemas import RestoreRequest


SOURCE_URL = (
    "https://huggingface.co/datasets/Hoshipu/RoboPro/resolve/"
    "90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/"
    "lerobot/roboreal_all_80tasks/videos/chunk-000/"
    "observation.images.cam_high/episode_000000.mp4"
)
SOURCE_SHA256 = "caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198"
INPUT_URI = "s3://seedvr2-golden/input.mp4"
OUTPUT_URI = "s3://seedvr2-golden/output/"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _LocalStorage:
    """Map the production storage boundary onto one private golden-eval directory."""

    def __init__(self, source: Path, root: Path) -> None:
        self.source = source
        self.root = root
        self.objects: dict[str, Path] = {}

    def read_bytes_with_etag(self, uri: str):
        path = self.objects.get(uri)
        if path is None:
            return None
        return path.read_bytes(), _sha256(path)

    def download_file(self, uri: str, local_path: str) -> str:
        source = self.source if uri == INPUT_URI else self.objects[uri]
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return str(target)

    def upload_file(self, local_file: str, uri: str) -> str:
        target = self.root / uri.rsplit("/", 1)[-1]
        shutil.copyfile(local_file, target)
        self.objects[uri] = target
        return uri


def _fetch_source(path: Path) -> None:
    if urlparse(SOURCE_URL).scheme != "https":
        raise RuntimeError("RoboPro golden-eval source must use HTTPS")
    # The URL is a fixed, hash-verified HTTPS fixture.
    with (
        httpx.stream("GET", SOURCE_URL, follow_redirects=True, timeout=120) as response,
        path.open("wb") as output,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes():
            output.write(chunk)
    if _sha256(path) != SOURCE_SHA256:
        raise RuntimeError("RoboPro golden-eval source hash mismatch")


def _degrade(source: Path, output: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "9",
        "-vf",
        "scale=320:240:flags=area",
        "-r",
        "50",
        "-c:v",
        "libx264",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)
    if _probe_video(output)["frames"] != 9:
        raise RuntimeError("golden-eval degradation did not retain nine frames")


def _require_h100() -> dict[str, str]:
    code = (
        "import json,torch; "
        "assert torch.cuda.is_available(); "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name':p.name,'capability':"
        "f'{p.major}.{p.minor}','torch':torch.__version__,"
        "'cuda':torch.version.cuda}))"
    )
    completed = subprocess.run(
        ["/opt/seedvr2-venv/bin/python", "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )
    inventory = json.loads(completed.stdout)
    if "H100" not in inventory["name"] or inventory["capability"] != "9.0":
        raise RuntimeError("SeedVR2 golden eval requires one H100 (sm_90)")
    return inventory


def main() -> None:
    """Execute and validate the smallest retained real-video restoration."""

    root = Path("/workspace/seedvr2-golden-eval")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(mode=0o700, parents=True)
    source = root / "source.mp4"
    degraded = root / "degraded.mp4"
    _fetch_source(source)
    _degrade(source, degraded)
    storage = _LocalStorage(degraded, root)
    result = restore(
        RestoreRequest(
            input_path=INPUT_URI,
            output_path=OUTPUT_URI,
            run_id="seedvr2-golden-eval",
            output_height=480,
            output_width=640,
            seed=666,
        ),
        storage_factory=lambda: storage,
    )
    restored = root / "restored.mp4"
    probe = _probe_video(restored)
    if probe["frames"] != 9 or (probe["width"], probe["height"]) != (640, 480):
        raise RuntimeError("real SeedVR2 golden output failed its media contract")
    artifact = {
        "solution": "seedvr2",
        "capability": "seedvr2_3b_video_restoration",
        "source_sha256": SOURCE_SHA256,
        "degraded_sha256": _sha256(degraded),
        "restored_sha256": _sha256(restored),
        "media": probe,
        "gpu": _require_h100(),
        "model": result["model"],
    }
    (root / "seedvr2.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
