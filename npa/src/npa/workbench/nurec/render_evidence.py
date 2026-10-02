"""Bind NuRec rendered media to its source scene and observed GPU telemetry."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time

EVIDENCE_FILENAME = "render-evidence.json"
EVIDENCE_SCHEMA = "npa.nurec.render-evidence.v1"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _media(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file()
    )


def _media_digest(paths: list[Path]) -> str:
    return hashlib.sha256(
        "\n".join(_sha256(path) for path in paths).encode()
    ).hexdigest()


def _gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    rows = []
    for values in csv.reader(result.stdout.splitlines()):
        if len(values) != 4:
            continue
        name, driver, utilization, memory = (value.strip() for value in values)
        if not re.fullmatch(r"[A-Za-z0-9 .()-]{1,100}", name) or not re.fullmatch(
            r"[0-9.]+", driver
        ):
            continue
        try:
            rows.append(
                {
                    "gpu": name,
                    "driver": driver,
                    "utilization_percent": int(utilization),
                    "memory_mib": int(memory),
                }
            )
        except ValueError:
            continue
    return rows


class RenderTelemetry:
    """Observe allocated-device activity while the native renderer is running.

    Args:
        None.

    Returns:
        A context manager retaining aggregate device telemetry without identifiers.

    Raises:
        None.
    """

    def __init__(self) -> None:
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.elapsed_seconds = 0.0

    def __enter__(self):
        self._started = time.monotonic()
        if _gpu_snapshot():
            self._thread = threading.Thread(target=self._observe, daemon=True)
            self._thread.start()
        return self

    def _observe(self) -> None:
        while not self._stop.wait(1):
            self.samples.extend(_gpu_snapshot())

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.elapsed_seconds = round(time.monotonic() - self._started, 3)

    def summary(self) -> dict:
        """Return device observations without host, account, or GPU identifiers.

        Args:
            None.

        Returns:
            Aggregate hardware observations; activity is device-wide, not process attestation.

        Raises:
            None.
        """
        return {
            "scope": "Allocated-device telemetry during NRE execution; not process-level attestation.",
            "sample_count": len(self.samples),
            "gpu_models": sorted({row["gpu"] for row in self.samples}),
            "driver_versions": sorted({row["driver"] for row in self.samples}),
            "peak_utilization_percent": max(
                (row["utilization_percent"] for row in self.samples), default=0
            ),
            "peak_memory_mib": max(
                (row["memory_mib"] for row in self.samples), default=0
            ),
            "elapsed_seconds": self.elapsed_seconds,
        }


def write_render_evidence(
    root: Path,
    artifact: Path,
    telemetry: RenderTelemetry,
    *,
    renderer: str,
    novel_view: bool,
) -> dict:
    """Record successful native rendering and bind its source and output bytes.
    Args:
        root: Render output directory, containing this invocation's frames only.
        artifact: Trained USDZ consumed by the renderer.
        telemetry: Observations collected during native execution.
        renderer: Selected native NRE renderer.
        novel_view: Whether the invocation requested novel camera views.
    Returns:
        A JSON-compatible rendering record with no infrastructure identifiers.

    Raises:
        ValueError: No render frames exist.
        OSError: Artifact bytes cannot be read or the record cannot be written.
    """
    paths = _media(root)
    if not paths:
        raise ValueError("render evidence requires actual output frames")
    record = {
        "schema": EVIDENCE_SCHEMA,
        "backend": "NVIDIA NRE",
        "renderer": renderer,
        "novel_view": novel_view,
        "artifact_sha256": _sha256(artifact),
        "frame_count": len(paths),
        "frames_sha256": _media_digest(paths),
        "telemetry": telemetry.summary(),
    }
    (root / EVIDENCE_FILENAME).write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    return record


def verified_render_summary(root: Path) -> dict:
    """Validate media bindings and return an allowlisted portable presentation.

    Args:
        root: Materialized canonical run containing reconstruction and novel_views.

    Returns:
        Sanitized render evidence, or an explicit legacy/unverified status.

    Raises:
        ValueError: Present evidence is invalid or no longer matches the media.
        OSError: Evidence or bound artifacts cannot be read.
    """
    path = root / "novel_views" / EVIDENCE_FILENAME
    if not path.exists():
        return {"render evidence": "Unavailable for this run"}
    record = _read_record(path)
    paths = _media(root / "novel_views")
    if record.get("schema") != EVIDENCE_SCHEMA or record.get("frame_count") != len(
        paths
    ):
        raise ValueError("render evidence schema or frame count mismatch")
    if record.get("frames_sha256") != _media_digest(paths):
        raise ValueError("render evidence frame digest mismatch")
    if record.get("artifact_sha256") != _sha256(root / "reconstruction" / "last.usdz"):
        raise ValueError("render evidence scene digest mismatch")
    return _portable_summary(record)


def _read_record(path: Path) -> dict:
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("backend") != "NVIDIA NRE":
        raise ValueError("invalid render evidence backend")
    if type(record.get("novel_view")) is not bool:
        raise ValueError("invalid render evidence view type")
    if type(record.get("frame_count")) is not int or record["frame_count"] < 1:
        raise ValueError("invalid render evidence frame count")
    telemetry = record.get("telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("invalid render telemetry")
    models = telemetry.get("gpu_models")
    if not isinstance(models, list) or not all(
        isinstance(model, str) for model in models
    ):
        raise ValueError("invalid render telemetry GPU models")
    return record


def _portable_summary(record: dict) -> dict:
    telemetry = record.get("telemetry", {})
    models = telemetry.get("gpu_models", [])
    from npa.workbench.nurec.nurec import has_rt_cores

    gpu = (
        "RTX PRO 6000"
        if any("6000" in str(value) and has_rt_cores(str(value)) for value in models)
        else "RT-capable GPU"
    )
    active = telemetry.get("peak_utilization_percent", 0)
    memory = telemetry.get("peak_memory_mib", 0)
    samples = telemetry.get("sample_count", 0)
    if (
        any(type(value) is not int or value < 0 for value in (active, memory, samples))
        or active > 100
    ):
        raise ValueError("invalid render telemetry measurements")
    observed = (
        samples > 0
        and active > 0
        and memory > 0
        and any(has_rt_cores(str(value)) for value in models)
    )
    return {
        "render evidence": "Media hashes verified; GPU activity observed"
        if observed
        else "Media hashes verified; GPU activity unverified",
        "render GPU": gpu if observed else "Unverified",
        "view type": "Novel views" if record["novel_view"] else "Training views",
        "observation scope": "Allocated-device activity during NRE execution; not process-level attestation.",
        "GPU telemetry samples": samples,
        "peak GPU utilization (%)": active,
        "peak GPU memory (MiB)": memory,
        "scene SHA-256": record["artifact_sha256"],
        "rendered frames SHA-256": record["frames_sha256"],
    }
