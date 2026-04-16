"""Auto-record torch CUDA memory stats for benchmarked Python processes."""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path


def _memory_output_dir() -> Path | None:
    raw = os.environ.get("LEROBOT_BENCHMARK_TORCH_MEMORY_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_payload(payload: dict[str, object]) -> None:
    output_dir = _memory_output_dir()
    if output_dir is None:
        return
    target = output_dir / f"{os.getpid()}.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(target)


def _collect() -> None:
    payload: dict[str, object] = {
        "pid": os.getpid(),
        "argv0": sys.argv[0] if sys.argv else "",
        "cuda_available": False,
        "cuda_initialized": False,
        "device_count": 0,
        "device_names": [],
        "max_memory_allocated_bytes": 0,
        "max_memory_reserved_bytes": 0,
    }
    try:
        import torch

        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_initialized"] = torch.cuda.is_initialized()
        if not torch.cuda.is_available():
            _write_payload(payload)
            return

        device_count = torch.cuda.device_count()
        payload["device_count"] = device_count
        device_names: list[str] = []
        allocated_bytes: list[int] = []
        reserved_bytes: list[int] = []

        for index in range(device_count):
            try:
                torch.cuda.synchronize(index)
            except Exception:
                pass
            try:
                device_names.append(torch.cuda.get_device_name(index))
                allocated_bytes.append(int(torch.cuda.max_memory_allocated(index)))
                reserved_bytes.append(int(torch.cuda.max_memory_reserved(index)))
            except Exception:
                continue

        payload["device_names"] = device_names
        payload["max_memory_allocated_bytes"] = max(allocated_bytes or [0])
        payload["max_memory_reserved_bytes"] = max(reserved_bytes or [0])
    except Exception as exc:  # pragma: no cover - best effort telemetry hook
        payload["error"] = repr(exc)

    _write_payload(payload)


atexit.register(_collect)
