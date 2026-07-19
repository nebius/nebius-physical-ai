"""Standalone Voxel51 FiftyOne functional smoke checks.

This script creates a tiny image dataset, launches the FiftyOne App, checks
that the web port responds, and shuts the session down.

Run with:
    python -m npa.smoke.test_fiftyone_functional
"""

from __future__ import annotations
import logging

import atexit
import base64
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from npa.smoke._versions import supported_tool_version

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    root: Path
    dataset_name: str
    port: int
    address: str
    dataset: Any | None = None
    session: Any | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def check_fiftyone_version(state: SmokeState) -> CheckResult:
    try:
        expected = supported_tool_version("fiftyone", __file__)
        version = metadata.version("fiftyone")
    except Exception as exc:
        return CheckResult("check fiftyone version", False, _format_exception(exc))

    if version != expected:
        return CheckResult(
            "check fiftyone version",
            False,
            f"expected version: {expected}; found: {version}",
        )
    return CheckResult("check fiftyone version", True, f"version: {version}")


def check_create_dataset(state: SmokeState) -> CheckResult:
    try:
        import fiftyone as fo

        images_dir = state.root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        dataset = fo.Dataset(name=state.dataset_name, persistent=False)
        for idx in range(3):
            image_path = images_dir / f"sample-{idx}.png"
            image_path.write_bytes(_TINY_PNG)
            dataset.add_sample(fo.Sample(filepath=str(image_path)))
        state.dataset = dataset
        return CheckResult("create sample dataset", True, f"samples: {len(dataset)}")
    except Exception as exc:
        return CheckResult("create sample dataset", False, _format_exception(exc))


def _write_synthetic_lerobot_dataset(root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    images_dir = root / "images" / "observation.image"
    images_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        (images_dir / f"frame-{idx}.png").write_bytes(_TINY_PNG)

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "features": {
            "observation.image": {
                "dtype": "image",
                "shape": [1, 1, 3],
            }
        },
    }))

    data = pa.table({
        "observation.image": pa.array(["frame-0.png", "frame-1.png"]),
        "episode_index": pa.array([0, 0], type=pa.int64()),
        "frame_index": pa.array([0, 1], type=pa.int64()),
        "timestamp": pa.array([0.0, 0.05], type=pa.float32()),
        "task_success": pa.array([True, True], type=pa.bool_()),
        "task_index": pa.array([0, 0], type=pa.int64()),
    })
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, data_path)

    tasks = pa.table({
        "task_index": pa.array([0], type=pa.int64()),
        "task": pa.array(["synthetic smoke task"]),
    })
    pq.write_table(tasks, meta_dir / "tasks.parquet")


def check_lerobot_import(state: SmokeState) -> CheckResult:
    try:
        import fiftyone as fo

        from npa.fiftyone_lerobot import import_lerobot_dataset

        source = state.root / "synthetic_lerobot"
        _write_synthetic_lerobot_dataset(source)
        result = import_lerobot_dataset(
            f"{state.dataset_name}-lerobot",
            str(source),
            state.root / "fiftyone-datasets",
        )
        dataset = fo.load_dataset(result["name"])
        first = dataset.first()
        try:
            if len(dataset) != 2:
                return CheckResult("import synthetic LeRobotDataset", False, f"samples: {len(dataset)}")
            for field in ("episode_index", "frame_index", "timestamp", "task_success"):
                if field not in dataset.get_field_schema():
                    return CheckResult("import synthetic LeRobotDataset", False, f"missing field: {field}")
            if first is None or first["episode_index"] != 0 or first["frame_index"] != 0:
                return CheckResult("import synthetic LeRobotDataset", False, "unexpected first sample metadata")
        finally:
            dataset.delete()
        return CheckResult(
            "import synthetic LeRobotDataset",
            True,
            f"samples: {result['samples']}; fields: {','.join(result['metadata_fields'])}",
        )
    except Exception as exc:
        return CheckResult("import synthetic LeRobotDataset", False, _format_exception(exc))


def check_launch_app(state: SmokeState) -> CheckResult:
    if state.dataset is None:
        return CheckResult("launch app", False, "skipped because dataset creation failed")

    try:
        import fiftyone as fo

        state.session = fo.launch_app(
            state.dataset,
            address=state.address,
            port=state.port,
            remote=True,
            auto=False,
        )
        url = f"http://{state.address}:{state.port}"
        deadline = time.monotonic() + 30
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= response.status < 500:
                        return CheckResult("launch app", True, f"url: {url}; status: {response.status}")
            except Exception as exc:
                last_error = _format_exception(exc)
            time.sleep(1)
        return CheckResult("launch app", False, last_error or f"no response from {url}")
    except Exception as exc:
        return CheckResult("launch app", False, _format_exception(exc))


def _cleanup(state: SmokeState) -> None:
    if state.session is not None:
        try:
            state.session.close()
        except Exception:
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    if state.dataset is not None:
        try:
            state.dataset.delete()
        except Exception:
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)
    shutil.rmtree(state.root, ignore_errors=True)


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="npa_fiftyone_functional_"))
    state = SmokeState(
        root=root,
        dataset_name=f"npa-fiftyone-smoke-{uuid.uuid4().hex[:8]}",
        port=int(os.environ.get("FIFTYONE_SMOKE_PORT", "5151")),
        address=os.environ.get("FIFTYONE_SMOKE_ADDRESS", "127.0.0.1"),
    )
    atexit.register(lambda: _cleanup(state))

    print(f"Temporary workspace: {root}")
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_fiftyone_version,
        check_create_dataset,
        check_lerobot_import,
        check_launch_app,
    ]
    results: list[CheckResult] = []

    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
    finally:
        _cleanup(state)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
