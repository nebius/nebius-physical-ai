"""FiftyOne container functional smoke checks."""

from __future__ import annotations

import atexit
import base64
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


EXPECTED_FIFTYONE_VERSION = os.environ.get("FIFTYONE_VERSION", "1.15.0")
TINY_PNG = base64.b64decode(
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
        version = metadata.version("fiftyone")
    except Exception as exc:
        return CheckResult("check fiftyone version", False, _format_exception(exc))
    if version != EXPECTED_FIFTYONE_VERSION:
        return CheckResult(
            "check fiftyone version",
            False,
            f"expected version: {EXPECTED_FIFTYONE_VERSION}; found: {version}",
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
            image_path.write_bytes(TINY_PNG)
            dataset.add_sample(fo.Sample(filepath=str(image_path), tags=["smoke"]))
        state.dataset = dataset
        return CheckResult("create sample dataset", True, f"samples: {len(dataset)}")
    except Exception as exc:
        return CheckResult("create sample dataset", False, _format_exception(exc))


def check_query_dataset(state: SmokeState) -> CheckResult:
    if state.dataset is None:
        return CheckResult("query sample dataset", False, "skipped because dataset creation failed")
    try:
        count = len(state.dataset.match_tags("smoke"))
        first = state.dataset.first()
        if count != 3 or first is None:
            return CheckResult("query sample dataset", False, f"count={count}; first={first!r}")
        return CheckResult("query sample dataset", True, f"tagged_samples={count}; first_id={first.id}")
    except Exception as exc:
        return CheckResult("query sample dataset", False, _format_exception(exc))


def check_launch_app(state: SmokeState) -> CheckResult:
    if state.dataset is None:
        return CheckResult("launch and stop app server", False, "skipped because dataset creation failed")
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
        deadline = time.monotonic() + 45
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if 200 <= response.status < 500:
                        state.session.close()
                        state.session = None
                        return CheckResult(
                            "launch and stop app server",
                            True,
                            f"url={url}; status={response.status}",
                        )
            except Exception as exc:
                last_error = _format_exception(exc)
            time.sleep(1)
        return CheckResult("launch and stop app server", False, last_error or f"no response from {url}")
    except Exception as exc:
        return CheckResult("launch and stop app server", False, _format_exception(exc))


def _cleanup(state: SmokeState) -> None:
    if state.session is not None:
        try:
            state.session.close()
        except Exception:
            pass
    if state.dataset is not None:
        try:
            state.dataset.delete()
        except Exception:
            pass
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
        check_query_dataset,
        check_launch_app,
    ]
    results = []
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
