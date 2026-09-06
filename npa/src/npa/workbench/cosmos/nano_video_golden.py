"""Generate and verify a real 30-second clip using a prestaged BF16 checkpoint."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import secrets
from typing import Any

from npa.workbench.cosmos.nano_video import (
    CHUNK_FRAMES,
    DEFAULT_PROMPT,
    NanoVideoError,
    artifact,
    run_rollout,
    validate_video,
    write_json,
)
from npa.workbench.cosmos.nano_video_server import NanoVideoRuntime


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NanoVideoError(message)


async def _generate(output: Path) -> dict[str, Any]:
    # The runtime validates READY.json before launching the GPU process. The
    # checkpoint mount is read-only and this command never stages/downloads it.
    runtime = NanoVideoRuntime()
    try:
        await runtime.start()
        await runtime.check_health()
        generated = output / "generation"
        report = await asyncio.to_thread(
            run_rollout,
            endpoint=runtime.endpoint,
            output_dir=generated,
            prompt=DEFAULT_PROMPT,
            seed=17,
            replica_id=runtime.replica_id,
        )
        _require(report["status"] == "succeeded", "rollout did not succeed")
        _require(
            report["pipeline"] == "Cosmos3OmniDiffusersPipeline"
            and report["dtype"] == "bfloat16"
            and type(report["tensor_parallel_size"]) is int
            and report["tensor_parallel_size"] == 1
            and report["guardrails"] is False,
            "rollout differs from the BF16 TP=1 diffusion contract",
        )
        _require(
            [chunk["requested_frames"] for chunk in report["chunks"]]
            == list(CHUNK_FRAMES),
            "rollout has incomplete or incorrect chunks",
        )
        for index, chunk in enumerate(report["chunks"], 1):
            _require(chunk["status"] == "succeeded", "a diffusion chunk failed")
            _require(
                all(
                    _positive(chunk[key])
                    for key in ("wall_seconds", "inference_seconds", "peak_memory_mb")
                ),
                "chunk lacks positive finite latency or memory measurements",
            )
            validate_video(generated / f"chunk-{index}.mp4", CHUNK_FRAMES[index - 1])
        _require(
            _positive(report["device_peak_used_mib"])
            and _positive(report["total_wall_seconds"]),
            "rollout lacks positive finite device memory or wall time",
        )
        result = {
            "validation": validate_video(generated / "video-30s.mp4", 720),
            "report": artifact(generated / "report.json"),
            "video": artifact(generated / "video-30s.mp4"),
        }
        _require(
            result["report"]["bytes"] > 0 and result["video"]["bytes"] > 0,
            "golden artifacts are empty",
        )
        return result
    finally:
        runtime.close()
        if runtime.process is not None:
            await asyncio.to_thread(runtime.process.wait)


def main() -> int:
    os.umask(0o077)
    root = Path(
        os.environ.get(
            "NPA_COSMOS3_GOLDEN_OUTPUT_ROOT", "/tmp/cosmos3-nano-video-golden"
        )
    )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = root / secrets.token_hex(12)
    result: dict[str, Any] = {"status": "running", "run_directory": str(output)}
    write_json(root / "result.json", result)
    os.environ.setdefault("NPA_COSMOS3_VIDEO_TOKEN", secrets.token_hex(32))
    os.environ["NPA_COSMOS3_VIDEO_OUTPUT_ROOT"] = str(output / "runtime")
    try:
        result.update(asyncio.run(_generate(output)), status="succeeded")
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        write_json(root / "result.json", result)
        raise
    write_json(root / "result.json", result)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
