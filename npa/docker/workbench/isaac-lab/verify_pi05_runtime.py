"""Build-time smoke for the pi0.5 Isaac client and review-artifact runtime."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import msgpack
import numpy as np
from websockets.sync.client import connect

from npa.workflows.sim2real.pi05_closed_loop import _write_review_artifacts
from npa.workflows.sim2real.pi05_closed_loop_entry import build_parser
from npa.workflows.sim2real.pi05_policy_client import _pack_array, _unpack_array


def _verify_wire_codec() -> None:
    signature = inspect.signature(connect)
    required = {"additional_headers", "compression", "max_size"}
    if not required.issubset(signature.parameters):
        raise RuntimeError("websockets.sync client lacks the pi0.5 transport API")
    source = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    payload = msgpack.packb({"wrist": source}, default=_pack_array)
    decoded = msgpack.unpackb(payload, object_hook=_unpack_array)["wrist"]
    if not np.array_equal(decoded, source):
        raise RuntimeError("msgpack changed pi0.5 image bytes")


def _synthetic_episode() -> dict:
    exterior = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.full((224, 224, 3), 127, dtype=np.uint8)
    return {
        "_exterior_frames": [exterior],
        "_wrist_frames": [wrist],
        "queries": [
            {
                "observation_time_s": 0.0,
                "inference": {"wire_input": {"capture_age_s": 0.0}},
            }
        ],
        "physics_trace": [
            {
                "object_position_m": [0.5, 0.18, 0.024],
                "object_speed_m_s": 0.0,
                "target_xy_distance_m": 0.0,
            }
        ],
    }


def main() -> None:
    _verify_wire_codec()
    if "--output-uri" not in build_parser()._option_string_actions:
        raise RuntimeError("pi0.5 closed-loop entrypoint parser is unavailable")
    with tempfile.TemporaryDirectory(prefix="npa-pi05-image-smoke-") as directory:
        artifacts = _write_review_artifacts(
            Path(directory), [_synthetic_episode()], "synthetic-build-smoke"
        )
        if artifacts["mp4"]["frames"] != 1 or not artifacts["rrd"]["rerun_verify"]:
            raise RuntimeError("pi0.5 review-artifact readback was incomplete")


if __name__ == "__main__":
    main()
