"""Execute XR1 camera-and-state actions in the same physical task used for demonstration collection."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import socket
import struct

import numpy as np

from .cameras import _Cameras
from .collection import _outcome
from .dataset import INSTRUCTION
from .scene import _Cell


def _receive(connection, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        block = connection.recv(size - len(payload))
        if not block:
            raise ConnectionError("XR1 policy disconnected during closed-loop evaluation")
        payload.extend(block)
    return bytes(payload)


def _predict(path: Path, images: dict, state: dict, noise_seed: int) -> dict:
    import cv2

    encoded = {}
    for name, pixels in images.items():
        ok, png = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode a policy camera observation")
        encoded[name] = base64.b64encode(png).decode()
    request = json.dumps({"images": encoded, "state": state, "instruction": INSTRUCTION,
                          "noise_seed": noise_seed}, allow_nan=False).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(path))
        connection.sendall(struct.pack(">I", len(request)) + request)
        size = struct.unpack(">I", _receive(connection, 4))[0]
        if not 0 < size < 1024 * 1024:
            raise ValueError("Policy response exceeds the 30-action output contract")
        result = json.loads(_receive(connection, size))
        if result.get("error", {}).get("kind") == "invalid_prediction":
            raise ValueError(result["error"]["message"])
        return result["targets"]


def _target(chunk: dict, index: int) -> dict:
    target = {name: np.asarray(values[index]).reshape(-1).tolist() for name, values in chunk.items()}
    for side, base_y in (("left", .4), ("right", -.4)):
        position = np.asarray(target[f"{side}_ee_pos"])
        rotation = np.asarray(target[f"{side}_ee_rotm"]).reshape(3, 3)
        aperture = np.asarray(target[f"{side}_gripper_pos"])
        if not all(np.isfinite(value).all() for value in (position, rotation, aperture)):
            raise ValueError("Policy produced a nonfinite actuator target")
        if np.any(position < [.10, base_y - .35, .005]) or np.any(position > [.85, base_y + .35, .85]):
            raise ValueError("Policy requested a Cartesian target outside the declared robot workspace")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
            raise ValueError("Policy rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError("Policy rotation contains a reflection")
    return target


def _control(cell: _Cell, cameras: _Cameras, policy_socket: Path, seed: int) -> dict:
    rows, chunk, failure = [], None, None
    started = cell.world.current_time
    for frame in range(500):
        cameras.follow_wrists(cell.robots)
        cell.world.render()
        images, state = cameras.capture(), cell.state()
        row = {"time": cell.world.current_time - started, "state": state, "measures": cell.measures()}
        rows.append(row)
        try:
            if frame % 6 == 0:
                chunk = _predict(policy_socket, images, state, seed * 10000 + frame)
            row["requested_action"] = _target(chunk, frame % 6)
        except ValueError as error:
            failure = str(error)
            break
        row["gripper_saturated"] = any(not 0 <= row["requested_action"][f"{side}_gripper_pos"][0] <= .08
                                         for side in ("left", "right"))
        for _ in range(3):
            cell.policy_step(row["requested_action"])
        if len(rows) >= 30 and _outcome([entry["measures"] for entry in rows])["success"]:
            break
    result = _outcome([row["measures"] for row in rows]) if len(rows) >= 21 else {"success": False}
    if failure:
        result.update(success=False, failure_reason=failure)
    return {"trace": rows, "recorded_frames": len(rows), **result}


def _rollout(args) -> None:
    from isaacsim import SimulationApp

    args.output_path.mkdir(parents=True, exist_ok=False)
    app = SimulationApp({"headless": True, "width": 960, "height": 720})
    cameras = None
    try:
        cell = _Cell(args.seed)
        cameras = _Cameras(cell.random, args.output_path)
        for _ in range(30):
            cameras.follow_wrists(cell.robots)
            cell.world.step(render=True)
        result = _control(cell, cameras, args.policy_socket, args.seed)
        result.update(schema="npa.xr1-antioch.rollout.v1", seed=args.seed,
                      checkpoint_sha256=args.checkpoint_sha256, control_hz=20,
                      replan_every_frames=6, horizon_seconds=25, expert_fallback=False,
                      grasp_mechanism="finger_contact", videos=cameras.frames)
        (args.output_path / "rollout.json").write_text(json.dumps(result, allow_nan=False))
        print(json.dumps({key: value for key, value in result.items() if key != "trace"}), flush=True)
    finally:
        if cameras is not None:
            cameras.close()
        app.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--policy-socket", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    _rollout(parser.parse_args())


if __name__ == "__main__":
    _main()
