"""Serve the real XR1 policy over a private local socket beside the Antioch simulator."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import socketserver
import struct

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from mibot.models import MIMODEL
from mibot.server.runtime.client import Client
from mibot.utils.io import (
    build_action_mask, compose_state, denormalize_action, normalize_quantile,
    recover_action, resize_image, validate_quantiles, validate_stats,
)

from .assets import PROCESSOR_REVISION
from .storage import _sha256


class _Policy:
    def __init__(self, checkpoint: Path, expected: str, statistics: Path, processor: Path):
        if _sha256(checkpoint) != expected:
            raise ValueError("Policy checkpoint differs from its verified S3 receipt")
        self.model = MIMODEL.build({"type": "xr1", "async_train": False}).to(torch.bfloat16)
        weights = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)["module"]
        state = {name.removeprefix("model."): value for name, value in weights.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to("cuda")
        self.processor = AutoProcessor.from_pretrained(str(processor), local_files_only=True,
                                                       revision=PROCESSOR_REVISION)
        stats = json.loads(statistics.read_text())
        self.mean, self.std = validate_stats(stats["mean"], stats["std"], 30)
        self.q01, self.q99 = validate_quantiles(stats["q01"], stats["q99"])
        self.mask = torch.from_numpy(build_action_mask(30)).to("cuda")[None]

    def _batch(self, request: dict):
        images = [Image.open(BytesIO(base64.b64decode(request["images"][name]))).convert("RGB")
                  for name in ("ego", "wrist_left", "wrist_right")]
        images = [resize_image(image, factor=32, max_pixels=160000) for image in images]
        messages = Client._messages(request["instruction"], *images)
        batch = self.processor.apply_chat_template(
            [messages], tokenize=True, return_dict=True, return_tensors="pt", padding=True,
            images_kwargs={"do_resize": False},
        ).to("cuda")
        state = request["state"]
        proprioception = compose_state(
            np.asarray(state["left_gripper_pos"]), np.asarray(state["left_arm_joint"]),
            np.asarray(state["right_gripper_pos"]), np.asarray(state["right_arm_joint"]),
        )
        normalized = normalize_quantile(proprioception, self.q01, self.q99)
        batch["state"] = torch.from_numpy(normalized).to("cuda")[None]
        batch["action"] = torch.zeros((1, 30, 60), device="cuda", dtype=torch.bfloat16)
        batch["action_mask"] = self.mask
        return batch

    @torch.no_grad()
    def predict(self, request: dict) -> dict:
        torch.manual_seed(int(request["noise_seed"]))
        action = self.model.generate(self._batch(request)) * self.mask
        action = denormalize_action(action.float().cpu().numpy()[0], self.mean, self.std)
        if not np.isfinite(action).all():
            raise ValueError("XR1 produced nonfinite actions")
        targets = recover_action(action, request["state"])
        return {"targets": {name: value.tolist() for name, value in targets.items()},
                "normalized_action_shape": [30, 60]}


def _receive(stream, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = stream.recv(size - len(result))
        if not block:
            raise ConnectionError("Policy socket closed before a complete message")
        result.extend(block)
    return bytes(result)


class _Request(socketserver.BaseRequestHandler):
    def handle(self):
        size = struct.unpack(">I", _receive(self.request, 4))[0]
        if not 0 < size < 4 * 1024 * 1024:
            raise ValueError("Observation exceeds the fixed three-camera input contract")
        observation = json.loads(_receive(self.request, size))
        try:
            result = self.server.policy.predict(observation)
        except ValueError as error:
            result = {"error": {"kind": "invalid_prediction", "message": str(error)}}
        payload = json.dumps(result, allow_nan=False).encode()
        self.request.sendall(struct.pack(">I", len(payload)) + payload)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "statistics", "processor", "socket"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    policy = _Policy(args.checkpoint, args.sha256, args.statistics, args.processor)
    args.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with socketserver.UnixStreamServer(str(args.socket), _Request) as server:
        os.chmod(args.socket, 0o600)
        server.policy = policy
        print("XR1_POLICY_READY", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    _main()
