"""OpenVLA predict_action on GPU — the production VLA inference workload.

The exact workload from the 2026-09-27 GPU validation (openvla-7b,
predict_action on cuda:0), onboarded onto the manifest path by descriptor
only. Dependencies (transformers/timm/tokenizers) arrive via
environment.pip; the model weights download from Hugging Face in-pod.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="openvla/openvla-7b")
    ap.add_argument("--prompt", default=(
        "In: What action should the robot take to pick up the red block?\nOut:"
    ))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-action", required=True)
    ap.add_argument("--out-obs", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda:0"
    assert torch.cuda.is_available(), "no CUDA device visible"
    gpu_name = torch.cuda.get_device_name(0)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    vla.eval()
    load_s = time.time() - t0

    keys = list(vla.norm_stats.keys())
    unnorm_key = "bridge_orig" if "bridge_orig" in keys else keys[0]

    rng = np.random.default_rng(args.seed)
    img = Image.fromarray(
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    ).convert("RGB")
    img.save(args.out_obs)
    inputs = processor(args.prompt, img).to(device, dtype=torch.bfloat16)

    t0 = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key)
    predict_s = time.time() - t0

    assert isinstance(action, np.ndarray), f"unexpected type {type(action)}"
    assert tuple(action.shape) == (7,), f"unexpected shape {tuple(action.shape)}"
    assert np.all(np.isfinite(action)), "non-finite values in action"
    np.save(args.out_action, action)

    summary = {
        "device_ok": True,
        "device": device,
        "gpu": gpu_name,
        "model_id": args.model_id,
        "torch": torch.__version__,
        "load_s": round(load_s, 1),
        "predict_s": round(predict_s, 2),
        "unnorm_key": unnorm_key,
        "prompt": args.prompt,
        "action_shape": list(action.shape),
        "action": [float(x) for x in action[:7]],
        "action_bytes": int(os.path.getsize(args.out_action)),
        "obs_bytes": int(os.path.getsize(args.out_obs)),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
