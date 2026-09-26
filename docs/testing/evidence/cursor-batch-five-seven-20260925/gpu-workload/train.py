"""Run real CUDA linear regression and publish the model, momentum, and measurements."""

import argparse
import importlib.metadata
import io
import json
import math
import os
import time

from object_storage import _client, _publish, _release_exists
from regression_contract import (
    IMAGE,
    INITIAL,
    RECIPE,
    TORCH_VERSION,
    _check_identity,
    _dataset,
    _json_bytes,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-uri", "run-id", "source-sha", "payload-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--post-output-verification", choices=("none", "until-release"), default="none"
    )
    parser.add_argument("--release-uri", default="")
    args = parser.parse_args()
    _check_identity(args.source_sha, args.payload_sha, args.run_id)
    if args.post_output_verification == "until-release" and not args.release_uri:
        parser.error("until-release requires --release-uri")
    return args


def _runtime(torch):
    if str(torch.__version__) != TORCH_VERSION or not torch.cuda.is_available():
        raise RuntimeError("the pinned Torch CUDA runtime and a real GPU are required")
    if os.environ.get("NPA_TASK_IMAGE", "").removeprefix("docker:") != IMAGE:
        raise RuntimeError("NPA_TASK_IMAGE must match the frozen image digest")
    torch.use_deterministic_algorithms(True)
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "device_type": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "image": IMAGE,
        "boto3_version": importlib.metadata.version("boto3"),
        "botocore_version": importlib.metadata.version("botocore"),
    }


def _tensors(torch, seed, samples):
    rows, targets = _dataset(seed, samples)
    return (
        torch.tensor(rows, dtype=torch.float64, device="cuda"),
        torch.tensor(targets, dtype=torch.float64, device="cuda").reshape(-1, 1),
    )


def _model(torch):
    model = torch.nn.Linear(8, 1, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([INITIAL[:8]], device="cuda", dtype=torch.float64)
        )
        model.bias.fill_(INITIAL[8])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=RECIPE["learning_rate"], momentum=RECIPE["momentum"]
    )
    return model, optimizer


def _train_step(torch, model, optimizer, inputs, targets, step):
    torch.cuda.synchronize()
    started = time.perf_counter()
    before = torch.cat(
        [parameter.detach().flatten() for parameter in model.parameters()]
    ).clone()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    gradient = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    gradient_norm = gradient.norm().item()
    optimizer.step()
    after = torch.cat(
        [parameter.detach().flatten() for parameter in model.parameters()]
    )
    movement = (after - before).norm().item()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    record = {
        "optimizer_step": step,
        "loss": loss.item(),
        "gradient_norm": gradient_norm,
        "parameter_delta": movement,
        "elapsed_seconds": elapsed,
        "samples_per_second": RECIPE["samples"] / elapsed,
    }
    if any(not math.isfinite(value) or value <= 0 for value in record.values()):
        raise ValueError("nonfinite or nonpositive training measurement")
    return record


def _checkpoint(torch, model, optimizer, binding):
    model_state = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    optimizer_state = optimizer.state_dict()
    for state in optimizer_state["state"].values():
        state["momentum_buffer"] = state["momentum_buffer"].detach().cpu()
    state = {
        **binding,
        "recipe": RECIPE,
        "optimizer_step": RECIPE["steps"],
        "initial_parameters": INITIAL,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _publish_training(torch, client, args, model, optimizer, metrics, binding):
    prefix = args.output_uri.rstrip("/")
    artifacts = {
        "checkpoint.pt": _checkpoint(torch, model, optimizer, binding),
        "metrics.json": _json_bytes(metrics),
    }
    records = {
        name: _publish(client, prefix + "/" + name, data)
        for name, data in artifacts.items()
    }
    manifest = {
        "schema": "cuda-regression-manifest/v1",
        **binding,
        "artifacts": records,
    }
    _publish(client, prefix + "/manifest.json", _json_bytes(manifest))
    print(
        _json_bytes(
            {"event": "OUTPUTS_VERIFIED", **binding, "artifacts": records}
        ).decode(),
        flush=True,
    )


def _post_output_verification(torch, client, args, model, binding):
    if args.post_output_verification == "none":
        return
    round_number = 0
    while True:
        round_number += 1
        inputs, targets = _tensors(torch, 1000 + round_number, RECIPE["samples"])
        with torch.no_grad():
            loss = (model(inputs) - targets).square().mean().item()
        if not math.isfinite(loss) or not 0 < loss < 0.02:
            raise ValueError("post-output CUDA generalization verification failed")
        progress = {
            **binding,
            "verification_round": round_number,
            "heldout_loss": loss,
            "classification": "post-output CUDA verification; no optimizer resume",
        }
        _publish(
            client,
            args.output_uri.rstrip("/") + "/post-output-progress.json",
            _json_bytes(progress),
        )
        release = _release_exists(client, args.release_uri)
        if release is not None:
            if json.loads(release) != {**binding, "action": "release"}:
                raise ValueError("post-output release marker identity mismatch")
            return


def main():
    """Train and publish measured CUDA artifacts, optionally continue validation.

    Args: None; arguments come from the command line.
    Returns: None.
    Raises: RuntimeError or ValueError for invalid runtime/evidence; S3 errors propagate.
    """
    args = _arguments()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    binding = {
        "run_id": args.run_id,
        "source_sha": args.source_sha,
        "payload_sha": args.payload_sha,
    }
    metrics = {
        "schema": "cuda-regression-metrics/v1",
        **binding,
        **_runtime(torch),
        "recipe": RECIPE,
    }
    client = _client()
    model, optimizer = _model(torch)
    inputs, targets = _tensors(torch, RECIPE["seed"], RECIPE["samples"])
    metrics["journal"] = [
        _train_step(torch, model, optimizer, inputs, targets, step)
        for step in range(1, RECIPE["steps"] + 1)
    ]
    with torch.no_grad():
        metrics["final_train_loss"] = (model(inputs) - targets).square().mean().item()
        heldout, expected = _tensors(
            torch, RECIPE["seed"] + 1, RECIPE["heldout_samples"]
        )
        metrics["heldout_loss"] = (model(heldout) - expected).square().mean().item()
    _publish_training(torch, client, args, model, optimizer, metrics, binding)
    _post_output_verification(torch, client, args, model, binding)


if __name__ == "__main__":
    main()
