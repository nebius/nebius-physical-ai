"""Download the CUDA artifacts and verify hashes, checkpoint state, and scalar CPU replay."""

import argparse
import io
import json

from artifact_validation import _validate_artifacts
from object_storage import _client, _publish, _read
from regression_contract import (
    RECIPE,
    _check_identity,
    _json_bytes,
    _validate_integrity,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-uri", "output-uri", "run-id", "source-sha", "payload-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    _check_identity(args.source_sha, args.payload_sha, args.run_id)
    return args


def _decode_checkpoint(data):
    import torch

    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    tensors = state["model_state_dict"]
    if set(tensors) != {"weight", "bias"}:
        raise ValueError("model checkpoint inventory differs")
    for name, shape in (("weight", (1, 8)), ("bias", (1,))):
        if tensors[name].shape != shape or tensors[name].dtype != torch.float64:
            raise ValueError("model checkpoint tensor shape or dtype differs")
    model = torch.nn.Linear(8, 1, dtype=torch.float64)
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=RECIPE["learning_rate"], momentum=RECIPE["momentum"]
    )
    optimizer.load_state_dict(state["optimizer_state_dict"])
    group = optimizer.param_groups[0]
    if len(optimizer.param_groups) != 1 or len(group["params"]) != 2:
        raise ValueError("optimizer parameter groups differ")
    for key, value in optimizer.defaults.items():
        if group.get(key) != value:
            raise ValueError("optimizer hyperparameters differ")
    parameters, momentum = [], []
    if set(optimizer.state) != set(model.parameters()):
        raise ValueError("optimizer momentum state is incomplete")
    for parameter in model.parameters():
        buffer = optimizer.state[parameter]["momentum_buffer"]
        if buffer.shape != parameter.shape or buffer.dtype != torch.float64:
            raise ValueError("optimizer momentum tensor shape or dtype differs")
        parameters.extend(parameter.detach().flatten().tolist())
        momentum.extend(buffer.flatten().tolist())
    return {
        key: value for key, value in state.items() if not key.endswith("state_dict")
    } | {
        "parameters": parameters,
        "momentum": momentum,
    }


def main():
    """Independently verify downloaded GPU results and publish a validation record.

    Args: None; arguments come from the command line.
    Returns: None.
    Raises: ValueError for invalid evidence; decoding and S3 failures propagate.
    """
    args = _arguments()
    client = _client()
    prefix = args.input_uri.rstrip("/")
    manifest = json.loads(_read(client, prefix + "/manifest.json"))
    artifacts = {
        name: _read(client, prefix + "/" + name)
        for name in ("checkpoint.pt", "metrics.json")
    }
    _validate_integrity(manifest, artifacts)
    metrics = json.loads(artifacts["metrics.json"])
    checkpoint = _decode_checkpoint(artifacts["checkpoint.pt"])
    expected = {
        "run_id": args.run_id,
        "source_sha": args.source_sha,
        "payload_sha": args.payload_sha,
    }
    result = _validate_artifacts(manifest, metrics, checkpoint, expected)
    result = {
        "schema": "cuda-regression-validation/v1",
        "status": "passed",
        **expected,
        **result,
        "artifacts": manifest["artifacts"],
        "limits": "Synthetic regression; no robot-policy, native Ray, or optimizer-resume claim.",
    }
    _publish(client, args.output_uri, _json_bytes(result))
    print(_json_bytes(result).decode(), flush=True)


if __name__ == "__main__":
    main()
