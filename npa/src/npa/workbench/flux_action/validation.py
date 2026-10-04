"""Verify fresh-process inference and independently read published export bytes."""

import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from .schemas import Recipe


def verify_reload(path: Path, recipe: Recipe) -> dict:
    """Require finite offline predictions with the declared action width.

    Args: path is the native evaluator report; recipe declares the robot.
    Returns: Non-identifying reload metrics.
    Raises: ValueError for missing windows, mismatched channels, or nonfinite errors.
    """
    report = json.loads(path.read_text())
    channels = report["action_mse_raw_per_channel"]
    values = [report["action_mse_normalized"], report["action_mse_raw"], *channels]
    if report["n_windows"] < 1 or len(channels) != len(recipe.robot.actions):
        raise ValueError("reload inference did not produce the declared action width")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("reload inference produced nonfinite metrics")
    return {
        "windows": report["n_windows"],
        "split": report["split"],
        "action_mse_normalized": report["action_mse_normalized"],
        "action_mse_raw": report["action_mse_raw"],
    }


def verify_remote_export(storage, output_uri: str, expected: dict) -> None:
    """Stream-hash S3 export files before writing the completion receipt.

    Args: storage is the operator client; output_uri is its run; expected maps hashes.
    Returns: None.
    Raises: ValueError for changed remote exports; provider errors propagate.
    """
    parsed = urlparse(output_uri)
    for name, wanted in expected.items():
        key = parsed.path.strip("/") + "/export/" + name
        stream = storage.s3.get_object(Bucket=parsed.netloc, Key=key)["Body"]
        digest = hashlib.sha256()
        for chunk in stream.iter_chunks(chunk_size=8 * 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != wanted:
            raise ValueError(f"published export SHA-256 mismatch: {name}")
