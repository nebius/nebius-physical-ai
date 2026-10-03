"""Freeze public training normalization across independent runs and resume."""

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_artifacts import sha256_file


def validate_normalization(request):
    """Require an exact S3 object and digest together before any GPU execution.

    Args:
        request: Training request with optional frozen normalization input.
    Returns:
        None.
    Raises:
        FlexPiError: The optional input lacks a safe object path or SHA-256.
    """
    if not request.normalization_path and not request.normalization_sha256:
        return
    parsed = urlparse(request.normalization_path)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.path.endswith("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or not re.fullmatch(r"[0-9a-f]{64}", request.normalization_sha256)
    ):
        raise FlexPiError("normalization requires an exact S3 object and SHA-256")


def stage_normalization(request, work, storage=None):
    """Download the original statistics and verify every byte before model setup.

    Args:
        request: Validated request with paired normalization identity.
        work: Private work directory.
        storage: Optional configured storage client.
    Returns:
        Verified local JSON path.
    Raises:
        FlexPiError: Storage returned different normalization bytes.
    """
    destination = Path(work) / "dataset_stats.json"
    (storage or StorageClient.from_environment()).download_file(
        request.normalization_path, str(destination)
    )
    if sha256_file(destination) != request.normalization_sha256:
        raise FlexPiError("normalization content verification failed")
    if not isinstance(json.loads(destination.read_text()), dict):
        raise FlexPiError("normalization must be a JSON object")
    return destination


def normalization_overrides(plan, work):
    """Select the exact frozen statistics using the upstream configuration field.

    Args:
        plan: Prepared training or restored-checkpoint plan.
        work: This phase's private work directory.
    Returns:
        Hydra overrides for the verified input, if supplied.
    Raises:
        FileNotFoundError: A promised normalization artifact is absent.
    """
    value = plan.get("normalization_file")
    if not value:
        return []
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError("frozen normalization artifact is missing")
    destination = Path(work) / "dataset_stats.json"
    if path != destination:
        shutil.copyfile(path, destination)
        if sha256_file(path) != sha256_file(destination):
            raise FlexPiError("normalization phase copy differs")
    return [
        f"data.{split}.pretrained_norm_stats=" + json.dumps(str(destination))
        for split in ("train", "val")
    ]
