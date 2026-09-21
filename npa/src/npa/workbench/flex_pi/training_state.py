"""Hash live optimizer and RNG state to prove complete checkpoint restoration."""

import hashlib
import json
import random

import numpy as np
import torch


def state_digest(value):
    """Hash nested training state by value, independent of object identities.

    Args:
        value: Optimizer, scheduler, or random-generator state.
    Returns:
        SHA-256 covering types, shapes, keys and every scalar or tensor byte.
    Raises:
        TypeError: The state contains an unsupported value.
    """
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _update_digest(digest, value):
    digest.update(type(value).__name__.encode() + b"\0")
    if isinstance(value, torch.Tensor):
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(str((value.dtype, value.shape)).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        digest.update(str(len(value)).encode() + b"\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(str(len(value)).encode() + b"\0")
        for item in value:
            _update_digest(digest, item)
    else:
        digest.update(json.dumps(value, allow_nan=False).encode() + b"\0")


def rng_digest(*, cuda_only=False):
    """Capture the random generators restored by Accelerate on this rank.

    Args:
        cuda_only: After continuation, inspect the model's current CUDA stream;
            fresh loader workers may legitimately consume a different CPU seed.
    Returns:
        SHA-256 of Python, NumPy, CPU Torch and all visible CUDA RNG states.
    Raises:
        RuntimeError: A CUDA generator cannot be inspected.
    """
    if cuda_only:
        return state_digest(torch.cuda.get_rng_state())
    return state_digest({"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
