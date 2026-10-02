"""Authenticate baseline anchor state without altering native model or optimizer tensors."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

PPO_SOURCE_SHA256 = "a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c"
ANCHOR_KEY = "npa_baseline_anchor"


def verify_native_source():
    """Reject native PPO versions or source differing from the reviewed algorithm.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: Native version or source bytes are unsupported.
    """
    import rsl_rl.algorithms.ppo as native

    digest = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if version("rsl-rl-lib") != "5.0.1" or digest != PPO_SOURCE_SHA256:
        raise ValueError("baseline anchor requires the exact RSL-RL 5.0.1 PPO source")


def state_digest(state):
    """Hash nested native checkpoint state including exact tensor contents.

    Args:
        state: Tensor, mapping, sequence or finite JSON scalar.
    Returns:
        Stable SHA-256 retaining mapping key and tensor layout identity.
    Raises:
        TypeError: State contains an unsupported value.
        ValueError: State contains a nonfinite scalar.
    """
    digest = hashlib.sha256()
    _feed_state(digest, state)
    return digest.hexdigest()


def _feed_state(digest, value):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(f"mapping:{len(value)}:".encode())
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            _feed_state(digest, key)
            _feed_state(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}:".encode())
        for item in value:
            _feed_state(digest, item)
    elif value is None or type(value) in (str, int, float, bool):
        digest.update(json.dumps(value, allow_nan=False).encode() + b";")
    else:
        raise TypeError("unsupported native checkpoint state value")


def implementation_digest():
    """Identify the complete local anchor implementation and redistribution notice.

    Args:
        None.
    Returns:
        SHA-256 of sorted source filenames and contents.
    Raises:
        OSError: A required source or notice cannot be read.
    """
    root = Path(__file__).parent
    paths = sorted(root.glob("anchor_*.py"))
    return state_digest(
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    )


def native_state(payload):
    """Select the untouched native payload independently of anchor metadata.

    Args:
        payload: Native checkpoint with optional namespaced anchor evidence.
    Returns:
        All original native fields.
    Raises:
        None.
    """
    return {key: value for key, value in payload.items() if key != ANCHOR_KEY}
