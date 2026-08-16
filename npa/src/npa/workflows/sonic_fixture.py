"""Build a tiny, real SONIC policy checkpoint (and its ONNX export) as a test fixture.

Why this exists
---------------
The live-submit matrix proves an ``npa.workflow`` twin by actually running it. The
SONIC twins (``sonic-export``, ``sonic-eval``, ``sonic-export-eval``) need a
**loadable torch policy checkpoint** as input, and the repo deliberately does not
vendor NVIDIA's public ``nvidia/GEAR-SONIC`` weights. Without a fixture those twins
can only ever be covered plan-only, which is not evidence.

``npa workbench sonic export`` accepts a checkpoint that stores a ``torch.nn.Module``
under ``policy`` / ``actor`` / ``model`` (see ``_load_policy_from_checkpoint`` in
``npa/src/npa/workbench/sonic/__init__.py``). This module builds exactly that: a small
deterministic MLP with the observation/action dimensions of a locomotion policy. The
export it produces is a *real* ONNX graph run through the shipped exporter, not a
stub — the point is to exercise the tool, not to imitate its output.

Standalone by design
--------------------
This module imports nothing from ``npa`` so it can run inside a SONIC container that
has torch but not (yet) this branch of the package —
``scripts/stage-sonic-export-fixture.sh`` mounts this single file into a pod through a
ConfigMap and runs it. Keeping the logic here (rather than inline in the script) is
what makes it unit-testable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

#: Unitree G1 whole-body-control shapes, which is what the SONIC specs target.
DEFAULT_OBS_DIM = 48
DEFAULT_ACT_DIM = 12
DEFAULT_HIDDEN = 32
FIXTURE_SCHEMA = "npa.sonic.export_fixture.v1"


class SonicFixtureError(RuntimeError):
    """Raised when the fixture cannot be built or published."""


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise SonicFixtureError(
            "building a SONIC checkpoint fixture requires torch; run this inside the "
            "SONIC workbench image or install the npa[sonic] extra"
        ) from exc
    return torch


def build_policy_module(
    *,
    obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
    hidden: int = DEFAULT_HIDDEN,
    seed: int = 0,
) -> Any:
    """Return a small deterministic MLP policy module.

    Deterministic (fixed seed) so a re-staged fixture is byte-comparable and a live
    run's numbers are reproducible.
    """

    if min(obs_dim, act_dim, hidden) < 1:
        raise SonicFixtureError("obs_dim, act_dim and hidden must all be >= 1")
    torch = _import_torch()
    torch.manual_seed(seed)
    policy = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ELU(),
        torch.nn.Linear(hidden, act_dim),
    )
    # The exporter resolves shapes from the policy when no --obs-spec is given, trying
    # `observation_dim` / `obs_dim` / `input_dim` / `num_observations` (and the action
    # equivalents). A bare Sequential exposes none of them, which is exactly how the
    # first staged fixture failed live: "observation dimension is required. Provide
    # --obs-spec or a policy with one of: ..." (SkyPilot job 188). Real SONIC policies
    # carry these, so the fixture must too.
    #
    # Plain ints land in the module's __dict__ (not _parameters/_modules), so they
    # survive torch.save/torch.load without making the checkpoint depend on any npa
    # class being importable at load time.
    policy.obs_dim = obs_dim
    policy.action_dim = act_dim
    return policy


def build_checkpoint(
    output_path: str | Path,
    *,
    obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
    hidden: int = DEFAULT_HIDDEN,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a ``{"policy": <nn.Module>}`` checkpoint and return its metadata."""

    torch = _import_torch()
    policy = build_policy_module(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, seed=seed)
    policy.eval()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": FIXTURE_SCHEMA,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": hidden,
        "seed": seed,
        "torch_version": str(torch.__version__),
    }
    torch.save({"policy": policy, "npa_fixture": metadata}, str(path))
    return {**metadata, "checkpoint_path": str(path), "bytes": path.stat().st_size}


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise SonicFixtureError(f"expected an s3:// URI, got {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise SonicFixtureError(f"s3 URI needs a key: {uri!r}")
    return parsed.netloc, key


def upload(local_path: str | Path, uri: str, *, client: Any | None = None) -> str:
    """Upload one file to ``s3://...`` and return the URI."""

    bucket, key = split_s3_uri(uri)
    if client is None:  # pragma: no cover - real S3 only
        import boto3
        from botocore.client import Config

        kwargs: dict[str, Any] = {"config": Config(signature_version="s3v4")}
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        client = boto3.client("s3", **kwargs)
    client.upload_file(str(local_path), bucket, key)
    return uri


def build_and_publish(
    *,
    checkpoint_uri: str = "",
    workdir: str | Path = "/tmp/npa-sonic-fixture",
    obs_dim: int = DEFAULT_OBS_DIM,
    act_dim: int = DEFAULT_ACT_DIM,
    hidden: int = DEFAULT_HIDDEN,
    seed: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Build the checkpoint locally and optionally upload it."""

    root = Path(workdir)
    checkpoint = root / "checkpoint.pt"
    result = build_checkpoint(
        checkpoint, obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, seed=seed
    )
    if checkpoint_uri:
        result["checkpoint_uri"] = upload(checkpoint, checkpoint_uri, client=client)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (used by scripts/stage-sonic-export-fixture.sh)."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-uri",
        default="",
        help="s3:// URI to upload the checkpoint to (omit to only write it locally).",
    )
    parser.add_argument("--workdir", default="/tmp/npa-sonic-fixture")
    parser.add_argument("--obs-dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--act-dim", type=int, default=DEFAULT_ACT_DIM)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        result = build_and_publish(
            checkpoint_uri=args.checkpoint_uri,
            workdir=args.workdir,
            obs_dim=args.obs_dim,
            act_dim=args.act_dim,
            hidden=args.hidden,
            seed=args.seed,
        )
    except SonicFixtureError as exc:
        print(f"Error: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
