"""Build-time proof that the pinned native Ray Serve path is complete."""

from __future__ import annotations

import os
from pathlib import Path

import ray
import torch
from cosmos_framework.inference.ray import serve as cosmos_ray_serve
from npa.workbench.cosmos import ray_serve as npa_ray_contract
from npa.workbench.cosmos import ray_server as npa_ray_server


def main() -> None:
    # Ray decorators replace the Python classes with Deployment/ActorClass
    # wrappers at import time. Prove the pinned implementation from its source
    # module, then separately prove that importing produced real Ray objects.
    source_path = Path(cosmos_ray_serve.__file__)
    source = source_path.read_text(encoding="utf-8")
    if "@ray.serve.batch" not in source:
        raise RuntimeError("upstream OmniModelDeployment no longer owns Ray Serve batching")
    if "class OmniModelDeployment" not in source or "OmniInference" not in source:
        raise RuntimeError("native Cosmos model generation path is incomplete")
    if not hasattr(cosmos_ray_serve.OmniModelDeployment, "bind"):
        raise RuntimeError("upstream model did not import as a Ray Serve deployment")
    if not hasattr(cosmos_ray_serve.OmniModelWorker, "remote"):
        raise RuntimeError("upstream model worker did not import as a Ray actor")
    if not hasattr(ray.serve, "run"):
        raise RuntimeError("Ray Serve runtime is missing")
    if npa_ray_contract.RAY_BATCH_SCHEMA != "npa.cosmos3.ray-serve.batch.v1":
        raise RuntimeError("NPA batch contract is missing from the framework environment")
    if not callable(npa_ray_server.main):
        raise RuntimeError("NPA authenticated Ray ingress is not importable")
    arches = set((torch._C._cuda_getArchFlags() or "").split())
    for required in {"sm_100", "sm_120"}:
        if required not in arches:
            raise RuntimeError(f"torch wheel lacks required Blackwell target {required}: {sorted(arches)}")
    if os.environ.get("NPA_COSMOS3_RAY_GUARDRAILS") != "true":
        raise RuntimeError("guardrails must default on")
    roots = [Path("/opt/cosmos3"), Path("/opt/npa"), Path("/outputs")]
    forbidden = []
    for root in roots:
        forbidden.extend(root.rglob("*.safetensors"))
        forbidden.extend(root.rglob("*.ckpt"))
    if forbidden:
        raise RuntimeError(f"model weights are baked into the image: {forbidden[:3]}")
    print(
        "[PASS] native Cosmos Ray Serve verified "
        f"ray={ray.__version__} torch={torch.__version__} arches={sorted(arches)}"
    )


if __name__ == "__main__":
    main()
