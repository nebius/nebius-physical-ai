# Public bootstrap with runtime-fetched serving components

This replacement inherits the accepted public `npa-cosmos3-serving` bootstrap,
not the former vendor runtime. The pinned Python/Debian base, Debian SkyPilot
bootstrap packages, and NPA source retain their licenses and notices. There are
no baked CUDA/vLLM wheels, model weights, upstream example media, or populated
runtime caches. Review the complete built image, including every ancestor layer,
with the serving payload scanner and mandatory security gates before publishing
an immutable development digest.

| Boundary | Scope and policy |
| --- | --- |
| Source | NPA bootstrap and runtime-fetched vLLM-Omni source use Apache-2.0. |
| Baked runtime | The accepted public parent and Debian packages retain their component notices. NVIDIA executable runtime bytes are fetched after launch. |
| Weights | [Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super/tree/e0262be9d8f7586bc24c069a2aed2b665bdff266) is pinned and fetched at runtime under OpenMDW-1.1. Public model files need no HF token; selected gated assets require actual upstream entitlement. Guardrails are disabled by default for this benchmark image. |
| Data | No dataset, benchmark prompt asset, example media, or customer payload is baked. The benchmark fetches pinned public model prompt assets at runtime. |
| Runtime caches | The operator accepts applicable software terms and supplies a writable cache. Never publish that populated cache or bake credentials or acceptance state. |
| Outputs | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) does not restrict generated outputs. Rights in input media remain separate. Retain artifacts and provenance privately. |

Anonymous pullability, runtime software acceptance, model access, and successful
GPU execution establish different facts. A successful development workload does
not reproduce the historical fixed benchmark or qualify a supported release.
