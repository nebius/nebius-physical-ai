# Public bootstrap with runtime-fetched Nano components

The replacement image inherits the accepted public `npa-cosmos3-serving`
bootstrap. It does not inherit any layer from the former
`vllm/vllm-omni:cosmos3` vendor image. CUDA, vLLM, Ray, models, upstream fixture
media, and populated caches are absent from the image's build closure. The
complete built archive must pass the serving payload and mandatory security
scanners before an immutable development digest is published.

| Boundary | Scope and policy |
| --- | --- |
| Source | NPA adapters and runtime-fetched [vLLM-Omni source](https://github.com/vllm-project/vllm-omni/blob/eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c/LICENSE) use Apache-2.0. |
| Baked runtime | The accepted Python/Debian bootstrap and added Debian packages retain their component notices. Serving and Ray dependencies are hash-locked recipes, fetched after launch under their own terms. |
| Weights | [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano/tree/7a312c868bcce8e40b3eb40861300a9d0ba3fde1) uses OpenMDW-1.1 and is fetched anonymously at its pinned revision. Disabled guardrails need no guardrail checkpoint. Enabling gated components requires their separate actual entitlement. |
| Data | No task/customer data, vendor fixture, or example media is baked. Operators supply source media they are entitled to process. |
| Runtime caches | The CPU staging Job populates the writable serving/Ray cache and stages the checkpoint with BF16 checks, file hashes, and atomic READY.json. Serving replicas read the model offline; credentials and acceptance values never enter the cache or image. |
| Outputs | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) imposes no restrictions on generated outputs. Input-media rights remain separate. Keep artifacts and provenance in operator storage. |

NVIDIA software acceptance authorizes the operator's runtime use; it is not a
grant to redistribute the fetched runtime. Public development validation of one
GPU does not establish the historical 16-replica acceptance matrix on replacement
bytes. Supported release promotion still requires its own exact-digest evidence.
