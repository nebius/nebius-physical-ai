# Operator-private runtime

The pinned upstream `vllm/vllm-omni:cosmos3` image includes a vendor runtime.
This derived image is restricted to the owning operator's registry and is
excluded from NPA's public publishing plan. The added adapter source is NPA
source; its license does not change the redistribution terms of inherited
image layers.

The model is delivered directly to the operator at runtime from
[NVIDIA's pinned Cosmos3-Nano release](https://huggingface.co/nvidia/Cosmos3-Nano/tree/7a312c868bcce8e40b3eb40861300a9d0ba3fde1),
under its [OpenMDW 1.1 terms](https://openmdw.ai/license/1.1).
The extension adds no model weights, task or customer media, credentials or
populated caches. The inherited upstream image includes public vendor fixtures
and example media; those bytes remain part of its restricted runtime. The shared
model cache is runtime storage and must not enter a build context or be published
as a derived image.

Review the six artifact boundaries separately:

| Boundary | Scope and policy |
| --- | --- |
| Source | NPA adapters and the pinned [vLLM-Omni source](https://github.com/vllm-project/vllm-omni/blob/9c1b7504b178afcf541867c1a2d30db48c69cda8/LICENSE) use Apache-2.0. These source licenses do not establish rights for every inherited binary. |
| Baked runtime | The digest-pinned vendor image, its CUDA/cuDNN and other installed components, and the added Ray/FFmpeg dependencies remain an operator-private runtime. Public availability of the base is not a redistribution grant for all its layers. Preserve the component notices; do not publish this image outside the owning organization without separately establishing those rights. |
| Weights | The pinned Nano model materials use OpenMDW-1.1 and are fetched anonymously at runtime. No model weights enter the Docker build; enabling gated guardrails requires separate access to their exact selected payloads. |
| Data | The extension bakes no task/customer media or dataset. Inherited vendor fixtures remain covered by their own applicable notices. Operators supply source media they are entitled to process. |
| Runtime caches | A single CPU staging Job populates shared persistent storage under a lock, checks BF16 tensors and hashes, and atomically publishes `READY.json`. Serving replicas consume the checkpoint offline from a read-only mount. Credentials and acceptance state must never be stored in or beside this cache. |
| Outputs | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) imposes no restrictions or obligations on use, modification or sharing of generated outputs. Rights associated with source media and other third-party materials remain separate. Keep output artifacts and their provenance in operator-controlled runtime storage. |

The restricted classification is enforced by both `packaging-contract.yaml`
and `RESTRICTED_PUBLICATION_TOOLS`; it is not relaxed by an internal registry or
by a successful capability test. Existing scan and B200 measurements apply to
their recorded immutable image digests. A source update does not refresh those
artifact scans or establish evidence for replacement image bytes.
