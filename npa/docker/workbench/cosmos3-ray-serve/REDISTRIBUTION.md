# Cosmos3 native Ray Serve redistribution decision

Decision: `public`, subject to exact-image payload/security scans and independent
real-GPU acceptance on every advertised CUDA target.

- The digest-pinned parent is the accepted public `npa-cosmos3` image. It carries
  NVIDIA cosmos-framework 1.2.2 source at commit
  `5e67049cd94acb667786f1e6dd0dab821cb90c97` under OpenMDW-1.1 and its frozen
  CUDA 13 inference environment.
- The new layer contains only NPA's Apache-2.0 CLI/client/ingress code. It uses
  upstream `OmniModelDeployment`, including upstream `@ray.serve.batch` and
  `OmniInference.generate_batch`; it contains no vLLM-Omni implementation.
- `nvidia/Cosmos3-Nano`, its VAE, and `nvidia/Cosmos-Guardrail1` are not baked.
  They download at runtime into the standard operator-owned NPA model cache with
  that operator's Hugging Face entitlement. Guardrails are enabled by default.
- Inputs, generated outputs, request/response records, and provenance are runtime
  data. The client persists them through the standard S3 contract; none enters an
  image layer.
