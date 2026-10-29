# Cosmos3 native Ray Serve redistribution decision

Decision: `public`, subject to exact-image payload/security scans and independent
real-GPU acceptance on every advertised CUDA target.

- The digest-pinned parent is the repaired main-source Cosmos3 development
  candidate, pending acceptance and not a public default. It carries NVIDIA
  cosmos-framework 1.2.2 source at commit
  `5e67049cd94acb667786f1e6dd0dab821cb90c97` under OpenMDW-1.1 and its frozen
  CUDA 13 inference environment.
- The application source added by this layer is NPA's Apache-2.0
  CLI/client/ingress code. It uses upstream `OmniModelDeployment`, including
  upstream `@ray.serve.batch` and `OmniInference.generate_batch`; it contains no
  vLLM-Omni implementation. The layer also installs the hash-bound serving
  dependency closure described below.
- The serving closure builds Apache-2.0
  `gradio==6.16.0+npa.pillow13.1` from the upstream `gradio==6.16.0` wheel with
  SHA-256
  `a5de01bf17336bf421d52af4b5dbd9fdbd530762c485067748456ccb797700bb`.
  The explicitly named derivative changes only the distribution version and
  Pillow constraint, regenerates the wheel RECORD, and has reproducible SHA-256
  `2523b8b34006626382b2afa315c229c3ef4515b17c6925b766308f32e161c749`.
  The build resolves the complete hash-locked environment against that wheel,
  runs `uv pip check`, and exercises a real Gradio image roundtrip on Pillow 13.
- `nvidia/Cosmos3-Nano`, its VAE, and `nvidia/Cosmos-Guardrail1` are not baked.
  They download at runtime into the standard operator-owned NPA model cache with
  that operator's Hugging Face entitlement. Guardrails are enabled by default.
- Inputs, generated outputs, request/response records, and provenance are runtime
  data. The client persists them through the standard S3 contract; none enters an
  image layer.
