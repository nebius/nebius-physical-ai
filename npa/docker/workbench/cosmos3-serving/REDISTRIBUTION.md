# Cosmos3 serving redistribution decision

Decision: `public`, subject to the repository's exact-image build, scan, GPU
acceptance, and digest promotion gates.

## Artifact classification

- Source: vLLM-Omni 0.28.0 at commit
  `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c`, Apache-2.0, fetched from
  upstream and verified by SHA-256 at runtime.
- Base: digest-pinned official Python 3.12 slim image. It is not an NVIDIA Deep
  Learning Container and contains no `NGC-DL-CONTAINER-LICENSE` payload.
- Serving closure: exact versions in `requirements.lock`, including vLLM 0.28.0,
  CUDA-enabled PyTorch 2.13.0, and Cosmos Guardrail 0.3.1. CUDA Python packages
  carry the NVIDIA Software License (v. May 12, 2021), whose downstream-terms
  requirements are not satisfied by anonymous GHCR. Therefore none of this
  closure is baked: the operator must explicitly set
  `NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES`, after which upstream delivers
  it directly into a writable runtime volume.
- Weights and data: none. `nvidia/Cosmos3-Super` and
  `nvidia/Cosmos-1.0-Guardrail` are fetched only at runtime into an
  operator-writable cache after authenticated access to each repository is
  independently confirmed.
- Credentials and terms: neither is accepted, persisted, or baked. The operator
  supplies a token for an account that already has each upstream entitlement.

The former `vllm/vllm-omni` parent is prohibited because its built filesystem
contained NVIDIA's Deep Learning Container license. Baking the replacement
CUDA Python closure is also prohibited. Mutation tests and built-image scanners
reject either runtime, their license markers, cached models, credentials, and
pre-accepted terms if any return.
