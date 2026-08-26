# Cosmos3-Nano native Ray Serve

`npa-cosmos3-ray-serve` is the persistent, batch-capable counterpart to the
single-run `npa workbench cosmos3 generate` path. It loads Cosmos3-Nano once and
uses NVIDIA cosmos-framework 1.2.2's native `OmniModelDeployment`, including its
real `@ray.serve.batch` → `OmniInference.generate_batch` path. It does not use
vLLM-Omni.

The service is intended for synthetic-data-generation queues that benefit from
resident weights and structured Cosmos outputs. The image is a thin derivative
of the exact accepted `npa-cosmos3` digest: it adds authenticated ingress and the
NPA S3 client/provenance contract, but no model, VAE, guardrail, credential,
dataset, or generated output.

## Runtime contract

The container starts `npa workbench cosmos3 ray-serve`. Important settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `NPA_COSMOS3_RAY_WORLD_SIZE` | `1` | GPUs used by one persistent model replica |
| `NPA_COSMOS3_RAY_MAX_BATCH_SIZE` | `4` | Maximum samples coalesced by upstream Ray Serve |
| `NPA_COSMOS3_RAY_BATCH_WAIT_TIMEOUT_S` | `0.05` | Coalescing window in seconds |
| `NPA_COSMOS3_RAY_PARALLELISM_PRESET` | `throughput` | Upstream Cosmos parallelism preset |
| `NPA_COSMOS3_RAY_GUARDRAILS` | `true` | Guardrails remain on unless explicitly disabled |
| `NPA_COSMOS3_RAY_TOKEN` | required | Bearer token for every API route |
| `HF_TOKEN` | required with guardrails | Operator identity used for gated guardrail fetches |

Mount a writable output volume at `/outputs`. For model reuse, apply
`npa/docker/workbench/common/model-weight-cache.yaml` and mount the standard
model-cache claim; NPA's cache plumbing sets the Hugging Face and Cosmos cache
family together. Set `NPA_MODEL_CACHE_PVC` for a Kubernetes-mounted claim or
`NPA_MODEL_CACHE_DIR` for an already-mounted filesystem; `ray-serve` creates and
exports the complete cache-variable family before model initialization. Mount a
memory-backed volume at `/dev/shm` sized for Ray's object store and the selected
batch profile. Runtime weights must never be copied into a derived image.

The service exposes authenticated `GET /health`, model-backed `GET /ready`,
`GET /models`, `GET /system-info`, `POST /v1/batches`, and artifact retrieval at
`GET /v1/artifacts/{path}`.

### Trusted batch callers and conditioning downloads

Treat the bearer token as a credential for trusted workflow pods, not as a
general multi-tenant API key. Each authenticated `/v1/batches` request is passed
to upstream `OmniSampleOverrides.download()`, which can fetch client-selected
conditioning inputs from HTTP(S) or S3 and can use files already mounted in the
container. NPA does not restrict those network destinations or claim an SSRF
defense at this API boundary.

Do not expose the endpoint to untrusted clients. Limit token distribution to the
workflow pods that own the generation queue, isolate the service from unrelated
tenants, and enforce any required destination allowlist with the deployment's
network-egress controls.

## Submit and persist a batch

Prepare an exact JSON object in S3:

```json
{"model":"Cosmos3-Nano","samples":[
  {"name":"cell-a","model_mode":"text2image","prompt":"a robot sorting bins","seed":17},
  {"name":"cell-b","model_mode":"text2image","prompt":"an autonomous forklift","seed":23}
]}
```

Then call:

```bash
npa workbench cosmos3 ray-batch \
  --input-path s3://<bucket>/<prefix>/batch.json \
  --output-path s3://<bucket>/<prefix>/outputs/ \
  --endpoint http://<service>:8000
```

The submitted samples become concurrent deployment-handle calls; upstream Ray
Serve performs the actual batching. NPA publishes the original request,
structured `SampleOutputs`, generated media, hashes, and `provenance.json`. The
equivalent declarative client is
`npa/workflows/workbench/npa-workflows/cosmos3-ray-batch.yaml`.

## GPU compatibility and evidence

B200 (`sm_100`) and RTX PRO 6000 Blackwell (`sm_120`) are separate targets. A
release claim requires the exact image digest to complete a guarded two-sample
generation on each claimed device, not merely import Torch or reach `/health`.
If an upstream kernel cannot execute on a target, NPA must refuse that placement
and document the real limitation rather than falling back to another backend.

Before launch, run `npa workbench health preflight --checks hf,ngc,s3 --json`
and `npa workbench health access --capability cosmos3 --json`. Keep exact cloud
resource and storage identifiers only in access-controlled validation evidence.

The supported public release is `ray1-cu130`. It and the retained immutable
development tag `dev-56d8c4f3f05db7aa3b03323441a3e0d7b97ac8da` resolve to
`sha256:6e42f553a0d14712dc1ed7fa42c72b0f083f4ae3f89b30eaf0e93cfdf64e820d`.
On 2026-08-26, that exact digest completed independent guarded two-sample
text-to-image batches on B200 and RTX PRO 6000. Each run returned two structured
outputs and two decodable images and persisted five S3 objects (request,
response, provenance, and media). B200 artifact hashes were
`3a91a993e19e…` and `8838f93a8831…`; RTX PRO 6000 hashes were
`357ca45a4121…` and `4345aac2743c…`. The measured runtime was Torch
2.10.0+cu130 with native `sm_100` and `sm_120` SASS. Concrete service, cluster,
and storage identifiers remain only in the access-controlled evidence record.
