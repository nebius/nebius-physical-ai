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
| `NPA_COSMOS3_RAY_MAX_BATCH_SIZE` | `4` | Maximum samples coalesced by upstream Ray Serve; also sets the model replica's admission capacity so the configured batch can form |
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

Before starting Ray, guarded serving prepares the pinned Guardrail1 tokenizer
subtree through the shared Cosmos materializer. It verifies the revision-scoped
cache and gives NLTK a regular-file data directory through `NLTK_DATA`. Hub
snapshot symlinks are incompatible with NLTK 3.10.3's enforced path checks;
the materializer preserves those checks and does not alter the upstream snapshot.
Missing access or an invalid cache prevents service startup. These assets are
fetched at runtime and must never be baked into the image.

The Ray batch toolRef uses the `cosmos3` access capability, matching the native
server's pinned `Cosmos-Guardrail1` payload. Workflow execution requires that
exact access probe to pass before starting the client.

The service exposes authenticated `GET /health`, model-backed `GET /ready`,
`GET /models`, `GET /system-info`, `POST /v1/batches`, and artifact retrieval at
`GET /v1/artifacts/{path}`. Readiness reports both `max_batch_size` and
`model_max_ongoing_requests`; they must match for this single-concurrent-batch
deployment.

### Trusted batch callers and conditioning downloads

Treat the bearer token as a credential for trusted workflow pods, not as a
general multi-tenant API key. Current source authorizes every conditioning input
against `NPA_COSMOS3_RAY_ALLOWED_S3_ROOTS` before staging it into a fresh request
directory. HTTP(S), server-local paths, custom defaults files, and decoded S3
keys that the storage parser would reinterpret (such as encoded query/fragment
characters or tabs/newlines) are rejected. The updated client checks that same
S3 key contract before submitting inference. These protections require the
current server source; the retained accepted image alone predates them.

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

Run the updated client from an editable installation of this checkout:

```bash
npa/.venv/bin/python -m npa workbench cosmos3 ray-batch \
  --input-path s3://<bucket>/<prefix>/batch.json \
  --output-path s3://<bucket>/<prefix>/outputs/ \
  --endpoint http://<service>:8000
```

The submitted samples become concurrent deployment-handle calls; upstream Ray
Serve performs the actual batching. NPA publishes the original request,
structured `SampleOutputs`, generated media, hashes, and `provenance.json`.

The declarative example is `workflows/testing/cosmos3-ray-batch.yaml`. Its default
catalog image is the accepted service image, which still bundles the older
client, and the example does not enable a source overlay. The integrity checks
below apply to the updated installed client shown above; the unchanged workflow
default does not provide them. Updating a host's NPA installation does not update
the client inside that image. The accepted service digest remains wire-compatible
with the updated client. It does not acquire the current source's Ray management
authentication, scoped S3 input staging, or Ray 2.58/Torch 2.13 runtime changes.
Those require validation of the actual service runtime; the retained two-image
proof applies only to its original immutable image.

The client assigns a request ID before submission when the input omits one and
persists that ID in `request.json`. It requires the supported response schema,
the supported framework revision, the requested model and ID, and one successful
result for every requested sample name. It also binds the sampling mode, supplied
seed and requested image/video frame category. Failed, skipped, duplicate,
foreign or incomplete results fail before any artifact download or publication.
Frame category includes the pinned mode defaults and single-WSM transfer default.
For implicit modes, S3 conditioning filenames use the same decoded object-key
parser as server input staging; the original URI remains in the submitted request.
Unrecognized implicit conditioning extensions fail before submission.
The CPU client mirrors only the pinned framework's mode and frame-category
defaults without importing its GPU runtime. Custom service defaults at the same
framework revision are outside this contract; provide explicit `model_mode` and
`num_frames` instead. A different framework revision is rejected until its
contract has been reviewed and validated.
Inline sample overrides instead of using `defaults_file`: server-local defaults
can introduce output-affecting settings that the client cannot independently bind.
Each declared file must have exactly one
matching artifact entry under that request and sample's directory; downloaded
bytes must then match its size and SHA-256.

File coverage follows the pinned framework's
[output writer](https://github.com/NVIDIA/cosmos-framework/blob/5e67049cd94acb667786f1e6dd0dab821cb90c97/cosmos_framework/inference/inference.py):
ordinary samples require `vision.jpg` for one resolved frame or `vision.mp4` for
multiple frames, while reasoner samples without transfer hints require
`reasoner_text.txt`. Transfer samples also require `control_<hint>.jpg` or `.mp4`
for every requested non-null `edge`, `blur`, `depth`, `seg` or `wsm` hint. The
returned hint set must match the request. Declared debug files are preserved and
verified too. The native Serve path
requires `num_outputs=1` per named sample; request several named samples for
several outputs. Use a new request ID for new inference; the service reserves
each request directory once.

Run the committed live client check against the current guarded service source
with an operator-owned S3 output prefix. Configure the service's
`NPA_COSMOS3_RAY_ALLOWED_S3_ROOTS` and storage credentials to read that prefix:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_COSMOS3_RAY_LIVE_OUTPUT_URI=s3://<bucket>/<prefix>/ \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_cosmos3_ray_batch_live_e2e.py -q
```

It uses the configured endpoint and token, generates one image from text and
one from synthetic conditioning pixels with an encoded S3 filename, reads back
the published records, decodes both images, and rejects a copy of
the real response with a missing artifact. The operator owns service and
storage cleanup after retaining the validation outputs.

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
