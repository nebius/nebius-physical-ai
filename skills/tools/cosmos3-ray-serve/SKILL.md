---
name: cosmos3-ray-serve
description: Operate, validate, or troubleshoot persistent Cosmos3-Nano generation through NVIDIA Cosmos Framework's native Ray Serve implementation, including authenticated readiness, dynamic request batching, B200 versus RTX PRO 6000 placement, runtime-fetched weights/guardrails, and durable S3 batch outputs.
---

# Cosmos3 Native Ray Serve

Use this service for repeated or batched synthetic-data generation where loading
Cosmos3-Nano once is materially better than starting one `cosmos3 generate` job
per sample. Do not substitute `npa-cosmos3-serving`: that image serves
Cosmos3-Super through vLLM-Omni and has a different model/API/runtime contract.

## Non-negotiable contract

- Run NVIDIA cosmos-framework at pinned commit
  `5e67049cd94acb667786f1e6dd0dab821cb90c97`.
- Bind upstream `OmniModelDeployment`; its `@ray.serve.batch` method must own
  coalescing and call `OmniInference.generate_batch`.
- Keep guardrails on unless the operator explicitly opts out.
- Fetch Cosmos3-Nano, VAE, and guardrail weights only at runtime with the
  operator's access. Use the standard NPA model-cache mount; never bake caches.
- Require `NPA_COSMOS3_RAY_TOKEN` for every API endpoint.
- Move batch inputs and outputs through S3. Never transfer data directly from a
  sibling workbench service.

## Preflight

Before provisioning or starting GPUs:

```bash
npa/.venv/bin/npa workbench health preflight --checks hf,ngc,s3 --json
npa/.venv/bin/npa workbench health access --capability cosmos3 --json
npa/.venv/bin/npa workbench golden-eval show cosmos3-ray-serve
```

Treat S3 failure, missing `Cosmos-Guardrail1` access, or an unpullable exact
image digest as a stop condition. A token's presence is not model entitlement.

## Start the service

Run the image by immutable digest, mount `/outputs` and the standard model cache,
and inject `HF_TOKEN` and `NPA_COSMOS3_RAY_TOKEN` as runtime secrets. The image's
default entrypoint starts:

```text
npa workbench cosmos3 ray-serve --world-size 1 --max-batch-size 4
```

Configuration is explicit: `--world-size` sets GPUs per replica;
`--max-batch-size` and `--batch-wait-timeout-s` are upstream batching knobs;
`--parallelism-preset` is the Cosmos placement preset; and
`--guardrails/--no-guardrails` is the explicit safety posture.

Use authenticated `GET /ready`, and verify that the exact image checks the native
Serve application and model replicas. Earlier implementations returned HTTP 200
while weights were still loading. Require the selected application to be
`RUNNING`, its model deployment `HEALTHY`, and a running replica before inference.
Retain checkpoint revision and payload verification separately; cache-resolution
events or filenames alone do not prove that every required payload is complete.
`GET /health` establishes liveness. Report configured guardrails separately from
evidence that the selected model actually applied them.

## Submit a durable batch

The input is JSON with one or more upstream `OmniSampleOverrides` objects:

```json
{"model":"Cosmos3-Nano","samples":[
  {"name":"sample-a","model_mode":"text2image","prompt":"a robot workcell","seed":17},
  {"name":"sample-b","model_mode":"text2image","prompt":"a warehouse aisle","seed":23}
]}
```

Submit it with the CPU client:

```bash
npa/.venv/bin/npa workbench cosmos3 ray-batch \
  --input-path s3://<bucket>/<prefix>/batch.json \
  --output-path s3://<bucket>/<prefix>/outputs/ \
  --endpoint http://<service>:8000
```

The client sends all samples concurrently so upstream Ray Serve can coalesce
them. It downloads each returned file, verifies bytes and SHA-256, and publishes
`request.json`, `response.json`, media under `artifacts/`, and
`provenance.json` (`npa.cosmos3.ray-serve.provenance.v1`). Use
`npa/workflows/workbench/npa-workflows/cosmos3-ray-batch.yaml` for the workflow
client; the persistent service must already be ready.

Preserve the pinned framework's sampling types. Its aspect ratios include
comma-delimited strings such as `"1,1"`, and resolution is a string enum such as
`"720"`; the latter does not specify a square image's measured pixel dimensions.
Validate against the [upstream sampling definitions](https://github.com/NVIDIA/cosmos-framework/blob/5e67049cd94acb667786f1e6dd0dab821cb90c97/cosmos_framework/inference/args.py)
and decode the actual media instead of inventing a numeric-string regex.

Measure request coalescing and model inference batches separately. Native worker
batch events can contain several requests while inference events remain size one.
State whether throughput includes initialization, cache verification and artifact
delivery; a client-request timing alone does not measure those phases.

If client validation fails after inference, preserve its failed result and inspect
the original native output directory before submitting again. Bind retained
`sample_args.json`, `sample_outputs.json` and media to the original request,
sample names, seeds and file hashes. Label any recovered qualification as derived
from those files; do not invent a missing HTTP response or change the original
client outcome. Generate a reviewable `.rrd` and contact sheet from qualified
outputs, retaining provenance and visible prompt-fidelity defects.

## GPU validation

Treat B200 (`sm_100`) and RTX PRO 6000 Blackwell (`sm_120`) as independent
targets. For each exact development digest, require image scans and anonymous
pull, `/system-info` on the intended device, guarded model readiness, a two-sample
batch producing structured outputs and two decodable artifacts, and S3
request/output/provenance persistence.

An import, server boot, `/health`, or CUDA probe alone is not acceptance. If an
upstream kernel rejects one compute capability, retain the failure and mark that
target unsupported; do not route around it with vLLM-Omni.

## Teardown and evidence

Delete only the validation service after client runs are terminal. Cancel
managed jobs before removing clusters or shared controllers. Preserve exact
operational identifiers only in access-controlled evidence; commits and PR prose
may record GPU family/count, hashes, timings, and image digests but never tenant,
project, cluster, bucket, endpoint, or node identifiers.
