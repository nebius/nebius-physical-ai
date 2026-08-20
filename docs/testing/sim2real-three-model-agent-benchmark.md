# Standalone Sim2Real model-agent benchmark

This benchmark compares open-weight coding models as operators of the canonical
Sim2Real workflow without using the NPA chat agent. Each trial starts from a
clean detached checkout of the same `origin/main` commit, receives the same
task, system instructions, tool schema, seed, workflow inputs, permissions, and
machine-verifiable success predicate, and writes its transcript outside Git.

The controller is `python -m npa.benchmarks.sim2real_model_agent`. It talks to
an OpenAI-compatible endpoint, maintains the full message/tool context, and
offers four workspace-scoped tools: shell, file read, file write, and file glob.
It does not import or call `npa.agent_backend` or any `npa agent` command.
Every shell call runs in a fresh user/mount namespace that masks the controller
repository and the entire private benchmark root, then bind-mounts only that
trial's evidence at `NPA_PRIVATE_EVIDENCE`. Other model workspaces, transcripts,
and evidence therefore remain inaccessible even if a model tries to traverse
outside its working directory. `HOME` is also redirected to that trial's private
evidence. Seed each private home with identical minimal NPA, Nebius, Kubernetes,
and SkyPilot configuration before the trial; never share mutable client state.

## Fairness contract

- Use the checked-in `system-prompt.txt` unchanged for every model. Record
  `origin/main`, the model revision, server image digest, serving
  arguments, seed, system-prompt hash, task hash, and tool-schema hash before a
  trial starts.
- Use a separate detached-HEAD checkout, workflow run ID, Kubernetes namespace,
  object prefix, logs, and output directory for every model. Never reuse a
  transcript or modified checkout.
- Run model servers sequentially on the same reserved serving capacity. Stop
  one server and confirm its GPU processes are absent before loading the next.
- Keep workflow GPU routing identical across trials. B200 is for language-model
  serving only; Isaac rendering remains on an RT-core GPU supported by the
  canonical workflow.
- Do not impose model-turn, token, workflow-job, cost, or wall-clock budgets.
  Transport failures are recorded and retried; a model that stops before the
  verifier passes receives the same verifier feedback and continues.

`npa/benchmarks/sim2real-three-model/models.json` records the immutable model
revisions, container-image digests, parser choices, and serving parameters used
by this benchmark. Verify each resolved image digest again in the private live
server receipt. Qwen uses `qwen3_coder`; vLLM exposes gpt-oss Harmony tool calls
through its `openai` parser; GLM uses SGLang's `glm47` tool parser with `glm45`
reasoning parsing. The pinned SGLang/FlashInfer build lacks one GLM-5.2 DSA
TRTLLM-GEN specialization used only by startup autotuning/CUDA-graph capture,
so autotuning plus ordinary and piecewise CUDA graphs are explicitly disabled;
the affected DSA decode path uses SGLang's sparse FlashMLA backend with its FP8
cache-dequantization path. Eager inference still runs the real immutable
checkpoint and avoids a deterministic missing-kernel crash in the pinned
TRTLLM backend.

Render the model-specific Kubernetes resources into private evidence, create the
`model-access` secret without putting its value on the command line, then apply
the manifest to the benchmark-owned B200 cluster:

```bash
npa/.venv/bin/python -m npa.benchmarks.sim2real_model_server \
  --models npa/benchmarks/sim2real-three-model/models.json \
  --model-index 0 --namespace <private-trial-namespace> \
  --service-name benchmark-model > /private/model-server.yaml
```

The renderer requests B200 only through the serving cluster's GPU nodes, pins
the image by digest, keeps a TP16 server on two anti-affined eight-GPU pods, and
routes the OpenAI-compatible Service only to rank zero. Port-forward the Service
to loopback on the dev VM; do not publish the endpoint. Delete the StatefulSet,
verify no serving GPU process remains, and clear the model-specific cache before
starting the next model.

## Success predicate

Pipeline completion or the canonical stable-placement promotion decision is
not sufficient for this task. The independent verifier requires all of:

1. a trained-policy, real-Isaac action rollout;
2. timestamped simulator ground truth showing stable grasp and at least 0.05 m
   object lift continuously for at least 2.0 seconds;
3. an independently decodable, non-empty `reports/sim2real.mcap`; and
4. an independently decodable, non-empty `reports/sim2real.rrd`.

The verifier refuses to infer elapsed time from sample count. If the canonical
artifacts lack physical timestamps, the trial must add honest timestamp or
simulation-step-duration evidence in its isolated workspace and rerun the real
component. It must not lower the 5 cm threshold or substitute a stub.

Run the verifier directly:

```bash
npa/.venv/bin/python -m npa.benchmarks.sim2real_model_agent verify \
  --artifact-root /path/to/downloaded/run
```

## Trial setup

Create each workspace from the recorded base without a branch or commit:

```bash
git clone --no-local <repository> /private/workspaces/trial-name
git -C /private/workspaces/trial-name checkout --detach <origin-main-sha>
git -C /private/workspaces/trial-name status --porcelain
```

Copy `config.example.json` into restrictive external storage and fill only that
private copy with the endpoint, project alias, namespace/object-prefix roles,
workspace, and evidence paths. Credentials stay in the process environment;
never put secret values in the JSON. Point `system_prompt_file` at the same
checked-in prompt in each detached checkout and record the fully resolved
serving configuration. Launch under a dedicated tmux session:

```bash
tmux new-session -d -s sim2real-model-trial \
  'npa/.venv/bin/python -m npa.benchmarks.sim2real_model_agent run --config /private/trial.json'
```

The evidence directory receives `run.json`, append-only `transcript.jsonl`,
per-request telemetry in `requests.jsonl`, and `success.json` only after strict
verification passes. Capture server `/metrics`, GPU samples, provider billing,
workflow status/logs, commands, diffs, run IDs, artifact hashes, and teardown
receipts beside these files. Publish only generic roles, non-identifying
measurements, and hashes.

Before provisioning, run the repository health, gated-access, image-pull, GPU,
and third-party EULA preflights. Teardown is cancel-before-destroy and applies
only to resources whose creation receipt belongs to this benchmark.
