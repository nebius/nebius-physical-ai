# Standalone Sim2Real model-agent benchmark

This benchmark compares open-weight coding models as operators of the canonical
Sim2Real workflow without using the NPA chat agent. Each trial starts from a
clean detached checkout of the same `origin/main` commit, receives the same
task, system instructions, tool schema, seed, workflow inputs, permissions, and
machine-verifiable success predicate, and writes its transcript outside Git.

The controller is `python -m npa.benchmarks.sim2real_model_agent`. It talks to
an OpenAI-compatible endpoint, maintains the full message/tool context, and
offers workspace-scoped shell, file read, file write, and file glob tools, plus
typed controller actions described below.
It does not import or call `npa.agent_backend` or any `npa agent` command.
Every shell call runs in a fresh user/mount namespace that masks the controller
repository and the entire private benchmark root, then bind-mounts only that
trial's evidence at `NPA_PRIVATE_EVIDENCE`. Other model workspaces, transcripts,
and evidence therefore remain inaccessible even if a model tries to traverse
outside its working directory. `HOME` is also redirected to that trial's private
evidence. Seed each private home with identical minimal NPA, Nebius, Kubernetes,
and SkyPilot configuration before the trial; never share mutable client state.

Run each trial's SkyPilot API server outside the tool-call PID namespace, on a
loopback-only endpoint with a restrictive trial-scoped HOME. Put that endpoint
in the trial's `SKYPILOT_API_SERVER_ENDPOINT` environment. The server HOME must
contain the same minimal Kubernetes/Nebius authentication material as the trial,
and its Kubernetes jobs-controller resources must omit `disk_size`. Before the
model starts, require `npa skypilot verify` plus workflow GPU discovery to
complete through that endpoint; `/api/health` alone does not prove executor
workers are functional. The model must not manage this benchmark-owned server.

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
  Per-response semantic-progress safeguards may discard a malformed or stalled
  response that has not completed a usable tool-call boundary; these are
  recovery boundaries, not overall benchmark limits. Transport failures are
  recorded and retried; a model that stops before the verifier passes receives
  the same verifier feedback and continues.

`npa/benchmarks/sim2real-three-model/models.json` records the immutable model
revisions, container-image digests, parser choices, and serving parameters used
by this benchmark. Verify each resolved image digest again in the private live
server receipt. Qwen uses `qwen3_coder`; vLLM exposes gpt-oss Harmony tool calls
through its `openai` parser; GLM uses SGLang's `glm47` tool parser with `glm45`
reasoning parsing. GLM is pinned to the official SGLang v0.5.17 CUDA 13.0 image,
whose release includes GLM-5.2 support on B200. SGLang selects its validated
FlashMLA sparse prefill and KV decode with BF16 KV; the 262k serving window
retains ample benchmark context beside the exact BF16 checkpoint. FlashInfer
all-reduce fusion is force-disabled for a two-node serving fabric without
MNNVL, leaving the validated NCCL/RDMA path to carry tensor-parallel
collectives.

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

The controller is restart-safe. Repeating the same command with unchanged
metadata resumes from the existing transcript and request sequence; a mismatch
in the recorded model, prompt, tool, or serving configuration is rejected.
The first launch requires a clean detached checkout. A restart requires the
same detached commit but preserves model-authored changes, recording a hash and
line count of the workspace status in private `resumes.jsonl` evidence.
When prompt telemetry reaches 85% of the configured context window, the
controller appends a deterministic, hash-linked checkpoint and continues from
that checkpoint plus a bounded verbatim tail. Compaction retains only complete
assistant/tool-result groups, the original task, current workspace-status
evidence, and discovered workflow run identifiers. A malformed or stalled
stream is closed when it crosses its recorded no-tool-progress, incomplete-tool
assembly, or idle boundary. The partial assistant message is never put in
history and no unvalidated arguments execute; recovery starts from the same safe
checkpoint mechanism. Each interruption records request index, elapsed time,
observed character and token-event lower bounds, reason, fingerprint, and action
in append-only `requests.jsonl`. Three identical consecutive malformations
terminate with `repeated_identical_malformed_response` in machine-readable
`failure.json`; any complete response resets that streak. These per-response
guards do not impose an overall completion, time, token, cost, or job limit.

The complete append-only transcript remains in private evidence. A checkpoint
is continuity evidence, not success evidence: it explicitly requires the model
to re-read durable state and forbids inferring success. Transport failures use
capped exponential retry backoff.

The evidence directory receives `run.json`, append-only `transcript.jsonl`,
per-request telemetry in `requests.jsonl`, and `success.json` only after strict
verification passes. Capture server `/metrics`, GPU samples, provider billing,
workflow status/logs, commands, diffs, run IDs, artifact hashes, and teardown
receipts beside these files. Publish only generic roles, non-identifying
measurements, and hashes.

## Prepared irreversible actions

Generic shell access remains available for diagnosis, targeted inspection, and
non-mutating preflights. For an irreversible workflow transition, prefer the
`submit_prepared_workflow` tool. It accepts only an operator-advertised
`action_id`; the model does not repeat private values or reconstruct shell
syntax.

An operator creates the action with the controller's `prepare-action`
subcommand from a mode-0600 request stored in an operator-only control directory
under the private root. That directory must be outside the trial evidence path
mounted into generic shell calls:

```bash
npa/.venv/bin/python -m npa.benchmarks.sim2real_model_agent prepare-action \
  --request /private/prepared-action-request.json \
  --output /private/prepared-action-receipt.json
```

The prepared tool is advertised only when such a receipt is configured; the
schema is identical for GLM, Qwen, and GPT-OSS trials. The closed receipt schema
binds the canonical spec path and digest, source and
benchmark-base commits, exact detached workspace state (including content hashes
for every tracked or untracked change), run identity,
project selection, staged-input manifest and identity, five immutable component
image digests, runtime/resume/no-deadline policy, scoped Isaac EULA acceptance,
required secret environment names (never values), the paths and hashes of eight
completed, passing preflight receipts, and the exact argv-array digest. The controller requires the
receipt to be an owner-owned regular file with mode 0600 under the configured
private root and outside the agent-visible evidence mount. It reopens, rehashes,
and verifies the pass state of every preflight immediately before execution.

Execution uses a closed option grammar and the receipt's argv array without
interpreting model-authored shell text. Help, version, plan-only, opt-out, and
unknown flags fail closed. The controller writes and fsyncs an occurrence-unique intent to the existing
tool WAL, then an append-only `execution_started` record before spawning NPA.
A restart may recover a durably finished result, may retry an occurrence proven
not to have started, and must stop on an indeterminate started occurrence. A
finished action cannot be submitted again, including through generic shell after
a WAL loss. Exit zero alone is insufficient: the bounded JSON result must
authoritatively identify the receipt-bound run. Conversely, an authoritative
receipt-bound run remains an accepted submission when monitoring returns a
nonzero exit for a terminal workflow failure; acceptance and terminal outcome
are reported separately. Existing NPA workflow launch
transactions, runtime resume, and durable submission records remain the source
of orchestration truth; the controller does not duplicate them.

The model receives only a concise result: accepted/rejected, safe run reference,
status, duration, and a typed error when applicable. Full stdout/stderr stays in
operator-control append-only `prepared-action-output.jsonl`; the result carries its
SHA-256 plus a bounded safe view. Safe context checkpoints state completed
preflights, the current blocker, durable submitted state, and the exact typed
action available. Adding the tool to an in-progress benchmark is recorded in
append-only `interventions.jsonl` and does not change the task or claim success.

For the narrower submission/monitoring benchmark, set `completion_mode` to
`workflow_terminal` and provide the identical shortened `task_text` in every
private trial config. The model finishes through the `complete_workflow` tool:
the controller queries authoritative NPA durable status using the configured
project and accepts only a terminal workflow state. A model-authored status file
or prose claim is never sufficient. The resulting `success.json` records both
terminal completion and whether the workflow itself succeeded, so operational
completion is not confused with policy efficacy.

Before provisioning, run the repository health, gated-access, image-pull, GPU,
and third-party EULA preflights. Teardown is cancel-before-destroy and applies
only to resources whose creation receipt belongs to this benchmark.
