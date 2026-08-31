---
name: token-factory
description: Use for zero-GPU hosted inference through Nebius Token Factory — captioning, batch text generation, and Cosmos physical-AI reasoning — including key setup, model selection, and the npa.workflow toolRefs that need no cluster.
---

# Token Factory (zero-GPU hosted inference)

Nebius Token Factory is an OpenAI-compatible hosted-inference API for open text
and vision models. It is the cheapest tier in the workbench that produces a real
artifact: **no cluster, no GPU, no provisioning**. Reach for it before standing up
anything, both for real work and to prove a toolchain end to end.

Full reference: `docs/workbench/token-factory.md`. Key setup only:
`docs/workbench/token-factory-key.md`.

## The credential is not a Nebius IAM token

This is the single most common failure. A Token Factory key is a separate
credential minted in the separate Token Factory console
(<https://tokenfactory.nebius.com/>). It is a long opaque token starting with
`v1.`, read from `NEBIUS_TOKEN_FACTORY_KEY` or `~/.npa/credentials.yaml`. Your
`nebius` CLI IAM token returns `403` here — having Nebius CLI access is not
enough, and no amount of re-authenticating the CLI will help.

Keys are shown once at creation. A project with no balance returns `402`/`403` on
inference even with a valid key.

```bash
npa workbench token-factory status    # connection settings, no network call
npa workbench token-factory verify    # live models call; non-zero on auth failure
npa workbench token-factory models    # what this key can actually reach
```

Run `verify` before a batch job and `models` before pinning a model name — model
availability is per-key, so a model in the docs may not be in your project.

Defaults: base URL `https://api.tokenfactory.nebius.com/v1/`, overridable with
`NEBIUS_TOKEN_FACTORY_BASE_URL`. Requests retry on 429 and 5xx.

## Commands

Every command takes local paths or `s3://` URIs for both input and output, and
supports `--dry-run` (compute without writing the artifact) and
`--output text|json`.

**Caption images** — default model `Qwen/Qwen2.5-VL-72B-Instruct`:

```bash
npa workbench token-factory caption \
  --input-path s3://<bucket>/frames/ \
  --output-path s3://<bucket>/captions.json \
  --max-images 50 --max-tokens 512 --temperature 0.2 \
  --instruction "Describe the scene, objects, and any action."
```

**Batch text generation** over a JSONL/text prompt file — default model
`meta-llama/Llama-3.3-70B-Instruct`:

```bash
npa workbench token-factory generate \
  --input-path prompts.jsonl \
  --output-path s3://<bucket>/generations.jsonl \
  --max-prompts 0 --max-tokens 512 --temperature 0.7 \
  --system-prompt "<applied to every request>"
```

`--max-prompts 0` means all of them. Set a small non-zero value first: this is
the command that turns a typo into a large token bill.

**Batch text generation** — same prompt file, same `generations.jsonl`, batch
token rates, default model `openai/gpt-oss-120b`:

```bash
npa workbench token-factory batch-generate \
  --input-path prompts.jsonl \
  --output-path s3://<bucket>/generations.jsonl \
  --model openai/gpt-oss-120b --completion-window 24h

# or submit now, collect later
npa workbench token-factory batch-generate ... --no-wait
npa workbench token-factory batch-status --operation-id <id> --output-path <same> --wait
```

Reach for `batch-generate` over `generate` whenever nothing is waiting on the
answer, which is most bulk stages. Three properties are unique to it, and each
one has already cost real debugging time:

- **Batch routing is a per-model entitlement, unrelated to real-time chat.** Most
  models that serve `generate` are rejected for batch. Measured live across eight
  text models on one key, exactly one — `openai/gpt-oss-120b` — was batch
  routable; `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen3-32B`,
  `Qwen/Qwen3-30B-A3B-Instruct-2507`, `Qwen/Qwen3-235B-A22B-Instruct-2507`,
  `google/gemma-3-27b-it`, `deepseek-ai/DeepSeek-V4-Flash`,
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, and `zai-org/GLM-5.1` were not. That
  is why `DEFAULT_BATCH_MODEL` is `openai/gpt-oss-120b` and not
  `DEFAULT_TEXT_MODEL`. Treat the routable set as per-key and verify on a couple
  of prompts before pointing a large run at a new model.
- **Batch is text-to-text only.** A vision model is rejected at submit with
  `Batch inference is only supported for text2text models`, so there is no batch
  captioning path; use `caption`, which is real-time.
- **The completion window is a deadline, not a latency.** Observed live: batches
  of one and three prompts sat `in_progress` with `completed: 0` for over an hour
  against a 24h window. Do not read a slow batch as a hung one, and never put
  `--wait` on a path that has its own timeout.

Where the failure reason actually lives matters. `GET /operations/{id}/errors`
returns a single empty string for a failed batch — useless. The real per-row
reason is in the batch record's error file
(`GET /batches/{id}` → `error_file_id` → `GET /files/{id}/content`, which
redirects, so redirects must be followed). `batch-generate` reads that file and
reports it, and also surfaces `request_counts` (`total`, `completed`, `failed`,
`invalid`) as the only genuine progress signal a pending batch offers.

**Distinguish a degraded platform from your own bug.** A batch that is accepted,
reports `in_progress` with rows validated (`total: 2, invalid: 0`), and holds
`completed: 0` is usually not your job's fault. Batch execution has been observed
unavailable while submissions were still accepted through the datasets/operations
route. The cheapest tell is `POST /v1/batches`, the OpenAI-compatible submit,
returning `403 Creating new batch job is temporarily unavailable`. Confirm it is a
server-side switch rather than your request by checking where the 403 lands: an
empty body returns `422` naming the missing fields, but a *valid* payload with a
genuinely uploaded `input_file_id` still returns `403`, so the gate sits ahead of
resource validation. Meanwhile the rest of the key stays healthy — real-time chat
on the same model, `POST /v1/files` with `purpose=batch`, `GET /v1/batches`, and
dataset create/delete all succeed — which rules out the key, the balance, the
model, and the payload. When you see this, stop debugging your spec, cancel what
you queued (`POST /batches/{id}/cancel`), and use `generate` until batch recovers.
Do not wait it out: the same 403 was still being returned eight days after it was
first seen, so "temporarily" can outlast any plausible stage timeout. Plan the
run on `generate` and re-probe later rather than leaving a stage parked.

**That 403 is not the quota rejection**, and conflating the two sends you down the
wrong path. The documented limits are 10 active batches per customer and 100
submissions per hour, a batch counts as active only until its processing
finishes, and rate limiting surfaces as `429`. So before blaming a limit, list
your batches (`GET /v1/batches?limit=100` — the default page is 10, which makes a
long history look artificially short) and count the non-terminal ones. All
terminal plus a 403 with no `x-ratelimit-*` headers means availability, not quota.

**Physical-AI reasoning over a scene** — default model
`nvidia/Cosmos3-Super-Reasoner`. Point it at scene images and ask what a robot
should do:

```bash
npa workbench token-factory reason \
  --input-path s3://<bucket>/scene/ \
  --output-path s3://<bucket>/plan.json \
  --task "Describe this scene and give a step-by-step plan of action." \
  --max-images 8 --max-tokens 1024 --temperature 0.2
```

## In workflows

These run as CPU-only `npa.workflow` steps with no accelerator request. The
renderer injects `NEBIUS_TOKEN_FACTORY_KEY` for `workbench.token_factory.*`
steps, so pass it as a secret at submit time and never in the YAML:

```bash
npa workbench workflow submit <spec.yaml> --secret-env NEBIUS_TOKEN_FACTORY_KEY
```

toolRefs: `workbench.token_factory.caption`, `.generate`, `.batch_generate`,
`.reason`, `.triage` (digest a run's textual artifacts into a triage report).

`npa workbench token-factory workflow` prints exactly four:
`token-factory-caption.yaml`, `token-factory-generate.yaml`,
`token-factory-cosmos-reason.yaml`, and `vlm-eval-token-factory.yaml`. Several
more are checked in but not listed by that command, so do not treat its output as
the full inventory:

- `token-factory-batch-generate.yaml` — the batch-inference twin of
  `token-factory-generate.yaml`.
- `token-factory-parallel-fanout.yaml` — parallel batches.
- `token-factory-gate-loop.yaml`, `tokenfactory-cosmos-gate.yaml` — a hosted
  model as a gate that decides whether the pipeline continues.
- `tokenfactory-rollout-judge.yaml`, `tokenfactory-scene-to-rollout-judge.yaml` —
  reason about a scene, then judge a rollout against that plan.
- `tokenfactory-train-triage.yaml` — triage a training run's artifacts.

All live under `npa/workflows/workbench/npa-workflows/`.

## Choosing between Token Factory and VLM eval

They overlap and are easy to confuse. `token-factory reason` **produces** an
analysis or plan. `vlm-eval` **scores** a rollout against a task and emits a
pass/fail gate with a threshold. When you want a judged number for a gate, use
`skills/tools/vlm-eval/SKILL.md` — and note it can consume a Token Factory
reasoning artifact directly through `--task-from`, so a judge scores against a
plan an earlier stage wrote rather than a hardcoded string.

## Gotchas

- **Canonical Sim2Real is scoring, not planning.** Stage 8 uses
  `nvidia/Cosmos3-Super-Reasoner` as its only Stage 8 evaluator, on CPU with no
  self-hosted evaluator image. It sends a bounded, deterministic rollout-wide
  frame sample and requires event-local structured scores. Stage 9 compares the
  single evaluator result with the authoritative Stage 7 rollout set and rejects
  missing, duplicate, or extra evaluations before PPO. Preserve request
  IDs, token usage, latency, retries, and an authoritative returned cost or
  explicit null separately from model-agent tokens.
- **Sim2Real preflight is stronger than model listing.** Its submit and prepared
  action paths declare `NEBIUS_TOKEN_FACTORY_KEY` by name only, then require
  both key-scoped model availability and a minimal inference before provisioning.
- **Model availability is per-key.** Confirm with `models` before pinning a name
  in a spec; a spec that names an unavailable model fails at run time, not at
  validation.
- **`--max-images` and `--max-prompts` are cost controls, not correctness knobs.**
  Defaults are 50 images and unlimited prompts. Always bound the first run.
- **`--dry-run` still calls the model.** It skips writing the artifact, so it is
  not a free syntax check. For a free check, validate the spec instead.
- **Hosted inference is not a rendering or simulation path.** It has no access to
  your cluster, your PVCs, or a GPU; give it S3 or local inputs it can read.
- **The key belongs in credentials, not in a spec or a shell history.** Persist it
  with `npa configure --save-env-credentials` (atomic `0600` write, never
  printed).

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
