# VLM-Eval Loop Runbook

[Cookbooks](README.md)

This runbook runs the sim-to-real VLM-eval loop on the self-hosted serving path:
serve a VLM with vLLM, score rollout directories with `vlm-eval`, and write a
task-success report.

Each evaluation records the requested `model` and the endpoint's returned
`served_model`. Self-hosted servers that omit the model identity leave
`served_model` null; NPA does not infer it from the requested alias. A supplied
identity must be a nonempty string. Retain the serving deployment's checkpoint
revision separately: a model name alone does not identify its weight bytes.

Successful real-backend results also contain an `evidence` record. It binds the
requested and returned model to:

- SHA-256 hashes, dimensions, media types, byte counts, and source-relative
  labels for the exact normalized image bytes submitted to the model;
- hashes of the prompt, rubric, and secret-free request manifest;
- request time, endpoint role, HTTP status when available, latency, finish
  reason, provider request ID and usage when returned;
- the exact provider response body, its SHA-256 hash, and parser version.

The request manifest intentionally excludes authorization, endpoint addresses,
input/output locations, prompts, and base64 image data. It contains enough
information to recompute what was submitted without copying pixels or secrets.
The full result still belongs in private run storage because the provider
response and existing task fields can describe operator data.

`stub` results and `--score` overrides do not contain provider evidence. They are
wiring checks, never visual proof. Real `benchmark` reports retain the evidence
for every case so calibration failures and model disagreement remain inspectable.
If a model wraps one complete JSON object in a single Markdown JSON fence, the
parser removes only that transport wrapper and appends `+markdown-fence-v1` to
the retained parser version. Prefixes, suffixes, duplicate keys, non-finite
numbers, invalid types, and partial output still fail rather than being repaired.
None of these fields turns a visual judgment into objective task, geometry,
collision, or safety evidence.

To verify this against your existing GPU endpoint, set
`NPA_INTEGRATION_E2E=1` and point `NPA_VLM_PROVENANCE_LIVE_CONFIG` at a private
JSON file containing `input_path`, `output_path` (a local JSON filename),
`endpoint_url`, `model`, `expected_served_model`, and `task`. Supply credentials
through the environment variable named by `api_key_env` (default
`VLM_EVAL_API_KEY`). Run
`npa/.venv/bin/python -m pytest npa/tests/e2e/test_vlm_served_model_live.py -q`.
The test calls the real endpoint and retains its verdict; it provisions and
destroys no resources.

## Prerequisites

- `npa` is installed from this repository.
- SkyPilot is configured and `sky check` shows Nebius enabled.
- A GPU with enough memory for the chosen VLM. No prebuilt serving image is
  required: the renderer installs vLLM (and `ninja`, which its JIT sampler needs)
  into whatever image the stage runs in.
- Object storage credentials are available through `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `AWS_ENDPOINT_URL`.
- Rollouts are available under one prefix, with one child directory per rollout.

## One Command

The loop is an `npa.workflow` spec, so a submit plus config overrides is the whole
invocation — no YAML rendering step:

```bash
export RUN_ID="vlm-eval-loop-smoke"
export NPA_S3_BUCKET="<your-bucket-name>"

npa workbench workflow submit \
  workflows/testing/vlm-eval-loop.yaml \
  --run-id "${RUN_ID}" \
  --var "bucket=${NPA_S3_BUCKET}" \
  --var "prefix=sim-to-real/${RUN_ID}" \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

That reads rollouts from `s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/rollouts/` and
writes to `.../scores/`; override `rollouts_uri` / `scores_uri` with `--var` to point
elsewhere. The stage runs on SkyPilot's default GPU image: the renderer installs vLLM,
starts it, health-checks `/health`, and tears it down when the stage exits, so no
prebuilt serving image is required. Set `--var vlm_model=<repo-id>` for a different VLM
and `--var vlm_serve_ready_seconds=<n>` if a cold checkpoint download needs longer than
the 900 s default.

The default model is `Qwen/Qwen2-VL-7B-Instruct`, the default frame selection is
`keyframes`, and the default success threshold is `0.8`.

To score a *single* rollout instead of a set, use
`workflows/testing/vlm-eval-single.yaml`, or call
`npa workbench vlm-eval run` directly.

## Inputs

`rollouts_uri` points to a local path or `s3://` prefix. The loop treats each direct
child directory as one rollout, and falls back to treating the prefix itself as a
single rollout when it has no child directories:

```text
rollouts/
  rollout-000/
    frame-000.png
    frame-001.png
    manifest.json
  rollout-001/
    frame-000.png
    frame-001.png
```

Each rollout can contain image files, RGB `.npy` or `.npz` arrays, or a video
file supported by the `vlm-eval` frame loader. If the task text is not supplied,
`vlm-eval` looks for it in `manifest.json`, `info.json`, or task metadata.

## Outputs

`scores_uri` receives:

- `rollouts/<rollout-id>/vlm_eval_stub.json`: one structured result per rollout.
- `task_success_report.json`: aggregate report with `total_rollouts`,
  `passed_rollouts`, `success_rate`, `mean_score`, `task_success`, and the
  per-rollout `{success, score, rationale}` records.

Read the report:

```bash
aws s3 cp "s3://${NPA_S3_BUCKET}/sim-to-real/${RUN_ID}/scores/task_success_report.json" -
```

Use `task_success` as the coarse gate, then inspect low-score rollouts and their
rationales before iterating on policy, simulation, or rubric.

## Plug In Real Labeled Rollouts

For unlabeled gating, point `rollouts_uri` at the rollout prefix and keep the loop
spec unchanged. For labeled calibration, create a benchmark manifest that
points at the same rollout directories and includes `expected_label` for each
item, then run the sweep below.

## Tune

Use neutral identify-then-judge task text. Add a source-matched missing-terminal
control that retains plausible progress but omits the requested outcome, an
ambiguous terminal control, and blank evidence. The default rubric instructs the
judge to assign `0.0` for missing or ambiguous terminal evidence with no
partial-progress credit. This is a prompt instruction, not an independent visual
validator. The gate still uses `score >= success_threshold`, so select a positive,
calibrated threshold. Custom rubrics replace the default instruction. Selected
stills cannot prove hidden state, continuous execution, or safety.

Sweep thresholds, rubrics, and models against labeled rollouts:

```bash
npa workbench vlm-eval benchmark \
  --dataset s3://${NPA_S3_BUCKET}/vlm-eval/benchmark/benchmark.json \
  --output s3://${NPA_S3_BUCKET}/vlm-eval/benchmark/results/ \
  --backend self-hosted \
  --endpoint-url http://127.0.0.1:8000/v1 \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --rubrics default,strict \
  --thresholds 0.5,0.8,0.9 \
  --format json
```

Use the best threshold and rubric from the benchmark report to update
`vlm_success_threshold` in the loop spec (or pass `--var` at submit time).
`workflows/testing/vlm-eval-benchmark.yaml` runs the same sweep as
a workflow stage.

The packaged benchmark includes `progress-without-terminal-fail`; its tiny
synthetic frames and prerecorded score test wiring only. Validate the exact
task, rubric, threshold, sampling and model on reviewed real inputs.

## Terminal-evidence live check

This hosted lane tests the default rubric on operator-reviewed real rollout
frames. It needs no local GPU or cloud resource creation. Stage four private
image directories before calling the model: a complete real sequence, a strict
prefix of that sequence ending during plausible progress, a real ambiguous
terminal sequence (for example, an occluded final placement), and repeated
blank frames. Use neutral filenames and identical task text. Review the actual
pixels and freeze labels before inference; the harness verifies byte bindings,
not the truth of the operator's visual labels.

Set `NPA_VLM_TERMINAL_LIVE_CONFIG` to an owner-only (`0600`) JSON file outside
Git with these fields:

| Field | Required value |
| --- | --- |
| `model` | Exact hosted vision model ID |
| `task` | One neutral identify-then-judge task for all controls |
| `max_frames` | Integer at least as large as the largest staged frame set |
| `success_threshold` | Frozen positive threshold no greater than 1 |
| `output_dir` | New private local directory; existing paths are refused |
| `cases` | Objects named `complete`, `missing-terminal`, `ambiguous-terminal`, and `no-evidence` |

Each case contains only `input_path` (its local image directory) and
`frame_sha256` (ordered SHA-256 digests of normalized submitted PNG bytes).
Compute the digests using `select_rollout_frames` with `frame_selection="sequence"`
and the frozen `max_frames`, then review and preserve that config. Every staged
frame must be selected. The truncated control must be a shorter byte-identical
prefix of the complete one. Frame counts and sampling differ between those
controls; this is a semantic regression check, not a one-factor experiment.
Credentials resolve through the existing environment or private NPA credential
store, never through the test JSON.

From the repository root, after selecting task-scoped private NPA configuration:

```bash
: "${NPA_VLM_TERMINAL_LIVE_CONFIG:?Set the private frozen control configuration}"
npa/.venv/bin/python -m npa workbench health preflight --checks token_factory --json
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_vlm_terminal_evidence_live.py -q
```

Keep logs private. When all four calls parse successfully, their real responses
are retained before checking visual expectations. Model substitution, changed
frame bytes, a false pass, or any nonzero negative-control score fails the lane.
Transport or parser failures
also fail and stop the lane. They have no result artifact; earlier successful
results remain on disk. These failures are not completed visual judgments. Do
not retry until a failure passes or edit the frozen threshold to fit results. Independent review
must still compare each rationale with the retained pixels: a correct label
with an invented visual explanation is not accepted evidence. This test is
separate from the nightly hosted smoke suite because it needs reviewed private
rollout inputs. A skipped or unconfigured run does not validate the rubric.
Retain the config, responses, and reviewed inputs; this lane creates no compute
resources to tear down. Four controls cannot qualify a model or estimate error
rates. Historical measurements in the [earlier control
review](../evidence/vlm-missing-terminal-control-review.md) do not validate a
changed rubric or current commit.

## Troubleshooting

- `sky check` does not show Nebius enabled: fix SkyPilot credentials before
  launching the workflow.
- vLLM never becomes healthy: the stage fails fast and prints the last 200 lines of
  `/tmp/npa-vlm-server.log`, which is where the cause almost always is (out of GPU
  memory, an unsupported model, or a missing CUDA toolchain).
- No rollouts are evaluated: confirm `rollouts_uri` points to a prefix with child
  rollout directories or directly to one rollout directory.
- Scores are all low or noisy: tighten `RUBRIC`, switch `FRAME_SELECTION`, or run
  `vlm-eval benchmark` on labeled rollouts before using the gate.
- S3 writes fail: verify `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud`
  and that the storage keys can read `rollouts_uri` and write `scores_uri`.
