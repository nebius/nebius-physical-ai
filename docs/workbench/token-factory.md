# Nebius Token Factory integration

Token Factory provides hosted text generation, image captioning, and scene
reasoning. Use these capabilities to annotate inputs or interpret results from
your Nebius GPU workloads. Direct CLI calls run from your machine; the
checked-in NPA workflows run their calling stages on Kubernetes CPUs.

## Configure and verify the key

Complete the [installation](../install.md), then create a key in the
[Token Factory console](https://tokenfactory.nebius.com/) and enter it through
`npa configure`. The key is separate from your Nebius Cloud IAM token. See
[key setup](token-factory-key.md) for credential-file and environment options.

```bash
npa workbench token-factory status
npa workbench token-factory verify
npa workbench token-factory models
```

`status` checks local settings. `verify` makes a live model-list request and
reports `authenticated` and `model_count`; it does not test inference or
billing access. Select compatible text and vision models from your key's
catalog. The implementation defaults below are not guaranteed to be available:

| Command | Default model |
| --- | --- |
| `generate` | `meta-llama/Llama-3.3-70B-Instruct` |
| `caption` | `Qwen/Qwen2.5-VL-72B-Instruct` |
| `reason` | `nvidia/Cosmos3-Super-Reasoner` |
| `batch-generate` | `openai/gpt-oss-120b` |

NPA reads `NEBIUS_TOKEN_FACTORY_KEY` from the environment or
`tokens.NEBIUS_TOKEN_FACTORY_KEY` in `~/.npa/credentials.yaml`. Keep the file
mode `0600`. The default API URL is
`https://api.tokenfactory.nebius.com/v1/`; use
`NEBIUS_TOKEN_FACTORY_BASE_URL` only when selecting another endpoint deliberately.

## Generate and inspect artifacts

These commands accept local paths or `s3://` URIs. Direct S3 calls require
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and the correct `AWS_ENDPOINT_URL`
in the process environment; their storage client does not load those values
from the NPA credential file. Supply them through your protected credential
loader. Workflow submission resolves named secrets from the selected project's
credentials instead. Use inputs you are authorized to send to the hosted model.

Set the model names selected above, then create a prompt file:

```bash
text_model="<available-text-model>"
vision_model="<available-vision-model>"
reason_model="<available-vision-reasoning-model>"
mkdir -p ./token-factory-results
cat > ./token-factory-results/prompts.jsonl <<'PROMPTS'
{"id":"pick-place","prompt":"Write a robot task instruction for moving a red block into a tray."}
{"id":"inspection","prompt":"List observable success criteria for a robot moving a red block into a tray."}
PROMPTS

npa workbench token-factory generate \
  --input-path ./token-factory-results/prompts.jsonl \
  --output-path ./token-factory-results/generations.jsonl \
  --model "$text_model" --output json
```

Each prompt is an independent request. A `.txt` file with one prompt per line
also works. Open `generations.jsonl`: it should contain one row per input with
`id`, `prompt`, and a nonempty `completion`. `generate` processes all prompts by
default.

For images, place your JPEG, PNG, WebP, BMP, or PPM files under `./frames`:

```bash
npa workbench token-factory caption \
  --input-path ./frames \
  --output-path ./token-factory-results/captions.json \
  --model "$vision_model" --output json

npa workbench token-factory reason \
  --input-path ./frames \
  --output-path ./token-factory-results/scene_reasoning.json \
  --task "Describe the scene and the steps a robot would need to move the red block into the tray." \
  --model "$reason_model" --output json
```

Captioning sends one request per image. Reasoning sends the selected images
together with the task. Neither command reads video files directly; extract
frames first. Check saved results against the source images:

| Artifact | Check |
| --- | --- |
| `captions.json` | `image_count` matches the processed inputs; every `captions` entry names an image and has a useful `caption`. |
| `scene_reasoning.json` | `images` names the intended inputs and `analysis` addresses the supplied `task`. |

The default image limits are 50 for `caption` and 8 for `reason`. Set
`--max-images` to your intended input count when processing more. A directory
output appends the filenames shown above. Use an explicit `.json` filename
for caption/reason results or `.jsonl` for generations. `--output json` formats status on stdout;
`--output-path` controls the saved artifact.

**`--dry-run` still calls the model.** It only skips saving the result. Use
`workflow validate-spec` or `plan-spec` for static workflow checks.

Scene reasoning produces a proposed plan. To score observed rollout frames
against a task, use `npa workbench vlm-eval run --backend api`; see the
[rollout cookbook](cookbooks/tokenfactory-compute-combos.md). A generated plan
does not establish task success.

## Batch generation

`batch-generate` submits text prompts asynchronously and writes the same JSONL
row format as `generate`. Batch access is a separate model entitlement, so a
model working for direct generation may still be rejected for batch.

```bash
npa workbench token-factory batch-generate \
  --input-path ./token-factory-results/prompts.jsonl \
  --output-path ./token-factory-results/batch/generations.jsonl \
  --model "<batch-enabled-text-model>" --no-wait --output json

npa workbench token-factory batch-status \
  --operation-id "<operation-id-from-submit>" \
  --output-path ./token-factory-results/batch/generations.jsonl \
  --wait --output json
```

Submission writes a `batch_operation.json` handle beside the eventual output.
Without `--no-wait`, submission waits for collection; `batch-status` without
`--wait` reports the current state. Pending is not completion. Check
`request_counts` and the collected `failures`, then inspect saved rows.
The completion window is a service deadline, not expected response latency.
Batch does not provide image captioning.

Request and response datasets are deleted after results are collected unless
`--keep-datasets` is set. A pending operation still needs its request dataset.
If the service rejects or cannot execute batches, follow its reported error and
cancel your pending batch through the provider before switching to `generate`.

## Run a workflow on Nebius

The files under `workflows/` are
`npa.workflow/v0.0.1` specs. Submit them through NPA, which renders SkyPilot
tasks. Passing these files directly to `sky jobs launch` does not work.

Complete [Workbench setup](getting-started.md), including the selected
Kubernetes context, writable S3 storage, and `npa skypilot bootstrap`. Upload
the prompt file to the S3 URI you supply below. Replace the placeholders and use
the same config overrides for planning and execution:

```bash
spec=workflows/testing/token-factory-generate.yaml
bucket="<your-bucket>"
run_id="<unique-run-id>"
prompts_uri="s3://${bucket}/inputs/prompts.jsonl"

npa workbench workflow validate-spec "$spec"
npa workbench workflow plan-spec "$spec" --run-id "$run_id" \
  --var "bucket=${bucket}" --var "prompts_uri=${prompts_uri}" \
  --var "generate_model=${text_model}" --json

npa workbench workflow submit "$spec" \
  --project "<project-alias>" --infra "k8s/<context>" \
  --run-id "$run_id" --stage-src \
  --var "bucket=${bucket}" --var "prompts_uri=${prompts_uri}" \
  --var "generate_model=${text_model}" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

`--stage-src` stages the current package for the calling stage. `--secret-env`
takes credential names, never values. For a rendered preview, add `--plan-only`
to the same submit command. Config keys are case-sensitive: `bucket`,
`prompts_uri`, and `generate_model` are declared by this spec.

The default result is
`s3://<your-bucket>/runs/<run-id>/token-factory/generations.jsonl`.
Use `workflow status`, `logs`, and `artifacts` with the returned run ID and
selected project, as described in the [run lifecycle](../run-lifecycle.md).
Verify the saved output after the stage succeeds. See the
[composition guide](composing-cloud-and-token-factory.md) for GPU producer and
hosted consumer stages.

## Python integration

The SDK wrapper performs the CLI operation, saves the artifact, and prints its
status; it returns `None`. Select an available model as above:

```python
from npa.sdk.workbench import token_factory

token_factory.generate(
    input_path="./token-factory-results/prompts.jsonl",
    output_path="./token-factory-results/sdk-generations.jsonl",
    model="<available-text-model>",
    output="json",
)
```

For in-memory results, use `generate_text`, `caption_images`, or `reason_scene`
from `npa.workbench.token_factory`. These return dataclasses; persistence
requires the corresponding `write_generations`, `write_captions`, or
`write_reason` function. `TokenFactoryClient.chat_completion_text` returns text
without creating an artifact.

## Troubleshooting and validation

| Symptom | Next step |
| --- | --- |
| Key missing | Run `npa configure`, then `token-factory status`. |
| Authentication fails | Check that you supplied the Token Factory key, not an IAM token; check revocation and project access. |
| Model or inference request rejected | Check `models`, the returned error, and your project's inference/billing entitlement. |
| Workflow says the key is required | Include `--secret-env NEBIUS_TOKEN_FACTORY_KEY` on `workflow submit`. |
| S3 read/write fails | Check storage credentials, endpoint, bucket, and input objects; direct calls need AWS variables in their process environment. |
| Batch remains pending | Inspect `batch-status` and provider status; acceptance alone does not prove execution. |

With the real key already configured, run the live integration tests from the
repository root:

```bash
npa/.venv/bin/python -m pytest npa/tests/e2e/test_token_factory_e2e.py -v
```

These tests exercise real model requests. They skip without a key, and the
reasoner test skips if that model is unavailable. Inspect pass/skip results;
a skip is not proof that inference worked.
