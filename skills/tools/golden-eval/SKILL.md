---
name: golden-eval
description: Use to prove a workbench container image actually works — the per-container hello-world manifest, dry-run/local/serverless execution tiers, batch runs across every image, and the offline manifest validation that gates CI.
---

# Golden eval (does this container actually work?)

A golden eval is the minimal tested rerun that proves one container image is
functional. It answers a narrow, valuable question — *does this image run its own
smoke on a real GPU?* — without standing up a cluster or a workflow. Reach for it
after building or republishing an image, and when triaging whether a failure is
the image or the pipeline around it.

The manifest is `npa/src/npa/smoke/golden_evals.yaml`
(format `npa_golden_evals_v1`); every container in `CONTAINER_IMAGE_NAMES` must
have an entry, enforced by `npa/tests/smoke/test_golden_eval_manifest.py`.

## Inspect before running

```bash
npa workbench golden-eval list                 # every container, kind, gpu, status
npa workbench golden-eval list --output json
npa workbench golden-eval show <container>     # full safety + Physical AI record
```

Read the `status` column before spending anything:

- `ready` — runnable now.
- `gpu-gated` — needs a real GPU; a local run without one proves nothing.
- `blocked-on-upstream` — excluded from batch runs unless you pass
  `--include-blocked`. A failure here is expected and is not your regression.
- `needs-image-update` — the manifest and the published image disagree; rebuild
  before drawing conclusions.

`kind` tells you what is actually exercised: `container-smoke`, `server-smoke`,
`entrypoint-smoke`, `workflow-smoke`, or `build-import`. A `build-import` passing
means the package imports, not that the tool works. `gpu` is `required`,
`optional`, or `none`.

## Three execution tiers, cheapest first

```bash
npa workbench golden-eval run <container>                      # dry run (default)
npa workbench golden-eval run <container> --execute            # local runtime
npa workbench golden-eval run <container> --serverless         # one GPU, real image
npa workbench golden-eval run <container> --serverless --gpu h200 --timeout 40m
```

**Dry run is the default and prints the command.** Use it to see exactly what
would execute — often enough to answer a question without running anything.

`--execute` runs locally and only works where the container runtime is present;
for a `gpu-gated` entry on a machine with no GPU it tells you nothing.

`--serverless` is the useful tier for verification: it submits the eval to a
Nebius Serverless Job **in the real container image on a GPU** and waits for
PASS/FAIL. No cluster, no node pool, no SkyPilot controller. This is the cheapest
way to get a truthful answer about an image. `--gpu` overrides the type (for
example `h200`, `h100`, `l40s`, `b300`) and `--timeout` defaults to `40m`.

## Batch across every image

```bash
npa workbench golden-eval run-all --serverless --parallel 4 \
  --json-out /tmp/golden-results.json
npa workbench golden-eval run-all lerobot groot --serverless
npa workbench golden-eval run-all --tools-only --serverless
npa workbench golden-eval run-all --include-blocked --serverless
```

Positional arguments restrict the run to a subset; the default is every manifest
entry. `--tools-only` skips foundation images. `--parallel` bounds concurrency
and applies to `--serverless` and `--execute`. Always write `--json-out` for a
batch — the console summary is not something you want to re-derive after a
40-minute sweep.

Like `run`, `run-all` defaults to dry-run. A batch with no `--execute` and no
`--serverless` costs nothing and validates that every command is well-formed.

## The offline gate

```bash
npa workbench golden-eval validate
```

Validates manifest completeness and consistency with no network and no GPU:
every container has an entry, entries map to known images or foundation images,
and each definition points at real Dockerfiles and entrypoints. This is the
nightly CI gate and belongs in any change that adds, renames, or retags an image.

## When to run which

- **Added or changed an image** → `validate`, then `run <container> --serverless`.
  Also update the catalogs; see `skills/atomic/audit-container-docs/SKILL.md`.
- **A workflow stage fails and you suspect the image** → `run <container>
  --serverless` isolates image from pipeline in one step.
- **Before publishing a batch of images** → `run-all --serverless --json-out`.
- **In a PR with no infra** → `validate` plus the pytest gate below.

## Gotchas

- **Dry run is the default everywhere.** A `run` that "passed" without
  `--execute` or `--serverless` only printed a command. Check which tier you
  actually used before reporting a result.
- **A local `--execute` pass on a GPU-gated entry is not evidence.** Neither is a
  `blocked-on-upstream` failure a regression.
- **Serverless still needs registry access.** The eval pulls the real image, so a
  `403` here is a pull-secret problem, not a broken smoke; see
  `skills/atomic/debug-failed-run/SKILL.md`.
- **Golden eval proves the container, not the pipeline.** It says nothing about
  spec wiring, S3 paths, or multi-stage behavior — that is
  `npa workbench workflow`.
- **Do not weaken a manifest entry to make a sweep green.** The entry is the claim
  about what the image can do.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/smoke/test_golden_eval_manifest.py \
  npa/tests/guardrails/test_skills_index.py -q
```
