---
name: quickstart
description: Use when guiding a user through first-time NPA setup, install, a first result, credential configuration, or the contributor dev/test loop. Covers the zero-credential first run and Nebius auth.
---

# Quickstart

Fast path to a productive `npa` user. Prefer the cheapest credible step first.

## Decision Order

1. Want a result now with nothing set up -> "Zero-Credential First Run".
2. Want to install or work on `npa` -> "Install" then "Dev Loop".
3. Want to run on Nebius (GPU, S3) -> "Authenticate", then load the `cookbooks`
   skill for validated end-to-end recipes.

## Zero-Credential First Run

No cloud, GPU, or credentials. Scores a shipped sample rollout set offline:

```bash
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

Expected: a ranked report with `accuracy: 1.0` over four labeled rollouts.

## Install

Python 3.10+. The venv can live anywhere:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa
npa --version
```

For repo validation use `npa/.venv/bin/python`; never bare `python`.

## Authenticate

```bash
npa configure
```

`npa configure` is interactive: it bootstraps a Nebius CLI profile when needed,
then writes NPA credential/config files. In non-interactive shells use
`npa configure --interactive`. Secrets live in
`~/.npa/credentials.yaml`; machine config in `~/.npa/config.yaml`. Never hardcode
project, tenant, registry, bucket, or secret values. Deploy workbench tools with
`--storage-endpoint storage.eu-north1.nebius.cloud` (the CLI default is wrong for
the primary cluster).

## Dev Loop

```bash
pip install -e "npa[dev]"
make test
```

Unit tests must not touch real infrastructure; mock SSH, S3, Nebius APIs, GPUs,
and network at the call site. Do not import GPU-heavy packages (`torch`,
`genesis`, `lerobot`) at module level in unit tests. CLI tests use
`typer.testing.CliRunner` against `npa.cli.main:app`.

## Where To Go Next

- Validated end-to-end pipelines and entrypoints: load the `cookbooks` skill.
- Architecture / review judgments: load `architecture` or `review-checklist`.
- Robotics/sim domain context: load `physical-ai-context`.
- Full docs: `docs/quickstart.md`, `docs/workbench/getting-started.md`.
