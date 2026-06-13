---
name: quickstart
description: Use when a user is setting up NPA for the first time, asking how to install, run a first result, configure credentials, or run the dev/test loop. Covers the zero-credential first run, install, Nebius auth, and contributor workflow.
---

# Quickstart

This skill is the fast path for getting a user productive with `npa`. Prefer the
cheapest credible step first: a real result with no cloud, GPU, or credentials,
then install/auth, then a working cookbook.

## Decision Order

1. User wants to *see it work now* with nothing set up -> "Zero-Credential First Run".
2. User wants to *install* or work on `npa` -> "Install" then "Dev Loop".
3. User wants to *run on Nebius* (GPU, S3) -> "Authenticate", then load the
   `cookbooks` skill for the validated end-to-end recipes.

## Zero-Credential First Run

No cloud, GPU, or credentials. Scores a shipped sample rollout set with the
offline stub backend:

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

Expected: a ranked report with `accuracy: 1.0` over four labeled rollouts. The
same command swaps `--backend stub` for `self-hosted` or `api` once credentials
exist.

## Install

Python 3.10+. Platforms: **macOS**, **Linux**, **Windows via WSL2 Ubuntu** (not
native Windows). Full copy-paste blocks: `docs/quickstart.md` § Fast install by
platform.

**Nebius CLI** (macOS, Linux, WSL — same install script):

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"
```

**npa** (venv can live anywhere; activating puts `npa` on `PATH`):

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa
npa --version
```

For repo validation, always use `npa/.venv/bin/python`; never bare `python`.

## Authenticate

```bash
nebius profile create
nebius iam get-access-token >/dev/null
npa configure
```

`npa configure` is interactive and bootstraps a Nebius profile. User secrets live
in `~/.npa/credentials.yaml`; machine-managed config lives in `~/.npa/config.yaml`.
Never hardcode project, tenant, registry, bucket, or secret values.

When deploying or configuring workbench tools, pass
`--storage-endpoint storage.eu-north1.nebius.cloud` (the CLI default
`storage.uk-south1.nebius.cloud` is wrong for the primary cluster).

## Dev Loop

```bash
pip install -e "npa[dev]"
make test
```

Use `RELAXED_DIRTY_TREE_MODE`: dirty files outside the run's target paths are not
blockers. Do not add time, cost, or job-count limits unless the operator asks.
For test conventions, lint gates, and the expected baseline, load
`.agents/skills/platform/testing-conventions/SKILL.md`.

## Where To Go Next

- Validated end-to-end pipelines and their exact entrypoints: load the
  `cookbooks` skill (`.agents/skills/workbench/cookbooks/SKILL.md`).
- Authoring/changing a workbench tool: load `workbench-tool`.
- Authoring/running SkyPilot workflow YAMLs: load `workflows` and
  `skypilot-workflows`.
- The zero-credential first run's tool (backends, benchmark sweeps, the loop):
  load `vlm-eval`.
- Full docs: `docs/quickstart.md`, `docs/workbench/getting-started.md`.
