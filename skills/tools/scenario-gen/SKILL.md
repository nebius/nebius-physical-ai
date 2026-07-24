---
name: scenario-gen
description: Use when generating adversarial scenarios via RL, ranking mined failures of a policy-under-test, or wiring the adversarial-scenario-hardening workflow.
---

# Scenario Gen (Adversarial Scenario Generation)

Adversarial scenario generation productizes the Isaac Lab RL capability as a
first-class hard-case miner: an adversary perturbs the environment / other
agents to *maximize failures* of a policy-under-test, surfacing hard scenarios
for regression and hardening.

The adversary backend is pluggable. The **intended production backend is an
Isaac Lab RL adversary** (reward = the policy-under-test's violation rate). The
**default backend is not RL** — it is a deterministic, GPU-free heuristic search
that acts as a functional scaffold/stand-in so the tool runs and is testable
without a GPU. Plug in the real backend via ``adversary_backend``.

## Three-access pattern

Source of truth is the FastAPI service
(`npa/src/npa/workbench/scenario_gen/service.py`). The CLI
(`npa/src/npa/cli/workbench/scenario_gen.py`) and SDK
(`npa/src/npa/sdk/workbench/scenario_gen.py`) are thin clients. Do not duplicate
logic across layers.

## Interfaces

CLI:

```bash
npa workbench scenario-gen generate --policy-uri <s3> --input-path <s3> --output-path <s3>
npa workbench scenario-gen rank --input-path <s3-manifest> --output-path <s3>
npa workbench scenario-gen status --run-id <id>
npa workbench scenario-gen system-info
npa workbench scenario-gen list
```

Endpoints: `/health`, `/status`, `/system-info`, `/list`, `POST /generate`,
`POST /rank`.

## API contract

- `POST /generate`: given a policy-under-test checkpoint URI (`--policy-uri`) and
  a base task/scene config (`--input-path`), train an adversarial RL agent whose
  reward is the failure/violation of the policy-under-test, then emit a ranked
  adversarial set to `--output-path`. Output schema
  `npa.scenario_gen.adversarial_set.v1` (S3 manifest + per-scenario configs and
  predicted failure metrics). Lineage (workflow run, input URIs, policy
  checkpoint, task) is threaded into every manifest.
- `POST /rank`: score/rank a generated set by weighted failure severity +
  diversity; emits `npa.scenario_gen.ranked_set.v1`.

The adversary backend is pluggable (`adversary_backend`). The default is a
deterministic, dependency-light heuristic (not RL) so the tool runs and tests
without a GPU; a live run swaps in the Isaac Lab RL backend.

## Visualization (Rerun)

`generate` also emits a Rerun recording at `{output_uri}/scenarios.rrd`
(disable with `--no-visualize`), visualizing the mined set: severity/diversity/
failure time-series over a `rank` timeline, a severity bar chart, a
severity-vs-diversity scatter, and a scenario×perturbation-axis heatmap tensor.
`rerun-sdk` is imported lazily; if it is absent, generation still succeeds and
`viz_uri` is empty. View it with `npa rerun host <viz_uri>` (prints an
app.rerun.io URL) or open the `.rrd` in the Rerun viewer; the NPA agent
classifies `.rrd` as a rerun artifact.

## GPU routing

Route the adversary training to **RTX PRO 6000** or **L40S** (RT-core capable
Isaac Lab build). Never route SONIC to L40S. General policy retraining uses the
existing `workbench.rl.policy_train` on the same RT-core class or H100.

## SkyPilot + workflow

- SkyPilot (headless, `cloud: kubernetes`, RTX PRO 6000):
  `npa/workflows/workbench/skypilot/scenario-gen-adversarial.yaml`
- Declarative hardening pipeline (generate -> rank -> retrain/evaluate/gate loop
  -> publish): `npa/workflows/workbench/npa-workflows/adversarial-scenario-hardening.yaml`

toolRefs: `workbench.scenario_gen.generate`, `workbench.scenario_gen.rank`,
`workbench.scenario_gen.write_hardening_decision`.

## Known issues

- The gate loops back to retrain while the measured failure rate stays above
  `config.failure_rate_threshold`; the outer loop is bounded by
  `config.outer_iterations`.
- Batch jobs must stay headless — never trigger a rendering path.
