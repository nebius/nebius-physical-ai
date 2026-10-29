# Sim2Real development image set — October 28, 2026

Use this source-attested image set with the [manual workflow guide](../guides/sim2real-workflow.md). It contains the five roles for Cosmos3-Nano generation, scenario generation, Isaac Lab rollout/PPO/evaluation, the CPU workflow runtime, and artifact visualization.

Source: `1f0a36b4973a8ba1d4e9e7c29a76a9db82b3d2b0`. All five exact development images passed anonymous OCI manifest/config reads, source-bound provenance, and SPDX SBOM verification. Development image publication and per-run policy effectiveness have separate evidence; use the guide's target/bootstrap preflight and retain the strict held-out Gold result with each run.

| Role | Immutable public image |
| --- | --- |
| Controller | `ghcr.io/nebius/nebius-physical-ai/npa-sim2real-control@sha256:9296a9a6296558e911770a060b9bb5712c67cc0341e95822fa231f11d1e4bb74` |
| Cosmos3 generation | `ghcr.io/nebius/nebius-physical-ai/npa-cosmos3@sha256:ba07281186b8f72f14d0dce1d17c347c741b70e4bebea4bbef461678cb750ba6` |
| EnvGen | `ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:4bce2169f7448eaa355b0740c9229d2fedb3f123e56250f1f936c37acf148d43` |
| Isaac Lab | `ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab@sha256:4673d22f550502f3bc33b4f11b7ae83736c8f54c5bbd6b45cf555f32f068cc45` |
| Viewer | `ghcr.io/nebius/nebius-physical-ai/npa-rerun-viewer@sha256:32c8e261767fbc30fc8990f92ffbcbcd05de96d8780612851d8b828290cd7535` |

The [October 29 manual qualification](sim2real-manual-qualification-20261029.md)
completed the canonical workflow with this exact image set and achieved 48/64
strict held-out Gold successes. Each new run measures its own result against the
unchanged 32/64 acceptance gate.

For earlier measurements, the [October 10 manual execution receipt](sim2real-manual-development-20261010.md) retains its own source, image set, and result. Those measurements are not attached to this image set.

For the faster full-pass cadence, declare `inner_iterations=1` and `allow_early_exit=1` while retaining 2,000 PPO updates, all 10,000 scenarios with disjoint splits, and 64 validation/64 Gold cases. The standard workflow performs validation-only selection and accepts at least 32/64 strict Gold successes at threshold `0.50`. Record the actual result and exact manual arguments in the run evidence.
