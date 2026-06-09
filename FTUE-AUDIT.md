# Sim2Real Cold-Start / FTUE Audit

A clean-room first-time-user pass over the public `npa` repo, attempting to stand
up the **sim2real** workflow across all three tiers (raw SkyPilot YAML, SDK,
CLI) by following the README and the sim2real runbook exactly. This documents
every cold-start friction point found, what this change fixes, and the gaps
deferred to follow-up.

No infrastructure was touched: this was a code- and docs-only review run in
parallel with live work. Examples use neutral placeholders; S3 is referred to as
a "non-default S3-compatible endpoint."

## Friction points (ordered)

1. **No preflight for the recurring blockers.** S3 endpoint/region mistakes,
   image-pull-secret expiry, missing HF/NGC tokens, an unpinned kube context
   pointing at the wrong cluster, and a schedulable-GPU count of zero were all
   only discoverable *mid-run*, after provisioning. A first-time user had no way
   to confirm readiness up front.
   **Fixed:** `npa doctor sim2real` runs these as explicit PASS/WARN/FAIL/SKIP
   checks (`config`, `coherence`, `s3`, `registry`, `tokens`, `cluster`), with
   `--checks`, `--json`, and `--warn-only`.

2. **Raw-SkyPilot tier was not independently runnable.** `runbook.yaml` set every
   `envs` value to `${VAR}` and `image_id` to `docker:${TRAINER_IMAGE}`. SkyPilot
   0.12.2 does not interpolate `${VAR}` inside `envs` or `image_id` at submission
   time, so the pod received the literal string `${NPA_SIM2REAL_BUCKET}` and an
   invalid image reference. The tier could not run via raw `sky`.
   **Fixed:** `runbook.yaml` now carries materialized literal defaults and a
   literal `example.invalid/...` image, and expands env vars only at container
   runtime in the `run` block. Operators override literals with `--env` /
   `--secret` and edit `image_id` for their registry.

3. **A test codified the broken `${VAR}` pattern.** `test_sim2real_loop.py`
   asserted `envs["VLM_IMAGE"] == "${VLM_IMAGE}"`, locking in the
   non-interpolating anti-pattern.
   **Fixed:** the test now asserts no `${` appears in `envs`/`image_id`, that the
   seam env names are still declared and consumed by the `run` block, and that
   the YAML stays endpoint-neutral.

4. **GPU-reaching path was documented dishonestly.** The README showed raw
   `sky jobs launch runbook.yaml` as the way to reach GPUs, but that path is
   currently blocked by the SkyPilot 0.12.2 pre-setup `getcwd()` bug.
   **Fixed:** the README now documents the materialized-runbook / direct-
   Kubernetes route as the GPU-reaching path and states the limitation inline.

5. **Three-tier coherence for the headline workflow was unenforced.** The
   contract guard covered `sonic`, `vlm-eval`, `trigger`, `cosmos*`, and
   `detection-training`, but `sim2real` was exempted as a free-form "seam" with
   no flag↔param↔env guard. A flag could drift from its YAML env or config field
   silently.
   **Fixed:** a canonical seam table (`SIM2REAL_SEAMS`) plus a `coherence_failures`
   check now validate, for all 20 BYO seams, that the CLI flag exists on
   `sim2real run`, the SDK/config field exists, and the runbook env is both
   declared and referenced. This guard runs inside `npa doctor` and as a
   guardrail test alongside the existing contracts.

6. **No one-value BYO seam map.** A new user had to read code to learn which CLI
   flag mapped to which SDK keyword and which YAML env.
   **Fixed:** the sim2real README now has a single "one seam, one value" table
   spanning all three tiers, plus a preflight-first section.

## Deferred gaps (follow-up)

- **Three-tier → GPU launch gap.** All three tiers reach GPUs only via the
  materialized-runbook / direct-Kubernetes path today, because raw
  `sky jobs launch` is blocked by the SkyPilot 0.12.2 pre-setup `getcwd()` bug.
  Closing this needs an upstream SkyPilot fix or a thin in-repo materializer that
  renders the runbook to a Kubernetes Job. Out of scope for this pass.
- **SDK seam discoverability.** `sim2real.run(**overrides)` forwards seams by
  keyword into `build_config_from_env`, so the seams are real and coherent but
  not discoverable from the function signature (and cannot use the inspect-based
  `CapabilityContract`). Promoting the seams to explicit keyword parameters would
  change the public SDK signature; deferred.
- **`sim2real` name collision.** Two surfaces share the "sim2real" name: the
  13-stage VLM-to-RL loop (`npa workbench sim2real run`) and the separate
  sim-to-real H100 quickstart/pipeline. A first-time user cannot tell which is
  canonical. Naming reconciliation is deferred.
- **Deeper GPU gate.** The cluster check counts schedulable `nvidia.com/gpu`; it
  does not yet match the requested GPU product node-selector label against
  available node products. A product-aware gate is a follow-up.
