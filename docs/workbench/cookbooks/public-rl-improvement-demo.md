# Public RL improvement demo

The reference prepares its own public RGB-D scan, trains a real navigation
baseline, reconstructs the capture, continues PPO with baseline replay, and
compares checkpoints on frozen cases. There is no pre-trained demo baseline to
find or operator adapter to write. On a configured RTX Workbench environment:

```bash
npa workbench workflow demo run rl-improvement
```

The shared demo command selects
[`field-failure-reference-demo.yaml`](../../../workflows/testing/field-failure-reference-demo.yaml).
The standard workflow interface also supports direct submission:

```bash
npa workbench workflow validate-spec workflows/testing/field-failure-reference-demo.yaml
npa workbench workflow plan-spec workflows/testing/field-failure-reference-demo.yaml --run-id preview
npa workbench workflow submit workflows/testing/field-failure-reference-demo.yaml \
  --run-id '<fresh-run-id>' --stage-src --runtime \
  --infra 'k8s/<configured-rtx-context>' --var 'bucket=<configured-bucket>' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Use the standard [Workbench setup](../getting-started.md) for authentication,
storage, and compute. The public sample needs no customer data or gated dataset
token. The runtime fetches the public robot and controller under the normal
Isaac runtime terms; neither is bundled into this repository.

## Training and replay

The full preset uses 4,000 concurrent robots, 1,500 baseline PPO iterations,
1,500 continuation iterations, and 300 control steps per evaluation episode.
These are training settings, not a wall-clock or spending cap. The spec exposes
`num_envs`, `baseline_iterations`, `candidate_iterations`, and `episode_steps`
for explicit experiments. `navigation_image` and `reconstruction_image` are
immutable runtime digests. `source_overlay` stages the exact source checkout.
Choose `cohort_seed` before training to freeze a new experiment's reset cohorts.
Do not reuse a final cohort for tuning once its outcomes have been inspected.

Preparation fetches and verifies the full public TUM office capture, validates
real TSDF reconstruction against excluded depth frames, and derives supported
reset footprints from that surface. The baseline learns public warehouse
routes. Continuation includes every baseline training route and every measured
office training route in one interleaved reset population. Both metric
collision scenes are packaged in one world, separated beyond the sensor range.
The office and its poses/goals are translated together; no floor is added and
no geometry is rescaled.

The earlier continuation trained only on the office before evaluating a
different warehouse layout. That candidate scored zero held-out successes.
Forgetting and distribution mismatch are plausible causes; replay is an
implemented mitigation, not proof that transfer has been fixed. The original
negative result and missing observation remain historical evidence.

## Development and final evaluation

Warehouse training, development, and final route cohorts have different
physical resets with disjoint identifiers and seeds. Their hashes are frozen
in `reference-plan.json` before baseline training. Final cases are outside the
learner's prepared input and never select a candidate.

The candidate must reach 80% development success, improve by at least one
percentage point, and avoid an aggregate increase in contacts or physical
failures. Failure publishes `reports/index.html` and stops before final
evaluation. Change the training recipe and run a new development experiment;
do not adjust final cases or lower quality gates to produce a passing demo.

An eligible candidate and baseline run on all 4,000 frozen final routes. The
existing comparison requires actual success improvement and enforces its
per-case safety and goal-distance regression limits. Completed simulation can
still fail policy quality. The workflow exits unsuccessfully on failed quality
after publishing its evidence. No stage deploys a policy.

Open `reports/index.html` for an offline page with scores, training settings,
cohort hashes, and limitations. `reports/result.json` is its structured result.
Native baseline, diagnostic replay, development, and final artifacts retain
rendered rollouts, checkpoints, learning curves, physical controls, and measured
episode trajectories.

## Scope

This demo evaluates new warehouse routes in a known layout while learning from
a public office reconstruction. It is a baseline-retention experiment, not
evidence of unseen-site transfer. The policy consumes range observations.
Camera-conditioned navigation, private robot integration, and adaptation to
customer raw failure logs require separate integration and qualification.

The capture is public reference RGB-D data, not recorded robot failure
telemetry. Native baseline replays measure actual failures on the derived
navigation task before continuation. Fresh GPU results, including failed
quality, must accompany any readiness claim.
