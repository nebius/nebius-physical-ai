# Public RL improvement demo

The reference prepares its own public RGB-D scan, trains a real navigation
baseline, observes actual failures on office training routes, admits the capture
for reconstruction, continues PPO with baseline replay and a frozen baseline
policy penalty, and compares checkpoints on frozen cases. There is no pre-trained
demo baseline to find or operator adapter to write. On a configured RTX Workbench
environment:

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

The continuation adds a KL-divergence penalty that discourages its action
distribution from drifting away from the original baseline. The full preset
sets `baseline_anchor_coefficient` to `10.0` before training; it is a learning
setting, not a quality threshold. The teacher is frozen after the exact native
baseline and optimizer have been restored and the initial checkpoint retained.
It sees the original training minibatches through its own fixed observation
normalizer. All office and warehouse routes remain in PPO training. Evaluation
uses the candidate alone; it does not switch between policies or select actions
from known route outcomes.

The anchor is a retention experiment and does not guarantee a passing policy.
Its checkpoint retains the original teacher identity and state so continued
training cannot silently replace the teacher with an adapted candidate. The
same per-case and per-region gates below decide whether it is effective.

The earlier continuation trained only on the office before evaluating a
different warehouse layout. That candidate scored zero held-out successes.
Forgetting and distribution mismatch are plausible causes; replay is an
implemented mitigation, not proof that transfer has been fixed. The original
negative result and missing observation remain historical evidence.

A subsequent full replay run improved development success from 82.475% to
93.0%, but failed 371 metric checks across 257 cases, including warehouse
regressions. Selection retained the baseline and never evaluated final outcomes.
That rejected run remains preserved separately from the current experiment.

The full coefficient-10.0 experiment completed 1,500 baseline and 1,500 new
candidate updates, with 48 million transitions per policy. It verified exact
baseline model/optimizer continuation and the original frozen teacher. On
4,000 separate development routes per policy, success declined from 82.475%
to 82.05% (−0.425 percentage points). Office success fell from 65.0% to 64.2%;
warehouse success fell from 99.95% to 99.90%. The unchanged regional and
per-case guards found 699 metric violations across 399 cases. Selection retained
the baseline, published the rejection report and exited unsuccessfully before
final evaluation. This verifies the workflow's training, comparison and rejection
path; the baseline penalty has not established an effective promotion.
See the [source-bound rejection evidence](../evidence/public-demos/rl-improvement-rejection.json).

## Development and final evaluation

Before adaptation, a separate native GPU stage evaluates the exact fresh baseline
on every one of the 4,000 frozen office **training** routes. It uses the full
300-step horizon and unchanged physical controls. The admission rule is frozen
before learning: at least one completed episode must be unsuccessful, including
a timeout or measured physical/contact failure. An incomplete rollout, crashed
process, or failed physical control cannot admit the capture.

The CPU admission stage recomputes all outcomes from native trajectories and
seals `observed-failures.json`: failed case IDs and seeds, ordered reset hash,
baseline checkpoint, raw outcome and native after-load evidence hashes, and
the exact public capture and scene identities. The public bundle reference
binds this admission's URI and SHA. Reconstruction and training reverify that
same evidence before accepting the capture or baseline. The whole admitted
capture and all original training routes remain in the experiment; the failure
count never changes route populations, training budgets, or quality gates.

If no failures were observed, `reports/index.html` says so and the workflow stops
without continuation or final evaluation. A zero-failure observation is not a
policy-improvement result. The initial prepared scene is observation scaffolding;
after admission, the actual capture is reconstructed again by the generic native
adapter before training.

Both development and final evaluation contain 4,000 routes: 2,000 measured
office routes and 2,000 warehouse routes each. Continuation retains all 4,000
training routes per region. Train, development, and final have different
physical resets with disjoint identifiers and seeds. The composite collision
geometry, region assignments, case identifiers, seeds, and case hashes are frozen
in `reference-plan.json` before baseline training. Reconstruction verifies exact
world-space collision triangles against that freeze, independently of USD ZIP
metadata or mesh ordering. Final routes are outside the learner's prepared
input and never select a candidate.

The candidate must reach 80% development success, improve by at least one
percentage point, and avoid an aggregate increase in contacts or physical
failures. Each region must retain its success rate and avoid increased mean
contacts, physical failures, or goal-distance regression beyond 0.25 metres.
Office gains cannot conceal warehouse regression. Failure publishes
`reports/index.html` and stops before final
evaluation. Change the training recipe and run a new development experiment;
do not adjust final cases or lower quality gates to produce a passing demo.

Development also applies the final comparison's identical per-case regression
guards using the frozen `reference-plan.json` metrics. Exact case IDs and seeds
pair the checkpoints: no case may lose success or increase contact or physical
failure steps, and no case may worsen goal distance by more than 0.25 metres.
Success gain is the mean of paired episode changes, so exactly 40 additional
successes among 4,000 cases meets the unchanged one-percentage-point gate.
The HTML report shows violation counts and every affected case when selection
is blocked; final cases remain unevaluated.

An eligible candidate and baseline run on all 4,000 frozen final routes. The
existing comparison requires actual success improvement and enforces its
per-case safety and goal-distance regression limits. Final promotion also
requires 80% overall success and the explicit per-region retention checks.
Completed simulation can
still fail policy quality. The workflow exits unsuccessfully on failed quality
after publishing its evidence. No stage deploys a policy.

Open `reports/index.html` for an offline page with overall and per-region
scores, warehouse retention, training settings, cohort hashes, and limitations.
The first, focal episode is an office adaptation route; its embedded scored
frames illustrate one route, while measured aggregate scores cover both regions.
`reports/result.json` is its structured result.
If report publication is interrupted, a retry can complete missing files only
when the retained selection, result, and HTML bytes match the regenerated
evidence exactly. Different retained bytes stop recovery without overwriting
the earlier result. A blocked selection remains blocked after report recovery.
Native baseline, diagnostic replay, development, and final artifacts retain
rendered rollouts, checkpoints, learning curves, physical controls, and measured
episode trajectories.

## Scope

This demo evaluates adaptation and retention on new routes in both known
public layouts. It does not establish unseen-site transfer. The policy consumes range observations.
Camera-conditioned navigation, private robot integration, and adaptation to
customer raw failure logs require separate integration and qualification.

The capture is public reference RGB-D data, not recorded robot failure
telemetry. Native baseline replays measure actual failures on the derived
navigation task before continuation. Fresh GPU results, including failed
quality, must accompany any readiness claim.

This public demo admits a prepared capture based on actual **simulation**
failures; it does not mine recorded physical robot failures. The generic
[field-failure workflow](field-failure-policy-improvement.md) accepts
operator-prepared recorded-failure capture bundles, but does not mine raw logs
or require evidence that a selected capture contains a measured failure.
The operator supplies capture selection, calibration, task/reset recipes,
baseline checkpoint, and separate held-out inputs.
