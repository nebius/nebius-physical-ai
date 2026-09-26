# Multiple models repairing a robot data workflow

On September 25, 2026, real Token Factory GLM Flash and full GLM workers edited
Workbench's simulation and dataset code, corrected failed attempts, and completed
four independently verified six-case runs. LangGraph owned their durable tool
loops. The branch was rebased on main `f8c9e4c3a` before preparing the campaign.
It was rebased again on `444a670e9` before final publication checks.
[Machine-readable evidence](specialists-multimodel-repair-results.json) retains
model usage, routing decisions, actual patches, failed checks, source hashes and
native artifact verification.

## Work and model assignment

| Task | Worker | Changes and completion |
| --- | --- | --- |
| Physics validation | GLM-5.3-Flash | Reject incomplete or misaligned physical evidence; recover from a native simulation failure caused by treating string phase labels as numeric data. |
| Dataset transaction | GLM-5.3-Flash | Validate episodes before encoding; stage the dataset and delay provenance indices until publication; repair two adapter compatibility regressions. |
| Integration review | GLM-5.3 | Reject missing/object-typed physical evidence, escaping or duplicated episode directories, and invalid frame rates; remove unnecessary float64 copies of camera recordings. |
| Review follow-up | GLM-5.3 | Preserve adapter error contracts, validate the first episode's two cameras, enforce real numeric vectors and scalar trace shapes, and split oversized helpers. |

Each task edited source, invoked fixed Workbench operations, and completed
simulation, conversion, independent replay and native LeRobot verification.
The initial physics and export workers ran concurrently. The integration worker
used an immutable snapshot of their edits while the export worker finished its
own compatibility checks; the follow-up incorporated those compatibility
requirements. These were new robustness requirements against real production
code, with deliberately malformed regression inputs and explicit failure
injection. Production source was not deliberately broken for this campaign.

**Lightning selected Flash for both initial tasks.** Its two calls took about
0.63 seconds each. Full GLM was then explicitly assigned the integration and
follow-up work. This proves multiple models contributing repairs, but does not
prove that Lightning can reliably distinguish cheap and difficult coding tasks.
The identical candidate criteria described Flash as suitable for local fixes
and full GLM as suitable for interacting lifecycle and data-format requirements.
No routing decision was overwritten or replaced in the evidence.

## Actual workflow and artifacts

The [six-case matrix](../../npa/examples/specialists/robot_workflow/multimodel-scenes.json)
varies two transfer layouts, RNG seeds, colors and illumination, including two
open-gripper negative controls. Every completed run contained:

- Six native MuJoCo Fetch episodes and **1,110 physics transitions**.
- **2,220 raw camera frames** plus 1,110 decoded side-by-side preview frames.
- Four accepted episodes and two rejected grasps retained as raw evidence.
- **740 training timesteps**, 1,480 aligned camera samples, correct language
  labels and a real four-sample batch from LeRobot 0.5.1.

The verifier independently replayed every action, state, contact and camera
frame, then decoded and checked the exported dataset. Candidate processes had
no network or host credentials, and could not modify the verifier. Submitted
snapshots bound all three edited source files; native manifests also retained
the simulation/export module hashes. This was local CPU Workbench execution of
the same simulator/exporter used by robot SDG. It did not submit GPU jobs,
train a policy, or perform physical robot transfer.

## Failures, review and scope

The initial regression contract exposed **36 failures**. The physics worker's
first native run failed, and it repaired the handling of string phase labels
before successfully rerunning. The export worker encountered and repaired two
existing API-contract regressions introduced by its patch. The integration
review exposed **22 additional failing checks**, and the final GLM follow-up
exposed **nine** under its expanded contract. These counts describe different,
partly overlapping test sets; they are not counts of independent production bugs.
The final GLM task passed 103 focused checks and a fresh six-case native run.
Every failed attempt remains in the results file.

The operator supplied tasks, regression cases and review feedback. Models
produced the recorded patches and drove their operations to completion; this
was not an autonomous discovery of every requirement. Maintainer review then
formatted the code, split a trace helper, included the goal in finiteness checks,
and added scalar/dtype input guards with seven more regressions. Final publication
validation is recorded separately from the model-authored revisions.

The first full Linux run exposed 15 failures in an existing stream-length test
module. Maintainer review restored its exact error messages, including empty
streams. Six cases also expected a partial first video when a later episode was
invalid; those now assert the stronger contract that no encoding starts until
every episode passes validation. The expanded focused set passed 128 checks.

Final validation at source commit `5cc83e841` passed **29,150 Linux tests**
(124 skipped, one xpassed) with **76.68% package-source coverage**, all **859
security regressions**, 221 Cypress tests and 28 native protocol tests.
Precheck passed 257 tests; CLI documentation drift and merge checks passed.
A fresh six-case simulation/export/replay run passed with the final three source
hashes. The separate real conversion-failure/retry test also passed and loaded
one 185-timestep episode through native LeRobot. These publication checks made
no additional model calls. Their receipts are separate from the model campaign
in the results file; they do not establish the hosted CI result.

Local dataset publication stages conversion and metadata on the same filesystem
before rename. Conversion errors preserve raw episodes and leave no new dataset
or provenance indices. Existing datasets and dangling destination links are
rejected. Use a single writer per run. Export retry is supported; the top-level
robot SDG command still requires a new output directory.

## Measured model usage

| Model | Responses | Input tokens | Output tokens | Public-tariff estimate |
| --- | ---: | ---: | ---: | ---: |
| Nemotron 3.5 Lightning | 2 | 549 | 40 | $0.000043 |
| GLM-5.3-Flash | 43 | 534,171 | 12,524 | $0.086388 |
| GLM-5.3 | 62 | 1,425,132 | 16,487 | $2.067728 |
| Total | 107 | 1,959,852 | 29,051 | **$2.154158** |

Rates were retrieved from the [official Token Factory catalog](https://tokenfactory.nebius.com/api/public/models_info)
on September 25. All classifier and worker responses, including failed repair
attempts, are included; no provider-usage records were missing. Full GLM reported
1,220,864 cached input tokens, but the estimate charges every input token at the
ordinary rate and assumes no cache billing discount. These are provider counters,
not a direct inspection of serving memory or a billing invoice.

There was no separate Astra coordinator subprocess in this campaign. The
operator/development conversation, host costs and setup are excluded. There was
no matched Astra-only arm, so this run establishes neither savings nor a speed
advantage over Astra. Jev was not invoked; routing used the Token Factory
classifier and explicit reviewer assignments. No private repository dependency
is required.

## Reproduce the workload and regression checks

Use the setup and independent-verifier commands in the
[prepared robot workflow guide](../../npa/examples/specialists/robot_workflow/README.md),
substituting `multimodel-scenes.json` for `scenes.json`. Keep a fresh run directory
and independently compute the expected source hashes. The source-repair workers
used operator-owned workspaces and fixed commands; source-edit scopes are not
an operating-system sandbox.

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_robot_physics_trace_contract.py \
  npa/tests/workbench/test_robot_export_transaction.py \
  npa/tests/workbench/test_robot_export_provenance.py \
  npa/tests/workbench/test_token_factory_robot_sdg.py \
  npa/tests/test_sim_episode_stream_lengths.py \
  npa/tests/test_adapter.py::TestConvertErrors -q
```

The opt-in live rollback test executes real simulation and conversion, injects
an error after native video/Parquet generation, checks that raw artifacts and
provenance remain intact, then retries and verifies the result with native
LeRobot. It uses no planning-model calls:

```bash
NPA_TOKEN_FACTORY_ROBOT_SDG_LIVE=1 \
NPA_LEROBOT_PROOF_PYTHON="$NATIVE_LEROBOT_PYTHON" \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_token_factory_robot_sdg_live.py \
  -k native_export_rolls_back_and_retries -q
```
