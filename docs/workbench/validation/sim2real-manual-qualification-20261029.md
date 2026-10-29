# Sim2Real manual qualification — October 29, 2026

The canonical [manual Sim2Real workflow](../guides/sim2real-workflow.md) completed
its 14-stage simulated stock-Franka task and achieved **48/64 strict held-out
Gold successes (75%)**. The acceptance gate was **32/64 (50%)**. Validation alone
selected the checkpoint; Gold supplied the independent final gate.

## Recipe and evidence scope

| Item | Qualified value |
| --- | --- |
| Workflow | `workflows/main/sim2real.yaml`, preset `public-franka-lift` |
| Workflow SHA-256 | `5141a06fcd6edda5be983e8533a7ded54d5763bf6d8e29b45456facb51269578` |
| Runtime and five-role image source | `1f0a36b4973a8ba1d4e9e7c29a76a9db82b3d2b0` |
| Qualified PR head | `517eeaa75d480e7be8d6993d0e9f0de550caab78` |
| Main integration | [PR #921](https://github.com/nebius/nebius-physical-ai/pull/921), merged October 29 |
| Image selection | All five explicit immutable pins in the [development image-set receipt](sim2real-development-image-set-20261028.md) |
| Measured GPU | RTX PRO 6000 Blackwell Server Edition |
| Scenario pool | 10,000 scenarios with sealed, disjoint training/validation/Gold splits |
| Training cadence | `inner_iterations=1`, `allow_early_exit=1`; each full PPO pass retains 2,000 updates |
| Validation and Gold evaluation | 64 cases in each split; strict stable placement within 5 cm |
| Selected checkpoint | `model_500.pt`, chosen by the automatic validation-only ranker |
| Checkpoint size | 2,319,962 bytes |
| Checkpoint SHA-256 | `37836196b337d69612899935054ce3ce41cc2c1d01ae65bb09b3b60282c9bd79` |
| Final strict Gold | 48 successes / 64 cases, above the fixed `0.50` gate |

The recorded runtime and image revision remains separate from the documentation
and repository dependency-security updates in the final PR head. Those updates
did not change the admitted Sim2Real executable source, image Docker content, or
workflow specification. No earlier checkpoint was imported into the new run and
no checkpoint was chosen manually.

## Run from main

Use the current manual guide to install the CLI, configure your own credentials,
storage and compatible RTX target, and select the five immutable image pins.
Keep the image receipt's `SOURCE_SHA` when using those published bytes; a newer
checkout does not change the source baked into an existing image. Preflight
checks the exact images and the selected target for each run.

Add the measured cadence to the same configuration used for preflight, planning,
and execution, then submit the ordinary workflow:

```bash
CONFIG_ARGS+=(--var inner_iterations=1 --var allow_early_exit=1)
WORKFLOW_ARGS=(--preset public-franka-lift "${CONFIG_ARGS[@]}")
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  "${TARGET_ARGS[@]}" "${WORKFLOW_ARGS[@]}" "${SECRET_ARGS[@]}" \
  --run-id "$RUN_ID" --runtime --durable-s3 --max-wait-seconds 0
```

This checks Gold after each full pass while keeping the dataset, update count,
checkpoint rule, and strict success metric. `--max-wait-seconds 0` lets a healthy
long-running wave finish without an operator-client observation deadline.

A new run should produce its own selected checkpoint, validation and Gold
reports, 14 ComponentRecords, camera evidence, and decoded RRD/MCAP outputs. The
success gate requires at least 32/64 Gold successes before promotion. **48/64 is
the measured reference result, not a guarantee that every new training run has
exactly 75% success.** Run-to-run training variability was not quantified by this
single qualification. Different tasks, objects, or embodiments need their own
training and held-out evaluation.

## Independent output checks

The audit checked all 64 Gold case records and 2,496 frame artifacts across the
primary, side, and overhead cameras. The selected checkpoint's downloaded byte
size and SHA-256 matched the identity actually loaded by Gold. The published
Rerun recording decoded to 6,317 chunks. MCAP decoded to 17,424 messages across
711 nonempty channels; independent regeneration was byte-identical.

The standalone HTML proof embeds the 192 terminal camera frames for all 64
cases, a real Cosmos3 output video, three-view success/failure review clips, the
validation ranking, and sanitized downloadable evidence. Desktop/mobile offline
browser checks verified video playback, evidence downloads, no external network
requests, and no page errors. Confidentiality and secret scans passed.

| Independently retained evidence | SHA-256 |
| --- | --- |
| Final simulation report | `5d9112cda8f72c5bd95e73da9f6658e6055330f1665ff18e58e84e0f95371951` |
| Strict held-out Gold report | `856d1f7f3dce669da2a34a8658249a685fc96bb8c2948251984d3d5ad2580141` |
| Independent artifact audit | `7a69aac66e3bcee296dd2766bf8004578d07097d75dc520a5511e64403ed1efd` |
| Final standalone HTML | `d672a675ce6e9a5f6699ad2295781db0241a430143f38f0b3bf084f6d7d35b69` |

Exact operator infrastructure coordinates and raw operational artifacts remain
outside Git. These hashes identify the reviewed evidence without publishing
those coordinates.

Stage 12 remains the designed external physical-robot validation handoff. This
qualification proves the declared simulated task, without a physical-hardware
or generalization claim. Older zero-success and 12/64 runs remain dated
historical evidence; neither describes this qualified run.
