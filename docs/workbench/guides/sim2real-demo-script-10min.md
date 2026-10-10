# Sim2Real: ten-minute result walkthrough

[Guides](README.md)

Present a verified run of the canonical
[`workflows/main/sim2real.yaml`](../../../workflows/main/sim2real.yaml).
The ten minutes describe presentation time, not the workflow's runtime.
Follow the [execution guide](sim2real-workflow.md) to prepare inputs, select a
qualified five-image bundle, provision capacity, submit, and resume. Run the
workflow on an always-on operator VM before the presentation.

## Before presenting

Use artifacts from one actual run. Check its final report, strict policy score,
image/source identities, split separation, and independently decoded recordings.
Keep private infrastructure identifiers, credentials, and customer inputs out of
slides and shared recordings. Follow the
[artifact sharing guide](../rerun-sharing.md) when sharing is requested.

Inspect durable status and artifact discovery with the same project configuration
used for submission:

```bash
npa workbench workflow status "<run-id>" --project "<project-alias>"
npa workbench workflow artifacts "<run-id>" --project "<project-alias>" --json
```

Download the exact run's `reports/sim2real.rrd` and `reports/sim2real.mcap` to a
private local directory. Independently inspect the Rerun bytes before opening
its viewer:

```bash
rerun rrd verify /path/to/private-run/sim2real.rrd
rerun rrd print -vv /path/to/private-run/sim2real.rrd
rerun /path/to/private-run/sim2real.rrd
```

A completed report must contain fourteen ordered ComponentRecords: thirteen
`WORKS` records and the Stage 12 external physical-validation `SEAM`. Stage 14
must publish both non-empty recordings; missing encoders or artifacts fail the
stage. A reduced orchestration test can complete with a weak policy. Present the
measured strict gold result separately from pipeline completion.

## 0:00–1:00: the graph and run identity

Show the [architecture diagram](sim2real-architecture.md) and the selected run's
final report. Explain that the standard workflow runtime owns state jobs,
parallel EnvGen waves, nested loops, and recovery. Its durable ledger is
`npa-workflow/runtime.json` under the run prefix. There is no private launch pack
or separate Sim2Real controller to reconstruct.

Show the exact source SHA and immutable image digests. Isaac states use RTX PRO
6000; compute-only Transfer and EnvGen may use available B200 capacity. The
workflow's `gpu_concurrency` must fit the actual pool.

## 1:00–3:00: inputs, augmentation, and sealed scenarios

| Stage | Evidence to open | What it proves |
| --- | --- | --- |
| 1: trigger | `stage_01_trigger/trigger.json`, `task-dataset-manifest.json` | A task-aligned, verified input dataset |
| 2: task and assets | `stage_02_assets/task-contract.json`, consumed spec files | Robot, object, physics, camera, and success identities |
| 3: augmentation | `augment/manifest.json`, actual generated frames | Real Cosmos Transfer output and its lineage |
| 4: EnvGen | Raw shard manifests and GPU ComponentRecords | Actual generated scenario configurations |
| 5: curation and split | `envs/manifest/curation-manifest.json`, `split-manifest.json` | Accepted/rejected cases, measured coverage, disjoint train/validation/gold digests |
| 6: feature lineage | `tokens/manifest.json` | Scenario configurations feed state PPO; pixels remain visualization and VLM evidence |

Show actual accepted and rejected counts. The default raw request is 10,000
scenarios across eight shards; report the measured retained counts rather than
assuming all generated cases passed curation.

## 3:00–5:00: rendered rollouts and hosted feedback

Open primary, side, and overhead views from one actual Stage 7 rollout. Show the
robot, cube, end effector, and intermediate actions. Inspect the primary view
specifically: Stage 8's hosted evaluator consumes primary frames.

The current default model is `MiniMaxAI/MiniMax-M3` on Token Factory. An
explicitly authorized endpoint may select Cosmos3; use the model identity in
this run's evidence when describing results. Stage 8 uses one model, all declared
primary frames by default, and concurrent independent rollout requests.

Show an evaluation's exact action-to-frame bindings and a verified receipt.
Receipts bind source, endpoint identity, model, action metadata, and frame bytes.
A retry reads back matching receipts before reusing results. Unsupported visual
actions retain neutral labels and zero confidence; do not describe them as
observed task success.

## 5:00–7:00: real PPO and validation selection

Show the temporal training signal and measured PPO telemetry. Production defaults
use 1,024 vector environments and 2,000 PPO updates per pass. Report actual
observed updates and finite loss/reward measurements from the run.

Show each candidate checkpoint's validation result and the deterministic
selection record. Checkpoint selection uses validation only. Demonstrate the
exact selected SHA-256 and byte size; avoid choosing a checkpoint using gold
outcomes. The learned actor alone must perform held-out inference, without a
scripted controller after its actions.

## 7:00–9:00: untouched gold and useful recordings

Open `eval/gold-heldout/outer-XX/report.json` for the actual selected outer
iteration. Show the strict result over 64 gold scenarios, its per-environment
physical measurements, and the selected checkpoint loaded for inference.

Success requires final object-to-goal distance below 5 cm with stable placement.
The quality threshold is 0.50. Wider distance diagnostics do not replace that
criterion. Explain the recorded loop decision: production defaults permit three
inner passes per outer iteration and three outer iterations, with early exit
when the quality threshold passes.

Use the independently decoded Rerun and MCAP to show actual primary/side/overhead
gold footage, camera timestamps, policy provenance, critiques, PPO curves, stage
progress, and held-out scores. Compare decoded metric coverage with the source
telemetry. A viewer opening alone does not prove artifact completeness.

## 9:00–10:00: recovery and physical handoff

Show the standard runtime's recovery evidence: completed waves and verified
artifacts were reused, an in-flight job was adopted when present, and incomplete
work continued under the same run identity. Verified Stage 8 receipts preserve
completed hosted work across retries.

Stage 12 remains an external physical-validation seam. Stage 13 records the
next-dataset handoff. State precisely whether the simulation policy passed the
strict threshold and whether external physical validation has been supplied.
Do not present simulation success as deployment on a real robot.
