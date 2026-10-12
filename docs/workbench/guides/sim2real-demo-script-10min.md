# Present a Sim2Real result in 10 minutes

[Guides](README.md) · [Manual setup and execution](sim2real-workflow.md) · [Data contracts](sim2real-data-contracts.md)

Prepare an actual completed canonical run before the presentation. Use the
manual runbook for installation, qualified images, access, cluster/cache setup,
seed staging, submission, and cleanup. A presentation does not require another
run. If showing a live run alongside completed artifacts, label each run clearly.

Retain the completed run's ID and project alias privately. From the checkout
root, inspect the terminal status and artifact inventory:

```bash
RUN_ID='<completed-run-id>'
NPA_PROJECT='<local-project-alias>'
npa/.venv/bin/npa workbench workflow status "$RUN_ID" --project "$NPA_PROJECT"
npa/.venv/bin/npa workbench workflow artifacts "$RUN_ID" --project "$NPA_PROJECT"
```

Download the selected report, validation/gold evidence, checkpoint metadata,
RRD, and MCAP through the Nebius storage console or your S3 client. The optional
[AWS CLI examples](scoped-storage-transfers.md) use explicit endpoint/profile selection.
Decode the recordings before rehearsal. Keep private S3 locations and provider
identifiers out of shared slides, screenshots, and exported reports.

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–1:00 | [14-stage graph](sim2real-architecture.md) and terminal runtime status | One standard workflow owns stages, parallel shards, iterations, and durable state |
| 1:00–2:30 | Trigger manifest, task contract, Transfer manifest, curation and split manifests | Real seed data leads to simulated scenarios; train, validation, and gold remain disjoint |
| 2:30–4:00 | Actual primary/side/overhead rollout frames and action metadata | Rendered task visibility and exact action/frame/episode bindings establish usable evidence |
| 4:00–5:30 | Hosted critique, training signal, measured PPO losses | Stage 8 records the actual model/provider; bounded grounded critique shapes real PPO |
| 5:30–7:00 | Validation selection and gold report | Validation chooses the checkpoint; gold measures strict 5 cm stable placement |
| 7:00–8:00 | Promotion decision and candidate metadata | Report the measured result, including below-threshold outcomes; completion alone does not prove efficacy |
| 8:00–9:00 | Stage 12 external seam and Stage 13 retrigger record | Physical-robot validation is external; retrigger requires verified new or corrected data |
| 9:00–10:00 | Decoded RRD/MCAP timeline and final report | All 14 ComponentRecords and accessible artifacts connect execution to the result |

The final report must show Stage 12 as `SEAM` and the other 13 canonical
components as `WORKS`. Missing RRD/MCAP is a verification failure, not an optional
visualization fallback. Playback durations do not establish physical grasp or
placement duration; use recorded simulation timestamps and measured predicates.

If a separate live run is still executing, show its real status/logs and continue
the artifact tour using the clearly identified completed run. If it failed, show
the failure honestly. Do not rewrite its decision, switch its checkpoint, or
present another run's artifacts as that run's result.

```bash
npa/.venv/bin/npa workbench workflow logs "$RUN_ID" --project "$NPA_PROJECT"
```

Do not run setup, rebuild images, regenerate invalid footage, or delete shared
infrastructure during the presentation. Resume and teardown use the exact
original arguments and the [manual runbook](sim2real-workflow.md#resume-and-verify).
