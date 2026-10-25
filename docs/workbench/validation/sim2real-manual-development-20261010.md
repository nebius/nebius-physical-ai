# Manual Sim2Real development execution: October 10, 2026

This receipt records a manual Linux run of the canonical 14-stage workflow
through the standard NPA CLI and SkyPilot runtime. No NPA agent was deployed.
The RTX PRO 6000 Blackwell pool and shared filesystem already existed;
the operator checkout, namespace, client state, controller, and selected cache
preparation were isolated. Fresh physical-cluster creation was not tested.

Native execution used CLI/workflow revision
`685d34d692ab94263fa6fc49ce1f603a766a4219`. Cleanup was verified after installing
`886f80b0a4c503f16fe73928a5211ca7eeeb99a2`, which fixes the pinned SkyPilot
status notice. The canonical Sim2Real spec and worker adapters are unchanged
between those revisions. The full GPU run was not repeated after the
operator cleanup fix.

For a development reproduction, select the checkout containing both operator
fixes after cloning and before installing:

```bash
git fetch origin 886f80b0a4c503f16fe73928a5211ca7eeeb99a2
git checkout --detach 886f80b0a4c503f16fe73928a5211ca7eeeb99a2
```

Continue with the [manual runbook](../guides/sim2real-workflow.md), using your own
project, storage, model credentials, and approved cluster inputs. All five
images share worker source `9d99205e6ae8beba85f2423c2d66cc865092f7d2` from
[PR #921](https://github.com/nebius/nebius-physical-ai/pull/921) and its
[secure publication run](https://github.com/nebius/nebius-physical-ai/actions/runs/38016195073).
The tested CLI revision and common worker source are separate provenance fields.
Anonymous image identity checks and actual Kubernetes pull/bootstrap passed for
each exact digest:

```bash
SOURCE_SHA=9d99205e6ae8beba85f2423c2d66cc865092f7d2
CONTROLLER_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-sim2real-control@sha256:0e94b8ebd15a4373dd8e5524b955c715fc4cb418a0be6798b27ef0b164ff315d
TRANSFER_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-cosmos2-transfer@sha256:761abca0b37e1a9a964e505d25edb77d2bc26ef7c9d97afb2b72ae69c6cb7c49
ENVGEN_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:cf78bb34dbf0415a3e5080879468f5f325c62831d30f9bbb947b24d9e312b252
ISAAC_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab@sha256:4747e52c7425a8752d24823e10d28e92e97a7a65b9167301f2416c58c4b8dd41
VIEWER_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-rerun-viewer@sha256:a45bdb0046732e2d4435b26cba22019dfb390edd03f59cae324966b8b1da41d6
```

These are development candidates with measured execution evidence. This receipt
does not promote release defaults, change publication quarantine, or mark the
checked-in readiness record's `source_image` gate verified.

The run retained 10,000 scenarios, eight shards, 64 rollouts with 32 action
samples per batch, 1,024 PPO environments, 24 steps per environment per update,
2,000 updates per pass, three inner and three outer iterations, and 64
validation/gold episodes. GPU concurrency was one for the selected one-GPU pool.
Strict stable placement remained 5 cm with promotion threshold `0.50`.

| Check | Measured result |
| --- | --- |
| Seed and Transfer | Real public Franka seed staged; real Transfer MP4 independently decoded |
| Isaac preparation | Cold bootstrap completed in 4,426 seconds as UID/GID 1000; exact versioned tree reported ready; CUDA and RTX graphics passed |
| Scenario splits | 10,000 unique scenarios; 8,000 train / 1,000 validation / 1,000 gold, with no overlap |
| Rollouts and hosted evaluation | Nine 64-rollout batches and 576 actual hosted requests; all 33 primary frames per request; exact action/frame bindings and bounded, non-degenerate signals |
| PPO | Nine complete passes: 18,000 contiguous updates and 442,368,000 measured environment steps; all 8,000 training scenarios applied each pass |
| Checkpoints | SHA/size, finite tensors, changed actor parameters, optimizer continuation, validation-only ranking, and exact held-out membership independently verified |
| Strict validation | All nine batches scored 0/64 |
| Strict gold | All three batches scored 0/64; 2,487 / 2,457 / 2,478 canonical camera images respectively across 64 rendered episodes per batch |
| Native completion | Runtime succeeded; exactly 14 ordered canonical records, Stage 12 SEAM and all others WORKS |
| Durable recovery | Native same-ID recovery after driver interruptions at Stages 7, 8, and 14; finalization job adopted without a duplicate driver |
| Recordings | Official readers decoded 13,069 RRD chunks, sampled pixels in all 1,024 camera entities, and matched 6,000 PPO samples to native telemetry; all 21,499 MCAP camera images decoded with consistent 10 FPS timestamps |
| Owned cleanup | Native cancel found all 50 recorded jobs terminal; native controller removal verified remote absence and local metadata cleanup; supported SDK stop removed the original owned API process/session; no active pods remain in the workload namespace |

Each PPO pass's full log retained 2,000 measured updates; its cumulative timer
was monotonic across phases. The passes took about 41–44 minutes each before
validation and uploads. Cold setup, simulation, evaluator requests, and
finalization add separate time. The full run takes hours. The measured RRD and
MCAP files occupy about 26 GB together; allow disk space for the selected
downloads and readers.

Camera samples were independently decoded and 14 frames visually reviewed from
each rollout and gold batch. Ordinary secondary-view arm occlusion and object
movement near the primary image edge were retained. Training rewards do not
replace the strict placement result: the policy missed the threshold. Stage 12
records the physical-robot validation seam; no physical robot was tested.

The recordings retain measured PPO curves from the final outer iteration's
three inner passes. Twenty-four decoded rollout pixel samples and eight gold
pixel samples exactly matched their canonical PNGs; six extracted RRD samples
were also visually reviewed.

| Recording | Bytes | SHA-256 |
| --- | --- | --- |
| `reports/sim2real.rrd` | 15,403,205,655 | `cf70282f45b88bbebd982c05cc376e2d32dbcaba83946292b54b29d253274ade` |
| `reports/sim2real.mcap` | 10,490,552,533 | `25008eaf67694e73893cccb9590f9e1fcbeb7e0ec0f0e5fdff2fac52f88cae0e` |

The original live queue-status failure was retained and recovered using native
`--retries 1` after the queue-reader fix. The first controller-cleanup attempt
also rejected the pinned configuration notice; the committed parser fix and
standard cleanup retry passed. Both original failures remain in private evidence.
No worker source overlays, manual cache
ownership changes, archived-output edits, relaxed success predicates, shortened
training settings, or hidden sibling GPU jobs were used. Shared pool/filesystem
resources and retained run outputs remain under their owners' control.
