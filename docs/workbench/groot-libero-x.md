# GR00T N1.7 LIBERO-X closed-loop evaluation

[Workbench docs](README.md) · [GR00T training cookbook](cookbooks/groot-1-7-training.md) · [strict workflow](../../workflows/testing/groot-libero-x-closed-loop.yaml) · [observed-paired workflow](../../workflows/testing/groot-libero-x-observed-paired.yaml)

`groot-libero-x-closed-loop.yaml` is a five-stage simulator evaluation for the
public `rohansiva/gr00t-libero-x` derivative.  It is deliberately separate
from the derivative card's reported open-loop MAE/MSE.  A completed run reports
actual native-LIBERO task success, the corresponding action MSE/MAE, decoded
MP4 rollouts, a hash-bound comparison, and an independently inspected Rerun
recording.  It never claims physical-robot success, benchmark convergence, or
that a local smoke is a representative benchmark.

## Lineage, credit, and terms

| Component | Exact identity and credit | Terms and delivery decision |
| --- | --- | --- |
| Derivative checkpoint | [Rohan Siva's `gr00t-libero-x` model card](https://huggingface.co/rohansiva/gr00t-libero-x) at `b9dfbdcce8da61950db4f34199fff30b286b8f13`; the card identifies it as a fine-tune of NVIDIA `libero_10`. | Model-card label: Apache-2.0. It is runtime-fetched at its immutable revision, never added to public image layers. |
| Base checkpoint | [NVIDIA `GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`, `libero_10`; NVIDIA is the model developer. | NVIDIA Open Model License Agreement. It is runtime-fetched at its immutable revision, never redistributed by this change. |
| GR00T evaluator | [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) `51d4c89f72fda44cbf77285c6a8114b52676b8a1`; Apache-2.0, copyright notices retained in the fetched source. | The existing immutable `npa-groot:0.1.0` bootstrap carries an older GR00T ref, so each policy stage fetches this exact public Apache source at runtime, verifies its Git SHA, then creates the upstream Python 3.12 server and LIBERO client environments. No new image is published and no upstream source is relabelled as NPA work. |
| Native simulator | [Lifelong Robot Learning's LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) submodule `8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT, copyright 2023 Lifelong Robot Learning. | Materialized only by Isaac-GR00T's documented `setup_libero.sh` inside the run-private/managed cache. It is not baked or redistributed by this workflow. |
| LIBERO-X evaluator | [Meituan's LIBERO-X source](https://github.com/meituan/LIBERO-X) `f528726421c7211d8eb05fe48e9e5e2535ccc813`; MIT; credit Wang, Zhang, Liu, Zhang, Cai, Liu, and Liu. | Runtime-fetched beside Isaac-GR00T, SHA-verified, and kept in the private cache. A hash-recorded NPA compatibility overlay registers only manifest-selected BDDL tasks in GR00T's existing native LIBERO workers; it does not claim that unmodified Isaac-GR00T supports LIBERO-X. |
| Evaluation data | [Meituan `LIBERO-X`](https://huggingface.co/datasets/meituan/LIBERO-X) `73053111f932d4dbaee995e3f06c2f42b3ad4adc`, CC-BY-4.0; credit Wang, Zhang, Liu, Zhang, Cai, Liu, and Liu, *LIBERO-X: Robustness Litmus for Vision-Language-Action Models* (2026). | Operator materializes the selected GR00T-format subset under a run-scoped S3 prefix. The workflow retains the card's CC-BY-4.0 attribution/revision in the protocol and does not package dataset bytes. |

The model/data licenses above are independently recorded rather than inferred
from access.  This change adds no EULA, `ACCEPT_*` variable, checkbox, or
telemetry consent.  The only access check rendered by these model stages is the
pre-existing exact-payload Hugging Face capability; successful access is not a
redistribution grant.  Model, dataset, cache, simulator output, and public OCI
image decisions remain distinct.

NVIDIA GR00T should be credited with *GR00T N1: An Open Foundation Model for
Generalist Humanoid Robots*, NVIDIA et al., arXiv:2503.14734 (2025).  Native
LIBERO should be credited with Liu et al., *LIBERO: Benchmarking Knowledge
Transfer for Lifelong Robot Learning*, arXiv:2306.03310 (2023).  The workflow's
output provenance includes model/data/source revisions, license labels,
upstream attribution, the resolved LIBERO submodule SHA, input-inventory hash,
and protocol hash.  NPA's modifications are the task-disjoint protocol,
orchestration, comparison, and artifact inspection; they are not a claim of
authorship over any upstream model, code, simulator, or dataset.

## Required inputs and workflow

The data card does not establish the derivative's exact 60 training task IDs
or a native simulator mapping.  Before GPU work, an operator must provide
three run-scoped JSON inputs:

1. A `npa.groot_libero_x.training_tasks.v1` manifest naming **exactly 60**
   derivative training task IDs.
2. A `npa.groot_libero_x.evaluation_tasks.v1` manifest with disjoint task IDs,
   `libero_x/<logical-task-id>` environments, an exact
   `libero/libero_x/bddl/LEVEL1...LEVEL4/<task>.bddl` path from the pinned
   evaluator source, and concrete global LeRobot trajectory IDs.  Classic
   `libero_sim/...` task names remain accepted for compatibility, but do not
   establish LIBERO-X coverage.
3. A `npa.groot_libero_x.evaluation_dataset.v1` manifest binding the selected
   GR00T-format dataset prefix, exact LIBERO-X revision/license, task-manifest
   hash, and object-inventory SHA-256.

The preparation stage rejects overlap, undeclared data, unsafe BDDL paths,
changed object inventories, and a training cohort that does not contain 60
unique tasks.  It writes a protocol consumed verbatim by both policy stages:

```text
prepare disjoint tasks/data
  -> NVIDIA base checkpoint: current GR00T server + LIBERO rollouts
  -> derivative checkpoint: same protocol and native runtime
  -> matched success + MAE/MSE comparison
  -> MP4 readback + inspected RRD evidence
```

Both policy stages call upstream
`gr00t.eval.rollout_policy.run_gr00t_sim_policy` through the upstream
server/client split.  For a `libero_x/...` task, the runtime derives its
instruction from the reviewed BDDL instead of trusting a manifest prompt and
maps the logical task to a stable internal `libero_sim/...` Gym identifier.
The registration bridge is present in every spawned vector-environment worker,
so the configured upstream `n_envs` is retained.  The evaluator refuses empty
episode sets, non-finite action metrics, task/result mismatch, runtime SHA
mismatch, and absent MP4s.  The runtime cache uses a per-revision lock and an
atomically published ready marker; without the managed GR00T data mount it is
private to the workflow pod.  Git LFS filters are disabled for the source-only
runtime fetch, so an absent `git-lfs` binary cannot silently turn a source
checkout into a failed or partial cache.
The GPU placement is an NPA workflow configuration input
(`evaluation_accelerator`), rather than a fixed graph resource.  Resolve it
against the selected target with `--var evaluation_accelerator=<catalog-name>:1`
or the existing `NPA_WORKFLOW_GPU_ACCELERATOR` operator override; no hardware
identifier or credential is embedded in a stage command.

### Observed paired mode when coverage is unknown

[`groot-libero-x-observed-paired.yaml`](../../workflows/testing/groot-libero-x-observed-paired.yaml)
is a separate five-stage native comparison for the case where the derivative's
exact training inventory remains unpublished. It accepts an
`npa.groot_libero_x.observed_tasks.v1` manifest with the same safe BDDL and
trajectory requirements as the strict evaluation manifest, binds the task/data
hash and run seed once, and passes that exact protocol to both policy arms.

Its protocol, rollout reports, comparison, RRD provenance, and evidence index
carry `training_coverage: unknown`, `task_disjointness: unverified`, and
`held_out: false` / `generalization: false`. It is useful factual closed-loop
evidence for the observed tasks, but never a substitute for the strict workflow
or a claim about unseen tasks. It adds no acceptance mechanism or new upstream
terms.

## Readiness and acceptance

Static workflow validation, planning, tool-argument coverage, native-handoff
contract tests, artifact comparison, and real RRD inspection are verified in
the adjacent readiness record.  They are not a live benchmark result.

Live execution remains unverified until the operator supplies the three
hash-bound inputs (including the authoritative 60-task training inventory),
passes the existing storage/Nebius/Hugging Face preflight, confirms the
unchanged immutable bootstrap image is pullable, and runs the five-stage
workflow on Kubernetes.  Acceptance then requires independent readback of the
final comparison, native MP4s, and RRD—not just a successful job status.
Preserve the produced provenance and licenses with any shared result; do not
reuse the reported derivative-card open-loop values as closed-loop evidence.
