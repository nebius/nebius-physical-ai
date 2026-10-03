# Task: reconstruct, compare and recover a NuRec campaign

Read the [common benchmark contract](README.md). Planning estimate: **4–8 hours
per arm**, subject to native calibration and declared GPU concurrency. This task
requires fresh NRE training and RT-core rendering, not only rendering a retained
checkpoint. NRE remains a separately licensed runtime.

## Agent goal

Build a campaign around Workbench's canonical NuRec workflow that reconstructs
two real captures in two exposure variants, renders a view sweep,
survives a controller interruption and rejects stale published media. Implement
the required orchestration and comparison behavior, fix encountered Workbench
issues, and deliver independently verified USDZ, MP4 and Rerun artifacts.

Use [the NuRec skill](../../../../skills/workflows/neural-reconstruction/SKILL.md)
and [canonical workflow](../../../../workflows/main/nurec-reconstruct.yaml).
Run `check → fetch → reconstruct → render → visualize → finalize` through real
Workbench execution. The vendor runtime must run on a verified RT-core GPU;
keep the same hardware and concurrency entitlement for both arms.

## Frozen workload

- Two scenes from the existing preconverted-NCore PPISP path: `struktur28` and
  `toro`. Pin the actual dataset revision, archives and all
  selected cameras after verifying access and calibration. Do not substitute
  the separate, unvalidated COLMAP conversion path.
- Two source variants per scene: `standard` (the original exposure-bracket
  sequence) and `auto` (the reprocessed sequence). The current NuRec path maps
  these to the scene directory and its `_auto` directory respectively. Calibrate
  all four combinations before measurement; an unavailable combination leaves
  this task unready rather than silently substituting a duplicate.
- Train every combination using the full native static 3DGUT recipe. Record
  every effective setting and native split/sampler evidence; a shortened smoke
  training cannot satisfy any combination.
- Four fresh training results per arm. Each checkpoint gets four translations
  `(0,+0.25,0)`, `(0,-0.25,0)`, `(+0.25,0,0)`, `(-0.25,0,0)` meters, at image
  scales `0.5` and `1.0`, with zero rig rotation: **32 render products**.
  Use the same camera selection, frame step and FPS across matched arms.
- Use new publication generations and separate prefixes for every product.
  Source capture reuse is allowed when verified; trained scenes from calibration
  or the other arm cannot replace any of the four required fresh trainings.

The dataset revision, effective recipe and quality floors are calibration inputs
still to be frozen, not claims of a completed sweep. The existing schema
does not by itself implement campaign scheduling or cross-run comparison.

## Engineering milestones

1. Implement a source-bound campaign plan and durable operation journal across
   training, render products and viewer construction. Produce real validation
   and planning receipts, then execute the four training runs. Inspect effective
   NRE settings and actual artifacts before considering training complete.
2. Fan out rendering from each verified checkpoint. Keep source/media identities
   distinct across scales, offsets, cameras and modalities. Collect stage
   evidence and let native jobs finish without repeated LLM polling.
3. Recover from the controlled failures below. Repair the Workbench behavior
   responsible, rerun only invalidated stages, and preserve all failed attempts.
   Integrated patches must pass source tests plus fresh affected native work.
4. Build a comparison index and Rerun recordings showing capture identity,
   exposure variant, validation metrics, novel-view products and failure/recovery
   history. Report the measured quality/time tradeoff without claiming that a
   larger offset is a better reconstruction.

## Controlled recovery exercises

After the first training job succeeds, interrupt only the campaign controller
before it records that terminal result. On restart, reconcile the real run and
checkpoint by digest. Do not pay for another training run merely because the
local acknowledgement is missing.

For one render publication, the harness retains a previously completed
generation and leaves a separate new generation partially uploaded. The agent
must detect the incomplete new inventory, preserve the old generation, and
publish a verified replacement under a fresh identity. Stale media or a success
marker copied from the prior generation must fail the independent grader.

These are deliberate recovery fixtures. They do not establish that current
Workbench has either defect. Existing generation-isolation repairs must remain
intact; focus fixes on behavior actually shown to be missing or broken.

## Independent acceptance

Use [the existing NuRec artifact reader](../workflows/VERIFICATION.md) for each
compatible run tree, with fresh trusted terminal/render receipts. Extend the
trusted campaign grader before measurement to require:

- Four fresh, source-bound training completions; correct dataset revision,
  scene/variant/cameras, effective recipe and native training/split evidence.
- All 32 render products, actual nonzero offset receipts, exact checkpoint
  binding, fully decoded MP4/PNG inventories and no training-view substitution.
- Readable USD stages with nonempty checkpoint storage and finite native
  PSNR/SSIM/LPIPS that meet the predeclared per-scene/variant quality floors.
  Compare identical validation splits between matched arms within each condition;
  exposure variants need not have identical splits. Retain every metric.
- Rerun entities separated by scene, variant, scale, offset, camera and
  modality, with decoded source-matched frames and correct endpoint sampling.
- Controller recovery without duplicate training, no cross-generation media,
  immutable preserved artifacts and successful final integrated-source checks.

The current reader does not prove dataset revision, sampled training coverage,
campaign-wide completeness or a quality floor; new independent evidence is
required. Native validation metrics are vendor measurements. Novel views lack
ground truth here, so do not claim independently measured novel-view accuracy.

## Suggested specialist ownership

- **Reconstruction operations:** inputs, training configurations, job reconciliation.
- **Render publication:** product fanout, generation isolation and publication repair.
- **Viewer integration:** production comparison artifacts and source lineage.

Relevant source includes `npa/src/npa/workbench/nurec/`,
`npa/src/npa/cli/nurec/`, `npa/src/npa/workflows/data_factory_viz.py` and the
canonical workflow. Grant only exact task paths after freezing the proposed
module split. No agent may edit the trusted grader or vendor payload.
