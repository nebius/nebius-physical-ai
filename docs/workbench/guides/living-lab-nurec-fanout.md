# Living-lab digital twin: parameterized neural reconstruction (16-GPU default)

A "living lab" is a research space observed by many cameras. This workflow turns
real multi-view captures of such a space into a **multi-zone digital twin**:
independent NVIDIA NuRec / NRE neural reconstructions, one per RTX PRO 6000
GPU, joined into one composite twin with a contact-sheet panorama and objective
per-zone GPU participation evidence.

The **shipped default is a 16-zone (16-GPU) twin** — eight real PPISP NCore
captures x two view sectors. The same generator also emits a **24-zone
topology** (eight capture pairs x three distinct view sectors) so the design
scales to 24 RTX PRO 6000 GPUs without changing the join or the proof contract.

It is a distinct workflow family from Sim2Real and the Physical AI Data Factory:
it is the shipped **neural-reconstruction** capability, fanned out N ways.

## Topology: derived from capture x sector inputs, 16 by default

Topology size is **derived from explicit capture and sector inputs**, not a
magic fixed count. The shipped default is the operator's reserved
**16 x RTX PRO 6000 Blackwell** capacity:

| | |
| --- | --- |
| Scenes | The 8 real, ungated (`CC-BY-4.0`) NCore V4 sequences in `nvidia/PhysicalAI-NuRec-PPISP` (4 scenes x 2 variants) |
| View sectors | 2 per sequence (`a` / `b`), distinguished by a distinct novel-view rig offset |
| Zones | 16 = 8 sequences x 2 sectors (8 public captures x 2 view sectors) |

Each zone is one **fully independent reconstruction shard** that runs the *entire*
real NRE pipeline (`check -> fetch -> reconstruct -> render -> visualize`) on its
own RTX PRO 6000, then publishes `zone_manifest.json` with objective evidence (GPU
name, timing, USDZ presence, val metrics). The N shards are one SkyPilot JobGroup
bounded by `max_concurrency`, so all N reserved GPUs are materially busy at once.

> Each view sector over the same sequence carries a **genuinely distinct rig
> offset**: the reconstruction itself is deterministic, while the sectors render
> distinct novel-view sweeps. A 24-zone topology is **eight public captures with
> three view sectors (a/b/c)** — not 24 independent physical captures. Every
> shard still runs the full real pipeline on its own GPU and contributes a
> distinct zone twin.

### Scaling to 24 zones (8 x 3)

`npa.workflows.living_lab.build_living_lab_workflow_spec(sectors=("a", "b", "c"))`
emits a 24-zone topology: 24 unique zone names, three distinct sector rig
offsets per capture, `expected_device_count: 24`, and a 24-member
`parallel:` JobGroup whose `parallelCount: "{{config.expected_device_count}}"` is
`24`. The join and proof contract are unchanged — they are size-neutral.

## Expected device count is explicit and validated

The generator derives the expected device count from the generated zone list
(`len(zones)`) and exposes it as `config.expected_device_count`. The fan-out
group wires it to the workflow **`parallelCount`** contract, so
`validate-spec` fails before plan/render/submit if an operator overrides it to
anything other than the actual member count. An operator therefore **cannot
weaken the proof by setting a smaller number than the actual zones**: under- or
over-stating `expected_device_count` (e.g. `8` or `32` against a 16-member
group) is rejected. `max_concurrency` remains a scheduling *cap* (it may be
less than the member count); it never lowers the required-device proof.

The `join` reads the comma-joined zone list (`config.zones`) that the generator
wrote, derives the required device count from it, and fails closed unless every
expected zone participates.

## Ingredients

| | |
| --- | --- |
| **Input** | `nvidia/PhysicalAI-NuRec-PPISP` (8 real NCore V4 sequences), CC-BY-4.0 |
| **Engine** | `nvcr.io/nvidia/nre/nre-ga:26.04` from NGC (pulled, never rebuilt) |
| **GPU** | N x RTX PRO 6000 Blackwell (RTXPRO-6000-BLACKWELL-SERVER-EDITION:1). **Must have RT cores**; never route at H100/H200/B200. Default N=16 |
| **Time** | ~45 min for the 16-zone fan-out on 16 GPUs in parallel (each shard is one full reconstruction) |
| **Credentials** | `NGC_API_KEY` and S3 keys; `HF_TOKEN` only for gated/private dataset overrides |

## The spec

**`npa/workflows/workbench/npa-workflows/living-lab-nurec-fanout.yaml`**

```
living-lab-zones (parallel: N GPU shards, each full nurec pipeline)
      │  parallelCount: {{config.expected_device_count}} (validated)
      │  maxConcurrency: N
      ▼
join (CPU barrier) ──▶ digital_twin.json + panorama.png
```

| Stage | What it does |
| --- | --- |
| `living-lab-zones` | Fans out N independent NRE reconstructions, one RTX PRO 6000 each; `parallelCount` guards the member count |
| `zone-<scene>-<variant>-<sector>` | Full real pipeline for its zone: `check`, `fetch`, `reconstruct`, `render`, `visualize`, `finalize`, then publishes `zone_manifest.json` |
| `join` | Barrier: asserts N/N zones present with real GPU + USDZ, aggregates metrics, builds `reports/digital_twin.json` and `reports/panorama.png` |

Each zone shard runs **inside the NGC NRE container** (`resources.gpu.image`).
The NRE image ships no `npa`, so the runtime stages it from `$NPA_SRC_S3_URI`;
the shard also installs ffmpeg (for `render --export-video`) and the NRE-runtime
Python deps (`nvidia-ncore`, `rerun-sdk`, `pillow`, `pyyaml`) before running the
pipeline, mirroring the shipped single-pod reference at
`npa/src/npa/workbench/nurec/examples/nurec-reconstruct.yaml`. The fetch →
reconstruct → render handoff is same-pod: `reconstruct` takes `--ncore-json`
(the local fetch meta-file) plus `--poses-component-group` / `--camera-id`, and
`render` takes `--artifact-path` (the trained `.usdz`) with `--camera-id`, exactly
as the reference does.

The spec is generated by `npa.workflows.living_lab.build_living_lab_workflow_spec()`
(the committed YAML is the generator's default output;
`test_committed_yaml_matches_generator` guards against drift).

## Fast path (no GPU)

```bash
npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/living-lab-nurec-fanout.yaml
npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/living-lab-nurec-fanout.yaml --run-id preview
```

## The real GPU run

Stage the npa source the pods install (the NRE image has no `npa`), then submit:

```bash
export NPA_SRC_S3_URI=s3://<your-bucket>/npa-src/<tag>
RUN_ID="living-lab-$(date -u +%Y%m%dt%H%M%S)z"
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/living-lab-nurec-fanout.yaml \
  --run-id "$RUN_ID" --runtime \
  --infra k8s/<your-rtx-16gpu-context> \
  --var bucket=<your-bucket> \
  --var prefix="checkpoints/living-lab/$RUN_ID" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env NGC_API_KEY
```

Watch it, then open the run in the NPA agent (it auto-selects the panorama / rrd).
The 16-GPU default remains the validated example; the existing 16-GPU live
evidence still applies to that default.

## Artifacts & validation

```
reports/digital_twin.json   composite twin: N/N zones, per-zone GPU evidence,
                            aggregate mean PSNR/SSIM, size-neutral proof fields
reports/panorama.png        contact-sheet of N zone previews (human-viewable)
zones/<zone>/reconstruction/last.usdz   each zone's renderable USDZ
zones/<zone>/novel_views/*.png          each zone's novel-view renders
zones/<zone>/input/*.png                each zone's real capture frames (GT evidence)
zones/<zone>/reports/sim2real.rrd       each zone's Rerun recording (agent panel)
zones/<zone>/reports/final.json         each zone's real finalize report
zones/<zone>/zone_manifest.json         per-zone objective evidence
```

The join requires **N/N** zone manifests, each with a real GPU identity and a
real USDZ, and it **fails the workflow closed** unless the proof demonstrates
exactly **N distinct, non-empty GPU UUIDs** (never inferred from model names)
with **material all-required-device temporal overlap**. The report records
`concurrency.required_device_count` and `concurrency.all_required_overlap`
instead of a fixed-name field, so the schema is size-neutral across the default
16-zone and any other (e.g. 24-zone) topology. A missing, invalid,
duplicate-device zone — or a zone whose observed unpacked capture disagrees
with its requested scene/variant — fails rather than producing a partial or
unproven twin. Non-positive counts, count/topology mismatches, duplicate
devices, missing timestamps, and insufficient overlap are all rejected with
sanitized diagnostic errors (counts / reasons only).

## Limitations

- Reconstruction is per-zone on one GPU (`--world-size 1`); multi-GPU NRE
  training needs aux-data enabled and is out of scope here (see the
  neural-reconstruction skill).
- Requires N RTX PRO 6000 GPUs (16 by default) and `NGC_API_KEY` (the `-ga` NRE
  image channel).
- Input is limited to the 8 real ungated PPISP sequences; adding a novel capture
  requires the upstream `ncore` authoring path.
- Parameterization changes topology size (zone count, expected device count,
  parallel member count) only; it does not change the per-shard NRE pipeline or
  the proof semantics.
