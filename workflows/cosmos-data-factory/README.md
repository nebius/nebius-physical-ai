# Cosmos Data Factory: IAA and EVG

Two separate Early Access NVIDIA Physical AI Data Factory workflows, adapted to
`npa.workflow/v0.0.1` and executed through NPA's standard SkyPilot runtime.

| Workflow | Guide | Purpose |
| --- | --- | --- |
| [Image Attribute Augmentation](paidf-image-attribute-augmentation.yaml) | [IAA getting started](image-attribute-augmentation.md) | Edit clothing, colors, and footwear in person images; verify attributes and publish a searchable image dataset |
| [Event Video Generation](paidf-event-video-generation.yaml) | [EVG getting started](event-video-generation.md) | Generate event videos from seed images; detect, track, caption, answer event/person questions, and assemble an annotated dataset |

This package covers these two workflows and their setup/operating skills from
NVIDIA's IAA/EVG Early Access announcement. Keep CDF-specific specs, guides, and
proof guidance in this directory. The current main PAIDF workflow remains at
`workflows/main/paidf-cosmos3.yaml`; its video-to-video graph and automatic edge
conditioning are a separate integration. IAA and EVG do not add that graph or
its edge-conditioning stages to their published task definitions.

For the right seed data and a reviewable standalone HTML, use the
[proof guide](proof.md). The local [HTML template](proof-template.html) and
[evidence packer](build-proof.py) produce one offline report from reviewed
operator-private media; they do not add another workflow or launch GPU jobs.
IAA needs person images; EVG needs scene images suitable
for the requested event. The workflow guides list the current adapter's sampling
coverage and implementation gaps; the announcement's full configurable attribute
and event vocabulary is not implemented. A robot-dataset frame can be evaluated
as an EVG seed, but that tests image-to-event-video synthesis. Robot-video augmentation,
action preservation, and policy generalization require their own evidence.

For recognizable Physical AI evaluation, the proof guide describes real
human–robot photographs for IAA and robot-workcell scene seeds for EVG. The
report takes source identity, sampled values, actual outputs, native acceptance,
and visual verdicts from reviewed run evidence. Pending runs never inherit a
historical fixture's successful metrics. Keep person-view limitations and
source-to-sampled-environment mismatches explicit.
For EVG, the proof template can show the photo-to-event-to-label transformation,
pause the original clip at reviewed timestamps, and explain model labels in
plain language. The [proof guide](proof.md#build-the-local-report) documents
the optional explanation and EVG-first display fields.

The implementations already existed in Workbench's testing catalog. These are
their canonical locations; workflow names, run prefixes, model revisions, stage
graphs, and existing live-matrix registrations are retained. The generic catalog
resolver and the agent's `iaa`/`evg` aliases select these files in source checkouts
and installed packages. Update direct references to the old testing paths.
Video-to-video augmentation remains a separate
[PAIDF + Cosmos 3 workflow](../guides/paidf-cosmos3.md).

## Robotics fit and reuse

The announcement highlights IAA and EVG. NVIDIA's broader
[PAIDF catalog](https://github.com/NVIDIA/physical-ai-data-factory#paidf-workflows)
lists four workflow families: DIG, VDA, IAA, and EVG; families can have multiple
execution modes. The two specs here extend that ecosystem using existing NPA
adapters. Robotics usefulness depends on the target task, not the workflow name.

| Target task | Existing path to reuse | Evidence needed for that use |
| --- | --- | --- |
| Robot-demonstration appearance augmentation | [Main PAIDF](../main/paidf-cosmos3.yaml), with its input-derived structural controls and quality gate | Selected ALOHA/other robot episodes retain motion timing, robot/object geometry, contacts, and padding; accepted data improves a held-out policy evaluation |
| Human/workcell safety perception | [EVG](paidf-event-video-generation.yaml) | Generated events visibly occur in representative workcell scenes, and labels support the intended safety detector |
| Human-appearance robustness near robots | [IAA](paidf-image-attribute-augmentation.yaml) | Person identity and useful labels survive the requested clothing edits; the target perception evaluation improves |
| Industrial visual inspection | [Existing DIG manual-ROI workflow](../testing/paidf-defect-image-generation.yaml) | Defect/clean images and native labels improve the intended inspection detector; the NPA spec covers the manual-ROI branch |

This is a task-fit assessment, not evidence of downstream performance. Current
`main` also contains `workflows/testing/nvidia-paidf-vda-cosmos-transfer25.yaml`
for the NVIDIA-derived Transfer 2.5 path. Reuse its existing implementation when
that backend is required; keep both VDA references in their current locations.
Do not create another robot augmentation graph inside this package.

Reuse the existing [LeRobot skill](../../skills/tools/lerobot/SKILL.md) and
[Sim2Real workflow](../main/sim2real.yaml) for relevant training/evaluation
operations. Generated MP4s alone are not a LeRobot training dataset. Preserve
source episode/camera/frame identity and explicitly verify action/timestamp
alignment before attaching demonstration labels. For a policy-training claim,
compare the same held-out task evaluation with original-only and augmented-data
training after the data-quality gate passes. Keep IAA/EVG's perception uses
distinct from that policy-training claim.

## YAML consumption decision

**Convert the orchestration to NPA YAML; reuse pinned component configuration.**
The [ecosystem overview](https://github.com/NVIDIA/physical-ai-data-factory)
links to [paidf-orchestration](https://github.com/NVIDIA/paidf-orchestration),
where IAA and EVG are Python Airflow DAGs. Their Kubernetes manifest YAMLs are
inputs to Airflow's custom operators, not complete portable workflow graphs.
Passing those manifests to `npa workbench workflow submit` would not execute
their DAG dependencies, XCom handoffs, service lifecycle, or cleanup rules.

| Upstream artifact | NPA handling |
| --- | --- |
| Python DAG and task groups | Translate into the two checked-in `npa.workflow` state graphs |
| `*_k8s_manifest.yaml`, Helm values, Airflow pools | Replace with NPA resource profiles and standard submit; do not import as workflow specs |
| `configs/cosmos_config.yaml` | Runtime-fetch at the pinned revision and customize media paths, models, endpoints, sampled variables, and output paths for the real `paidf-augmentation` client |
| Auto-label configs, prompt/question banks, and client protocols | Reuse the pinned upstream contracts through the existing `workflow.paidf.*` adapters |
| Airflow payload JSON and XCom | Map supported settings to NPA `config`; persist typed manifests in S3 |
| Upstream agent skills | Adapt into [skills/cosmos-data-factory](../../skills/cosmos-data-factory); replace Airflow setup/trigger/monitor instructions with NPA commands |

The authoritative orchestration revision is
`f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9`, verified against upstream `main`
when this directory was introduced. Client source, model snapshots, image
parents, licensing, and runtime adaptations are independently pinned in
[NOTICE-NVIDIA-PAIDF](../../skills/NOTICE-NVIDIA-PAIDF). Do not replace a pin
with a moving branch just because the Early Access guide changes.

The adaptation preserves the real generation and labeling components and their
three retries with a 30-second delay. Generation service and consuming batch run
in one state; the adapter owns shutdown on completion or failure. Airflow's
deployment scheduler, pools, XCom, `all_done` callbacks, and REST-derived HTML
timing dashboard are not reproduced as controller features. NPA uses durable
S3 handoffs, standard runtime state, stage timings, and terminal validation JSON.
The workflow guides describe data-preparation and inference differences; this
is semantic translation, not control-plane equivalence.

## Shared NPA setup

Use the [NPA installation guide](../../docs/install.md), configure the intended
project with `npa configure`, and bootstrap SkyPilot with `npa skypilot bootstrap`.
The upstream [shared setup guide](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/docs/getting-started.md)
explains NVIDIA's reference deployment. This integration uses NPA rather than
installing that Airflow Helm release or importing its `secrets.env`.

For an actual run, supply:

- A configured project, exact Kubernetes context, worker-readable S3 input
  prefix, and operator-owned output bucket. Each fresh run gets its own prefix.
- S3 credentials, `NEBIUS_TOKEN_FACTORY_KEY`, `HF_TOKEN`, and `NGC_API_KEY`
  through NPA's credential store or private environment. Forward secret names
  with `--secret-env`; never put values in YAML or command arguments.
- Exact scanned operator-private image digests for every selected service.
  Shipped `registry.example.invalid`/zero-digest values deliberately cannot run.
  Follow the [compatibility-image recipes and acceptance requirements](../../docs/workbench/guides/physical-ai-data-factory.md#native-dig-iaa-and-evg-execution-contracts).
  These images are restricted; upstream access does not grant redistribution.
- Capacity for IAA's one-B200 generation state or EVG's two-B200 generation
  state on one node. EVG detection, captioning, and Visual QA each request one
  B200 in subsequent states; hosted VLM inference does not remove the decoder's
  GPU requirement. CPU states and SkyPilot's controller also need capacity.
  These serial graphs do not sum all stage GPU requests into simultaneous demand.

Validate credentials and exact capability access before provisioning or GPU
submission. IAA uses `paidf-iaa,paidf-label-attribute-search`; EVG uses
`paidf-evg,paidf-label-detection,paidf-label-captioning,paidf-label-visual-qa,paidf-label-attribute-search`.
See the workflow-specific commands below. Image building/publishing is a
separate operation; a spec does not build its private dependencies.

Both specs accept these existing configuration keys:

| Setting | Meaning / shipped default |
| --- | --- |
| `bucket` | Output bucket; `example-bucket` is planning-only |
| `input_uri` | Staged S3 image prefix; replace the placeholder with authorized inputs |
| `prefix` | Output prefix, default `<workflow-name>/{{run.id}}` |
| `num_augmentations` | Positive variants per prepared image, default `1` |
| `seed` | Deterministic attribute/event sampling, default `42` |
| `vlm_model`, `llm_model` | Explicit hosted models; verify availability in the selected account |
| `generation_image`, `attribute_search_image` | Required private digest references |
| `detection_image`, `captioning_image`, `visual_qa_image` | Additional private digest references required by EVG |

The shipped hosted models are reference selections, not an account availability
guarantee. VLM/LLM URLs must use the supported Token Factory HTTPS origin.
The generation model and revision are bound to the reviewed Qwen/Cosmos service;
arbitrary generation backend substitution is rejected. The generation endpoint
is loopback inside the generation state.

## Monitoring and outputs

Use the same run ID, project, input, and image overrides from the selected guide:

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
npa workbench workflow logs "$RUN_ID" --project "$PROJECT" --stage validate-final-outputs
npa workbench workflow artifacts "$RUN_ID" --project "$PROJECT" --json
```

Inspect `reports/upstream.json`, the final dataset manifest, and
`reports/terminal-validation.json` under the run prefix. Required sidecars must
reopen and validate; a generated filename alone is not success. Inspect actual
images/videos for quality and read QA coverage separately from protocol success.
Raw generated media and any separately produced Rerun recording are distinct
artifacts; neither workflow declares an automatic Rerun stage.

Resume interrupted work through generic `submit --resume-run "$RUN_ID"` with
the same immutable input/source/image configuration. Cancel with
`npa workbench workflow cancel "$RUN_ID" --project "$PROJECT" --json`, then
verify terminal state before destroying owned infrastructure.

The [existing native live evidence](../../docs/workbench/guides/physical-ai-data-factory.md#native-live-validation-evidence)
describes accepted digests, representative results, and visual/QA limitations.
It does not qualify a replacement image or a new dataset.

## Agent skills

Start with the local [CDF entry point](SKILL.md) when working in this directory.
Load [shared setup](../../skills/cosmos-data-factory/cosmos-data-factory-setup/SKILL.md),
then [IAA](../../skills/cosmos-data-factory/physical-ai-image-attribute-augmentation/SKILL.md)
or [EVG](../../skills/cosmos-data-factory/physical-ai-event-video-generation/SKILL.md).
All three are registered in `skills/index.yaml`. They adapt NVIDIA's pinned
skill content; Airflow commands and payload schemas are not NPA commands.
