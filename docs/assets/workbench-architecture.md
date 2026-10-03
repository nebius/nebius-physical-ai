# Workbench architecture diagram reference

The [PNG overview](workbench-architecture.png) maps the solution ecosystem,
container distribution, control layer, and Nebius services. The
[README task flow](../../README.md#how-workbench-runs-a-task) describes how an
individual workflow executes.

## Platform architecture and sources

| Element | Meaning | Source |
| --- | --- | --- |
| NVIDIA | Selected integrations: Isaac Lab / Isaac Sim, GR00T, SONIC, Cosmos 3, and Transfer 2.5. | [Public container catalog](../workbench/container-image-catalog.md) |
| Open ecosystem | Selected integrations: LeRobot, Genesis, FiftyOne, LanceDB, Rerun, and Foxglove. | [Public container catalog](../workbench/container-image-catalog.md) |
| NPA packaging → GHCR | Repository-selected public runtime images use `ghcr.io/nebius/nebius-physical-ai`. Resolve release tags to digests for reproducibility. | [Image resolver](../../npa/src/npa/deploy/images.py), [release manifest](../../npa/src/npa/deploy/public_release_manifest.json) |
| Runtime fetch | Some public images contain a bootstrap or source runtime; models and vendor runtimes are obtained separately at execution time where required. | [Packaging contract](../workbench/container-packaging.md), [machine-readable inventory](../../npa/docker/workbench/packaging-contract.yaml) |
| npa | Operators and coding agents use the CLI and supported Python interfaces to control workloads. | [Agent first run](../workbench/agent-first-run.md), [CLI and Python interfaces](../workbench/cli-sdk-yaml-walkthrough.md) |
| npa.workflow YAML | Workflow specifications declare states, tool references, and resource profiles for npa to plan and execute. | [Workflow guide](../workbench/npa-workflow-guide.md) |
| Workflow engine + SkyPilot → Kubernetes | The NPA workflow engine plans execution waves and SkyPilot submits jobs; containerized GPU and CPU workloads execute on Nebius Kubernetes. Kubernetes pulls the selected images from GHCR. | [SkyPilot setup](../orchestration/skypilot-setup.md), [workflow runtime](../../npa/src/npa/orchestration/npa_workflow/runtime.py) |
| Object Storage | S3 stores inputs, outputs, checkpoints, reports, and durable workflow state. Workload containers read and write artifacts; NPA also persists and reads run state. | [Workflow guide](../workbench/npa-workflow-guide.md), [run lifecycle](../run-lifecycle.md) |
| Token Factory | A separate hosted inference API, callable directly through NPA or from workload containers. The detailed API connection appears in the README task flow. | [Token Factory integration](../workbench/token-factory.md) |

The integrations column is a selection of integrations represented in the public
catalog, not an exhaustive inventory. Names identify integrations; the
packaging contract determines which upstream bytes are included. For example,
the Foxglove image hosts the Embed SDK and NPA glue rather than redistributing the
Foxglove application. Restricted operator-built images and development
candidates retain their separate status in the catalog.

The cloud boundary contains Kubernetes execution, Object Storage, and Token
Factory. GHCR is an external distribution service. The Workbench column groups
image distribution and job submission responsibilities; it does not specify
where every SkyPilot component is hosted. Image pull and job submission are
separate connections into Kubernetes. Hardware support remains specific to each
image and workload; see the
[image/GPU matrix](../workbench/image-gpu-compatibility-matrix.md).

## Task-flow reference

The README Mermaid diagram covers requests and results, workflow YAML, the NPA
engine's execution waves, SkyPilot jobs, S3 artifacts and durable state, and the
separate Token Factory API. Its sources are the
[workflow guide](../workbench/npa-workflow-guide.md),
[tool catalog](../workbench/npa-workflow-tool-catalog.md), and
[run lifecycle](../run-lifecycle.md). Individual VM, container, and service
deployment modes are documented in the [Workbench index](../workbench/README.md).

## Visual maintenance

The PNG uses three columns: selected integrations, Workbench, and Nebius AI
Cloud. Its proportional sans-serif typography and navy, lime, and pale-blue
palette draw from the [Workbench UI styles](../../npa/src/npa/cli/agent_ui.html).
The npa control panel and job-submission path are the visual focus; GHCR is a
quieter supporting component. Keep labels readable at the README's 960-pixel
display width and retain only a few integration examples. The public catalog
holds the inventory and the README holds the registry namespace. Preserve the
accessible image description and the Mermaid `accTitle` / `accDescr`.

The PNG was produced with the built-in image-generation tool. The reusable
generation brief below records its content and layout. Update this reference,
the PNG, and its accessible description when the architecture changes. Update
the Mermaid source when execution relationships change. Check every label and
connector against the sources above after generation.

<details>
<summary>PNG generation brief</summary>

Create a landscape platform architecture graphic for Nebius Physical AI. Use
Workbench navy `#0D2A3D`, dark navy `#0A1C2B`, white type, lime `#E5FF4F`, and
pale blue `#DCEEFF`. Use smooth Inter-style proportional sans-serif typography,
medium body labels, semibold headings, a precise three-column grid, generous
spacing, thin borders, and flat panels with restrained six-pixel corners.
Make npa the focal component and GHCR a quieter, shorter supporting card.
Keep the title moderate and all labels readable at 960px. Render sharp lettering,
orthogonal connectors, and compact arrowheads without texture or glow.
Avoid redundant nested boxes:
the integration groups and SkyPilot have no inner borders, and the Workbench
column has no enclosing boundary.

- Header: "WORKBENCH ARCHITECTURE" and "Nebius Physical AI". No tagline.
- Left column, "Selected integrations": one panel with "NVIDIA" and "Open ecosystem"
  groups, separated by a fine rule. NVIDIA rows: "Isaac Lab / Isaac Sim",
  "GR00T · SONIC", and "Cosmos 3 · Transfer 2.5". Open ecosystem rows:
  "LeRobot · Genesis", "FiftyOne · LanceDB", and "Rerun · Foxglove".
- Middle column, "Workbench": a secondary "GHCR" panel with "Public container
  images". Below it, the primary panel with "npa", "CLI · Python · coding agents",
  "npa.workflow YAML", a downward arrow, and "Workflow engine + SkyPilot".
  Give the npa panel a restrained lime accent. Do not connect GHCR to npa.
- Right column, "Nebius AI Cloud": an enclosing boundary with three services:
  "Kubernetes" / "GPU + CPU workloads", "Object Storage" / "S3 artifacts + run
  state", and "Token Factory" / "Hosted inference API", stacked vertically.
- Connect the outer integrations panel to GHCR with a rightward pale-blue
  "package" arrow representing both integration groups. Align GHCR and Kubernetes
  for a straight pale-blue "image pull" arrow. Route a lime "jobs" elbow from
  the workflow engine and SkyPilot through the column gutter into Kubernetes,
  clear of other lines and text. Connect Kubernetes and Object Storage with a
  short vertical bidirectional pale-blue "read / write" arrow. Token Factory
  stands separately.
- Footer: "Models and vendor runtimes fetched at runtime where required."

Use only these five connections, including npa to the workflow engine and
SkyPilot. Keep solution names verbatim. Omit the registry namespace, decorative
icons, circuits, gradients, shadows, invented logos, numbered stages, hardware
models, and inventory counts.

</details>
