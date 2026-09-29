# Workbench architecture diagram reference

The [PNG overview](workbench-architecture.png) maps the solution ecosystem,
container distribution, control layer, and Nebius services. The
[README task flow](../../README.md#how-workbench-runs-a-task) describes how an
individual workflow executes.

## Platform architecture and sources

| Element | Meaning | Source |
| --- | --- | --- |
| NVIDIA | Selected integrations: Isaac Lab / Isaac Sim, Isaac Lab-Arena, GR00T, SONIC, Cosmos 3, Transfer 2.5, Cosmos Evaluator, Cosmos Curator, Alpamayo 2 Super, and Content Agents. | [Public container catalog](../workbench/container-image-catalog.md) |
| Open ecosystem | Selected integrations: LeRobot, Genesis, FiftyOne, LanceDB, Wan 2.2, LTX-2.5, Diffusers, LingBot World, SAM 2.1, OpenArm, Rerun, Foxglove, and Lichtblick. | [Public container catalog](../workbench/container-image-catalog.md) |
| NPA packaging → GHCR | Repository-selected public runtime images use `ghcr.io/nebius/nebius-physical-ai`. Resolve release tags to digests for reproducibility. | [Image resolver](../../npa/src/npa/deploy/images.py), [release manifest](../../npa/src/npa/deploy/public_release_manifest.json) |
| Runtime fetch | Some public images contain a bootstrap or source runtime; models and vendor runtimes are obtained separately at execution time where required. | [Packaging contract](../workbench/container-packaging.md), [machine-readable inventory](../../npa/docker/workbench/packaging-contract.yaml) |
| npa | Operators and coding agents use the CLI and supported Python interfaces to control workloads. | [Agent first run](../workbench/agent-first-run.md), [CLI and Python interfaces](../workbench/cli-sdk-yaml-walkthrough.md) |
| SkyPilot → Kubernetes | SkyPilot submits jobs; containerized GPU and CPU workloads execute on Nebius Kubernetes. Kubernetes pulls the selected images from GHCR. | [SkyPilot setup](../orchestration/skypilot-setup.md), [workflow runtime](../../npa/src/npa/orchestration/npa_workflow/runtime.py) |
| Object Storage | S3 stores inputs, outputs, checkpoints, reports, and durable workflow state. Workload containers read and write artifacts; NPA also persists and reads run state. | [Workflow guide](../workbench/npa-workflow-guide.md), [run lifecycle](../run-lifecycle.md) |
| Token Factory | A separate hosted inference API, callable directly through NPA or from workload containers. The detailed API connection appears in the README task flow. | [Token Factory integration](../workbench/token-factory.md) |

The ecosystem shelf is a selection of integrations represented in the public
catalog, not an exhaustive inventory. Names identify integrations; the
packaging contract determines which upstream bytes are included. For example,
the Content Agents image uses runtime-fetched vendor components, and the
Foxglove image hosts the Embed SDK and NPA glue rather than redistributing the
Foxglove application. Restricted operator-built images and development
candidates retain their separate status in the catalog.

The cloud boundary contains Kubernetes execution, Object Storage, and Token
Factory. GHCR is an external distribution service. The control band represents
submission responsibilities; it does not specify where every SkyPilot component
is hosted. Image pull and job submission are separate connections into
Kubernetes. Hardware support remains specific to each image and workload; see
the [image/GPU matrix](../workbench/image-gpu-compatibility-matrix.md).

## Task-flow reference

The README Mermaid diagram covers requests and results, workflow YAML, the NPA
engine's execution waves, SkyPilot jobs, S3 artifacts and durable state, and the
separate Token Factory API. Its sources are the
[workflow guide](../workbench/npa-workflow-guide.md),
[tool catalog](../workbench/npa-workflow-tool-catalog.md), and
[run lifecycle](../run-lifecycle.md). Individual VM, container, and service
deployment modes are documented in the [Workbench index](../workbench/README.md).

## Visual maintenance

The PNG uses a midnight navy canvas, white type, electric lime for NVIDIA and
GHCR, periwinkle for the open ecosystem and control layer, and fine rules.
Keep the solution names and public registry namespace readable at the README's
960-pixel display width. Preserve the accessible image description and the
Mermaid `accTitle` / `accDescr`.

The PNG was produced with the built-in image-generation tool. The reusable
generation brief below records its content and layout. Update this reference,
the PNG, and its accessible description when the architecture changes. Update
the Mermaid source when execution relationships change. Check every label and
connector against the sources above after generation.

<details>
<summary>PNG generation brief</summary>

Create a landscape platform architecture graphic for Nebius Physical AI. Use
midnight navy `#071D2B`, lighter navy `#102A3A`, white type, electric lime
`#D8F85B`, and periwinkle `#9BAFFF`. Use crisp sans-serif typography, a precise
column grid, generous spacing, thin rules, and restrained geometric line icons.

- Header: "WORKBENCH ARCHITECTURE", "Nebius Physical AI", and
  "Containerized solutions. Shared control. Nebius compute."
- Upper left: "SELECTED SOLUTION INTEGRATIONS", divided into "NVIDIA" and
  "Open ecosystem" columns. Use the exact solution names from the source table
  above. Present them as lists without workflow ordering.
- Upper right: a prominent "GHCR" panel, "Public container images", the public
  namespace `ghcr.io/nebius/nebius-physical-ai` split after `nebius/`, and
  "Release tags + digests". Beneath a divider: "Runtime fetch where required"
  and "Models / vendor runtimes".
- Connect the solution shelf to GHCR with a rightward "NPA packaging" arrow.
- A slim control band below the solution shelf: "npa", "CLI / Python / coding
  agents", "SkyPilot", and "Job orchestration".
- A wide foundation labeled "Nebius AI Cloud" contains three service panels:
  "Kubernetes" / "GPU + CPU workloads", "Object Storage" / "S3 artifacts + run
  state", and "Token Factory" / "Hosted inference API".
- Route a blue "jobs" connector from SkyPilot leftward through the gap and
  down into Kubernetes. Route a lime "image pull" connector from GHCR leftward
  and down into Kubernetes, to the right of the jobs arrow. Keep them separate
  and clear of text. Connect Kubernetes and Object Storage with a bidirectional
  "read / write" arrow. Token Factory stands separately in the cloud boundary.

Use only these connections. Keep the registry namespace and solution names
verbatim. Omit decorative circuits, gradients, shadows, invented logos,
workflow stages, hardware model numbers, and inventory counts.

</details>
