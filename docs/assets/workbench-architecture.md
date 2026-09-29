# Workbench architecture diagram reference

The [root README](../../README.md#how-workbench-runs-a-task) contains the Mermaid
flow and [PNG overview](workbench-architecture.png). Both describe the standard
Kubernetes workflow path. They are a map of responsibilities, not a catalog of
every tool, deployment mode, or supported GPU.

## Architecture and sources

| Element | Meaning | Source |
| --- | --- | --- |
| You + your coding agent | An external coding agent invokes `npa`; operators can use the CLI and supported Python interfaces directly. | [Agent first run](../workbench/agent-first-run.md), [CLI and Python interfaces](../workbench/cli-sdk-yaml-walkthrough.md) |
| npa control plane | Configures access, checks readiness, plans work, submits jobs, and inspects results. | [Run lifecycle](../run-lifecycle.md) |
| npa.workflow YAML | Declares states, tool references, and resource profiles. It is input to the control plane, not a deployed service. | [Workflow guide](../workbench/npa-workflow-guide.md), [tool catalog](../workbench/npa-workflow-tool-catalog.md) |
| Workflow engine + SkyPilot | The NPA engine evaluates the state graph and plans execution waves; SkyPilot launches the jobs. The operator environment is distinct from the workload containers. | [Runtime implementation](../../npa/src/npa/orchestration/npa_workflow/runtime.py), [SkyPilot setup](../orchestration/skypilot-setup.md) |
| Containerized tools | GPU and CPU stages run in the selected Nebius Kubernetes environment. The capability labels are categories, not claims that every image supports every GPU. | [Tool catalog](../workbench/npa-workflow-tool-catalog.md), [image/GPU matrix](../workbench/image-gpu-compatibility-matrix.md) |
| Object Storage · S3 | Tools exchange inputs and outputs; NPA stores and reads durable workflow state and decision artifacts. | [Workflow guide](../workbench/npa-workflow-guide.md), [run lifecycle](../run-lifecycle.md) |
| Token Factory | Hosted inference is a separate API. Direct CLI calls need no cluster; workflow stages can also call the API from their containers. The clients handle artifact persistence. | [Token Factory integration](../workbench/token-factory.md) |
| Artifact inspection | Rerun, Foxglove, media viewers, and reports expose the outputs for review. | [Agent viewers](../agent.md), [Foxglove](../workbench/foxglove-export.md) |

The cloud boundary encloses workload execution, storage, and hosted inference.
The workflow/SkyPilot box represents operator-side submission, not the physical
location of every SkyPilot component: managed-job controllers also run remotely.
The PNG's downward job arrow emphasizes submission; the Mermaid flow also labels
returning status and logs. The dashed S3 connection represents artifact and
durable-state exchange, rather than job scheduling.

Individual tool VM, container, and service paths are documented in the
[Workbench index](../workbench/README.md). The overview does not imply a universal
HTTP API, an OSMO integration, Token Factory model export, or a uniform hardware
support matrix.

## Visual maintenance

Use a white canvas, navy text, electric lime for the control plane, pale blue for
orchestration and cloud grouping, and a dark S3 card. The Mermaid theme uses the
navy `#052B42`, lime `#E0FF4F`, and pale blue `#EEF3FF` colors from the
[Nebius website](https://nebius.com/); the PNG uses a matching inspired palette.
Keep labels readable at the README's display width and preserve the accessible
image description and Mermaid `accTitle` / `accDescr`.

The PNG was redrawn with the built-in image-generation tool. Its generation
brief follows; architecture changes should update this reference, the Mermaid
source in the README, and the PNG together. Inspect all labels and arrowheads
after generation. A generated image is not evidence that a capability works.

<details>
<summary>PNG generation brief</summary>

Redesign the old architecture diagram completely as a modern technical diagram
for the root GitHub README. Remove the old content. Use a crisp, flat editorial
style, modern sans serif, large readable text, generous spacing, rounded
rectangles, and thin orthogonal arrows. Use deep navy `#0B2436`, electric lime
`#D6F84C`, off-white `#F5F7F8`, and pale blue `#E6F2FA`. Avoid invented logos,
illustrations, gradients, shadows, tiny text, 3D, and decorative circuits.

- Header: "Nebius Physical AI"; subtitle: "Workbench architecture"; top-right
  pill: "STANDARD WORKFLOW PATH".
- White access card: "You + your coding agent" and "Direct CLI and supported
  Python interfaces". Connect bidirectionally to the control plane.
- Lime control-plane card: "npa control plane" and "Configure · preflight ·
  plan · submit · inspect".
- White YAML card to its right: "npa.workflow YAML", with an arrow into the
  control plane.
- Pale-blue orchestration card below: "Workflow engine + SkyPilot" and
  "Plan waves · submit jobs · track status". Connect bidirectionally to the
  control plane and downward to the tools card.
- A large region titled "Nebius AI Cloud" encloses only the tools, Token
  Factory, and S3 cards.
- White tools card: "Containerized tools", "Kubernetes · GPU + CPU",
  "Simulate · train · generate", and "Curate · reconstruct · evaluate".
- White Token Factory card to its right: "Token Factory", "Hosted inference
  API", and "No customer GPU required". A separate bidirectional "Direct API"
  path connects it to the control plane, bypassing orchestration.
- Navy storage card beneath the tools: "Object Storage · S3" and "Inputs ·
  checkpoints · media · reports · run state". Connect bidirectionally to the
  tools. A dashed bidirectional "Artifacts + durable state" path connects it
  to the control plane along the left margin. Do not connect Token Factory
  directly to S3.
- Footer: "Inspect artifacts with Rerun, Foxglove, media viewers and reports."

Keep labels verbatim. Add no invented connections or components, OSMO, Vault,
hardware model numbers, partner marketplace, or model export claims.

</details>
