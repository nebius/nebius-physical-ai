# Solutions Model

> Last updated: 2026-06-11

Nebius Physical AI is organized as a platform with one or more solutions. The
platform owns the shared command model, repository conventions, architecture
docs, and integration boundaries. Each solution owns a coherent user-facing
capability set inside that platform.

## What A Solution Is

A solution is a named product surface that groups related physical AI tools,
workflows, documentation, and validation under a stable CLI namespace.

The current solution is Workbench:

- CLI namespace: `npa workbench`
- User model: containerized workbench tools that can be deployed, queried, and
  invoked from the CLI or SDK
- Documentation: [docs/workbench/](../workbench/)
- Reference commands: `npa workbench <tool> <verb>` for deployable tools;
  `npa workbench workflow submit <yaml>` for multi-stage pipelines

A solution is not just a directory or a label. It must define who it serves, the
tools it exposes, the contracts those tools honor, how operators validate it,
and where users find the solution-specific docs.

## Platform Vs Solution Boundary

The platform owns shared concerns that must stay consistent across solutions:

- the top-level `npa <solution> ...` CLI shape;
- authentication and configuration conventions that are reused across
  solutions;
- shared error formatting and SDK expectations;
- repository-wide architecture docs and contribution guidance;
- cross-solution command naming conventions;
- linkable documentation indexes from the root `README.md` and `docs/README.md`.

A solution owns the behavior behind its namespace:

- its tools and tool-specific verbs;
- service, workflow, or job implementation details;
- container image expectations and runtime dependencies;
- SDK modules for solution-specific clients;
- operator runbooks, troubleshooting docs, and cookbooks;
- validation evidence for its workloads.

Workbench is the reference implementation of this split. The platform exposes
the `npa workbench` namespace and common CLI conventions. Workbench owns tools
such as Isaac Lab, FiftyOne, LeRobot, LanceDB, Cosmos, GR00T, Genesis, and
SONIC, along with their deployment, status, run, train, and data-flow behavior.

## How To Add A Solution

Use Workbench as the reference implementation and add the new solution in small,
reviewable steps.

1. Define the solution scope.

   Write down the target user, the physical AI workflow the solution supports,
   and the first complete tool or workflow it will expose. Do not create a
   solution namespace until there is a real user-facing capability to document
   and validate.

2. Choose the CLI namespace.

   Reserve a short solution name that fits `npa <solution> ...`. The namespace
   should describe the product surface, not an implementation detail. Workbench
   uses `npa workbench`.

3. Define the first tools.

   List each `<tool>` under the namespace and identify the verbs each tool must
   support. Prefer standard verbs where they apply, then add tool-specific verbs
   only when the workflow needs them.

4. Specify the tool contract.

   Document inputs, outputs, runtime assumptions, artifacts, and failure modes.
   For service-backed tools, include health, status, list, and system-info
   surfaces. For data-processing tools, include the S3 input and output paths
   that connect the tool to the rest of the platform.

5. Add implementation behind one source of truth.

   Keep core behavior in the service or shared implementation layer, then have
   CLI and SDK clients call that behavior. Workbench follows this model for
   containerized FastAPI services where the container is the deployment unit,
   the endpoint is the invocation unit, and CLI/SDK code is client code.

6. Add solution documentation.

   Create a solution docs area under `docs/<solution>/`. Workbench uses
   [docs/workbench/](../workbench/) for getting started guides, cookbooks, and
   troubleshooting. Link the new docs from `docs/README.md` and from the root
   `README.md` solutions table.

7. Add architecture notes when the platform contract changes.

   If the solution requires a new shared CLI convention, credential convention,
   packaging rule, or cross-solution behavior, document that under
   `docs/architecture/` rather than inside solution-only docs.

8. Add validation and reference commands.

   Provide the smallest repeatable validation path for the first tool. Include
   local help commands, offline checks, and live infrastructure checks only when
   they are required for that solution.

9. Update indexes and links.

   Update the root `README.md`, `docs/README.md`, CLI reference docs, and any
   affected architecture pages. Run a Markdown link audit for moved or newly
   introduced docs before handing the change off.

## Tool Contract

The standard invocation shape is:

```bash
npa <solution> <tool> <verb> [options]
```

Workbench examples:

```bash
npa workbench isaac-lab deploy
npa workbench isaac-lab status
npa workbench fiftyone list
npa workbench lancedb system-info
```

Standard verbs should keep these meanings across solutions:

| Verb | Contract |
| --- | --- |
| `deploy` | Create or update the service, job runner, managed VM, container, or workflow runtime required by the tool. Repeated deploys should update in place unless the tool documents a replacement flow. |
| `status` | Report current health and operational state. Include enough identifiers for an operator to find the underlying deployment, job, endpoint, or artifact prefix. |
| `list` | Enumerate user-visible resources such as configured projects, deployments, datasets, runs, jobs, or artifacts. Empty states should be explicit and non-fatal. |
| `system-info` | Report runtime details needed for support and reproducibility, such as image tags, dependency versions, accelerator visibility, storage endpoints, and platform-specific environment facts. |

Workbench service-backed tools should expose the same concepts over their API
surface where applicable:

- `GET /health`
- `GET /status`
- `GET /list`
- `GET /system-info`
- `POST /train` or `POST /run` for workload submission

Tools that pass data between stages should use object storage paths, not direct
tool-to-tool data transfer. Workbench commands use S3-style `--input-path` and
`--output-path` options for pipeline data flow so one tool's artifacts can
become the next tool's inputs.

## Workbench Namespace Layout

Everything Workbench-related stays under `npa workbench`. Only platform
infrastructure (`npa configure`, `npa cluster`, `npa skypilot`, …) sits
directly on `npa`.

| Surface | Examples | Purpose |
| --- | --- | --- |
| **Tools** (slot 3) | `lerobot`, `sonic`, `lancedb`, `vlm-eval` | Deployable capabilities with `deploy` / `run` / `status` |
| **Tool subcommands** | `npa workbench sonic retargeting run` | Job-shaped steps scoped to a parent tool |
| **Workflows** | `npa workbench workflow submit <yaml>` | Multi-stage SkyPilot pipelines (sim-to-real, BDD100K, SONIC finetuning) |
| **Workflow helpers** | `npa workbench workflow trigger watch` | Long-running watchers that resubmit pipeline YAML |
| **Hidden compat** | `npa workbench data` (hidden) | S3 bridge for scripts/SDK; not advertised in `--help` |

Do not register pipeline orchestrators (`sim2real`, `sim2real-envgen`) or
operator tooling as top-level workbench tools. Use workflow YAML + cookbooks
instead.

Tool docs should state:

- required credentials and non-secret identifiers;
- required runtime or accelerator assumptions;
- accepted input paths and produced output paths;
- success artifacts and their names;
- idempotency behavior for `deploy`;
- expected empty-state behavior for `list`;
- troubleshooting data to collect for `status` or failed runs.
