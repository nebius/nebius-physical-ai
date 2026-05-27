# Nebius Physical AI

Nebius Physical AI is a multi-solution platform for physical AI workloads on
Nebius infrastructure. The repository provides a shared `npa` command model,
platform architecture guidance, and solution-specific implementations for
robotics, simulation, perception, and synthetic-data workflows.

The current solution is Workbench. Workbench packages containerized tools,
SkyPilot workflows, CLI commands, SDK entry points, and operator documentation
for running physical AI workloads on Nebius.

## Solutions

| Solution | CLI namespace | Purpose | Documentation |
| --- | --- | --- | --- |
| Workbench | `npa workbench <tool> <verb>` | Containerized tools and workflows for robotics, simulation, perception, dataset curation, and model training. | [docs/workbench/](docs/workbench/) |

New solutions should follow the platform model in
[docs/architecture/solutions-model.md](docs/architecture/solutions-model.md)
before adding a new CLI namespace, docs area, tool contract, or packaging
surface.

## CLI Model

The platform CLI is organized as:

```bash
npa <solution> <tool> <verb> [options]
```

For Workbench, `<solution>` is `workbench`, `<tool>` is a capability such as
`isaac-lab`, `fiftyone`, `lerobot`, or `lancedb`, and `<verb>` is an operation
such as `deploy`, `status`, `list`, `system-info`, `train`, or `run`.

Examples:

```bash
npa workbench --help
npa workbench isaac-lab --help
npa workbench fiftyone list
npa workbench lancedb status
```

Solutions may add tool-specific verbs, but shared verbs should keep the same
meaning across tools:

| Verb | Expected meaning |
| --- | --- |
| `deploy` | Create or update the service, job runner, or managed runtime for a tool. |
| `status` | Report health, endpoint, job, or deployment state. |
| `list` | Enumerate configured projects, deployments, runs, datasets, or artifacts. |
| `system-info` | Report runtime, image, dependency, accelerator, and environment details. |

## Getting Started

For the first Workbench install, credential setup, deployment validation, and
BDD100K pipeline path, follow
[Workbench Getting Started](docs/workbench/getting-started.md). For a broader
CLI walkthrough, see the [npa quickstart](docs/quickstart.md).

To reproduce the Cosmos, Isaac Lab, GR00T, and FiftyOne Workbench demo in your
own Nebius project, follow the [8-GPU H200 demo runbook](docs/demo/8gpu-h200.md).

## Repository Layout

```text
.
|-- README.md                         # Platform overview and solution index
|-- docs/
|   |-- README.md                     # Documentation index
|   |-- architecture/                 # Platform architecture and solution model
|   |-- workbench/                    # Workbench-specific guides, cookbooks, and troubleshooting
|   |-- cli/                          # Generated and curated CLI reference pages
|   |-- demo/                         # Demo runbooks
|   |-- demos/                        # Demo walkthroughs and visual assets
|   |-- orchestration/                # SkyPilot and runtime setup docs
|   |-- sdk/                          # SDK-facing docs
|   |-- security/                     # Security and reproducibility docs
|   `-- testing/                      # Test and validation docs
`-- npa/                              # Shared CLI, SDK, workflows, and tool implementations
```

Workbench-specific content lives under
[docs/workbench/](docs/workbench/). Platform-level architecture stays under
[docs/architecture/](docs/architecture/), including the
[solutions model](docs/architecture/solutions-model.md) and
[CLI namespace conventions](docs/architecture/cli-namespaces.md).

CLI reference pages are generated from Typer help output in
[docs/cli](docs/cli/README.md). Browser-based Rerun review workflows are
covered by the `npa rerun host` and `npa rerun share` CLI references.

## Working With AI Agents

This repo is designed to be navigated by AI coding agents (Codex and Claude
Code) as well as humans. Agent behavior is configured by a small, structured
set of files rather than embedded in prompts:

- [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) are lightweight indices
  loaded automatically by Codex and Claude Code respectively. They point at
  the relevant skill for any given task.
- Skills live under [.agents/skills/](.agents/skills/) (Codex) and
  [.claude/skills/](.claude/skills/) (Claude). Each `SKILL.md` is a focused,
  versioned reference for one topic — e.g. workbench tools, SkyPilot
  workflows, Nebius infrastructure, testing conventions, or review checks.
- Skills are living documents. Every run captures lessons to
  `.agents/runs/<run-id>/` (durable, repo-visible). A reviewer different
  from the builder triages those logs and promotes them into the affected
  skills, drops them with a reason in
  [.agents/curation-log.md](.agents/curation-log.md), or escalates them as
  Open Questions. See `super-prompt-patterns` and `skill-curation` for the
  full loop.

Contributors editing code under a skill's `applies_to` paths should read the
skill first, note the self-review outcome in the PR description, and bump the
skill's `last_verified` / `version` / `## Changelog` when reality drifts.

## Contributing

We welcome contributions from the community. Whether you are adding a solution,
improving Workbench documentation, or reporting issues, align new work with the
solution model and keep platform concerns separate from solution-specific
behavior.

## License

Copyright (c) 2026 Nebius BV

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

```text
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
