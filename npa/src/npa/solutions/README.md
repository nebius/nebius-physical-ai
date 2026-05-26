# NPA Solutions

Validation context: [docs/workbench/solutions-validation.md](../../../../docs/workbench/solutions-validation.md)
documents the current solutions framework validation state.

A solution is a top-level product namespace on the NPA platform. It groups a
related CLI surface, optional SDK namespace, workflows, containers, manifests,
and agent skills around one implementation domain.

Workbench is the reference implementation. It owns robotics and physical AI
workflow tooling under `npa workbench`, with SDK clients under
`npa.sdk.workbench`, workbench-specific CLI internals under
`npa.cli.workbench`, and tool skills under `.agents/skills/workbench/`.

## Adding A Second Solution

1. Add `npa/src/npa/cli/<solution>/__init__.py`.
2. Register the solution CLI namespace in `npa/src/npa/cli/main.py`.
3. Add a `[[solutions]]` entry in `npa/src/npa/solutions/solutions.toml`.
4. Add an SDK namespace if applicable.
5. Add `.agents/skills/<solution>/` skill files.

Future solution examples include `datalake` for dataset and storage workflows
and `simfarm` for simulation fleet workflows.
