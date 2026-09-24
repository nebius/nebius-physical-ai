# NPA solutions

[Contributor guide](../../../../CONTRIBUTING.md) · [Solutions architecture](../../../../docs/architecture/solutions-model.md)

A solution is a top-level NPA product namespace. It groups commands, optional
Python clients, workflows, images, and agent skills for one domain. **Workbench**
is the current reference: its commands start with `npa workbench`.

To add a robotics tool to Workbench, use
[add a Workbench tool](../../../../skills/workflows/add-workbench-tool/SKILL.md).
Create a new solution only when it needs its own top-level namespace.

## Files to change for a new solution

1. Implement `npa/src/npa/cli/<solution>/__init__.py` and register it in
   [`npa.cli.main`](../cli/main.py).
2. Add a `[[solutions]]` entry to [`solutions.toml`](solutions.toml).
3. Add a Python namespace if the solution supports programmatic access.
4. Add its root skill under `skills/tools/` or `skills/workflows/`, then register
   it in [`skills/index.yaml`](../../../../skills/index.yaml).
5. Document and test the supported commands and integration paths using the
   [contribution checks](../../../../CONTRIBUTING.md).

See [solutions validation](../../../../docs/workbench/solutions-validation.md)
for what the framework has exercised. Names such as `datalake` and `simfarm`
are design examples, not installed solutions.
