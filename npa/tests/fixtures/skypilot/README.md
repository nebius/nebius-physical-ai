# Raw SkyPilot test fixtures

[Contributor guide](../../../../CONTRIBUTING.md)

These frozen tasks exercise raw-YAML handling in
[`test_workflow_cli.py`](../../cli/test_workflow_cli.py). They are test inputs,
not examples to deploy. For current runnable workflows, use the
[workflow catalog](../../../../workflows/README.md).

The fixtures keep coverage of image resolution, registry authentication, S3
wiring, and `${PLACEHOLDER}` substitution independent of the shipped catalog.
That allows production templates to change or retire without removing coverage
of customer-owned raw SkyPilot tasks.

When editing a fixture, run its CLI tests and explain which input contract the
change exercises. Keep private infrastructure values and credentials out of
fixture content. See the historical R51 record in
[EVIDENCE.md](../../../../EVIDENCE.md) for the catalog retirement.
