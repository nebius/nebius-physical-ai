# Raw SkyPilot task fixtures

These are **not** shipped workflows. They are frozen copies of two retired catalog templates,
kept only as fixtures for `tests/cli/test_workflow_cli.py`.

`npa workbench workflow submit <a raw SkyPilot YAML>` remains a supported path — a customer can
bring their own task and have the wrapper resolve images, registry auth, S3 wiring and
`${PLACEHOLDER}` substitution for it. That contract needs a raw task to exercise, and it should
**not** be exercised against a shipped template: doing so is what made these tests block the
templates' retirement even though nothing in the product referenced them.

Keeping the fixtures here says the wrapper's contract is independent of the catalog, which is
the point of retiring the catalog in the first place. See EVIDENCE.md §R51.
