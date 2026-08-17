---
name: toolref-argv-contract
description: Use when adding or changing an npa.workflow toolRef argv template, or when changing a CLI option that a toolRef already passes — the contract that stops a stage from rendering cleanly and then crashing in the pod.
---

# The toolRef Argv Contract

A `toolRef` is the only way an `npa.workflow` spec invokes a workbench tool. The
engine expands it from `TOOL_CATALOG` in
`npa/src/npa/orchestration/npa_workflow/catalog.py` into an argv list and runs
it inside the task pod.

Nothing in `validate-spec` or `plan-spec` compares that argv against the CLI
command it will be handed. A template can validate, plan, and render perfectly
and still die with `No such option` on real infrastructure after a GPU has been
provisioned. `npa/tests/guardrails/test_tool_catalog_argv.py` exists to close
that gap. Treat it as the real gate, not the spec validators.

Read this before writing an entry, and again whenever you rename or retype a CLI
option that any toolRef already passes.

## Shape

```python
"workbench.<tool_snake>.<verb>": ToolEntry(
    name="workbench.<tool_snake>.<verb>",
    description="...",
    argv_template=[
        "npa", "workbench", "<tool>", "<verb>",
        "--input-path", "{{config.input_uri}}",
        "--output-path", "{{config.output_uri}}",
    ],
),
```

The catalog key is snake_case; the argv inside it is the kebab-case CLI path.
`workbench.scenario_gen.generate` invokes `npa workbench scenario-gen generate`.

## Rules

**Every flag must exist on the target command.** Verify against the live
signature, not memory and not the docs. `npa workbench <tool> <verb> --help` is
the source of truth.

**Include required flags.** A template missing `--run-id` or `--workflow-run`
fails at runtime, not at render.

**Never pass a format word to a path option.** `--output json` on an option that
takes a path is the worst failure mode in this class, because the stage
*succeeds* and the declared artifact silently never appears. When a command has
both, `--output` takes the path and `--output-format` takes the word.

**Do not pass infrastructure.** Image, accelerator, and GPU type belong in the
spec's `resources.<profile>`, not in the stage argv. Passing them again nests
infrastructure selection inside the pod.

**Booleans cannot be templated.** A v0.0.1 argv template is a fixed list with no
conditional rendering, so `--flag {{config.x}}` cannot express a paired boolean
such as `--headless/--no-headless`. Either hardcode the flag or record the
parameter as an unreachable `spec_gap` in the three-tier contract.

**Prefer `npa ...` over a wrapper.** Templates that start with `bash` or
`python` are harder to check. The `bash -c` forms have their embedded `npa`
commands extracted and audited, and `python -m` forms are parsed against their
module's argparse by `npa/tests/guardrails/test_module_toolref_argv.py`, but
inline `python -c` source is genuinely unchecked. The exemption lists in the
argv guardrail are pinned to shrink and never grow; a new non-CLI template fails
the build.

**Multi-node is fail-closed.** `ToolEntry.multi_node_mode` defaults to
`forbidden` because the same command running on every node would publish racing
outputs. A sharded tool must declare its rank-aware activation and its durable
join through `shard_activation_config` and `shard_output_config`.

## Registering The Entry

An argv template alone is not enough. Three other surfaces must agree:

- **Reachability** — every catalog entry must be reachable from a shipped spec
  under `npa/workflows/workbench/npa-workflows/`, or be listed in
  `PUBLIC_REUSABLE_TOOLREFS` in `catalog.py` as a deliberate public primitive.
  Enforced by `npa/tests/guardrails/test_shown_workflow_catalog.py`.
- **Docs** — one row per toolRef in
  `docs/workbench/npa-workflow-tool-catalog.md`. Enforced by
  `npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py`.
- **Image routing** — a containerized or GPU stage needs its toolRef prefix
  mapped to an image in
  `npa/src/npa/orchestration/npa_workflow/skypilot_render.py`. Without it the
  stage runs on the default image, which usually fails late and confusingly.

## Verify Locally

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_tool_catalog_argv.py \
  npa/tests/guardrails/test_module_toolref_argv.py \
  npa/tests/guardrails/test_shown_workflow_catalog.py -q
```

Then validate the spec that uses the entry:

```bash
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/<spec>.yaml
```

Remember what each proves. The guardrail proves the argv can run; `validate-spec`
proves the spec is well formed. Neither proves the stage does the right thing —
that needs live coverage, per `skills/atomic/testing-conventions/SKILL.md`.

## Failures This Contract Has Already Caught

Each of these shipped or nearly shipped, and each is now pinned by a test:

- `workbench.rl.policy_train` passed `--learning-rate`, `--batch-size`, and
  `--input-path` to `npa workbench isaac-lab train`, which has none of them. The
  real options are `--steps`, `--num-envs`, and `--data-path`.
- `workbench.sonic.eval` passed `json` to `--output`, a path option. The stage
  reported success and the declared `eval.json` never appeared.
- `workbench.lancedb.create_failure_views` passed `--table` to
  `lancedb create-mv`, whose option is `--source-table`. It hid inside a
  `bash -c` wrapper, which the audit did not read at the time.
- `sim2real_envgen.raw_shard` omitted the required `--run-id` from a `python -m`
  invocation.
