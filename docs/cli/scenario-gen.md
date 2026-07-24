# `npa workbench scenario-gen`

## Command Tree

```text
Usage: npa workbench scenario-gen [OPTIONS] COMMAND [ARGS]...

Adversarial scenario generation: mine hard scenarios that fail a policy-under-test (pluggable Isaac Lab RL backend;
deterministic default).

Options
--help  Show this message and exit.
Commands
generate  Mine ranked adversarial scenarios against a policy-under-test.
rank  Score and rank generated adversarial scenarios.
status  Fetch a scenario-gen run status.
system-info  Show scenario-gen runtime information.
list  List service-managed scenario-gen runs.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `generate` | Mine ranked adversarial scenarios against a policy-under-test. |
| `rank` | Score and rank generated adversarial scenarios. |
| `status` | Fetch a scenario-gen run status. |
| `system-info` | Show scenario-gen runtime information. |
| `list` | List service-managed scenario-gen runs. |

## Examples

```bash
npa workbench scenario-gen --help
npa workbench scenario-gen generate --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `scenario-gen`.
