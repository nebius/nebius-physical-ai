---
name: npa-cli-conventions
description: Use when writing or modifying npa CLI commands or SDK functions — registration, option naming, path contract, output format, error handling, exit codes, the decorators that are easy to forget, and config/credential access.
---

# npa CLI And SDK Conventions

Conventions for changing the `npa` package itself. For the full path of adding a
new tool, use `skills/workflows/add-workbench-tool/SKILL.md`.

The codebase is mixed: some modules predate the current pattern. Where this
skill and an existing module disagree, follow this skill and leave the old module
alone unless you are deliberately migrating it.

## Where Code Goes

Behavior belongs in `npa/src/npa/workbench/<tool_snake>/`. The CLI and SDK are
clients of it, not second implementations. Three integration patterns exist:

- **Workbench module first** — the SDK builds a Pydantic request and calls the
  shared implementation, or HTTP when `service=True`. This is the pattern for
  new work. See `npa/src/npa/sdk/workbench/dataset.py`.
- **CLI wrapper** — `make_cli_wrapper` in `npa/src/npa/_sdk.py` exposes an
  existing CLI callback as an SDK function. Used by older orchestration-heavy
  tools such as LeRobot. Do not choose it for new tools.
- **Direct module import** — for platform code with no service, such as
  `npa/src/npa/sdk/provisioning.py`.

## Registration

Register new command groups under `npa workbench` in
`npa/src/npa/cli/workbench/__init__.py`. Do not add top-level groups: the
platform utilities registered in `npa/src/npa/cli/main.py` predate the solution
namespace model and are marked as legacy in that file.

## Options

The cross-tool handoff flags are `--input-path` (reads an `s3://` URI) and
`--output-path` (writes one). Validate them with the shared contract in
`npa/src/npa/cli/path_contract.py`. Public handoff paths must not require
VM-local paths, `file://` URIs, or plain HTTP URLs. A few legacy tools use
`--input-uri` / `--output-uri`; keep them working with aliases rather than
spreading the older spelling.

Conventional flags: `-p/--project` for the config alias, `-n/--name` for the
workbench instance, `--endpoint` to override a saved endpoint, `--service` to
call the deployed service instead of running in-process, and `--token-env` for
the env var holding a bearer token. Not every tool has all of them; check the
sibling module rather than assuming.

## Output

There is no shared output-format module. Each CLI package defines its own
`OutputFormat` enum and an `emit()` helper — copy the one next to you rather
than inventing a third shape. `npa/src/npa/cli/workbench/lancedb/helpers.py` is
a compact reference.

JSON flag spellings differ by area: `--output-format json` and `--json` on
workflow and platform commands, `--output json` on several workbench tools. For
anything an agent or script will parse, prefer `--output-format` plus
`@json_stdout_contract` from `npa/src/npa/lifecycle_intent.py`, which guarantees
exactly one JSON document on stdout.

## Errors And Exit Codes

Raise a domain error from the workbench module, map it to `HTTPException` in the
service, and let the CLI turn it into a message. `npa/src/npa/errors.py` holds
the shared base (`NpaError`); most tools define their own subclasses locally.
`npa/src/npa/cli/_error_formatting.py` renders serverless errors for humans or
as JSON.

The top-level handler is `app_entry()` in `npa/src/npa/cli/main.py`:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Serverless/client errors, and most command-local `typer.Exit(1)` |
| 2 | Unexpected exception |
| 130 | SIGINT |

On exit code 2 the CLI prints a hint instead of a traceback; `NPA_DEBUG=1`
restores the traceback. Never swallow an exception silently — a bare
`except Exception: pass` is rejected by
`npa/tests/guardrails/test_hygiene_guards.py`.

## Three Decorators That Are Easy To Forget

- `@resolve_typer_defaults` (`npa/src/npa/cli/_typer_defaults.py`) — required
  when one Typer command calls another as a plain Python function. Without it an
  unresolved `OptionInfo` object leaks into downstream code such as Terraform
  variables or shell commands. Enforced by
  `npa/tests/guardrails/test_typer_command_calls.py`.
- `@json_stdout_contract` (`npa/src/npa/lifecycle_intent.py`) — on any command
  exposing a JSON flag.
- `@intent_boundary` (`npa/src/npa/lifecycle_intent.py`) — on entrypoints that
  provision or destroy, so the lifecycle intent is inherited by child processes.

## Config And Credentials

Never hardcode project, tenant, registry, or bucket identifiers. Read them
through `npa/src/npa/clients/config.py` and
`npa/src/npa/clients/credentials.py`:

```python
from npa.clients.config import resolve_config, resolve_environment
from npa.clients.credentials import load_credentials

cfg = resolve_config(project=project, name=name)
env = resolve_environment(project=project)
creds = load_credentials()
```

Precedence is CLI flags, then environment variables, then
`~/.npa/credentials.yaml`, then `~/.npa/config.yaml`. Other accessors worth
knowing: `resolve_ssh_config`, `resolve_project_storage`,
`resolve_container_registry`, and the deep-merging `write_config` /
`write_credentials_file`.

Committed examples use placeholders such as `<your-project-id>` and
`<your-bucket>`. Concrete values are caught by gitleaks and the confidentiality
scan — see `skills/atomic/protect-nebius-infra-details/SKILL.md`.

## Subprocesses

To re-invoke the CLI internally, use `internal_cli_argv()` from
`npa/src/npa/cli/invocation.py`. It runs `python -m npa` on the active
interpreter rather than trusting an `npa` binary on `PATH`.

## Before You Push

Changing CLI options invalidates generated docs and can break catalog argv.
Regenerate with `bash scripts/build_docs.sh` and re-check the catalog contract
in `skills/atomic/toolref-argv-contract/SKILL.md`. Full gate order is in
`skills/atomic/pre-pr-validation/SKILL.md`.
