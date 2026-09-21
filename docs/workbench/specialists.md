# Self-hosted Workbench specialists

Run independent agents on your own Linux or macOS host. LangGraph checkpoints
persist each task's conversation and next step; Workbench owns task assignment,
tool permissions, operation receipts and source patches. Model inference uses
explicit OpenAI-compatible endpoints, including Token Factory. Optional Jev
classification selects an already configured specialist once per task.

The runtime is an optional agent layer. It does not replace the existing
Workbench grounded chat or change its default model routing. It uses the public
LangGraph library and SQLite, without requiring LangGraph Platform or a
LangSmith service account. Jev and Token Factory are external hosted APIs.

## Start a team

Install into the controller checkout's own environment:

```bash
uv venv npa/.venv --python 3.12
uv pip install --python npa/.venv/bin/python -e 'npa[agent-specialists]'
```

Create a separate checkout or Git worktree for each specialist. Keep the
controller installation outside those editable workspaces. Copy
[`team.json`](../../npa/examples/specialists/team.json) to an operator-owned
configuration location and adjust the workspace directories. The example gives
GLM a Cosmos role and DeepSeek V4 a Sim2Real role. Model availability depends on
your Token Factory account; verify exact IDs before running.

```bash
npa/.venv/bin/python -m npa workbench health preflight --checks token_factory --json
npa/.venv/bin/python -m npa workbench token-factory models
export NPA_SPECIALISTS_TOKEN="$(openssl rand -hex 24)"
npa/.venv/bin/python -m npa workbench specialists --config "$TEAM_CONFIG" serve
```

Use `NEBIUS_TOKEN_FACTORY_KEY` through the existing private NPA credential store
or environment. A profile's `key_env` identifies its credential without storing
the value in JSON. `NPA_SPECIALISTS_TOKEN` authenticates the monitor's API; keep
it in a private environment file when installing a persistent service. The
default browser address is `http://127.0.0.1:8790`. Paste the service token to
connect; the browser keeps it only in the current tab's memory. For remote use,
forward this port over SSH or place it behind your authenticated TLS proxy.

`serve` starts and supervises one process per profile. Workers continue when a
browser closes. For independent lifecycle management, run `serve --no-workers`
and one `worker <profile>` command per process supervisor/systemd service.
The process supervisor should restart the service on failure and preserve its
state directory. SIGTERM requests a graceful stop after the current node;
it does not cancel an already submitted external workload.

## Prompt to result

Choose a specialist or automatic routing, enter a goal, and submit. The monitor
shows durable task state, the actual responding model, token usage, tool
receipts, failures, final answers and source patches. The CLI and Python SDK
share exactly the same coordinator:

```bash
npa/.venv/bin/python -m npa workbench specialists --config "$TEAM_CONFIG" \
  submit "Inspect and repair the workflow; run validation and planning." \
  --specialist cosmos --task-id cosmos-repair
npa/.venv/bin/python -m npa workbench specialists --config "$TEAM_CONFIG" status
npa/.venv/bin/python -m npa workbench specialists --config "$TEAM_CONFIG" \
  pause --specialist cosmos
npa/.venv/bin/python -m npa workbench specialists --config "$TEAM_CONFIG" \
  pause --specialist cosmos --resume
```

```python
from npa.sdk.workbench.specialists import SpecialistTeam, load_config

team = SpecialistTeam(load_config(config_path))
task = team.submit("Diagnose the pipeline", specialist="sim2real", task_id="diagnose")
print(team.status(task["id"]))
```

An identical task ID and request returns the existing task. Reusing that ID for
a different goal is rejected. A completed task can receive a follow-up using
`submit --parent-id <task-id>` or the monitor. A pause takes effect at the next
model/tool boundary; it does not abort an in-flight command. Paused or unresolved
tasks hold their specialist's queue while other specialists continue.
Use `cancel <task-id>` or the monitor to stop a task at its next durable boundary.
Cancellation preserves existing edits and receipts and does not cancel external
workloads; use the corresponding Workbench workload controls for those.

`completed` means the agent returned its final answer. It is not an independent
claim that a GPU workflow completed successfully. Use the recorded Workbench
status, test and artifact receipts to assess that claim.

## Tools and authorization

Each profile defines `read_paths`, `write_paths` and named `operations`. Source
access is restricted to explicit workspace-relative paths. Edits require the
current file SHA-256 and one exact matching replacement. Symlinks, hard links,
parent traversal and `.git` access are rejected. Each task retains a unified
diff against its own first-edited file contents for human review.

An operation contains fixed `argv`, a description and optional `pass_env` names.
The model chooses its name; it cannot supply command arguments or shell text.
`{python}`, `{workspace}`, `{task_id}` and `{run_id}` are the only substitutions.
`{python}` identifies the controller interpreter. `{run_id}` is stable for the
task, allowing submit/status/artifact operations to refer to the same workflow.
Commands run with `shell=False` in that specialist's workspace. They inherit
only PATH, HOME, locale, temporary-directory settings and explicitly listed
environment names. Command exit code, stdout and stderr become durable receipts.

The example authorizes validation and planning. To authorize execution, add
fixed Workbench `health preflight`, `workflow submit`, `workflow status` and
artifact commands with your already verified project, workflow and storage
configuration. Add only the credential environment names those operations need.
Follow the normal Workbench access and workflow preflight procedures. Likewise,
a named test command can use the workspace's own `npa/.venv/bin/python` to test
edits there. Commit/push/PR publication is an explicit operator operation, not
an implicit consequence of model text. Patches are available through the SDK,
monitor and authenticated `/api/tasks/{id}/patch` endpoint.

These controls constrain the model's tool interface; they are **not an OS
sandbox**. Tests and authorized commands execute code with the service user's
permissions. Use an isolated machine/account for untrusted repositories. Keep
configuration, state, credentials and the controller install outside editable
workspaces. Workspaces must not overlap. Runtime storage is owner-only and
contains private task/checkpoint data; do not publish it as review evidence.

## Restart and reconciliation

Every model or tool node ends at a synchronous LangGraph checkpoint. A separate
SQLite journal records a tool call before execution and its result afterwards.
On restart, completed tool receipts are reused even if the graph checkpoint was
interrupted. If a call started but has no result, the task enters
`needs_attention`; the runtime does not automatically repeat the effect.

Inspect the external result, then supply the interrupted call ID and its verified
JSON result through the monitor or `reconcile --call-id ... --result ...`.
Alternatively, `reconcile --retry` explicitly authorizes retry after inspection.
Provider failures with no tool effect can be retried through the same control.
Changed profile/model/tool policy blocks existing tasks until the original
configuration is restored. File locks prevent two host-local workers from
advancing the same specialist simultaneously.

This version supports one host with durable local storage. It does not claim
distributed failover or exactly-once remote execution. A worker crash can leave
a command running; inspect it before reconciliation. Model requests can be
repeated after a crash before their checkpoint. There are no default task-count,
runtime or token-generation caps; `model_options.max_tokens` is optional and
operator-controlled. Provider context limits still apply and surface as failures.

## Optional Jev

Set `router` to `jev` and provide `TYPESAFE_API_KEY` through the private credential
store or environment. Workbench reuses its existing Jev HTTP classifier with
profile descriptions as criteria. It sends only redacted task text, not source
files, system messages or tool transcripts. Only use this option for task text
permitted for that provider; pattern redaction cannot classify arbitrary
proprietary prose. Explicit assignments bypass Jev. Missing credentials,
abstention and provider failure select `default_profile` without changing its
permissions. The routing receipt records the choice and fallback.

See [Jev routing](jev-routing.md) for the classifier contract and evaluation
limits. Live Jev inference is a separate test from successful Token Factory
inference and is not implied by transport mocks.

## Validation

Hermetic tests exercise concurrent graph execution, process-safe task ownership,
checkpoint restart, interrupted operations, duplicate tasks, scope enforcement,
real local edits/commands and authenticated HTTP controls. Run:

```bash
npa/.venv/bin/python -m pytest npa/tests/agent_eval/test_specialists.py -q
NPA_SPECIALISTS_LIVE=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_specialists_live.py -q -s
```

The opt-in live test uses GLM and DeepSeek through Token Factory in separate
worker processes. Each repairs a deliberately invalid copy of a public Workbench
workflow and runs real validation and planning. It pauses and restarts both
workers before completion. It does not submit GPU workloads. Exact operational
evidence stays outside the repository; publish only reviewed aggregate results.
