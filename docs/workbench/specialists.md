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
share exactly the same coordinator.

For large source files, agents can request an inclusive, 1-based range with
`read_file(path, start_line=100, end_line=160)`. Omitting the range reads the whole
file; omitting `end_line` reads through the end. The receipt includes the actual
range, total line count, and the **complete file's SHA-256**, so an edit still
detects changes outside the excerpt. The same file grants apply to every range.

Worker process IDs make restarts visible. Model activity shows provider-reported
cached prompt tokens, including an explicit zero; absent telemetry reads
`cache unreported`. A cache hit is evidence of reused prompt computation, not a
claim about a billing discount. Cache behavior and native tool-call support vary
by model; a successful model-list request alone proves neither. For example:

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

By default, `completed` means the agent returned its final answer. A profile can
set `required_operations`, for example `["validate", "plan"]`, to also require
successful command receipts after its latest successful file edit. A later edit
invalidates earlier checks, and a failed rerun invalidates that operation's
success. These gates use checkpointed tool results, including journal receipts
recovered after a restart. They certify the configured commands' exit status;
use commands that inspect actual remote status/artifacts when workload completion
is required. A submission receipt alone cannot establish that a GPU job finished.

## Tools and authorization

Each profile defines `read_paths`, `write_paths` and named `operations`. Source
access is restricted to explicit workspace-relative paths. Edits require the
current file SHA-256 and one exact matching replacement. Symlinks, hard links,
parent traversal and `.git` access are rejected. Each task retains a unified
diff against its own first-edited file contents for human review.

`list_files` accepts an ancestor of a granted path, including `.` for the
workspace root. For example, with `read_paths: ["src/render.py"]`, listing `src`
returns `src/render.py` if it exists, without revealing sibling files or
directories. Directory grants still include their readable descendants, and
write grants also permit reading and listing those files. Listings exclude
symlinks, hard links, `.git` entries and nonexistent files. Listing an ancestor
does not grant permission to read or edit any additional path.

An operation contains fixed `argv`, a description and optional `pass_env` names.
The model chooses its name; it cannot supply command arguments or shell text.
`{python}`, `{workspace}`, `{task_id}` and `{run_id}` are the only substitutions.
`{python}` identifies the controller interpreter. `{run_id}` is stable for the
task, allowing submit/status/artifact operations to refer to the same workflow.
Commands run with `shell=False` in that specialist's workspace. They inherit
only PATH, HOME, locale, temporary-directory settings and explicitly listed
environment names. Command exit code, stdout and stderr become durable receipts.

`observation_only` is an optional strict boolean, defaulting to `false`. Set it
to `true` only after auditing a fixed command that observes existing state,
such as status or logs. It must not submit, cancel, edit, start a verifier or
otherwise change the workload. Keep its implementation outside model-editable
paths. Local diagnostic output and audit logs are allowed; this declaration is
operator trust, not automatic analysis or an OS sandbox. Models cannot supply
or change this flag. Default configurations retain their existing policy hashes;
opting in changes the policy and applies only to newly created calls/tasks.

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

For an operation explicitly marked `observation_only` before execution, the SDK
method `team.dismiss_interrupted_observation(task_id, call_id)` can discard a
lost result. It requires exclusive profile ownership, an unchanged policy and
a task awaiting attention. A transaction records a failed `InterruptedObservation`
receipt and its original classification, digest and error. The journal call is
terminal (`completed`), but the result is failed; it cannot satisfy a successful
operation gate. The task stays `needs_attention`. No command runs, no result is
invented, and no task is requeued. A second identical dismissal is idempotent.

Older calls without an original classification, writes, unclassified operations,
active tasks and profiles with mixed unsafe uncertainty are rejected. The
additive journal migration leaves old receipts intact and never upgrades old
calls into observations. The [workflow experiment bridge](../../npa/examples/specialists/workflows/README.md)
grants this narrow dismissal tool only when hybrid profiles opt in, allowing its
existing takeover control after all uncertain calls are resolved. Broad operator
reconciliation remains separate.

Storage failures report the affected phase and SQLite error class/code without
raw exception text. A failed receipt write remains uncertain. When storage is
unavailable, diagnostics explicitly distinguish an unpersisted receipt from a
durable result. Restore healthy storage before recovery; observation dismissal
does not repair an underlying filesystem or SQLite I/O failure.

Inspect the external result, then supply the interrupted call ID and its verified
JSON result through the monitor or `reconcile --call-id ... --result ...`.
Alternatively, `reconcile --retry` explicitly authorizes retry after inspection.
Provider failures with no tool effect can be retried through the same control.
Rejected model responses retain their reported usage and finish reason in the
activity log, while none of their proposed tools execute. This makes failed
generation costs visible; missing usage counters still mean unknown cost.

For automatic recovery, configure `fallback_models` as an ordered list of
explicit endpoints. Each entry accepts only `model`, `base_url`, `key_env` and
`model_options`, inheriting the endpoint defaults when omitted. It cannot change
workspace or tool grants. Truncated generations, malformed native calls, empty
answers or missing completion checks advance to the next configured model at a
durable boundary. That model stays active for the rest of the task, including
after restart. The monitor records the handoff, reason and all reported usage.
Exhausted candidates leave the task in `needs_attention`. There are no implicit
backup models. Identity mismatches, provider refusals, transport failures and
uncertain tool effects do not trigger this handoff.

Backups receive the authorized task's conversation and tool results; configure
only providers permitted to receive that data. Provider-specific reasoning
fields are removed on handoff, while tool calls and receipts remain intact.
The rejected generation is excluded. Recovery does not undo existing edits or
repeat a tool by itself. Model requests can still be repeated across a crash
before their graph checkpoint, as with the primary endpoint.

`model_options` supports `temperature` (0–2), `top_p` (0–1),
`reasoning_effort`, `chat_template_kwargs` and optional `max_tokens`.
The example uses DeepSeek's recommended agent sampling of temperature 1.0 and
top-p 0.95 from its [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/README.md).
Use settings appropriate to each endpoint; sampling behavior differs by model.
The example requires validation and planning; remove those completion gates for
profiles whose tasks should finish without running those commands.

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
worker processes. They repair deliberately broken transitions in copies of the
public Cosmos3 PAIDF and Sim2Real workflows, then run real validation and planning.
The test checks that each repaired file exactly matches its healthy original.
It pauses and restarts both workers before completion. It does not submit GPU
workloads. Exact operational evidence stays outside the repository; publish only
reviewed aggregate results.

The [simulation fanout experiment](specialists-simulation-experiment.md) compares
six real GLM/DeepSeek specialists against one Astra agent with matched tools and
simulator concurrency. It includes all trials, measured token estimates, actual
MuJoCo action replay and the observed reliability failures. The
[reproducible example](../../npa/examples/specialists/simulation/README.md) is an
optional paid workload, separate from the hermetic tests.

The [workflow experiment runner](../../npa/examples/specialists/workflows/README.md)
compares Astra alone with Astra supervising independent specialists through the
same Workbench operation grants. It records supervisor and specialist usage,
preserves failed attempts, and supports exclusive takeover after a worker needs
attention. Its [NuRec artifact verifier](../../npa/examples/specialists/workflows/VERIFICATION.md)
checks actual reconstructed scenes, rendered media, Rerun recordings and native
quality metrics against independently collected execution receipts. Submission
acceptance alone does not count as a completed workflow.

The [real NuRec experiment](specialists-nurec-experiment.md) exercised GPU
reconstruction and rendering on four scenes. Both original agent runs ended
with zero verified completions; four hybrid artifact sets passed separate
operator verification. The report retains the failures, costs and unequal
conditions, which prevent a superiority or savings claim.

The [NuRec source-repair experiment](specialists-nurec-repair-experiment.md)
required source edits and freshly verified stages: both arms completed 2/2
repairs. GLM authored the hybrid render fix; Astra repaired the viewer after
specialist escalation. The hybrid cost more in model-price equivalents.
