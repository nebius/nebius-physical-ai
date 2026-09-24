# Astra with independent Workbench specialists

This opt-in runner compares **Astra alone** with **the same Astra coordinator
delegating to configured Token Factory specialists**. Both arms receive the same
scoped file tools, named operations and concurrent `run_operations` tool. The
hybrid adds durable `delegate`, `specialist_status`, `wait_specialist`, `wait_specialists` and
`take_over` tools; its GLM/DeepSeek workers use the existing LangGraph runtime.
Optional per-profile Jev routing chooses an endpoint within the assigned
workspace's existing grants. Selecting a workspace explicitly does not bypass
that endpoint policy. A live Jev result requires an accepted provider receipt;
missing credentials or fallback are reported separately.

Install the `agent-specialists` extra and the optional MCP dependency described
in [the simulation example](../simulation/README.md). The runner uses an
authenticated `codex exec -m gpt-6-astra` with native shell and additional Codex
workers disabled. The recorded tool-scope check accepts only observed Workbench
MCP tools granted to that arm, plus messages and reasoning. Other native tools,
unknown item types, other MCP servers and malformed events invalidate the check.
Configure each specialist's explicit
model endpoint and sampling policy in an operator-owned `team.json`, following
[the specialist configuration guide](../../../../docs/workbench/specialists.md).
Keep credentials in the NPA credential store or the endpoint's `key_env`.

## Prepare and run a matched pair

Prepare a separate team configuration, fresh state directory and disjoint
workspaces for each arm. Provide **identical task files, instructions and
operation grants**; only workspace, run identity and artifact destination may
differ.

Both coordinators and specialists can read large files in inclusive, 1-based
line ranges with `read_file(..., start_line=100, end_line=160)`. The returned hash
always covers the full file, including unread lines, for subsequent guarded edits.

The default `--coordination completion` separates hybrid planning from review.
Astra delegates complete tasks and exits its planning turn. Python then waits
for all workers to finish or one to need attention without a running Astra
process or model calls. Intermediate successful completions accumulate without
another coordinator call. A fresh Astra turn receives the common requirements, workspace
policies and compact host-generated task reports. It reviews verification
receipts or recovers a failed worker; unrelated active workers continue. Full
worker traces stay in the private journal. A model's completion message alone
does not establish artifact acceptance.

Use `--coordination continuous` to reproduce the original single-conversation
supervisor. The baseline performs the same task through its direct tools in
either mode. Give both arms the same playbook, source, operation grants and
acceptance criteria. The completion mode changes orchestration, not permission
or the independent grader. `coordination.json` records the host waits, and
`coordinator-config.json` retains every fresh turn's invocation. All turns enter
the cost calculation, including unsuccessful planning and recovery.

The runner takes the same common prompt file for both arms:

```bash
umask 077
npa/.venv/bin/python npa/examples/specialists/workflows/experiment.py --live \
  --team-config "$ASTRA_TEAM_CONFIG" --prompt "$COMMON_TASK_PROMPT" \
  --output "$ASTRA_EVIDENCE" --arm astra-only --effort medium
npa/.venv/bin/python npa/examples/specialists/workflows/experiment.py --live \
  --team-config "$HYBRID_TEAM_CONFIG" --prompt "$COMMON_TASK_PROMPT" \
  --output "$HYBRID_EVIDENCE" --arm astra-tofa --effort medium
```

`--output` must be a new directory outside editable workspaces. A team with
existing tasks is rejected so previous results cannot enter a new trial. The
runner has no inference-token, workload-runtime or job-count budget. The
`--live` flag authorizes the configured operations as well as paid inference;
inspect those commands before running. Tool grants are not an OS sandbox.

For a workflow experiment, configure fixed `validate`, `plan`, `submit`,
`status`, `wait`, `logs` and `verify` operations. Those commands can call a
remote Linux operator host through fixed argv; the coordinator itself may run
on another OS. The adapter owns exact remote run identities and must preserve
Workbench's submit reconciliation, durable status and artifact verification.
Never pass arbitrary model-supplied shell commands. Forward needed environment
names explicitly with `pass_env`; secrets must not appear in argv or task files.

Direct operations should return observations within the MCP transport deadline
(120 seconds). A long-running workflow submission needs a durable asynchronous
adapter, and a `wait` operation should return new stage evidence within a short
observation interval. This transport deadline does not cancel or constrain
remote jobs. `wait_specialist` waits locally for an event or terminal task state
for up to the requested `observation_seconds` (default 30, maximum 60), spending
no model tokens during that wait. Its elapsed observation interval leaves the
worker running. Supply `after_sequence` from the previous response to consume
only new receipts.

For supervision across several workers, prefer `wait_specialists`. It waits for
any new terminal state or a paused task and returns compact states and cursors;
routine model/tool events keep accumulating privately without waking Astra.
Pass its `after_sequences` mapping into the next call. The default observation
interval is 60 seconds, after which running tasks are reported without being
stopped. Inspect full `specialist_status` receipts when attention is needed or a
task finishes, and remove ended tasks from the next wait. These waits invoke no
models themselves; coordinator calls before and after each wait still consume
tokens and belong in the cost comparison.

Use `required_operations` such as `validate` and `verify` for specialist
completion. `verify` must fail unless the actual remote workflow and its
artifacts satisfy the task. A successful submit, a specialist's final answer,
or the coordinator's exit code is insufficient evidence of workload success.
Both arms still need the same independent artifact grader.
The example [NuRec verifier](VERIFICATION.md) decodes downloaded USDZ, images,
video and Rerun data, and checks their agreement with retained workflow and
render receipts. Its verdict does not replace the experiment's quality and
cost comparison.
For an older pinned viewer, optional `legacy_rrd_review_evidence` supplies its
verified producer settings while retaining the same recording, frame and hash
checks. See [the evidence contract](VERIFICATION.md) for the required binding.

## Ownership and retained evidence

Delegation uses a stable `task_id`; repeating the identical request reuses its
assignment. While a delegated task is queued, running or awaiting attention,
the coordinator cannot act on that workspace. A host-local profile lock also
excludes delegation during a direct coordinator operation. An uncertain tool
effect blocks further effects, including a new delegated task, until externally
reconciled. The bridge exposes no automatic replay tool.
The hybrid coordinator initially delegates independent workspace tasks. After a
specialist enters `needs_attention`, Astra may inspect its receipts and call
`take_over`. This requires the profile lock and fully resolved tool effects,
preserves the prior error and receipts, and cancels only the local specialist
task. Astra can then repair the workspace and continue the existing remote
workflow through its configured operations. Takeover never cancels a remote
job, replays a tool or resolves an uncertain effect.

Status and single-task wait responses omit source text from earlier `read_file`
receipts by default. They retain file hashes, line ranges, omitted byte counts,
operation output, failures and uncertain calls. This avoids sending whole source
files repeatedly to the coordinator. The original text remains in the durable
journal; request `specialist_status(..., include_read_content=True)` to retrieve
it explicitly. Use `after_sequence` to consume subsequent receipts, or targeted
`read_file` calls for current source once the workspace is available.

Operators may explicitly declare audited read-only commands with
`"observation_only": true` in their operation configuration (default `false`).
Only existing-state observations qualify; a command that starts verification,
submits work, edits files or cancels jobs must remain unclassified. The runtime
trusts this declaration, so keep the implementation outside model-editable paths.
Both arms may invoke declared observations for fresh diagnostics while a
specialist is awaiting attention, even when an effect remains uncertain. The
profile lock still excludes active workers; reads do not release write ownership
or resolve uncertain submissions.

With this opt-in, the hybrid additionally receives
`dismiss_interrupted_observation(task_id, call_id)`. It accepts only a lost
observation classified under the original, unchanged task policy, with no active
work or mixed unsafe uncertainty. It atomically records a **failed** observation
receipt and preserves history; it does not replay, requeue or claim success.
The task remains `needs_attention` until the coordinator explicitly takes over
after all calls are resolved. Old/unclassified calls require separate operator
reconciliation. Configure identical observation declarations in both trial arms
and retain all failed receipts when comparing reliability and cost.

SQLite failures surface safe phase/error-class diagnostics. If a receipt cannot
be saved, the response marks `receipt_persisted: false`; it is not durable success
and must not authorize repeating an effect. Storage recovery is an operator task.

The output directory retains:

- `protocol.json`, `team.json`, `prompt.txt`: exact configuration, shared prompt
  digest and source hashes; `coordinator-config.json` records the executed argv.
- `codex.jsonl` and `codex.stderr`: every observed Astra event and diagnostic.
- `coordinator-end-task-receipts.json`: worker state when Astra exits.
- `task-receipts.json` and `coordinator-receipts.json`: final timestamped model,
  tool, delegation and observation receipts, including failures.
- `usage.json`: each Astra turn and every specialist response, including rejected
  generations. Missing or interrupted usage is marked incomplete.
- `execution.json`: coordinator and full local shutdown timing, error class,
  snapshot failures and unfinished delegated tasks.

When the coordinator exits or raises, local specialists stop at their current
durable node. The runner pauses their profiles, preserves the coordinator-end
snapshot, joins the local workers, then records final usage. It does not cancel
remote workloads. A failed coordinator can therefore leave real workflows
running; reconcile their exact identities through Workbench. Catastrophic host
loss may prevent final snapshots, but the private journals retain prior receipts.
Do not silently rerun an interrupted trial or replace it with a successful retry.

## Fair interpretation

Match data, GPU types and concurrency, source/image identities, training budgets,
quality requirements and injected faults. Alternate arm order and distinguish
warm caches from cold startup. Count Astra **plus** all specialist calls, failed
attempts, GPU allocation and other metered resources; token-only savings are
not total savings. Keep unknown cost unknown when billing evidence is missing.
This runner records observations; it does not claim the hybrid is faster,
cheaper or more reliable until an independently verified comparison establishes
that result.

After a trial finishes, the [usage summary script](USAGE.md) combines retained
Astra and specialist generations using an operator-supplied price table. It
reports unknown counters and ranges for unresolved cache or context-tier
pricing; GPU and artifact-quality comparisons remain separate.

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/agent_eval/test_specialist_workflow_experiment.py -q
```

The separate [NuRec repair report](../../../../docs/workbench/specialists-nurec-repair-experiment.md)
compares actual source edits followed by affected-stage execution. Both arms
verified 2/2 repairs; the hybrid needed an Astra takeover and cost more.
It preserves the original measured patches separately from later production
hardening and runtime improvements.
