# Astra with independent Workbench specialists

This opt-in runner compares **Astra alone** with **the same Astra coordinator
delegating to configured Token Factory specialists**. Both arms receive the same
scoped file tools, named operations and concurrent `run_operations` tool. The
hybrid adds durable `delegate`, `specialist_status`, `wait_specialist` and
`take_over` tools; its GLM/DeepSeek workers use the existing LangGraph runtime. Jev is optional in
Workbench and is not used by this explicitly assigned experiment.

Install the `agent-specialists` extra and the optional MCP dependency described
in [the simulation example](../simulation/README.md). The runner uses an
authenticated `codex exec -m gpt-6-astra` with native shell and additional Codex
workers disabled. Native file edits are outside the matched grant and invalidate
the recorded tool-scope check. Configure each specialist's explicit
model endpoint and sampling policy in an operator-owned `team.json`, following
[the specialist configuration guide](../../../../docs/workbench/specialists.md).
Keep credentials in the NPA credential store or the endpoint's `key_env`.

## Prepare and run a matched pair

Prepare a separate team configuration, fresh state directory and disjoint
workspaces for each arm. Provide **identical task files, instructions and
operation grants**; only workspace, run identity and artifact destination may
differ. The runner takes the same common prompt file for both arms:

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

Operations should return observations within the MCP transport deadline
(120 seconds). A long-running workflow submission needs a durable asynchronous
adapter, and a `wait` operation should return new stage evidence within a short
observation interval. This transport deadline does not cancel or constrain
remote jobs. `wait_specialist` waits locally for an event or terminal task state
for up to the requested `observation_seconds` (default 30, maximum 60), spending
no model tokens during that wait. Its elapsed observation interval leaves the
worker running. Supply `after_sequence` from the previous response to consume
only new receipts.

Use `required_operations` such as `validate` and `verify` for specialist
completion. `verify` must fail unless the actual remote workflow and its
artifacts satisfy the task. A successful submit, a specialist's final answer,
or the coordinator's exit code is insufficient evidence of workload success.
Both arms still need the same independent artifact grader.
The example [NuRec verifier](VERIFICATION.md) decodes downloaded USDZ, images,
video and Rerun data, and checks their agreement with retained workflow and
render receipts. Its verdict does not replace the experiment's quality and
cost comparison.

## Ownership and retained evidence

Delegation uses a stable `task_id`; repeating the identical request reuses its
assignment. While a delegated task is queued, running or awaiting attention,
the coordinator cannot act on that workspace. A host-local profile lock also
excludes delegation during a direct coordinator operation. An uncertain tool
effect blocks further effects, including a new delegated task, until externally
reconciled. The bridge exposes no automatic reconciliation or replay tool.
The hybrid coordinator initially delegates independent workspace tasks. After a
specialist enters `needs_attention`, Astra may inspect its receipts and call
`take_over`. This requires the profile lock and fully resolved tool effects,
preserves the prior error and receipts, and cancels only the local specialist
task. Astra can then repair the workspace and continue the existing remote
workflow through its configured operations. Takeover never cancels a remote
job, replays a tool or resolves an uncertain effect.

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
