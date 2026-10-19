# Specialist configuration routing

`Profile.configuration_routing` is an opt-in policy for selecting a model together
with its reasoning options, output reservation and observation-compaction policy.
It runs inside the durable specialist worker. Token Factory supplies inference;
the operator host runs routing, the journal, tools and verifiers. This change does
not deploy a hosted harness or provision compute.

The default is `None`, preserving existing profiles and task fingerprints. This
policy replaces legacy `model_router`, `fallback_models` and profile-wide
`compact_context`; do not combine them. The profile's `model` must name a candidate
and remains its display identity. The selected candidate supplies the actual
endpoint and options. Candidates never supply workspace or execution grants.

## Configuration

Install `npa[agent-specialists]` and construct the policy through the existing
Python, JSON configuration, CLI or specialist-service interfaces. For example:

```python
from npa.agent_backend.specialists.config import (
    AgentConfiguration, ConfigurationRouting, ContextEndpoint, Profile,
)

fast = "nvidia/Nemotron-3_5-Lightning"
deep = "zai-org/GLM-5.3"
policy = ConfigurationRouting(
    classifier=ContextEndpoint(
        model=fast, context_window=1048576,
        model_options={
            "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    candidates=[
        AgentConfiguration(
            name="fast-full", model=fast,
            description="Localized repairs with straightforward verification.",
            context_window=1048576,
            model_options={
                "max_tokens": 32768,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tools_verified=True,
            input_price_per_million=0.06,
            output_price_per_million=0.24,
        ),
        AgentConfiguration(
            name="deep-compact", model=deep,
            description="Difficult diagnosis or recovery after failed verification.",
            context_window=1024000,
            model_options={
                "max_tokens": 32768,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
            compact_context=True, tools_verified=True,
            input_price_per_million=1.4,
            output_price_per_million=4.4,
        ),
    ],
)
profile = Profile(
    name="geometry", description="Inspect sensor geometry",
    model=fast, workspace="/path/to/isolated-workspace",
    read_paths=["src"], configuration_routing=policy,
)
```

Replace the workspace and declare only required reads, edits and operations.
The model IDs, capacities and USD prices above are a public catalog snapshot from
2026-10-18. Recheck the [public model catalog](https://tokenfactory.nebius.com/api/public/models_info)
and key-scoped availability before use. `tools_verified` defaults to false: set it
only after verifying the exact served endpoint/options honor native tool calls.
This is an operator attestation, not an automatic capability probe.

Every candidate requires a unique `name`, nonempty `description`, declared
`context_window`, explicit positive integer `model_options.max_tokens`, and finite
nonnegative input/output prices per million tokens. Multiple configurations may
use the same model. Existing `base_url` and `key_env` endpoint fields apply; keys
remain in environment/credential storage. The classifier also requires its own
capacity and explicit output reservation. `framing_tokens` defaults to 4096.

## Decision and execution

Before every worker model call, the runtime builds each candidate's request view
and checks native-tool attestation, exclusions, context capacity and output space.
Admission estimates ASCII JSON bytes for messages and tool schemas, plus
`framing_tokens`, plus 128 per message. This estimate and the output reservation
must fit together in the declared window. Classifier admission includes its full
messages and response schema.

This is a conservative estimate, not exact served tokenization or a guarantee
about an undocumented provider template. Receipts set `exact_tokenization=false`.
The policy supports text and native tool messages; image, audio and video blocks
are rejected before inference. Artifact paths returned by tools remain text.

When several configurations are feasible, a structured classifier request
receives visible execution history, candidate criteria, reasoning settings,
declared prices and capacity receipts. It can select only an eligible route ID or
abstain. Missing credentials, malformed/truncated responses, identity mismatches,
abstention and classifier overflow block generation. A single feasible route
needs no classifier call.

The current configuration stays selected while feasible. A rejected generation,
or a terminal operation failure with `handoff_on_failure=true`, hands off when
another configuration remains. That excludes the previous configuration for the
rest of the task. A final backup can self-repair after verification failure.
Subsequent selection sees the recorded
failure and completed work. Context exhaustion can also force a different
configuration. This avoids routine paid routing and unnecessary switches; it
does not estimate cache reuse or optimize an empirical whole-task cost model.

An identical failed source-edit invocation repeated without a successful edit or
new user/recovery boundary also triggers handoff. This detects a concrete mutation
cycle; it is not a wall-time or model-call limit. Different corrective edits and
first failures remain opportunities for the current model to recover.

For deterministic local verification, `Operation.deterministic_inputs` declares
the complete data-file dependencies of the fixed verifier. These must be scoped,
readable files. The runtime hashes them before and after execution. A check that
changes inputs cannot pass, and completion rechecks the recorded hashes against
current files, including files the agent did not edit. Repeating a failed check
against identical declared input hashes is a no-progress cycle. This declaration
must not be used for remote or stochastic work, mutable external services, or
observation waits; those outcomes depend on more than local file contents.

The classifier cannot execute tools, expand grants or change fixed operation
arguments. Handoffs retain tool effects and never replay them. Hidden reasoning
is omitted from routing input and from handoffs across model/provider boundaries;
no encrypted compaction state or KV cache crosses providers. Pause/cancel controls
are checked again after classification and before generation.

## Context, completion and evidence

`compact_context=true` uses existing request-view pruning of superseded successful
source reads and repeated explicitly observational successful logs. Failed
receipts, current source, tool-call IDs and the full durable journal remain.
This is not semantic summarization. If required evidence cannot fit any candidate,
the task needs attention.

For bounded tasks, combine declared `required_operations` with
`finish_on_verified_operations=true` as documented in [specialists](specialists.md).
Completion then requires authoritative current-source verification and resolved
tool effects. Routing never substitutes a model claim for verification.

`Profile.verify_on_completion=true` additionally queues those checks when the
model returns a final answer without requesting them itself. This is opt-in and
requires configuration routing, `finish_on_verified_operations=true`, nonempty
`required_operations`, and deterministic input declarations on every required
check. Host-generated verification calls use the same durable tool journal and
pause/cancel boundaries as model-generated calls. Success is reported as
`runtime_verified`; a failure returns its actual receipt for repair or handoff.
Fixed-model policies and the final routed backup can both repair failed checks.
For comparisons, keep these recovery and verification rules the same across arms.

`task_report.configuration_routing` includes the current configuration, classifier
attempts and receipts, admission count and handoff count. Admissions include
checks later interrupted by operator controls and are not inference-call counts. `configuration_request`
events expose admission estimates for eligible and rejected candidates. Classifier
attempts are journaled before inference; an interrupted paid decision is not
silently repeated. Accepted decisions can be reused after a restart before
generation. Unknown usage remains unknown. Count classifier usage and latency
alongside worker inference. Declared token prices establish neither actual
whole-task savings nor outcome quality.

Worker inference attempts are journaled before requests as well. If execution
loses a response or fails to commit its graph checkpoint, the same request boundary
requires attention rather than silently paying again. An available usage receipt
still counts; an unfinished attempt has unknown usage. Tool effects remain governed
by their separate journal.

`task_report.configuration_routing.inference_accounting` separates worker and
classifier requests, normalizes standard and legacy usage counters, and retains
unknown attempts. For prices, call the shared
`npa.agent_backend.specialists.accounting.summarize_inference(events, rates)` with
a public-catalog snapshot keyed by model ID. Rate cards use
`input_price_per_million_tokens`, `output_price_per_million_tokens` and optional
`input_cached_price_per_million_tokens`. Missing usage or prices produce an unknown
cost, not zero. A missing cached rate prevents a cache-adjusted estimate whenever
cached usage is present or unknown. The uncached token-rate estimate is still
available when all input/output counters and ordinary prices are known. Retries
without separate usage make the complete estimate unknown. These are token-cost
estimates, not invoices; coordinator, host, workload compute and taxes are excluded.

## Editing redacted JSON

`replace_json` sets an existing value using `path`, full-file `expected_sha256`,
an RFC 6901 `json_pointer`, and native JSON `value`. It accepts strings, numbers,
booleans, null, arrays and objects; an empty pointer replaces the document. Use
it for lists or when the complete desired value is known from the requirements
but its previous value is redacted. It requires the same write grant, source
hash, durable receipt and subsequent verification as other source edits. It
rejects missing targets, duplicate original keys and nonfinite values. The tool
does not reveal an unknown old value or infer a hidden URI prefix.

URI redaction can make a literal whole-file replacement impossible. The existing
`edit_file` tool therefore accepts optional `json_pointer` (RFC 6901). It replaces
exactly one `old` string fragment with `new` inside the selected existing JSON
string. For example, change a known shard suffix by pointer while the surrounding
URI remains hidden. The full raw source SHA-256 must still match. Write grants,
original-source capture, edit receipts, check invalidation and replay rules remain
the same. New files, non-string targets, duplicate keys, nonfinite JSON values,
ambiguous matches and malformed pointers are rejected for this mode. Existing
plain-text `edit_file` calls keep their behavior.

## Research basis and validation scope

[HM-Router](https://arxiv.org/abs/2609.32213) motivates selecting model/harness
pairs. Here the choices are context/reasoning configurations within one Workbench
harness; this does not port the paper's learned scorer, weights or benchmark.

[Context-guided routing for stateful agents](https://arxiv.org/abs/2610.19292)
motivates execution-history input and accounting for switching costs. This policy
is zero-shot with sticky boundary decisions, not a reproduction of its evaluation.
[RouteLLM's controller](https://github.com/lm-sys/RouteLLM/blob/main/routellm/controller.py)
uses the last message and documents its first-turn training limitation; its
pretrained preferences are not assumed calibrated for these endpoints or tasks.

No external router code, weights or training data are bundled. Tests cover context
rejection, compaction integrity, history-aware failure handoff, endpoint identity,
paid-attempt recovery, grant preservation, operator controls and compatibility.
Functional integration evidence is distinct from a held-out quality comparison.
This is not a learned or calibrated claim of model superiority.
