# Literal evidence and failure policy

Use `npa.literal_values` at evidence boundaries: `require_boolean`,
`require_integer`, and `require_number` return unchanged literal values or raise
`ValueError` naming the actual input field. Numbers exclude booleans and numeric
strings; floats must be finite. Optional minimum and maximum bounds are inclusive.
Integers remain integers, including values too large to represent as floats.
Choose domain bounds at the caller, such as positive counts or nonnegative distance.

Callers own presence and default semantics: an omitted optional field can use a
documented default, while a present `null` must be validated as supplied. Do not
use truthiness to substitute defaults. Provider-specific normalization belongs at
its authoritative boundary, before evidence is exposed under the application schema.

Choose failure behavior by the operation:

- Reject malformed ingress before side effects. Never silently reinterpret it.
- Discovery quarantines or degrades the malformed record, retains healthy
  inventory, and emits explicit diagnostics identifying the rejected evidence.
- A transient ambiguous probe resets the success streak and retries within the
  existing deadline. It cannot count as successful evidence.
- Mutating reconciliation validates all static shape evidence before the first
  mutation, so a later malformed record cannot leave a partially applied plan.

Standalone generated backend modules that cannot import the package may maintain
an intentional local adapter. Document that packaging boundary and test its
behavior against this scalar contract; do not introduce alternate coercion rules.

Foxglove MCAP conversion validates every present simulator ground-truth flag
and the ground-truth object before creating or truncating the output. Missing
flags default to false; present null, strings, numbers, and containers fail with
the source filename, action index, and flag name. Completion reasons still
control the phase, while `success` retains its measured `placement_stable`
meaning. Phase and status fields consume the same validated evidence.
