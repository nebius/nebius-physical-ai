# Caption thinking: current source and review bridge

This is an additive source/review receipt, not another inference run or a
merge-readiness verdict. Its measured source is
`46735ee9e4af85020aaad5ac7a28ec239af5b469`, tree
`3b1472b054b74703f897dc13e92377738746be0f`, integrating main
`a5a934af1516b78d88b2b2755247ec59bb1559b5`.
[Source hashes, review identities and gate counts](summary.json) are separate
from the [original eight-response hosted proof](https://github.com/nebius/nebius-physical-ai/blob/32febf7bae6b45f4ed58a50d5dad7d0338ebfbc9/docs/testing/evidence/cursor-vlm-disclosures-20261003/pr832/readme.md).
The original public synthetic inputs and decoded captions remain at that
immutable proof; no private images are published here.

The original eight inference calls ran at
`26e6f8abec78d4c7bf0bd988c61b0a21d65d5122`, with one separate model-list
request. Six positive images produced visible captions; two valid blank PNGs
produced failed unavailable-image items. MiniMax's blank answer omitted the
final period. These are false-unavailability observations, not proof that the
image pixels were absent. A separate earlier reasoning-only failure remains
retained. MiniMax enabled reported 28 reasoning tokens; MiniCPM's generic
control reached the request, but provider honor is unestablished.

The caption core, CLI and SDK bytes still equal the original eight-call source.
The client is not byte-identical to that original: its only changed function
AST is `thinking_chat_extra`, which now refuses explicit booleans for known
`reasoning_effort` profiles, including Kimi-K3, before HTTP. No supported boolean
mapping is invented. Omitted controls preserve centralized defaults; the
MiniMax, Lightning and generic branches remain distinct. All four caption
source files equal the independently reviewed marker successor
`6712138684724625e27067980f8d4a541cf93ce6`. That review passed 114 controls and
replayed the eight original responses with unchanged request objects and
outcomes, zero new provider calls. This is CPU replay, not fresh inference or a
wire-level request capture.

Whole-path hermetic tests cover positive, exact sentinel, normalized nested
sentinel and mixed positive/sentinel inputs across omitted/true/false controls.
They check provider JSON through the extracted helper, processing after a
sentinel, failed counts, saved partial manifests and nonzero CLI/SDK outcomes.
The current main-integration supplement passed 2,579 tests with one inherited
opt-in GPU skip at the measured source above. It also exercises landed
completion/identity/grade and protected live-caller contracts; it is not a full
suite or GPU execution. Earlier affected 1,322-pass, 270-pass precheck and docs
checks retain their own execution source in the summary. The complete original
local full failed: one failure, 40,089 passes, 213 skips, one unexpected pass,
77.48% coverage. The inherited temporary-root oracle was subsequently repaired,
but that original full remains failed. A genuine passing full is still pending
the exact third configure-catalogue fixture repair and consuming runner/source
closure; no test is omitted or failure waived.

Claude availability changed after the historical proof's unavailable
observation. The operator's first-party Claude Code worker returned actual
`claude-opus-5` review at the primary source and one focused material-repair
source. Both original verdicts were changes required, not approval. The Kimi
override, historical evidence wording and default credentialed-test opt-in
findings have separate immutable independent Codex dispositions. Subsequent
source bridges are Codex review, not a new Claude review or future-head
approval. Returned packet/review/result hashes are retained; the original Mac
raw-output hashes are worker-reported because those bytes are unavailable on
the VDI. Binary media was omitted, so no Claude pixel inspection is claimed.

The original hosted record lacks contemporaneous after-state source fields;
later clean observations and hash bridges are not backdated proof of historical
immutability. Historical dependency packs bind their stated sources, not every
later head. No calibrated model quality, physical-task correctness, robot
safety, GPU/image release or human approval is established. Final published-head
CI, current-main integration and lifecycle/queue gates remain separate.
