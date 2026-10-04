# Missing-terminal VLM control review

This report records one narrow hosted check of the default rollout rubric:
intermediate progress is not task completion when the submitted evidence stops
before the requested terminal state. It does not calibrate a threshold, qualify
a judge, prove physical correctness, or certify robot safety.

## Hardware and workload applicability

| Item | Evidence |
| --- | --- |
| Source commit | `f3a8cfc42c19ae8dd499b566a39a2b3e13113d96` |
| Container image | Not applicable; the installed NPA client made hosted API requests |
| Local CPU | Real video decode, frame selection, PNG encoding, hashing, strict parsing, and private evidence validation |
| Local GPU | Not used |
| Hosted API | Six real image requests: three each to `MiniMaxAI/MiniMax-M3` and `openbmb/MiniCPM-V-4_5` |
| Provider hardware | Unavailable and unattested by the hosted API |
| Full workload | One seven-frame complete sequence, one seven-frame source-matched truncated sequence, and one seven-frame uniform-gray no-evidence control |
| Focused exact-head tests | 169 passed and 1 skipped |
| Independent review | Plan/labels, source/harness, exact-call permission, and retained pixels/rationales were reviewed separately |

This is not a hardware-performance result. No container or local accelerator
was involved.

## Frozen control

The task was to pick up a red cube and place it into a white box. Labels were
fixed before calls:

- `C01`, pass: the final submitted frame visibly shows the cube in the box and
  the gripper retracted;
- `C02`, fail: the sequence ends during held-cube transfer and shows no
  placement, release, retraction, or stable terminal state;
- `C03`, fail: seven byte-identical, uniform RGB `(128, 128, 128)` frames
  provide no task evidence.

The two real controls came from one SO-100 top-camera episode. They are
source-matched semantic controls, not a one-factor matched experiment. Each
model received the same neutral task and rubric text; private case identifiers,
roles, labels, source names, and paths were absent from provider-facing text.

## Hosted outcome

Exactly six approved calls completed without retries or errors. Every response
had HTTP 200, `finish_reason=stop`, non-empty usage, a distinct request
identity, the exact requested model, and strict JSON parsing.

| Model | C01 | C02 | C03 | Objective matrix | Rationale review | Narrow result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `MiniMaxAI/MiniMax-M3` | `1.0`, pass | `0.1`, fail | `0.0`, fail | TP=1, TN=2, FP=0, FN=0 | failed | rejected |
| `openbmb/MiniCPM-V-4_5` | `1.0`, pass | `0.0`, fail | `0.0`, fail | TP=1, TN=2, FP=0, FN=0 | passed | accepted |

The score matrix alone would have accepted both models. Independent pixel and
rationale review rejected MiniMax for this control: its `C02` rationale denied
visible grasp and transfer and claimed the robot moved away without the cube,
and its `C03` rationale called the byte-exact gray frames white. Its final
labels were correct, but the visual explanation was not.

MiniCPM's three rationales were grounded in the submitted pixels. Its `C02`
phrase “fails to place” is interpreted only as no placement shown in the
truncated submitted evidence, not as a claim that the underlying episode could
not later complete.

## Bounded claim

At the pre-frozen `0.8` threshold, on these exact three seven-frame inputs,
MiniCPM distinguished visible terminal placement from truncated intermediate
transfer and absent evidence, with grounded rationales. MiniMax produced the
expected score-derived labels but did not pass rationale grounding.

This is a regression control for missing terminal evidence, not threshold
calibration or broad model qualification. Other retained controls may still
disqualify either model for acceptance use.

## Public evidence bindings

- request freeze:
  `8d402a5048187396947be58072963b703887e69041104b14b89364a8db11a648`;
- exact-call approval:
  `7a857f70742ee068d8d068ef27247d951da905e2460482d52139d5aab1f1800b`;
- hosted state:
  `aec32006dbc48048bd111d1765d116893a94655184c81833aebf1386363dc77d`;
- immutable hosted summary:
  `aa68b2f1990f825c6b8b834cd406b5202f9647605f4450e263b8a0febc8dea8f`;
- independent retained-evidence review:
  `790c094673dd58197fba704e4a6aff4d0561b1287f7db69c19d298eab334cc60`.

Raw responses, complete rationales, provider request identities, source pixels,
private mappings, paths, and endpoint details remain outside the public
repository. Independent review recomputed all 21 control files, all 42
submitted frame byte/RGB bindings, all six raw-response hashes, and every
transport/result/state/summary chain without mismatch.

## Limitations

- Three items cannot estimate false-positive, false-negative, or
  generalization rates.
- The real controls differ in selected intermediate frames and motion blur.
- Selected stills cannot establish continuous execution, hidden state, gripper
  opening, release mechanics, long-term stability, physical correctness, or
  safety.
- Visible cube-in-box placement and gripper retraction do not establish future
  stability.
- Six provider responses do not independently audit provider-side billing or
  exactly-once delivery.
- No deployment recommendation, policy-quality claim, or robot-safety claim
  follows. A vision judgment cannot certify robot safety.
