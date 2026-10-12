# RoboCasa readiness continuation

PR #595 remains a draft. Source validation, exact-image qualification, real GPU
execution, visual usefulness, and image release are separate decisions.

## Source identities and repairs

| Execution source | Scope |
| --- | --- |
| `d24786597608e2267ef9a9a938ead074413ce5e7` | Initial main integration and native scanner controls |
| `9e12f0b29e1d26dc5446a220cb2341c26a62aa89` | Asset-publication and native-outcome repairs; Datasets 5.0.1; updated image recipe |
| `0a3a778bdd22f6b3d53aa16985873daf340bd54a` | Direct local dataset verifier and integration of published VLM provenance change |
| `d31d451b4b19711f2866e6c22cc41dde2668a31b` | Explicit local verifier inputs and reconciled image metadata; final image-build source |
| `fc6eedbe6406ef8442171528be3760b9747f705b` | External byte-scanner encoded-layer admission fix; image/runtime bytes unchanged |
| `3370510d2fd1edd4a810835dc012584c6d745006` | FIFO regression coverage correction only; all image/runtime/scanner bytes unchanged |

The final source integrates main
`f1ecd4255131374c417cdbb1f6c3e0d84b2a3514`. These commits are signed.
No source from another owner's uncommitted checkout was used.

An independent Codex review reproduced two defects in the first candidate:
an ancestor symlink could redirect asset publication and deletion outside the
asset root, and malformed native success values could be coerced into success.
The successor holds no-follow directory descriptors, checks ancestor identities,
performs descriptor-relative publication, and rejects malformed native scalar
evidence. Negative controls preserve the outside marker, reject namespace swaps
and FIFO receipts, and cover malformed success, reward, and termination values.
Independent successor review reproduced the repairs and closed all three
blocking source findings, including the vulnerable dependency. A nonblocking
test-coverage note led to a stronger FIFO regression: it first publishes a valid
tree, replaces the receipt with a FIFO, and proves the held-receipt helper is
reached before rejection. Independent review reproduced all eight boundary tests
and closed that fourth finding at `3370510d`. The scanner successor's ten new
tests and three additional independently constructed gzip controls also passed.
Runtime bytes are unchanged by the FIFO test correction. Native Claude was unavailable;
neither Claude review nor independent human approval is claimed.

The actual source-security comparison rejected the original Datasets 4.8.5 lock
for CVE-2026-66007. The successor pins 5.0.1 and updates the packaging-only
LeRobot and RoboCasa derivative identities consistently. The pinned upstream
LeRobot archive hash and patch applicability were verified. An intermediate
local-only dataset fixture triggered the generic Hub-download scanner; the final
verifier constructs the installed image-folder builder directly, with no scanner
suppression. Its first replay omitted the required file list and failed; the
corrected explicit two-file fixture passes offline in the actual image. The final
base/candidate security comparison reports zero blocking
findings. These failures remain retained in private evidence.

## Local results

- Initial full suite: 39,215 passed, six failed, 166 skipped, one xpassed.
  One real failure identified a retired image label, removed in the successor.
  Five failures arose from temporary fixture paths matching private/home-path
  assertions; an unchanged 30-test replay under task-owned temporary storage
  passed. The original failures were not rewritten or relabeled as passes.
- Repaired runtime and packaging focused tests: 354 passed.
- Final affected VLM and RoboCasa boundary checks: 339 passed, one skipped.
- Final precheck: 270 tests passed, with Ruff, formatting and dependency checks.
- Image/catalog/runtime checks: 1,055 passed; supplemental golden manifest,
  skills and image contract checks: 357 passed.
- Final aggregate confidentiality and reachable-commit Gitleaks checks passed.
  These local scans do not claim access to the operator's complete CI policy.
- Complete successor suite at `9e12f0b29e1d26dc5446a220cb2341c26a62aa89`:
  40,236 passed, 181 skipped, one xpassed, zero failures. The original execution
  SHA is retained; later changes are covered by the affected checks above, not
  presented as a fabricated rerun of the full suite at the final SHA.
- A fresh full required suite at `3370510d2fd1edd4a810835dc012584c6d745006`
  passed: 40,299 passed, 181 skipped, one xpassed, zero failures and 339 warnings.
  This run covered the substantive scanner change on its unchanged candidate;
  its log SHA-256 is
  `7b39a4fa8f9e946d34670d8f19b4450f975359f14bae6177342a993bd78493e7`.
  A later proof-only commit is not presented as that test's execution SHA.
- Freshly fetched main and the committed `3370510d` candidate passed the virtual
  merge-tree and dependency-fingerprint precheck; the working tree was not altered.

The native scanner implementation is byte-identical across these successors.
Its retained real controls include 114 Go subtest passes, 200 differential
literal cases, and whole reference/native/duplex populations. Reference and
native each scanned 47 records and 2,118,866 bytes with 3,524 expected findings;
the duplex control scanned 98 records and 480,026 bytes with 4,095 expected
findings. Both cancellation controls remain incomplete and invalid. Raw
`valid=false` on expected findings is not automated image acceptance. The
independent lane reviewed these full populations and their source binding.
The actual native integration sequence was also rerun successfully at `fc6eedbe`,
retaining the same expected-findings and cancellation boundaries. The unchanged
native helper does not imply an unchanged Python scanner.

## Image and workload scope

The earlier historical image was not accepted and is absent from the current
local image store. Its historical hash is not fresh proof of replacement bytes.
The new recipe changes the CUDA base and dependency closure, so it requires its
own exact-image qualification.

The local-only builds at sources `9e12f0b29e1d26dc5446a220cb2341c26a62aa89`
and `d31d451b4b19711f2866e6c22cc41dde2668a31b`
passed actual CPU ACT inference and checkpoint round-trip through the production
adapter, two positive and five negative dataset-metadata controls, and decoded
NPA-to-LeRobot conversion. Conversion covered two episodes, six state/action
rows, four videos and all 12 camera frames. Maximum pixel error was 1/255 against
the frozen tolerance of 4/255; state/action rows and pooled statistics were
compared numerically. A non-finite frame rate was rejected before output.
The non-root image smoke passed four of four checks. Metadata controls cover
lexical traversal, absolute paths and URL schemes; they do not establish symlink
containment or a sandbox for untrusted datasets.

Those build controls used temporary directories: their decoded-video and
checkpoint payloads were not retained. Independent review verified the actual
full build log and source assertions, not an independent redecoding of those
discarded controls. No GPU-rendered media exists for this candidate.

A subsequent network-disabled CPU run of the unchanged image verifier functions
retained its complete 35-file population: four control videos, inputs, dataset,
untrained ACT checkpoint, selected action, and missing-weights negative. It again
decoded all 12 frames and compared the six state/action rows and pooled statistics
using the verifier's numeric tolerances; maximum pixel error remained 1/255.
Missing images and model weights were rejected. Its report SHA-256 is
`45b310c45dfefced6cc19484bcdfbf27066b7ad54d5bc7065e1dec86c6e195e9`.
One original report field uses the word `exact`; the actual comparisons are
`torch.testing.assert_close` and pooled-statistic `numpy.allclose` with
`rtol=atol=1e-6`, not a zero-tolerance statistic claim. The original artifact is
unchanged. Independent review rehashed all 35 files (45,089,740 bytes), decoded
all four MP4s and all 12 frames, checked every state/action and ACT dataset row,
recomputed pooled statistics at the stated tolerances, and fully decoded all
five retained safetensors files. The selected finite seven-dimensional action
matches the original report. These are deterministic, untrained controls, not
learned-policy quality.

The original completed execution event is retained privately with SHA-256
`c7b984471dbe04ca7e6da8ea8ea9816e02ebc7d92094f96c35ee149315c39ede`.
It records the exact image, network-disabled Docker invocation, non-root user,
read-only runner, no GPU-device request and successful exit. This is original
execution-argument evidence, not a retrospective container-inspection receipt.

The four clips below are the unchanged 32×32, three-frame CPU conversion
controls. They contain solid-color numeric test vectors, not kitchen imagery,
learned behavior, or GPU rendering. Each is 0.3 seconds at 10 fps.

| Control clip | SHA-256 |
| --- | --- |
| [Workspace, episode 0](synthetic-cpu/workspace-episode-0.mp4) | `8ad7f00c18c83129cbd4007d6aef427a752d7e6762cc14206eff1ce309af0e24` |
| [Workspace, episode 1](synthetic-cpu/workspace-episode-1.mp4) | `a4841cc422a62f003ce6b93c4b35edd17df5d5784ffb9b0407336469481e4319` |
| [Wrist, episode 0](synthetic-cpu/wrist-episode-0.mp4) | `648a4a07e4e9975ba7e30d9b50588b86fa714f8ad248a909f5ac43b2197d88fc` |
| [Wrist, episode 1](synthetic-cpu/wrist-episode-1.mp4) | `0c04ba6b27e929ed93d21ec9f086fdf873ee3275395a7e46cda5243158476d54` |

The final local image manifest is
`sha256:e17848b74e5f89af640221dc575ec03d52d0d18d10800e4946b2cc4d30e000fb`;
its config is
`sha256:23edc66e8a14b0f4d7da82ae452f6565c7f5fca0acd4bcff0a79f9e1dcec9be6`.
The 9,850,598,912-byte Docker archive has SHA-256
`70c8ba138737e31e805eb0c893cdd4083deebdae1336561e680c5af4a0addd88`.
All-layer graph verification passed: 33 layers, 83,626 entries and
18,012,213,421 content bytes read. The separate restricted-payload scan completed
over 82,437 layer entries with no restricted-payload or history hits. These
different entry counts reflect different graph and payload scan scopes.

Pinned Trivy 0.70.0 passed the configured critical/fixed-vulnerability policy
with the repository ignore file and the separate all-severity secret scan
without that ignore file. Both had zero policy findings. This is not a claim of
zero vulnerabilities at every severity. The SPDX 2.3 SBOM contains 576 packages
and 1,555 relationships, with SHA-256
`953e77a51ac53fbce7d3babba80878574096adf69727e32ad61cd884e07feb8d`.

## Actual pinned asset population

The final image anonymously fetched and rehashed all 91 source-declared public
archives against the exact upstream LFS SHA-256 and size metadata: 11,213,250,500
compressed bytes, 123,399 receipt file records, and 91 valid published-tree
receipts. The pinned sources are
[`robocasa/robocasa-assets@1b92c3d02ca4354984fec961357db0bff7b32166`](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166)
and
[`nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF@420a04af939c34873e6839a586b70844baf28aab`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/tree/420a04af939c34873e6839a586b70844baf28aab).
Pinned source metadata and attribution were retained privately; no asset payload
was published by this continuation.

The actual downloaded Objaverse archive was rejected when its publication
ancestor was a symlink; the outside marker remained byte-identical with no new
outside files. A separate deliberately altered receipt digest was also rejected;
that negative changed the receipt, not the original asset tree. The complete
population report SHA-256 is
`2ac818294f14347683941a1c4d938849717152c2779b3217b540a3119e60087e`;
the frozen resumed runner SHA-256 is
`99950d07a2bb60192ab6067d43b114153a9cf03beb0db1f5c2d2d0e89416ac8c`.

Independent review rehashed all 91 archives and all 123,399 installed records
(24,419,152,636 bytes), reproduced all 91 canonical receipts, and matched every
one of the 123,387 regular decoded ZIP members plus 12 pinned XML overlays to
the installed population. No archive members were excluded from that comparison.

An initial attempt was interrupted during fourth-archive extraction after three
valid publications and four complete downloads. Its original script and log are
retained. The resumed attempt used the final image and vendor-default anonymous
transport, reused completed cache bytes, rehashed every archive, and passed the
same fixed criteria. This is asset-fetch and publication proof, not a kitchen
reset, rendered workload, or policy-quality result.

## Remaining qualification

Final exact-image qualification is pending. The first complete-byte attempt
failed with `complete_record_limit`, incomplete and invalid, before decoding
large encoded layer blobs. Both workers joined. Investigation found the outer
archive walker prematurely applied its 1 GiB logical-record limit to exact
graph-bound compressed containers. The signed scanner successor preserves that
limit for decoded files and unknown outer files, and routes only exact verified
encoded-layer members through existing size, digest and full-decoding checks.
Ten new boundary controls and 1,176 affected tests pass. Independent source
review accepted the scoped fix without changing matching or finding policy.
The actual whole-image rerun is active but incomplete; an admission-only
large-header test is not full-byte evidence.

The local complete-byte scan uses
a frozen inventory of 129 actual noncredential operational metadata literals
and the unchanged `exact-substring-v1` policy. That bounded scope is not the CI
regex policy, a complete customer inventory, or public-release authorization.
Every finding must remain available for independent whole-population review.

No current-candidate GPU kitchen rollout or policy-quality result is available.
The historical real environment reset failed result publication; it remains a
failed run, not a completed rollout. The frozen live protocol remains 200 ACT
updates on three training tasks, two held-out tasks, six matched policy/random
reset-seed pairs, five workflow states, 12 decoded videos and native outcomes,
then positive/negative calibration and blinded order-balanced hosted
`google/gemma-3-27b-it` review. No labels, thresholds, or success criteria were
tuned. VLM judgments cannot override native task failures.

The final workflow passes specification validation and plain planning, but its
production render check fails closed: `tool://lerobot` resolves the quarantined
default image. A separately accepted, compatible immutable trainer override must
therefore be selected and actually validated; package-version availability alone
does not demonstrate checkpoint or runtime compatibility.

The remaining external dependencies are an approved nonpublishing delivery
route for the exact qualified image, an accepted compatible trainer-image choice,
and verified service/trainer GPU capacity. No image was published and no GPU resource was allocated by this
continuation. Published source `d31d451b` has 31 successful checks and three
explicit skips: built-image payload scanning, PR smoke and queue guardrails.
Required `gitleaks`, `scan` and `security-regression` contexts all passed under
GitHub Actions app identity 15368. That source is clean and mergeable, but these
results do not transfer to an unpublished successor or establish runtime
readiness. Final published-head CI and merge-state audit remain pending.
No useful-policy, generalization, sim-to-real transfer, robot-safety, or
public-image-release claim is made.
