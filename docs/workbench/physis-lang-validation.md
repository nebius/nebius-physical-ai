# Physis-Lang experiment qualification

The complete [physical-prompting workflow](physis-lang.md) passed on Nebius
on 2026-10-02. All four executable stages completed through the standard
SkyPilot runtime, and the submit process exited with code 0. Independent
storage readback verified all 36 original videos, 2,916 decoded frames,
36 raw judge responses, and 108 assertion verdicts.

Keep this capability **experimental**. The description-only arm gained two
passing assertions in this small comparison; adding negative guidance did not
improve the aggregate pass rate. This is an independent paper-inspired
implementation, not a reproduction of the unreleased upstream implementation.

## Measured results

Six original scenarios and two seeds produce 12 videos per arm. Each video has
three assertions fixed before generation. Unknown verdicts remain in the
denominator and count as unproven.

| Conditioning arm | Pass | Fail | Unknown | Assertion pass rate |
| --- | ---: | ---: | ---: | ---: |
| Original scenario | 26 | 9 | 1 | 72.2% |
| Scenario plus physical description | 28 | 6 | 2 | 77.8% |
| Description plus negative guidance | 26 | 4 | 6 | 72.2% |

The description-only gain was 5.6 percentage points. Its cloth-draping results
improved, while its ramp-rolling results regressed. Negative guidance had zero
aggregate gain over baseline and more unknown judgments. The
[machine-readable evidence](validation/physis-lang.json) retains all paired
scores, individual verdicts, and original video hashes, including failures.

The judge received 16 chronological frames per video, the original scenario,
and its assertions. It did not receive the arm label or expanded prompt.
These sampled-frame VLM scores do not establish continuous physical correctness,
benchmark-equivalent performance, statistical reliability, or robot-policy
transfer. Review the original videos alongside the judgments.

## Full native GPU execution

| Property | Verified configuration |
| --- | --- |
| GPUs | Two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs |
| Work distribution | One resident Wan 2.1 14B pipeline per GPU; 18 requests per seed |
| Comparison | Six scenarios × two seeds × three arms = 36 videos |
| Every video | 1280×720, 81 frames, 16 fps, 50 denoising steps |
| Native pipeline | BF16 weights, FP32 VAE, UniPC flow shift 5, guidance 5 |
| Runtime | PyTorch 2.13.0, CUDA 13.0, Diffusers 0.38.0, Transformers 5.5.0 |
| Prompt model | `nvidia/Nemotron-3_5-Lightning` |
| Blinded judge | `MiniMaxAI/MiniMax-M3` |

Both live containers matched the accepted immutable Diffusers image digest and
had zero restarts. Generation used the full native settings without smoke
reductions. The per-video elapsed times averaged 2,121.5 seconds and summed to
76,374.1 seconds across the two workers; these are execution measurements, not
billing measurements or a throughput benchmark.

The audit re-read every stage checksum manifest, verified exact paired coverage,
checked prompts and native settings against the frozen recipe, decoded every
frame, and checked dimensions, frame rate, content, and SHA-256 hashes. It also
reparsed every raw judge response, recomputed each assertion score and summary,
and verified that every gallery MP4 matches its original generation receipt.

## Provenance and checks

The live run used source commit
`a364dea310ffb50bdecdac5c87f111f758ac8116`. The source digest, native module
hashes, workflow hash, image digest, pinned weight revision, recipe hash, report
hash, and all video hashes are recorded in the linked evidence. After merging
new main changes, the Physis runtime modules, workflow YAML, and focused tests
still matched the tested bytes; the workflow registration conflict was resolved
by retaining both main's new entries and this experiment.

The tested implementation passed the
[complete hosted candidate CI](https://github.com/nebius/nebius-physical-ai/actions/runs/36965743360),
including the full Python 3.12 coverage suite, browser tests, Python 3.10/3.14
compatibility, security regression, lint, documentation drift, and secret and
confidentiality scans. The merged branch also passed 103 focused tests and the
270-test precheck before the evidence update.

A prior prepare attempt failed its strict prompt-length contract before GPU
generation. The successful comparison used a fresh run and a frozen recipe.
A transient Kubernetes command-session authentication failure during generation
recovered without restarting either GPU workload. Neither event was rewritten
as a successful test.

Provider tenant membership was verified before launch. Concrete infrastructure
identities, credentials, raw operational records, and the original gallery stay
in private operator evidence. The public receipt contains no live infrastructure
identifiers. Upstream PhysThinker, PhysCapBench, self-evolving guidelines,
retrieval, and training remain deferred as described in the capability guide.

Exact-run cancellation checks and controller cleanup passed. The owned local
API was stopped, and the final residue audit found zero experiment pods while
the existing cluster remained Ready. Run state, original media, and private
cleanup receipts were retained.
