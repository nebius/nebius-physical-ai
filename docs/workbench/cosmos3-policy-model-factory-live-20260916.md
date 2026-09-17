# Cosmos 3 policy model factory: live validation

[Workflow guide](cosmos3-policy-model-factory.md)

**Result:** a functional learning round completed across the original training
run and an explicit evaluation continuation. Eight-GPU training produced a real
checkpoint; simulator evaluation measured 0 successes in one episode; feedback
returned `needs_work`; and Cosmos generated one review candidate. Policy
qualification failed, and the candidate remains ineligible for policy training.

## Scope

This execution tests the new native LIBERO policy workflow, independently of
the earlier PAIDF video-generation exercise. Training uses eight reserved
RTX PRO 6000 GPUs on one worker; evaluation and generation each request one GPU.
Training used source revision `f6f886557`. The corrected evaluation continuation
uses `044a840fa` and explicitly references that completed training manifest.

The functional recipe uses five optimizer iterations, one sample per rank,
gradient accumulation one, and seed 42. Evaluation selects task 0 and one
native initial state. This small execution cannot qualify the policy against
the workflow's full ten-task, 500-episode benchmark, or prove improvement.
The default eight-B200, 2,000-iteration recipe remains unqualified.

## Testing-workflow registration

The executable spec is
[`workflows/testing/cosmos3-policy-model-factory.yaml`](../../workflows/testing/cosmos3-policy-model-factory.yaml).
It remains an experimental testing workflow, listed in the
[workflow catalog](../../workflows/README.md) and registered in
[`SUBMIT_LIVE_MATRIX`](../../npa/src/npa/orchestration/npa_workflow/submit_matrix.py)
with tier `gpu` and `plan_only: false`. No existing main-workflow graph changes.
Its adjacent [readiness record](../../workflows/testing/cosmos3-policy-model-factory.readiness.json)
distinguishes execution evidence from the unqualified default training recipe.

On 2026-09-16, `validate-spec` and `plan-spec` both passed again, and all 42
submit-matrix tests passed. These were local checks of registration and planning;
the live results below come from the retained training and continuation runs.

## Proof fingerprints

The retained workflow and MP4 bytes were hashed again when recording this proof.
The video hashes identify the fully decoded artifacts described below; they do
not establish policy success or candidate task quality.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Testing workflow YAML | — | `48c9ef1df693503acb42128256fab7efb37c85ae71dbb94b2573403b9d778893` |
| Policy rollout MP4 | 524,162 | `c25993879f4f9d90e23e81d9b17c195ed42d389e1a19bbb5940191ef9d9c1629` |
| Generated candidate MP4 | 55,991,236 | `4c429947e4b8534aa9a9a022d37c3e03a11805c2c0a466983b3a35e6e7e44b61` |

## Training evidence

The native framework checkout, training dependency sync, pinned dataset/model
downloads, base-checkpoint conversion, and configuration dry run succeeded.
Training used Torch 2.10 with CUDA 13 and Transformer Engine 2.12 in the
framework's separate environment. All eight FSDP ranks completed.

| Measurement | Observed result |
| --- | --- |
| Completed optimizer iterations | 5 |
| Per-iteration loss | 13.4251, 17.5446, 12.2693, 15.8544, 12.7394 |
| Training subprocess elapsed time | 583.55 seconds, including native checkpoint save |
| Native checkpoint save | 411.76 seconds |
| Sampled GPU memory during training | Approximately 34.9–35.3 GB per GPU |
| Final distributed checkpoint | 36 files; 177,284,648,556 bytes |
| Complete published training prefix | 48 objects; 180,103,771,949 bytes, including VAE, logs and manifest |
| Retained native state | Model, optimizer, scheduler and trainer |

These timings exclude environment setup, input downloads, base conversion,
artifact publication, and controller overhead. The manifest's
`training_gpu_seconds` is subprocess elapsed time multiplied by eight; it is
not a measurement of reserved-pool utilization or end-to-end factory efficiency.
The loss sequence does not establish policy quality.

The native processor registry resolved the same revision as the pinned model
weights, verified from the live worker. A subsequent adapter change enforces
that check automatically before training and moves the completed checkpoint
within its workspace instead of making a second disk copy. Those two changes
have contract-test coverage; this execution used the earlier copy-based path.

## Evaluation and feedback

Training job completion and its published manifest are verified. The original
workflow resumed by adopting that exact completed job and proceeded to
evaluation; it did not repeat training. That worker downloaded and verified all
47 listed artifacts before starting its native environment setup.

The first evaluation attempt failed while building LIBERO's `egl-probe`
dependency because CMake was absent from the inference image. The corrected
bootstrap installs the system build tools and constrains Torch 2.5.1 plus
torchvision 0.20.1 to their CPU builds, preventing an indirect dependency from
upgrading the simulator's Torch stack. It records the simulator environment
and verifies that Torch has no CUDA runtime.

The corrected bootstrap passed a separate ARM64 Linux CPU preflight: native CLI
import, LIBERO task 0 reset, both 256-pixel cameras, and an OSMesa simulator
step. This setup check used a fixed test action, not the trained policy, and
is not counted as policy evaluation.

The next evaluation worker passed that simulator setup on AMD64, but the native
policy server could not import `nltk`. Its default guardrail runners require the
framework's `guardrail` dependency extra in addition to `train`. The corrected
bootstrap installs both extras from the frozen lockfile and checks guardrail
imports before downloading the checkpoint. Guardrails remain enabled. The
adapter also checks the native server's canonical `model` directory when
verifying the loaded checkpoint identity.

That worker then verified the complete training bundle and initialized native
guardrail assets, but config resolution required `LIBERO_ROOT` from the previous
training worker. Evaluation now clears only that unused dataloader root while
retaining the action/prompt metadata and native EMA checkpoint-loading path.
A native configuration preflight runs before simulator setup and checkpoint
download, so a resolution failure is detected without another full readback.

The corrected native preflight and policy server passed. The server's `/info`
identity matched the verified checkpoint's `model` directory. The simulator
then completed one real episode for task 0: put both the alphabet soup and the
tomato sauce in the basket.

| Measurement | Observed result |
| --- | --- |
| Native simulator steps | 520 |
| Task success | 0 of 1 episodes |
| Episode runtime error | None |
| Episode elapsed time | 170.488 seconds |
| Policy inference calls | 32 |
| First policy request | 25.915 seconds |
| Median policy request | 2.562 seconds |
| Native rollout GIF | 521 frames, 256 × 256, 26.05 seconds |

The GIF and its local H.264 MP4 conversion both decode completely and retain
all 521 frames and the same duration. The native GIF shows the third-person
camera; policy inference consumed both third-person and wrist observations.
Visual inspection of beginning, middle, and end frames showed arm motion with
the target objects remaining outside the basket, consistent with the failed
task result. This single episode is not a policy benchmark or an improvement
claim. Request timings describe this episode only, including its cold first
call, and exclude environment setup and checkpoint transfer.

Feedback completed with `decision: needs_work`, `qualified: false`,
`benchmark_complete: false`, and `improvement_proven: false`. It retained the
failed episode ID and produced one generation target from the task description.
The generation stage completed using the existing Cosmos3-Nano text-to-video
path. It published one 1280 × 720 MP4 with 189 frames at 24 FPS, lasting 7.875
seconds. Full decoding passed. Its guardrail receipt records successful
Blocklist and Qwen3Guard prompt checks, VideoContentSafetyFilter checks on all
189 frames, and RetinaFace postprocessing.

Content safety does not establish task fidelity. Sampled beginning, middle, and
end frames show a person operating a handheld device over loose food, rather
than a clear robot-arm placement of the two requested packaged objects. This
manual review found a task mismatch; no task-quality acceptance or action-label
validity is claimed. The candidate manifest retains `training_eligible: false`.
No retraining or policy promotion followed.

The evaluation, feedback and candidate-generation continuation finished with
workflow status `succeeded`. This describes successful execution and artifact
handoffs, while the measured policy and candidate-quality outcomes remain poor.

## Software validation

The current adapter passes 53 focused policy and CLI tests on both the local
development host and Linux Python 3.10. The broader Linux run at `6654e1a5a`
reported 21,226 passed, 56 failed, 221 skipped and one xpassed. All 56 failing
cases also failed on the unchanged `d743f1853` baseline in the same environment.
This is a baseline comparison, not a clean full-suite pass. Documentation
examples passed 589 checks before this report was finalized. Repository-wide
CI is tracked on the pull request; these local results do not stand in for the
full matrix.

## Runtime recovery

The local Linux submission service exhausted memory while the GPU worker
continued. Its supported CPU and memory discovery hints corrected local API
worker sizing. SkyPilot state was kept on a native Linux filesystem after a
macOS bind-mounted SQLite database failed. Neither change altered GPU job
resources or introduced job duration limits.

Before resuming, the exact managed training job was confirmed successful,
the training completion manifest was read back, and the eight-GPU workload
pod had disappeared. This proves workflow adoption of completed training.
It does not prove restoration of optimizer state after interrupted training.

The human cloud credential later expired. After normal sign-in, its changed
cache correctly invalidated the original local API identity binding. That
history was preserved. Read-only controller database inspection established
that the training job succeeded and the evaluation job failed, both terminal,
with no GPU workload pods remaining. Standard NPA orphan-controller cleanup
then removed the three idle controllers and verified remote absence. The
continuation uses a new isolated session and a direct reference to the completed
training artifact; it is not presented as transparent recovery across credential
rotation or as an uninterrupted four-stage run.

## Cleanup and retained evidence

All managed jobs reached terminal states. Standard workflow cancellation
confirmed no active job, and controller cleanup verified remote absence before
worker removal. The eight-GPU training pool and original one-GPU worker were
removed, followed by the remaining test cluster. The final provider inventory
verified zero clusters, compute instances, disks, filesystems, IP allocations,
AI endpoints, and AI jobs in the task's project.

The project, artifact storage, storage identity, and default network remain.
The trained checkpoint and review videos are retained. Before removing the
temporary local operator container, its run history was archived, all 2,096
archived file hashes were verified, and all 20 SQLite databases passed integrity
checks.

Exact job identities, private storage locations, raw logs, model artifacts,
and infrastructure inventory are retained in operator-controlled evidence.
Design provenance is recorded in the
[architecture assessment](../architecture/cosmos3-model-factory.md#design-provenance).
