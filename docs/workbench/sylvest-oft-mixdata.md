# Sylvest OpenVLA-OFT mixed-data checkpoint comparison

This is a source-pinned paired evaluation for
[`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata).
It does not treat fine-tuning as improvement. A result is a reproducible test
against the stated OpenVLA-OFT baseline only when both checkpoints complete the
same source-pinned simulator cases and produce the declared artifacts.

The primary executable spec is
[`sylvest-oft-mixdata-original-libero-comparison.yaml`](../../workflows/testing/sylvest-oft-mixdata-original-libero-comparison.yaml).
It uses the original MIT-licensed
[`Lifelong-Robot-Learning/LIBERO@8f1084e3`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01),
not the separately deferred LIBERO-Plus source. Its successful path has five
connected, substantive stages:

1. prepare a hash-bound original-LIBERO task/initial-state-byte protocol, either
   excluding an authoritative externally supplied mix-SFT training inventory or
   explicitly marked `training_coverage_unknown`;
2. run upstream OpenVLA-OFT evaluator baseline simulator rollouts;
3. run the identical cases with the candidate checkpoint;
4. calculate per-case paired differences, category summaries, intervals, and
   exact McNemar statistics; and
5. publish the measured report, checksums, a Rerun `.rrd`, upstream MP4 rollout
   artifacts, and the preparation-stage third-party notice record.

The default `libero_spatial`, along with `libero_object`, `libero_goal`, and
`libero_10`, is an upstream original-LIBERO suite. One rollout per selected
task/initial-state index follows the OpenVLA-OFT evaluator configuration. The
workflow pins each selected `.pruned_init` file SHA-256 and passes one matching
configuration (seed, camera resolution, crop, input-frame count, and open-loop
action count) to both checkpoint rollouts. Upstream trial indices are contiguous
zero-based indices into stored initial states, not independently sampled seeds.
This is not a substitute for a full benchmark.

The related
[`sylvest-oft-mixdata-libero-plus-comparison.yaml`](../../workflows/testing/sylvest-oft-mixdata-libero-plus-comparison.yaml)
retains the same five-stage contract but is deliberately **plan-only**. Its
pinned LIBERO-Plus source has no authoritative source license; it is not a
substitute for, nor evidence of, the original-LIBERO result.

### Author-published asset compatibility is a separate, narrow capability

The author-published [`Sylvest/LIBERO-plus` asset
card](https://huggingface.co/datasets/Sylvest/LIBERO-plus) declares `mit`.
That permits a distinct operator-private **asset-only** profile: hash an exact
selected archive and card, assemble a selected scene through original
MIT-licensed LIBERO's `TableArena`, and retain native MuJoCo/EGL render
evidence. It does **not** grant a license to
[`sylvestf/LIBERO-plus`](https://github.com/sylvestf/LIBERO-plus), whose source
remains deferred.

The scoped native compatibility check used original LIBERO at the pinned
revision, compiled the selected scene with MuJoCo 3.3.2, passed `mj_forward`,
and emitted two distinct decoded 256×256 EGL camera views. Its selected asset
archive/card hashes, scene hash, images, and renderer receipt are retained in
operator-private run evidence. The inspected asset profile has no BDDL task
definition or benchmark randomizer. It is therefore not a LIBERO-Plus rollout,
policy evaluation, 10,030-task benchmark, success metric, or generalization
claim; those remain blocked on separately licensed benchmark-source execution.

## Evidence boundary

The checkpoint card's leaderboard is upstream authorship, not an NPA result. A
run supports only the paired simulator result it emits. The checkpoint card does
not publish a training-task split. The default protocol is therefore
`training_coverage_unknown`: it selects the named suite but may include
in-distribution candidate tasks and must never be called a held-out
generalization result. `comparison_scope=held_out` is accepted only with a
supplied immutable inventory and excludes its exact task names.

The report does not establish training convergence, full-benchmark performance,
or physical-robot success. Its RRD logs measured paired cases, not a synthetic
visualization.

## Sources, credit, and provenance

| Component | Pinned source/revision | Credit and role |
| --- | --- | --- |
| OpenVLA-OFT evaluator | [`moojink/openvla-oft@e4287e94541f459edc4feabc4e181f537cd569a8`](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8) | Moo Jin Kim, Chelsea Finn, and Percy Liang; `GenerateConfig`, model initialization, `run_task`, and rollout videos are upstream-native. |
| OFT bidirectional Transformers dependency | [`moojink/transformers-openvla-oft@bc339d9ad707454c0c115970db43c260067c61ab`](https://github.com/moojink/transformers-openvla-oft/tree/bc339d9ad707454c0c115970db43c260067c61ab) | Hugging Face team and Moo Jin Kim; Apache-2.0 fork used by OFT for documented full bidirectional attention. The primary workflow pins and license-verifies it rather than resolving OFT's mutable dependency URL. |
| OpenVLA foundation | [`openvla/openvla`](https://github.com/openvla/openvla) | Moo Jin Kim and OpenVLA collaborators; architecture/base-model lineage. |
| OFT data dependency | [`kvablack/dlimp@92e3eca97af3b14d0b6aa15182c0dc240407698d`](https://github.com/kvablack/dlimp/tree/92e3eca97af3b14d0b6aa15182c0dc240407698d) | Kevin Black; licensed Apache-2.0 parent. NPA derives a private runtime copy with only `dlimp/dataset.py`'s `options.deterministic = False` changed to `True`, preserves `LICENSE`, and writes `NPA_MODIFICATIONS.md`. It does not fetch the unlicensed `moojink/dlimp_openvla` fork. |
| Original LIBERO benchmark | [`Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) | Yifeng Zhu and LIBERO authors; MIT-licensed benchmark source, task map, simulator, and stored initial states for the primary paired route. |
| LIBERO-Plus benchmark (deferred) | [`sylvestf/LIBERO-plus@4976dc30028e805ff8094b55501d532c48fec182`](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) | Senyu Fei and collaborators; robustness framing and separate source. It is neither executed nor redistributed by the primary route. |
| Candidate checkpoint | [`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata@a85655ec941bae6644c9fbdf62db02b9726d7cf5`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/a85655ec941bae6644c9fbdf62db02b9726d7cf5) | Sylvest; mixed-data OpenVLA-OFT candidate. |
| Baseline checkpoint | [`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10@638918f3d1c2e43a39a8a20772bdb8b91835e4b7`](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/tree/638918f3d1c2e43a39a8a20772bdb8b91835e4b7) | OpenVLA-OFT authors; paired baseline. |

NPA adds task/initial-state pairing, an inventory-gated held-out rule, an
explicit unknown-coverage alternative, source and revision checks, complete
checkpoint-inventory provenance, cache ready markers, and report artifacts.
The primary runtime fetches source-only OpenVLA-OFT, its pinned Apache-2.0
bidirectional Transformers dependency, dlimp, and original LIBERO
checkouts by immutable revision to lock-protected, atomically marked worker
caches; it does not assume a source checkout is baked in the image. Preparation
writes a third-party notice record that the final report stage consumes and
republishes alongside its own provenance.
The Apache dlimp derivative is content-inventoried and atomically marked ready;
the rollout refuses an already-imported unverified dlimp module. The immutable
checkpoint cache is read-only to the evaluator: its small mutable configuration
files are copied into a stage-local workspace while the model payload is linked
or copied privately. This avoids the upstream local-checkpoint compatibility
path modifying a cached snapshot. It does not modify the checkpoints or relabel
upstream research as NPA work.

Preserve the upstream citations with reports:

```bibtex
@article{fei25libero-plus,
  title={LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models},
  author={Senyu Fei and Siyin Wang and Junhao Shi and Zihao Dai and Jikun Cai and Pengfang Qian and Li Ji and Xinzhe He and Shiduo Zhang and Zhaoye Fei and Jinlan Fu and Jingjing Gong and Xipeng Qiu},
  journal={arXiv preprint arXiv:2510.13626}, year={2025}
}
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310}, year={2023}
}
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645}, year={2025}
}
@article{kim24openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
  journal={arXiv preprint arXiv:2406.09246}, year={2024}
}
```

## License and redistribution boundary

| Boundary | Current authoritative finding | Decision |
| --- | --- | --- |
| OpenVLA-OFT source | Pinned `LICENSE` is MIT, copyright Moo Jin Kim, Chelsea Finn, and Percy Liang (2025). | Attribution retained; source-only checkout is fetched at the exact revision into the operator-owned runtime cache. |
| OFT bidirectional Transformers dependency | Pinned `moojink/transformers-openvla-oft@bc339d9a…` has Apache-2.0 `LICENSE` (SHA-256 `77fd4710…2049`) and identifies the documented Transformers 4.40.1 line. | Fetch the exact source-only revision into the operator-owned runtime cache and reject a pre-imported or unpinned Transformers module; it is not a model payload. |
| OpenVLA source | Upstream [`LICENSE`](https://github.com/openvla/openvla/blob/main/LICENSE) is MIT, copyright Moo Jin Kim, Karl Pertsch, and Siddharth Karamcheti (2024). | Lineage recorded; base-model/dependency rights remain separate. |
| dlimp data dependency | Pinned `kvablack/dlimp@92e3eca…` carries Apache-2.0 (`LICENSE` SHA-256 `c71d239d…d0ab4`). The former OFT fork differs only by omitting that license and setting `options.deterministic = True`. | Runtime-fetch the licensed parent only. Derive a private content-inventoried copy with that exact one-line change plus a modification notice; preserve Apache notice. |
| Original LIBERO source | Pinned source `LICENSE` is MIT (SHA-256 `e2885fd3…68ff6`). | Runtime-fetch or use in an operator-private runtime; preserve its copyright/license and source revision in protocol provenance. |
| Original LIBERO benchmark data and stored initial states | The pinned upstream README separately identifies datasets as CC-BY-4.0. | Keep initial-state data and run outputs operator-private for this qualification; preserve source/data license attribution in protocol provenance and do not infer a public redistribution decision for resulting media. |
| LIBERO-Plus source | Pinned source has no `LICENSE`, `NOTICE`, or `COPYING`; GitHub supplies no license metadata. | Do not bake, distribute, or execute it until the authors publish or identify a license. This blocks only the separate LIBERO-Plus workflow. |
| Author-published LIBERO-Plus task assets (narrow asset-only profile) | The [`Sylvest/LIBERO-plus` dataset card](https://huggingface.co/datasets/Sylvest/LIBERO-plus) declares `mit`; its exact selected archive and card revision are hash-bound in private run provenance. The archive does not license the separate GitHub benchmark source. | Runtime-fetch an exact selected asset only into an operator-private original-MIT-LIBERO scene-compatibility run. Do not bake or redistribute it, and do not call the resulting MuJoCo/EGL render a benchmark, rollout, policy result, or source grant. |
| Candidate and baseline weights | The pinned public Hugging Face model cards declare `mit` and were readable without a gated-access prompt. | Operator runtime fetch only; no weights, adapters, or cache in a public image. |
| Other LIBERO-Plus assets/training data | The public cards for `Sylvest/libero_plus_rlds`, `Sylvest/libero_plus_data_4suite`, and `Sylvest/libero_plus_lerobot` declare `mit`. Their task/benchmark relationship and output scope are separate questions. | Do not bake or redistribute. This onboarding has not accepted them for benchmark-source execution, training, or policy evaluation. |
| Runtime cache | Revision-keyed cache uses a lock, validates an inventory including OFT adapter/action-head/proprioception-projector files, and atomically records content hashes. | Operator-owned, non-public; publish only checksum-bearing provenance. |
| Run outputs | MP4s, paired data, RRD, and report are from a future run. | Keep run-scoped and label simulator-only; access does not imply output redistribution rights. |

No new NPA EULA, `ACCEPT_*` variable, generic legal checkbox, telemetry opt-in,
or duplicate per-image attestation is added. The current public model and data
cards expose no documented acceptance mechanism. If one becomes concretely
necessary, use only that upstream mechanism and preserve its exact evidence.

One task-owned neutral private bootstrap was built, pushed, and pulled for
control-plane qualification only; NPA Kubernetes preflight then rejected that
digest because it lacked the required SkyPilot bootstrap-contract attestation.
No workload was launched from that digest and it is not qualified. The primary
MIT-LIBERO route requires an operator-private, attested runtime selected through
NPA configuration; weights, adapters, caches, and run outputs remain
runtime-owned. No public image publication is authorized.

## Executing the primary route

Use the original-LIBERO spec with an operator-private immutable runtime digest
that has passed the required image pull/bootstrap checks. NPA configuration
selects storage, credentials, project, runtime image, and GPU; none is
hardcoded in the spec. Obtain an authoritative mix-SFT task inventory only when
making a held-out claim; otherwise retain `training_coverage_unknown`.

```bash
npa/.venv/bin/npa workbench health preflight --checks nebius,s3,hf --json
npa/.venv/bin/npa workbench workflow validate-spec \
  workflows/testing/sylvest-oft-mixdata-original-libero-comparison.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec \
  workflows/testing/sylvest-oft-mixdata-original-libero-comparison.yaml \
  --run-id "<owned-run-id>" --json
npa/.venv/bin/npa workbench workflow stage-src \
  --bucket "<run-scoped-bucket>" --run-id "<owned-run-id>" --project "<project>"
npa/.venv/bin/npa workbench workflow submit \
  workflows/testing/sylvest-oft-mixdata-original-libero-comparison.yaml --runtime \
  --run-id "<owned-run-id>" --var bucket="<run-scoped-bucket>" \
  --var runtime_image="<reviewed-image@sha256:...>" \
  --var comparison_scope=held_out \
  --var training_task_ids_uri="s3://<run-scoped-bucket>/<inventory>.json"
```

Independently inspect both `rollouts.json` files, decode every MP4 referenced in
the manifests, inspect `comparison.json`, report, checksums, and decoded RRD
before marking any live readiness verified. Retain the run prefix for
diagnosis/resume; cancel owned work before tearing down only owned resources.
