# Sylvest OpenVLA-OFT mixed-data checkpoint comparison

This is a deferred, source-pinned evaluation candidate for
[`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata).
It is not an accepted model, image, or benchmark result. It records a
reproducible test of the published robustness claim against the stated
OpenVLA-OFT baseline; fine-tuning alone is not treated as improvement.

The executable spec is
[`sylvest-oft-mixdata-libero-plus-comparison.yaml`](../../workflows/testing/sylvest-oft-mixdata-libero-plus-comparison.yaml).
Its successful path has five connected, substantive stages:

1. prepare a hash-bound LIBERO-Plus task/initial-state protocol excluding an
   externally supplied mix-SFT training inventory;
2. run upstream OpenVLA-OFT evaluator baseline simulator rollouts;
3. run the identical cases with the candidate checkpoint;
4. calculate per-case paired differences, category summaries, intervals, and
   exact McNemar statistics; and
5. publish the measured report, checksums, a Rerun `.rrd`, and the upstream
   MP4 rollout artifacts.

The default `libero_spatial` is one of the four expanded suites for which the
pinned LIBERO-Plus source publishes classifications. `libero_object`,
`libero_goal`, and `libero_10` are also valid. `libero_90` has no matching
published LIBERO-Plus classification and is deliberately not a workflow target.
One rollout per selected task/initial-state index follows the upstream
LIBERO-Plus evaluation configuration. The upstream evaluator maps its trial
index directly to a stored initial state, so `initial_state_indices` must be a
contiguous zero-based list; it is not presented as independently sampled random
seeds. This is not a substitute for a full benchmark.

## Evidence boundary

The checkpoint card's leaderboard is upstream authorship, not an NPA result. A
run supports only the paired simulator result it emits. The checkpoint card
does not publish a training-task split, so the prepare stage calls a task held
out only if its exact name is absent from a supplied immutable inventory.

The report does not establish training convergence, full-benchmark performance,
or physical-robot success. Its RRD logs measured paired cases, not a synthetic
visualization.

## Sources, credit, and provenance

| Component | Pinned source/revision | Credit and role |
| --- | --- | --- |
| OpenVLA-OFT evaluator | [`moojink/openvla-oft@e4287e94541f459edc4feabc4e181f537cd569a8`](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8) | Moo Jin Kim, Chelsea Finn, and Percy Liang; `GenerateConfig`, model initialization, `run_task`, and rollout videos are upstream-native. |
| OpenVLA foundation | [`openvla/openvla`](https://github.com/openvla/openvla) | Moo Jin Kim and OpenVLA collaborators; architecture/base-model lineage. |
| LIBERO-Plus benchmark | [`sylvestf/LIBERO-plus@4976dc30028e805ff8094b55501d532c48fec182`](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) | Senyu Fei and collaborators; task maps, classifications, simulator, assets, and robustness framing. |
| Candidate checkpoint | [`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata@a85655ec941bae6644c9fbdf62db02b9726d7cf5`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/a85655ec941bae6644c9fbdf62db02b9726d7cf5) | Sylvest; mixed-data OpenVLA-OFT candidate. |
| Baseline checkpoint | [`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10@638918f3d1c2e43a39a8a20772bdb8b91835e4b7`](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10/tree/638918f3d1c2e43a39a8a20772bdb8b91835e4b7) | OpenVLA-OFT authors; paired baseline. |

NPA adds task/initial-state pairing, a training-inventory exclusion rule, source
and revision checks, complete checkpoint-inventory provenance, cache ready
markers, and report artifacts. The immutable checkpoint cache is read-only to
the evaluator: its small mutable configuration files are copied into a
stage-local workspace while the model payload is linked or copied privately.
This avoids the upstream local-checkpoint compatibility path modifying a cached
snapshot. It does not modify the checkpoints or relabel upstream research as
NPA work.

Preserve the upstream citations with reports:

```bibtex
@article{fei25libero-plus,
  title={LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models},
  author={Senyu Fei and Siyin Wang and Junhao Shi and Zihao Dai and Jikun Cai and Pengfang Qian and Li Ji and Xinzhe He and Shiduo Zhang and Zhaoye Fei and Jinlan Fu and Jingjing Gong and Xipeng Qiu},
  journal={arXiv preprint arXiv:2510.13626}, year={2025}
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
| OpenVLA-OFT source | Pinned `LICENSE` is MIT, copyright Moo Jin Kim, Chelsea Finn, and Percy Liang (2025). | Attribution retained; not baked by this deferred candidate. |
| OpenVLA source | Upstream [`LICENSE`](https://github.com/openvla/openvla/blob/main/LICENSE) is MIT, copyright Moo Jin Kim, Karl Pertsch, and Siddharth Karamcheti (2024). | Lineage recorded; base-model/dependency rights remain separate. |
| LIBERO-Plus source | Pinned source has no `LICENSE`, `NOTICE`, or `COPYING`; GitHub supplies no license metadata. | Do not bake, distribute, or execute it until the authors publish or identify a license. This blocks the live run. |
| Candidate and baseline weights | The pinned public Hugging Face model cards declare `mit` and were readable without a gated-access prompt. | Operator runtime fetch only; no weights, adapters, or cache in a public image. |
| LIBERO-Plus assets/training data | The public cards for `Sylvest/LIBERO-plus`, `Sylvest/libero_plus_rlds`, `Sylvest/libero_plus_data_4suite`, and `Sylvest/libero_plus_lerobot` declare `mit`. | Runtime fetch only after source execution is lawful; no data/assets baked or redistributed. |
| Runtime cache | Revision-keyed cache uses a lock, validates an inventory including OFT adapter/action-head/proprioception-projector files, and atomically records content hashes. | Operator-owned, non-public; publish only checksum-bearing provenance. |
| Run outputs | MP4s, paired data, RRD, and report are from a future run. | Keep run-scoped and label simulator-only; access does not imply output redistribution rights. |

No new NPA EULA, `ACCEPT_*` variable, generic legal checkbox, telemetry opt-in,
or duplicate per-image attestation is added. The current public model and data
cards expose no documented acceptance mechanism. If one becomes concretely
necessary, use only that upstream mechanism and preserve its exact evidence.

No image has been built, pushed, or added to the public image catalog. This is
intentional: the source-license gap prevents a lawful functional runtime image.

## Resuming the deferred validation

First obtain a published source license and an authoritative mix-SFT task
inventory. Then build, scan, pull, and record a reviewed immutable runtime image
with only lawful runtime contents. NPA configuration selects storage,
credentials, project, and GPU; none is hardcoded in the spec.

```bash
npa/.venv/bin/npa workbench health preflight --checks nebius,s3,hf --json
npa/.venv/bin/npa workbench workflow validate-spec \
  workflows/testing/sylvest-oft-mixdata-libero-plus-comparison.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec \
  workflows/testing/sylvest-oft-mixdata-libero-plus-comparison.yaml \
  --run-id <owned-run-id> --json
npa/.venv/bin/npa workbench workflow stage-src \
  --bucket <run-scoped-bucket> --run-id <owned-run-id> --project <project>
npa/.venv/bin/npa workbench workflow submit \
  workflows/testing/sylvest-oft-mixdata-libero-plus-comparison.yaml --runtime \
  --run-id <owned-run-id> --var bucket=<run-scoped-bucket> \
  --var runtime_image=<reviewed-image@sha256:...> \
  --var training_task_ids_uri=s3://<run-scoped-bucket>/<inventory>.json
```

Independently inspect both `rollouts.json` files, decode every MP4 referenced in
the manifests, inspect `comparison.json`, report, checksums, and decoded RRD
before marking any live readiness verified. Retain the run prefix for
diagnosis/resume; cancel owned work before tearing down only owned resources.
