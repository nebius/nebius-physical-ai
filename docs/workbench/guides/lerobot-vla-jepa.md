# LeRobot VLA-JEPA: pinned simulator fine-tuning candidate

[`lerobot-vla-jepa.yaml`](../../../workflows/testing/lerobot-vla-jepa.yaml)
is a five-stage native LeRobot workflow for a separately tested VLA-JEPA
candidate. It is deliberately separate from the accepted generic LeRobot 0.5.1
and optional 0.6.0 images: VLA-JEPA first appears in LeRobot **v0.6.1** at
source revision `7e241bd630a3719a56157a497ce5d08f244784f1` and requires the
`vla_jepa` extra.

The successful path is substantive and connected:

1. fetch and validate pinned LIBERO simulations, split held-out task ids, and
   recompute action/state statistics on training episodes only;
2. fine-tune the upstream VLA-JEPA pretraining checkpoint with native
   `lerobot-train`;
3. run native `lerobot-eval` held-out LIBERO policy rollouts and decode MP4s;
4. validate native bounded success metrics and checkpoint lineage; and
5. write checkpoint provenance plus a decoded factual Rerun `.rrd`.

The defaults match the documented 30,000-update, batch-32 fine-tuning shape,
but a single-GPU run is an operational workload, not a reproduction of the
paper's eight-GPU result or a convergence claim. LIBERO is a simulator dataset;
no physical robot result, deployment readiness, or benchmark score is implied.
The declared GPU resource is one RTX PRO 6000 Blackwell (`RTXPRO6000:1`) with
16 CPUs and 96 GiB of memory. A native run must record the actual device and
compute capability; it is not B200 evidence or a hardware-equivalence claim.

## Pinned upstream work and terms

| Artifact | Exact revision | Declared terms / credit |
| --- | --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot) | `7e241bd630a3719a56157a497ce5d08f244784f1` | Apache-2.0; Hugging Face LeRobot contributors |
| [VLA-JEPA Pretrain](https://huggingface.co/lerobot/VLA-JEPA-Pretrain) | `e946c3e5b538d760f4b4ff239d1b1c12090c041d` | Apache-2.0; LeRobot model authors and [ginwind/VLA-JEPA](https://huggingface.co/ginwind/VLA-JEPA) lineage |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `89644892e4d85e24eaac8bacfd4f463576704203` | Apache-2.0; Qwen |
| [V-JEPA2 ViT-L](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | MIT; Meta/Facebook |
| [HuggingFaceVLA LIBERO](https://huggingface.co/datasets/HuggingFaceVLA/libero) | `86958911c0f959db2bbbdb107eb3e17c5f9c798e` | Apache-2.0; HuggingFaceVLA |

The VLA-JEPA method is credited to Jingwen Sun, Wenyao Zhang, Zekun Qi,
Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen,
*VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model*,
arXiv:2602.10098 (2026). The full notice and LeRobot BibTeX are in
[`THIRD_PARTY_NOTICES.md`](../../../npa/docker/workbench/lerobot-vla-jepa/THIRD_PARTY_NOTICES.md).
NPA only supplies packaging, orchestration, and provenance adapters.

The three model repositories and a pinned `HuggingFaceVLA/libero` parquet member
accepted anonymous exact-revision byte-range probes with no cached login. The
adapter and workflow do not read `HF_TOKEN`, so the native submit forwards only
the genuine S3 credentials. This is not an NPA EULA, a new acceptance checkbox,
or a statement of use rights. Telemetry remains disabled, and no credential is
copied into image layers. Model weights, datasets, caches, checkpoints, videos,
and run output are never image layers. Public
redistribution remains **unvalidated** until the exact built bytes, source
closure, and runtime evidence are reviewed; this workflow uses an
operator-private image candidate only.

## Execute and inspect

Build and push the candidate image only to an operator-controlled registry. Do
not alter `lerobot_version_manifest.json` or treat `main`, 0.5.1, or 0.6.0 as
VLA-JEPA support. Supply the resulting immutable digest as an exact toolRef
override; the checked-in image value is intentionally non-routable.

```bash
VLA_JEPA_IMAGE='private-registry.invalid/operator/lerobot-vla-jepa@sha256:REPLACE_WITH_64_HEX_DIGEST'
NPA_PROJECT='operator-project'
NPA_K8S_CONTEXT='operator-kubernetes-context'
NPA_BUCKET='operator-authorized-bucket'

npa/.venv/bin/npa workbench health preflight --checks nebius,s3 --json
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/lerobot-vla-jepa.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/lerobot-vla-jepa.yaml \
  --run-id vla-jepa-operator-run --waves --json \
  --var "vla_jepa_image=${VLA_JEPA_IMAGE}"
npa/.venv/bin/npa workbench workflow preflight-images workflows/testing/lerobot-vla-jepa.yaml \
  --infra "k8s/${NPA_K8S_CONTEXT}" \
  --image-override "workbench.lerobot.vla_jepa=${VLA_JEPA_IMAGE}"
npa/.venv/bin/npa workbench workflow submit workflows/testing/lerobot-vla-jepa.yaml \
  --project "$NPA_PROJECT" --infra "k8s/${NPA_K8S_CONTEXT}" --runtime --stage-src \
  --run-id vla-jepa-operator-run --var "bucket=${NPA_BUCKET}" \
  --image-override "workbench.lerobot.vla_jepa=${VLA_JEPA_IMAGE}" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

After completion, materialize the report prefix and independently check
`report.json`, the checkpoint hash manifest, decoded rollout MP4s, and
`vla-jepa.inspection.txt`. The `.rrd` is acceptable only when both `rerun rrd
verify` and `rerun rrd print -vv` confirm its VLA-JEPA provenance and held-out
metric. Use the run's exact `npa workbench workflow status`, `logs`, and
`artifacts` output while it is running; cancel only that run if it must stop.

## Configuration boundary

`heldout_task_ids` defaults to `[0, 1]` in `libero_spatial`; preparation rejects
them from training and rollout passes those same ids to native LeRobot. For an
operator-owned physical robot dataset, replace only the dataset repo/revision
after verifying its LeRobot v3 features, action/state dimensions, camera keys,
terms, and source provenance. Cross-embodiment models may need the exact
upstream `policy.reinit_modules` configuration; that is intentionally not
assumed by this LIBERO recipe.
