# LeRobot GPU Benchmarks: Reproducing the Research via Nebius Workbench

This cookbook documents how to reproduce the May 2026 LeRobot GPU benchmark
research by @tle and @mnrozhkov with Nebius Physical AI Workbench. It is written
for robotics teams evaluating Nebius GPU options, partners coordinating case
studies with Nebius, and solution architects who need a technical artifact they
can run or hand to a customer.

The benchmark compares four LeRobot policy architectures across four NVIDIA
GPUs. It measures training throughput, profiling behavior, and single-sample
inference latency. It does not measure convergence, final loss, policy quality,
power draw, or energy efficiency.

## TL;DR

For the headline result, reproduce Diffusion Policy on H200 first:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-h200 deploy \
  --project-id <NEBIUS_PROJECT_ID> \
  --tenant-id <NEBIUS_TENANT_ID> \
  --region <NEBIUS_REGION> \
  --runtime container \
  --gpu-type gpu-h200-sxm \
  --gpu-preset 1gpu-16vcpu-200gb \
  --disk-size 500

npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-h200 profile-train \
  --run diffusion:lerobot/pusht:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0
```

The benchmark's H200 Diffusion result is 33.4 step/s at batch size 8. The B300
result with stock PyTorch is 12.2 step/s, approximately 2.5x slower, because the
B300 path falls back through PTX JIT for `sm_103` kernels.

| Policy | Parameters | H200 | B300 | RTX PRO 6000 | L40S | Guidance |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VQ-BeT | 38M | 119.7 step/s | 113.0 step/s | 102.8 step/s | 90.4 step/s | H200 and B300 are close for training. B300 has the lowest VQ-BeT inference latency. |
| ACT | 52M | 35.5 step/s | 37.0 step/s | 30.2 step/s | 27.7 step/s | H200 and B300 are comparable on a single run. |
| Diffusion Policy | 263M | 33.4 step/s | 12.2 step/s | 23.0 step/s | 14.0 step/s | Use H200 unless you have a native `sm_103` PyTorch build for B300 or need B300 for availability. |
| SmolVLA | 450M | 10.5 step/s | 11.8 step/s | 9.2 step/s | 7.1 step/s | H200 and B300 are comparable; choose based on availability, memory needs, and cost. |

Use `profile-train --mode wallclock` for throughput comparisons. Use
`profile-train --mode profiler` only for per-stage diagnosis on a single GPU,
because `torch.profiler` synchronization can distort cross-GPU comparisons by up
to 40%.

## Prerequisites

- A Nebius AI Cloud account with quota for at least one of the benchmark GPUs.
- The Nebius CLI installed and authenticated. See the
  [npa quickstart](../../quickstart.md) for setup.
- Terraform on `PATH` for VM or container workbench deploys.
- `npa` installed from this repository.
- A Hugging Face token in `~/.npa/credentials.yaml` or `HF_TOKEN` for datasets
  and model weights that require authentication.
- An object-storage bucket for checkpoints and benchmark artifacts.
- Local or project configuration values for `<PROJECT_ALIAS>`,
  `<NEBIUS_PROJECT_ID>`, `<NEBIUS_TENANT_ID>`, and `<NEBIUS_REGION>`.

The research used single-GPU runs, batch size 8 for the main comparison, local
dataset cache, and max dataloader workers for each node. Node CPU counts differ
by GPU shape, so use the wall-clock profiler command when comparing GPUs.

## GPU Selection Guide

| GPU | Architecture | SM | VRAM | Memory bandwidth | Node CPUs in research | Workbench platform |
| --- | --- | --- | ---: | ---: | ---: | --- |
| L40S | Ada Lovelace | `sm_89` | 48 GB | 864 GB/s | 40 | `gpu-l40s-a` |
| H200 | Hopper | `sm_90` | 141 GB | 4.8 TB/s | 16 | `gpu-h200-sxm` |
| B300 | Blackwell | `sm_103` | 270 GB | 8.0 TB/s | 24 | `gpu-b300-sxm` |
| RTX PRO 6000 | Blackwell | `sm_120` | 96 GB | 1.8 TB/s | 24 | `gpu-rtx-pro-6000` |

| Policy | Recommended | Acceptable | Avoid for this benchmark | Notes |
| --- | --- | --- | --- | --- |
| Diffusion Policy | H200 | RTX PRO 6000, L40S | B300 with stock PyTorch | B300 is approximately 2.5x slower than H200 because stock PyTorch lacks native `sm_103` SASS. |
| ACT | H200 or B300 | RTX PRO 6000 | L40S when top throughput matters | B300 leads H200 by 4% in the single run, which is within likely run-to-run variance. |
| SmolVLA | H200 or B300 | RTX PRO 6000 | L40S when top throughput matters | B300 leads H200 by 12% in the single run; treat the ordering as directional. |
| VQ-BeT | H200 or B300 | RTX PRO 6000, L40S | None at this scale | H200 leads B300 by 6% for training; B300 is 2.8x faster than H200 for VQ-BeT inference. |

### The B300 Diffusion Caveat

The benchmark ran PyTorch 2.10.0 with CUDA 12.8. Stock PyTorch provided native
SASS for `sm_89`, `sm_90`, `sm_100`, and `sm_120`, but not for B300's `sm_103`.
B300 therefore JIT-compiled kernels from `sm_100` PTX at runtime. Diffusion
Policy dispatches many small convolution and attention kernels, so this penalty
is visible in both training and inference.

Building PyTorch with native `sm_103` support changed B300 Diffusion from
12.11 step/s to 21.36 step/s, a 1.76x speedup. It closed about 73% of the H200
gap, but B300 remained 1.40x slower than H200 for Diffusion. ACT and SmolVLA did
not materially change under native `sm_103`, which supports the conclusion that
the penalty is specific to kernel-dispatch-heavy models.

### torch.compile Guidance

For Diffusion Policy, `torch.compile` improved H200 throughput by 29.3%, B300 by
5.6%, L40S by 20.4%, and RTX PRO 6000 by 12.8%. Enable it for H200 Diffusion
experiments when you are comparing optimized training loops. Do not expect it to
fix B300 Diffusion under stock PyTorch, because graph-level fusion does not
remove the `sm_103` PTX JIT path.

## Reproduction Workflow

There are two useful reproduction paths.

Use the VM or container path when you need benchmark-equivalent profiling. It
deploys a GPU workbench, caches datasets locally, and runs
`profile_train.py` through `npa workbench lerobot profile-train`. This is the
closest path to the research timing methodology.

Use the serverless Jobs path when you want a managed training job without
manually operating the VM. The current checkout includes
`npa workbench lerobot train --runtime serverless`. Older published releases may
require the VM or container path until the LeRobot Jobs work is included and
validated in that release.

### VM or Container Path

Deploy one workbench per GPU type. The examples below use H200; replace the GPU
platform and preset with the Nebius shape available in your project for B300,
L40S, or RTX PRO 6000.

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-h200 deploy \
  --project-id <NEBIUS_PROJECT_ID> \
  --tenant-id <NEBIUS_TENANT_ID> \
  --region <NEBIUS_REGION> \
  --runtime container \
  --gpu-type gpu-h200-sxm \
  --gpu-preset 1gpu-16vcpu-200gb \
  --disk-size 500
```

For B300 and RTX PRO 6000 VM deploys, use a CUDA 13 image family when your
project requires it:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-b300 deploy \
  --project-id <NEBIUS_PROJECT_ID> \
  --tenant-id <NEBIUS_TENANT_ID> \
  --region <NEBIUS_REGION> \
  --runtime container \
  --gpu-type gpu-b300-sxm \
  --gpu-preset 1gpu-24vcpu-346gb \
  --disk-size 500 \
  -v image_family=ubuntu24.04-cuda13.0
```

Run the benchmark-style profiler:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-h200 profile-train \
  --run vqbet:lerobot/pusht:100 \
  --run act:lerobot/pusht:100 \
  --run diffusion:lerobot/pusht:100 \
  --run smolvla:lerobot/aloha_sim_insertion_human:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0
```

`--num-workers 0` on `profile-train` means max CPUs. The authoritative metric is
`throughput_steps_per_sec` in `wallclock_results.json`, uploaded under the
profile artifacts when workbench storage is configured.

### Serverless Jobs Path

Use this path for managed training runs and checkpoint upload. It is not the
stage-level profiler path, but it is the forward-compatible reproduction path
for LeRobot training on Nebius Serverless Jobs.

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type diffusion \
  --dataset lerobot/pusht \
  --job-name diffusion-h200-100 \
  --steps 100 \
  --batch-size 8 \
  --gpu-type h200 \
  --gpu-count 1 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/diffusion-h200-100/
```

For B300:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type diffusion \
  --dataset lerobot/pusht \
  --job-name diffusion-b300-100 \
  --steps 100 \
  --batch-size 8 \
  --gpu-type b300 \
  --gpu-count 1 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/diffusion-b300-100/
```

The CLI warns when you run Diffusion Policy on B300 because of the PTX JIT issue.
If you omit `--output-path`, serverless training needs
`storage.checkpoint_bucket` in the selected project configuration.

## Per-Policy Recipes

### VQ-BeT (38M Parameters, Hybrid)

**What this policy does:** VQ-BeT uses a VQ-VAE to discretize actions into a
learned codebook, then predicts action tokens with a transformer decoder.

**Computational profile:** Mixed small operations: codebook lookups,
convolutions in the VQ-VAE path, and transformer attention.

**Recommended GPU:** H200 or B300. H200 measured 119.7 step/s and B300 measured
113.0 step/s, a 6% training margin on a single run. B300 measured 4.18 ms
single-sample inference latency, faster than H200's 11.58 ms.

**Benchmark-style profile:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run vqbet:lerobot/pusht:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0
```

**Serverless training:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type vqbet \
  --dataset lerobot/pusht \
  --job-name vqbet-h200-100 \
  --steps 100 \
  --batch-size 8 \
  --gpu-type h200 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/vqbet-h200-100/
```

**Validation:** Compare `throughput_steps_per_sec` against the research values:
H200 119.7, B300 113.0, RTX PRO 6000 102.8, L40S 90.4 step/s. Treat small
differences as directional unless you repeat the run.

### ACT (52M Parameters, Transformer)

**What this policy does:** ACT maps visual observations and robot state into a
chunk of future actions using a transformer encoder-decoder.

**Computational profile:** Attention-heavy, with relatively small matrix
dimensions. It is a lightweight imitation-learning baseline for LeRobot users.

**Recommended GPU:** H200 or B300. B300 measured 37.0 step/s and H200 measured
35.5 step/s; the 4% margin is not enough to claim a stable ranking from a single
run.

**Benchmark-style profile:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run act:lerobot/pusht:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0
```

**Serverless training:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type act \
  --dataset lerobot/pusht \
  --job-name act-h200-100 \
  --steps 100 \
  --batch-size 8 \
  --gpu-type h200 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/act-h200-100/
```

**Validation:** Expected research throughput is H200 35.5, B300 37.0,
RTX PRO 6000 30.2, and L40S 27.7 step/s.

### Diffusion Policy (263M Parameters, U-Net)

**What this policy does:** Diffusion Policy uses a conditional U-Net to denoise
action trajectories through repeated denoising steps.

**Computational profile:** Many small, heterogeneous convolution and attention
kernels. This is the most kernel-dispatch-intensive model in the benchmark.

**Recommended GPU:** H200. It measured 33.4 step/s, compared with B300 at
12.2 step/s, RTX PRO 6000 at 23.0 step/s, and L40S at 14.0 step/s.

**Caveat:** Avoid B300 with stock PyTorch for Diffusion unless availability,
memory, or a native `sm_103` PyTorch build changes the tradeoff. B300 Diffusion
inference also measured 36.48 ms versus H200 at 10.06 ms.

**Benchmark-style profile:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run diffusion:lerobot/pusht:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0
```

**With torch.compile on H200:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run diffusion:lerobot/pusht:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0 \
  --compile
```

**Serverless training:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type diffusion \
  --dataset lerobot/pusht \
  --job-name diffusion-h200-100 \
  --steps 100 \
  --batch-size 8 \
  --gpu-type h200 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/diffusion-h200-100/
```

**Validation:** This is the headline reproduction. H200 should be around
33.4 step/s in wall-clock profiler output. B300 with stock PyTorch should be
around 12.2 step/s. The H200/B300 ratio is the main signal; it should remain
large even if absolute throughput moves with software versions.

### SmolVLA (450M Parameters, VLM)

**What this policy does:** SmolVLA maps image observations and optional language
instructions to robot actions with a vision-language-action transformer.

**Computational profile:** Large, regular matrix operations in attention and
feed-forward layers. It dispatches fewer, larger kernels than Diffusion Policy.

**Recommended GPU:** H200 or B300. B300 measured 11.8 step/s and H200 measured
10.5 step/s; the 12% margin is directional from one run. B300's 270 GB VRAM is
useful when you increase batch size, sequence length, or model size beyond this
benchmark.

**Benchmark-style profile:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run smolvla:lerobot/aloha_sim_insertion_human:100 \
  --mode wallclock \
  --batch-size 8 \
  --num-workers 0
```

**Serverless training:**

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type smolvla \
  --dataset lerobot/aloha_sim_insertion_human \
  --job-name smolvla-b300-100 \
  --steps 100 \
  --batch-size 8 \
  --gpu-type b300 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/smolvla-b300-100/
```

**Validation:** Expected research throughput is H200 10.5, B300 11.8,
RTX PRO 6000 9.2, and L40S 7.1 step/s. The research num-workers sweep on B300
showed SmolVLA throughput roughly flat from 1 to 24 workers, which indicates the
run is GPU-bound after the dataset is warm.

## Profiling Methodology

Use `profile-train --mode wallclock` for cross-GPU comparisons. This path times
the measured training loop with CUDA events and a single synchronization after
the loop. Warmup steps are excluded. The metric to compare is
`throughput_steps_per_sec`.

Use `profile-train --mode profiler` when you need per-stage diagnostics:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run diffusion:lerobot/pusht:100 \
  --mode profiler \
  --batch-size 8 \
  --num-workers 0
```

Do not use profiler totals for cross-GPU ranking. `torch.profiler` inserts
synchronization barriers between labeled stages. The research found this can
overstate or understate cross-GPU comparisons by up to 40%, because different
GPU architectures benefit differently from kernel pipelining.

For inference latency, use:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n <WORKBENCH_NAME> profile-train \
  --run diffusion:lerobot/pusht:100 \
  --mode inference
```

Inference mode forces batch size 1. The research measured H200 Diffusion
inference at 10.06 ms and B300 Diffusion inference at 36.48 ms. For non-Diffusion
models, B300 and H200 inference were comparable except VQ-BeT, where B300
measured 4.18 ms and H200 measured 11.58 ms.

## Common Measurement Pitfalls

- Pre-cache datasets. Early cold-cache runs were inflated by up to 3x when
  dataset download happened inside the timed run.
- Prefer at least 100 measured training steps. At low step counts, setup time
  can hide real GPU differences.
- Confirm evaluation is disabled or moved outside the measurement window. One
  early run added an 18 second evaluation phase to a 4 second training segment,
  causing a 10x underestimate of actual throughput.
- Keep batch size consistent. The main comparison uses batch size 8.
- Record whether `--compile` was enabled. For Diffusion, H200 benefits much more
  from `torch.compile` than B300.

## Troubleshooting

### Job Stuck in `queued`

Check quota, regional capacity, and GPU type. Serverless Jobs may need an
explicit subnet if the project has multiple VPC subnets:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type act \
  --dataset lerobot/pusht \
  --job-name act-h200-queued-debug \
  --steps 100 \
  --batch-size 8 \
  --gpu-type h200 \
  --subnet-id <VPC_SUBNET_ID> \
  --submit-only \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/act-h200-queued-debug/
```

### Training Reaches `failed`

Common causes are a missing Hugging Face token, an output path that is not an
`s3://` URI, unavailable dataset access, or a GPU/image mismatch. Re-run with
text output and inspect the Job in Nebius:

```bash
npa workbench lerobot -p <PROJECT_ALIAS> -n lerobot-jobs train \
  --runtime serverless \
  --project-id <NEBIUS_PROJECT_ID> \
  --policy-type diffusion \
  --dataset lerobot/pusht \
  --job-name diffusion-h200-debug \
  --steps 100 \
  --batch-size 8 \
  --gpu-type h200 \
  --output-path s3://<YOUR_BUCKET>/lerobot-benchmarks/diffusion-h200-debug/
```

If the command returns a Job ID, fetch logs with the Nebius CLI available in
your environment. The exact log command can vary by Nebius CLI version, so use
the Jobs command help in your installed CLI.

### Throughput Does Not Match the Research

First compare ratios, not just absolute numbers. Software versions, image
families, and dataloader worker counts can move absolute step/s. Then check:

- Dataset is warm on local storage.
- `--mode wallclock` was used for throughput ranking.
- Batch size is 8.
- `--num-workers 0` was used on `profile-train` for max CPUs, or worker count
  was explicitly recorded.
- `--compile` setting matches the row you are comparing.
- B300 Diffusion is not running on stock PyTorch when you expect native
  `sm_103` behavior.

### B300 Diffusion Is Slower Than Expected

Confirm the PyTorch build. Stock PyTorch in the benchmark did not ship native
`sm_103` SASS, so B300 used PTX JIT from `sm_100` intermediate code. A native
build with `TORCH_CUDA_ARCH_LIST=10.3` changed B300 Diffusion from
12.11 step/s to 21.36 step/s in the research, but still did not catch H200.

### VM Deploy Works but Serverless Does Not

Use the VM/container profile path to reproduce benchmark numbers while
serverless Jobs issues are triaged. Serverless training and VM profiling use
different control planes: a Jobs failure does not invalidate the GPU throughput
result from the VM profiler.

## Current Status and Roadmap

The VM/container path is the benchmark-equivalent reproduction path today:
deploy a LeRobot workbench, run `profile-train --mode wallclock`, and compare
`throughput_steps_per_sec` to the research table.

This checkout also contains LeRobot training on Nebius Serverless Jobs through
`npa workbench lerobot train --runtime serverless`. When using a release that
does not yet include that work, use the VM/container path and update the command
to `--runtime serverless` after the LeRobot Jobs support is included in your
installed version.

Cosmos serverless Endpoints are a related serving workflow and demonstrate the
same serverless control-plane family, but they are not a substitute validation
for LeRobot training throughput.

## FAQ

**Can I reproduce this on a single GPU?**

Yes. All benchmark results in the research are single-GPU measurements.

**Do I need to build PyTorch from source?**

Only for B300 Diffusion if you need native `sm_103` behavior. The main published
stock-PyTorch result intentionally shows the PTX JIT penalty.

**Should I use `benchmark` or `profile-train`?**

Use `profile-train --mode wallclock` for the research throughput table. Use
`benchmark` for operational sweeps that collect system info and full training
process timings.

**Can I use my own dataset?**

Yes. Replace `--dataset` or the `POLICY:DATASET:STEPS` middle field with any
LeRobot-compatible Hugging Face dataset. For a dataset staged in object storage,
use `train --input-path s3://...` on the training path.

**What about multi-node training?**

Multi-node training is out of scope for this benchmark. The results here are
single-GPU architecture comparisons.

**Are the H200/B300 transformer rankings definitive?**

No. ACT, SmolVLA, and VQ-BeT have margins of 4% to 12% between H200 and B300 in
single-run measurements. Treat those as directional unless you run repeated
trials under your exact workload.

## Acknowledgments

- LeRobot and Hugging Face for the policy implementations and datasets.
- Nebius AI Cloud teams supporting the GPU environments and Workbench tooling.
- Benchmark presenters: @tle and @mnrozhkov.

## References

- Original LeRobot GPU benchmark research document, May 2026: publication URL
  pending.
- [npa quickstart](../../quickstart.md)
- [npa LeRobot CLI reference](../../cli/lerobot.md)
- [LeRobot project](https://github.com/huggingface/lerobot)
- [Hugging Face tokens](https://huggingface.co/settings/tokens)
