# LingBot-VA LIBERO-Long (quarantined candidate)

This is a source-only onboarding candidate for [Robbyant's LingBot-VA](https://github.com/Robbyant/lingbot-va), not an accepted NPA model, benchmark, image, or robot capability. The executable specification is [`workflows/testing/lingbot-va-libero-long.yaml`](../../workflows/testing/lingbot-va-libero-long.yaml). Its successful path has five connected substantive stages:

1. Select, reindex, and split actual operator-staged CC-BY LeRobot v2.1
   LIBERO-Long episodes; decode both embedded cameras to 10 Hz MP4 and use the
   native Wan2.2 VAE/text encoder to make fresh latents.
2. Continue the official LIBERO-Long derivative with upstream `wan_va.train`.
3. Serve and roll out the upstream websocket policy on the native ten-task
   LIBERO-Long (`libero_10`) suite.
4. Aggregate actual success JSON and emitted action-prediction validity metrics.
5. Produce a Rerun recording and a copied, decoded real rollout MP4.

The implementation deliberately invokes upstream training/server/client code; NPA only supplies runtime isolation, S3 handoff, telemetry removal, provenance, validation, and output checks. It does not use a RoboTwin task. RoboTwin remains outside this scope until a separately pinned, real proof exists.

## Exact pinned lineage

| Component | Immutable reference | Terms / treatment |
| --- | --- | --- |
| LingBot-VA source | [`Robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`](https://github.com/Robbyant/lingbot-va/tree/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb) | Apache-2.0; source-only image input |
| Base checkpoint | [`robbyant/lingbot-va-base@68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c`](https://huggingface.co/robbyant/lingbot-va-base) | Apache-2.0 card; runtime only |
| Official LIBERO-Long derivative | [`robbyant/lingbot-va-posttrain-libero-long@0e89d1e753019988aba484e8da2dc0810e264d9f`](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long) | Apache-2.0 card; runtime only and continuation origin |
| LIBERO source | [`Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) | MIT; runtime-only dependency |
| Raw LIBERO LeRobot data | [`HuggingFaceVLA/libero@affa19c0de0f6bce2a7edd26dddef8a532e7e6f6`](https://huggingface.co/datasets/HuggingFaceVLA/libero/tree/affa19c0de0f6bce2a7edd26dddef8a532e7e6f6) | CC-BY-4.0 card; runtime-staged raw input only |

The raw input is the HuggingFace VLA Team's LeRobot v2.1 conversion of the
original LIBERO data. The stage requires its exact metadata, selects task IDs
0–9 only after confirming their authoritative multi-step labels, and creates a
contiguous derivative because those source episodes are interleaved with the
other 30 tasks. The run manifest preserves every source episode ID, task ID,
source revision, action-statistics digest, and derived-tree inventory. It
decodes the embedded `image` and `image2` bytes and never reads or derives from
`robbyant/libero-long-lerobot`. CC-BY-4.0 attribution remains required for the
raw source and derived run artifacts; the raw data, derived MP4s, latents, and
outputs are not packaged in an image. No additional NPA EULA, acceptance
variable, checkbox, or telemetry consent is used.

The raw-source card asks users to cite the original LIBERO work: Liu, Zeng,
Patil, Mu, Xu, Liu, Wu, Liu, Tenenbaum, et al., “LIBERO: Benchmarking Knowledge
Transfer for Lifelong Robot Learning,” arXiv:2306.03310 (2023). That credit,
the linked card, and its CC-BY-4.0 declaration must travel with any derived
artifact provenance.

LingBot-VA credits Robbyant and inherits Wan-Video and Mixture-of-Transformers (MoT) components. The image's notices preserve this credit, the upstream citation (`Lin et al., arXiv:2601.21998, 2026`), source revision, model lineage, and NPA-only modifications. NPA does not relabel the underlying research as original work.

## Runtime and numerical contract

Upstream's documented requirements are Python 3.10.16, PyTorch 2.9.0/CUDA 12.6, `diffusers==0.36.0`, `transformers==4.55.2`, `lerobot==0.3.3`, and Flash Attention. This candidate therefore has an isolated runtime cache; it does not reuse NPA's CUDA 13 Wan 2.2 environment or newer LeRobot tool runtime. NPA retains the CUDA 12.6/PyTorch 2.9 ABI but applies security-maintenance pins (`torch==2.9.1+cu126`, `diffusers==0.38.0`, `transformers==5.10.0`, and their reviewed runtime closure) rather than knowingly shipping reported vulnerable versions. Native GPU validation of that modified closure is required before any capability claim. Its resolved `huggingface-hub==0.36.2` meets the updated Diffusers lower bound, and `wandb==0.20.1` is the lowest release compatible with LeRobot 0.3.3; the source did not pin either transitive dependency. LeRobot 0.3.3 declares Torch `<2.8` and Transformers `<4.52`, which conflicts with LingBot-VA's required Torch 2.9 and Transformers 4.55.2; the upstream instructions explicitly use `pip install lerobot==0.3.3 --no-deps`. This candidate follows that exact limited mechanism after installing its non-Torch data dependencies. The source imports Flash Attention even for the documented Flex/Torch modes; NPA softens only that unused import, so this candidate does not claim the optional `flashattn` mode.

The following values are hard-checked against upstream `va_libero_cfg.py` before preparation, training, and serving:

- Training attention: `flex`; inference/server attention: `torch` (never Flex evaluation).
- Window 30, four frames/chunk, 128×128 two-camera inputs, 30 action channels, four actions/frame.
- First seven action channels active; remaining 23 zero-padded.
- Quantile normalization with upstream `q01`/`q99`; video SNR shift 5.0 and action SNR shift 0.05.
- The official recipe's eight-GPU, 5,000-step `libero_train` configuration. The upstream shell's example W&B credentials are not used; W&B is explicitly disabled.

The evaluation stage reports real LIBERO-Long result JSON aggregation and numerical validity of model-emitted normalized action tensors. It explicitly does **not** call those validity values action accuracy or calibrated visual prediction quality: an aligned held-out action/video comparator must be added and independently inspected before such a claim. Smoke, full benchmark, convergence, and physical-robot success remain distinct.

## Current acceptance status

The checked-in image is a source-only candidate and has no public release
route. Its private-validation disposition remains fail-closed: an immutable
candidate digest, built-byte/security/license/SBOM scans, a native GPU
preparation run, and a real NPA/SkyPilot/Kubernetes execution are still
required before any live-ready claim. The RRD is bound to the
renderer-provided workflow run ID and carries sanitized source/provenance plus
its explicit prediction-quality limitation. After a run, independently
download and inspect the final evaluation manifest, RRD, and MP4 before
changing any status to accepted or live-ready. The hash-bound readiness record
names the pending qualification evidence.
