# FLUX 3 Action fine-tuning integration

Research date: 2026-09-23. Source pin: [`black-forest-labs/flux-action` at `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f`](https://github.com/black-forest-labs/flux-action/tree/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f).

## Decision and scope

Use the standalone **full fine-tuning** trainer with its native LeRobot indexer for configurable robot embodiments. BFL's fine-tuning guide explicitly supports new controls and tasks; the SO-101 LoRA route instead adapts an existing robot-specific checkpoint. The base is an adaptation component with fresh embodiment heads, not a ready robot policy. This integration should claim support for datasets satisfying the contract below, rather than every robot without preprocessing. [BFL fine-tuning guide](https://docs.bfl.ai/flux_3/flux3_action_finetuning), [base model card](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/main/README.md).

Recommended workbench boundary: local/materialized LeRobot dataset + explicit robot/training configuration → native index → native trainer → complete resumable checkpoint + BF16 inference export + provenance/report. A task-performance claim requires separate evaluation; successful training or offline error alone does not establish robot success. [SO-101 evaluation caveat](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/docs/so101-lora.md).

## Dataset and embodiment contract

The native indexer reads LeRobot v2.1/v3.0 metadata, parquet state/actions and video camera features. Configure `--camera STREAM=FEATURE` in physical layout order; training exposes each stream as `images.STREAM`. Preserve joint names, order, units, control rate and measured-state semantics. The index records dataset identity/revision, dimensions, camera mapping, frame counts and optional source hashes. It can reject/drop malformed episodes; report those counts. Its FPS handling casts to integer, so an integration should reject fractional metadata FPS rather than silently change timing. [Indexer](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/data/lerobot/index.py).

The trainer requires **state width = action width = `policy.action_dim`**. Equal width does not establish compatible semantics. No automatic projection, padding, unit conversion or resampling should be inferred. Custom modalities initialize their own heads while reusing the shared action trunk. [Trainer reconciliation](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/training/trainer.py), [head wiring](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/models/wiring.py).

For the ordinary single-observation profile, each window supplies `images.<camera>` as uint8 `(chunk_size + 1, 3, H, W)`, float32 `state (D,)`, float32 `action (chunk_size, D)`, task text and a deterministic window seed. Frame `s` conditions commands `s..s+chunk_size-1`; subsequent frames are their visual targets. The indexed loader also supplies the command before the window. LeRobot starts at frame 1, so it requires at least `chunk_size + 2` frames per episode, unlike the custom game adapter's `chunk_size + 1`. History profiles have additional requirements and should be configured explicitly. [Window loader](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/training/data.py).

Required explicit robot fields:

| Field | Contract |
| --- | --- |
| `action_dim`, `action_modality` | Positive width; modality is an unreserved identifier. |
| `camera_keys`, `camera_layout` | `single`: one; `side_by_side`: two; `grid`: any positive number; `droid`: three with its fixed geometry. |
| `canvas_hw` | Height then width, both multiples of 32. |
| `fps`, `chunk_size`, `n_action_steps` | Actual recorded rate; chunk divisible by four outside history profile; execution horizon no larger than chunk. |
| `action_parameterization` | `absolute` means values as recorded; `joint_delta` computes consecutive command differences. |
| `absolute_action_dims` | Channels retained as absolute under delta parameterization; empty for fully absolute actions. |
| `gripper_flip_dims` | Use `[]` unless the robot contract deliberately needs inversion; upstream defaults to last-channel inversion. |
| `single_frame_encode` | Set explicitly and retain in inference; source default is `true` despite stale prose in `embodiments.md`. |

[Policy configuration and validation](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/config.py).

## Normalization and training

Pass parameterization explicitly to indexing: its CLI defaults to `joint_delta` with the final dimension absolute, which is an SO-101 assumption. Keep `manifest.json`, `rows.f32.npy` and `statistics.json` together. Statistics use per-channel q01/q99 range normalization, including **both training and held-out episodes**. This upstream behavior must be disclosed for evaluation; a rigorously independent test set must remain outside the indexed corpus. [Indexer CLI](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/cli.py), [statistics implementation](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/data/lerobot/index.py).

The trainer adopts indexed bounds when unspecified and rejects inconsistent configured bounds, clipping, action parameterization or width. Preserve that native reconciliation. `steps` counts optimizer updates; global batch is ranks × windows per rank × accumulation. Worker count must be positive. Small corpora may need more `visits_per_epoch` to supply complete updates. Exact-position resume needs unchanged data/policy/loader topology and `reseed_on_resume=false`. Default schedule phases target a long DROID recipe and can leave a short run's trunk frozen; expose schedule settings instead of silently inventing a run budget. [Trainer](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/training/trainer.py).

## Checkpoints and exports

Only `step-N` directories containing `COMPLETE` are eligible for resume/export. Preserve distributed model/optimizer state, EMA profiles, training state, saved config and RNG for resume. Export after training with `flux-action export-checkpoint --checkpoint ... --output ... --profile model --dtype bfloat16`; `ema_0p10` and `ema_0p05` are alternatives when configured. Export defaults otherwise retain FP32 masters. Exporting runs on CPU and requires sufficient host memory; it is a real conversion, not a checkpoint copy. [Checkpoint implementation](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/training/checkpoint.py).

The inference package contains `config.json`, `model.safetensors` and `manifest.json`; the manifest hashes the first two and records the selected profile. Verify those hashes before reporting success. Normalization remains in the exported config. Frozen VAE/text encoders remain external references, not bundled or checksummed by upstream export: preserve immutable encoder identities and accessible paths when moving artifacts. Do not label the export self-contained or resumable. [Export writer](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/src/flux_action/policy.py).

## Runtime and licensing

The pinned reference environment uses Linux/NVIDIA, Python 3.12, CUDA 12.8, torch 2.10.0, torchvision 0.25.0, transformers 5.16.1 and NATTEN `0.21.6+torch2100cu128`; NATTEN is separate from `uv.lock`. Install `encoders` and `data` extras plus FFmpeg. Training memory differs substantially from inference; published GPU examples are not a verified minimum. Measure real training/export before claiming hardware support. [Setup](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/docs/setup.md), [dependency declaration](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/pyproject.toml).

Recommended packaging: public image containing Apache-2.0 source/runtime only; fetch weights at runtime or use operator-provided compatible paths. Preserve source LICENSE/NOTICE. The current model card declares FLUX Kommunity License v1.0 for weights and derivatives; commercial rights are conditional, and redistribution has license/attribution obligations. The unmodified Qwen encoder has its separate Apache-2.0 designation. Do not infer model licensing from the source license or bake weights into a public image. [Source LICENSE](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/LICENSE), [source NOTICE](https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/NOTICE), [model card at verified revision](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/eb267865d35e49e4066bde4936237f8d9f15a68c/README.md), [license at verified revision](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/eb267865d35e49e4066bde4936237f8d9f15a68c/LICENSE.md).

**License revision caveat.** The upstream recipe pins base revision `62878e2925e59b7a89ec14463ce89932624c490d`. That revision has no `LICENSE.md` or model-card license metadata; its README says repository access and usage restrictions apply. Current main was verified as `eb267865d35e49e4066bde4936237f8d9f15a68c`, which includes the license above. Official tree metadata shows identical action-base and VAE LFS hashes, and an identical text-encoder directory object, across these revisions:

| Artifact | Identity shared by both revisions |
| --- | --- |
| Action base SHA-256 | `eade59cbc25f2fdc8f5c0a5a8f7c7976cf10dc513105ec415aeccc21d3ad633f` |
| Video VAE SHA-256 | `db1c3211548371cf42c7230d52b5a99df5072f246d6bdfdc8fee99208ef58f01` |
| Text encoder Git directory object | `d09ba473c8d7b4d29f0b4b6584bab971dee5acd9` |

[Older revision tree](https://huggingface.co/api/models/black-forest-labs/flux-3-action-base/tree/62878e2925e59b7a89ec14463ce89932624c490d), [licensed revision tree](https://huggingface.co/api/models/black-forest-labs/flux-3-action-base/tree/eb267865d35e49e4066bde4936237f8d9f15a68c), [older README](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/62878e2925e59b7a89ec14463ce89932624c490d/README.md).

Recommendation: pin `eb267865d35e49e4066bde4936237f8d9f15a68c` for a snapshot containing explicit terms with unchanged model bytes. If retaining the older recipe pin, record the newer license source separately. Identical artifact identities support correspondence to the currently licensed release; no explicit retroactive licensing statement for the older snapshot was found, so do not claim that snapshot itself contains the license or infer unrestricted rights from its omission.

## Validation boundary

CPU tests can establish mapping, validation, command construction, config fidelity and artifact checks. Mocked execution cannot establish model loading, GPU memory suitability, finite real training losses or usable trained weights. Live acceptance needs a complete native checkpoint, finite metrics and a checksum-valid export from the intended hardware. Held-out closed-loop evaluation is a separate gate for robot performance.
