# flex-pi

NPA packages [flex-pi](https://github.com/geyan21/flex-pi) as a single-GPU,
runtime-fetch workbench for real robot-policy inference. The default path uses
the released RoboTwin checkpoint in its action-only regime: three RGB cameras,
a 14D robot state, and a language instruction produce a 32-step bimanual action
chunk. It does not run a simulator and does not claim task success from one
observation.

## Architecture and target

Flex-pi is a 6B multi-stream world-action model. A Wan2.2-TI2V-5B video expert
and roughly 1B-parameter ActionDiT share a 30-layer mixture-of-transformers
backbone; DINOv3 supplies frozen semantic features and an optional pointmap
stream supplies geometry. One checkpoint selects observed and generated streams
at inference time.

The reference configuration observes RGB and DINO, omits pointmap, and disables
future video/DINO/pointmap denoising. It keeps the trained action horizon and
four Euler steps. Upstream reports the same 94.6% average RoboTwin success for
action-only and full-joint modes, while action-only avoids unnecessary stream
generation. One RTX PRO 6000 supplies the required Blackwell `sm_120` target and
ample memory above upstream's 16–26 GB inference range.

## Packaging and terms

`npa-flex-pi:0.1.0-cu128` contains pinned MIT flex-pi source, configs, the
RoboTwin policy adapter, and its CUDA/Python runtime. It contains no checkpoint,
Wan/T5/DINOv3 weights, observation media, credentials, actions, or populated
cache. A narrow maintained patch makes upstream inspect the lexical Hugging Face
snapshot path before resolving checkpoint symlinks into the blob store; this
ensures the released checkpoint's adjacent `config.yaml` defines the model
architecture. The upstream follow-up is intentionally kept in this integration
branch rather than a second pull request.

The checkpoint repository is MIT-labelled. The selected public RoboTwin dataset
card declares no license; NPA does not infer one from the simulator's MIT source,
does not redistribute its bytes, and fetches only the five hash-pinned objects
needed for the authorized observation. The ModelScope converted Wan VAE/T5
repository is Apache-2.0; DINOv3 retains its separate model license. Ancillary
assets are fetched from immutable revisions and verified by SHA-256 before model
construction. ModelScope exposes only its `master` branch to the SDK for this
repository, so NPA first requires that branch to resolve to the pinned Git
commit and then verifies both selected files by SHA-256. See the image's
`REDISTRIBUTION.md` and `THIRD_PARTY_NOTICES.md` for the maintained boundary.

## CLI and SDK

Plan locally without downloading model assets:

```bash
npa workbench flex-pi infer \
  --input-path "s3://${OPERATOR_BUCKET}/inputs/flex-pi-observation.json" \
  --output-path "s3://${OPERATOR_BUCKET}/plans/flex-pi/" \
  --dry-run
```

Run inside the GPU image with a local directory or S3 prefix:

```bash
npa workbench flex-pi infer \
  --input-path /opt/flex-pi/npa/public_robotwin_sample.json \
  --output-path "s3://${OPERATOR_BUCKET}/runs/${RUN_ID}/flex-pi/" \
  --expected-gpu RTXPRO6000
```

The SDK exposes the same implementation as
`npa.sdk.workbench.flex_pi.infer(...)`. For serving, set an owner-controlled
`NPA_FLEX_PI_TOKEN`, configure `NPA_FLEX_PI_OUTPUT_ROOT` to an authorized S3
prefix or local mode-0700 directory, and keep transport private or terminate
TLS. `/health`, `/status`, `/system-info`, `/list`, and `/run` all require bearer
authentication. HTTP callers cannot change the checkpoint or startup-snapshotted
input manifest.

## Workflow

Validate, plan, and submit the reference workflow:

```bash
npa workbench workflow validate-spec workflows/testing/flex-pi-rtxpro-inference.yaml
npa workbench workflow plan-spec workflows/testing/flex-pi-rtxpro-inference.yaml
npa workbench workflow submit workflows/testing/flex-pi-rtxpro-inference.yaml \
  --infra "$CONFIGURED_TARGET" --var "bucket=$OPERATOR_BUCKET" \
  --secret-env HF_TOKEN
```

The spec requests exactly one RTX PRO 6000 and routes
`workbench.flex_pi.infer` to the flex-pi image. It publishes only verified
artifacts to operator-owned object storage. The released checkpoint is public,
so `HF_TOKEN` is not an access gate; forwarding an operator read token through
the secret channel avoids anonymous multi-shard download throttling. Never put
the token in the workflow YAML or an artifact.

## Artifacts and acceptance

- `input.json`: immutable public-observation manifest, including object hashes;
- `actions.json`: 32×14 finite denormalized actions, real latency, peak allocated
  GPU memory, device/compute capability, and source/input/checkpoint provenance;
- `result.json`: request, image, model/data identity, metrics, and publication
  provenance under `npa.workbench.flex_pi.inference.v1`.

A passing run requires terminal job success, all three durable objects,
read-after-write verification, RTX PRO 6000 identity with compute capability
12.0, and exact source/checkpoint/data revisions. Raw logs and S3 locations stay
outside Git; reports and pull requests use only sanitized summaries.

## Build qualification

Build only from `npa/docker/workbench/flex-pi/Dockerfile` with the immutable
`dev-<full-git-sha>` tag. Before pushing, run dependency, vulnerability, secret,
license, and `scan_image_flex_pi_payload.py` gates over every layer. Resolve the
pushed digest anonymously and run the real workflow against that digest. Only
the exact digest that passed those gates may be promoted to the supported tag.

Cancel a live run before teardown and remove only resources created for that
run. Do not destroy shared clusters or buckets.
