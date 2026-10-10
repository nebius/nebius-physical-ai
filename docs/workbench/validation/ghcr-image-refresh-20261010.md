# GHCR image refresh — October 10, 2026

This refresh covers the 18 supported images with confirmed source drift in the
[parity audit](ghcr-image-parity-20261010.md). The starting checkout is main commit
`348449e8752e6b3c10e62b072ce599646537f084`. MJLab is outside this batch.

The official publication workflow builds additive `dev-<full-git-sha>` tags,
checks image bytes before pushing, publishes provenance and SBOM attestations,
and verifies anonymous exact-digest readback. Supported release pins continue to
identify their previously qualified digests; new development bytes require
functional qualification before promotion.

## Build repairs

- Cosmos Transfer: pin the fixed PyJWT 2.15.1 wheel with its upstream hash.
- Envgen: select the October 9 Ubuntu snapshot and fixed userspace headers;
  retain the existing exact-source scikit-image recipe removal before flattening.
- FiftyOne: pin GraphQL Core 3.2.11, compatible with its Strawberry 0.316 dependency,
  install its multimodal extra with the upstream-pinned Protobuf runtime, and
  require all four environment checks including the LeRobot temporal API.
- Lyra: install the hash-pinned cryptography 50.0.2 wheel required by current NPA.
- OpenArm: refresh unavailable deadsnakes artifacts using exact package-index
  hashes, fixed userspace headers, and the October 9 Ubuntu snapshot; add the
  cryptography dependency required by current NPA image routing.
- SONIC MuJoCo and Cosmos3 serving: refresh the Debian snapshot to include the
  fixed Perl packages.
- Diffusers, LingBot World, and SAM2: derive from the verified refreshed Wan
  digest rather than the older supported Wan runtime.
- Alpamayo: exclude host bytecode recursively, disable runtime bytecode creation,
  and remove dependency test/example data and UV caches before installation
  layers commit; upgrade inherited Ubuntu packages from the October 9 snapshot.

## Reviewed public content

The unchanged credential detector flags expressions such as
`hf_token = credentials.hf_token`, library parameter documentation, embedded
font bytes, Unicode name tables, and cryptographic self-test constants. These
matches cannot be accepted by filename or by excluding binary files.

[The review catalog](../../../npa/scripts/image-payload-content-reviews.json)
binds each reviewed member's complete SHA-256, byte count, finding kinds and
counts to its exact public repository or package artifact. GnuTLS key-body
constants were decoded from adjacent C string literals and compared in full
with official known-answer self-test source; the
libunistring match was reconstructed from successive Unicode name tokens.
The font match occurs in Pillow's embedded base64 Aileron font. SDK and NPA
matches reference runtime inputs, type annotations, or documented placeholders.
No operational credential is authorized by these records.

Alpamayo and Cosmos3 serving retain their existing detector rules. On a match,
the disposition helper reads and hashes every remaining member byte, then
requires exact reviewed identity. Unknown content, a changed size or kind,
any modified byte, and an appended credential all remain failures. Conventional
credential paths, gated payload, image config/history, and the other mandatory
vulnerability, secret, license and publication gates remain enforced. Reports
retain the exact disposition receipts alongside unresolved findings.

## Published development images

All **18 of 18** selected images passed their official publication jobs and
were anonymously verified at `2026-10-10T18:00:42.763116+00:00`. All **36** digest-bound
SLSA provenance and SPDX 2.3 SBOM signatures independently verify against the
official publication workflow and each exact producer commit.

Canonical COPY/ADD inputs and recipe files match the refreshed branch at
`88ca5a94f94ed77ea5c33b8c8633cfd30afd5d2f` for every producer version below. This is source-input parity with
the frozen snapshot plus repairs; later main commits are outside this batch.

The 27 supported release tags were also rechecked anonymously and still match
their recorded digests. This refresh adds development bytes while preserving
those accepted release identities.

| Image | Producer commit | Published digest | Publication |
|---|---|---|---|
| `alpamayo2-super` | [88ca5a94f94e](https://github.com/nebius/nebius-physical-ai/commit/88ca5a94f94ed77ea5c33b8c8633cfd30afd5d2f) | `sha256:68ac9e35188bc89eda9d40f3ab8651568a41bfdf87d52a658d0d9b2d16f3388e` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38069511382/job/114264172598) |
| `content-agents` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:b95c70872e93b72f9d6e46b35fce19e9866ea3a748f0d005d1df2dbf6d6f677c` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467728) |
| `cosmos2-transfer` | [1fa22c19a191](https://github.com/nebius/nebius-physical-ai/commit/1fa22c19a19102b1df68110ab65a49d38478a388) | `sha256:50b914df30f68c4954f94e290ac553b4108922362e919fea710866d5a1ae3ab4` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062275093/job/114243083569) |
| `cosmos3-serving` | [57a47bc4600c](https://github.com/nebius/nebius-physical-ai/commit/57a47bc4600c3e52f15a17f3097914294456925e) | `sha256:b8eda0eb3457acc6d845dfd8f0432c5c8baa6ce9cfbce3c89ec9fa0d36f44ab8` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38065075422/job/114251302114) |
| `detection-training` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:0dde62cafe3652f531a54fdeb2860eaaecd8031113f8f67e25acb48e9bdef4d6` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467737) |
| `diffusers` | [b0e9dddb74d3](https://github.com/nebius/nebius-physical-ai/commit/b0e9dddb74d349c6e6d532d699f0a8e990171570) | `sha256:daabe1ec816eed2d3f4fd8697ef26f072eee1be0298d71d293199e9d21eb84b4` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062887743/job/114244858778) |
| `envgen` | [1fa22c19a191](https://github.com/nebius/nebius-physical-ai/commit/1fa22c19a19102b1df68110ab65a49d38478a388) | `sha256:d7f774ee20f1edae4f776a4b2af151bd88bdb8a2a3e12b28268791aa5f4556ab` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062275093/job/114243083619) |
| `fiftyone` | [010646731c10](https://github.com/nebius/nebius-physical-ai/commit/010646731c10e22e8161c77447274e62ff010c4a) | `sha256:5b947cef34be21601bdda732c7f50fb341d78759cfd85bffd7ec4970069e97a0` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38067878475/job/114259473414) |
| `flex-pi` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:0420399bf9c4df2f01f7486d13cd45643814c3ef1ddc31cf3083215a211b9c13` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467744) |
| `lingbot-world` | [b0e9dddb74d3](https://github.com/nebius/nebius-physical-ai/commit/b0e9dddb74d349c6e6d532d699f0a8e990171570) | `sha256:6a2a6ef5a0587b289f3b74521a2ddf7836887e2c4709bdc2910e0cd62cd18877` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062887743/job/114244858740) |
| `ltx2` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:aff04b7a68acf99bc6ad58d8cfdcac65d97f2c1350e3741ba6589cf86f7bda47` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467713) |
| `lyra2` | [1fa22c19a191](https://github.com/nebius/nebius-physical-ai/commit/1fa22c19a19102b1df68110ab65a49d38478a388) | `sha256:3e4fdbaf597046ba47fcce16c671495355da0384aa6a0922431e330a6a424c0b` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062275093/job/114243083596) |
| `openarm` | [57a47bc4600c](https://github.com/nebius/nebius-physical-ai/commit/57a47bc4600c3e52f15a17f3097914294456925e) | `sha256:4fb518acdacf0f649e081466a72695395eda03d3bb1d5a60ce4dd07819c4c47d` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38065075422/job/114251302156) |
| `rerun-viewer` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:d826fd7ccb6dee00fcaedafa54e99980cdac6cdc91c9c0a626029f57b7534ed3` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467699) |
| `sam2` | [b0e9dddb74d3](https://github.com/nebius/nebius-physical-ai/commit/b0e9dddb74d349c6e6d532d699f0a8e990171570) | `sha256:0eaf4beb6cfec4670d21766d03d4df6dd2943aa47f5ffbbb7de1bf3c6527e05f` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062887743/job/114244858794) |
| `sim2real-control` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:629e342deda13f2b04000fac6205b007fdc529543f6af9266d1008d2f094ae38` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467752) |
| `sonic-mujoco` | [1fa22c19a191](https://github.com/nebius/nebius-physical-ai/commit/1fa22c19a19102b1df68110ab65a49d38478a388) | `sha256:a2839731e873561fe08d71fc93f559f15f709665dabdf2969c1c9a25e81e6ade` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38062275093/job/114243083559) |
| `wan2-2` | [348449e8752e](https://github.com/nebius/nebius-physical-ai/commit/348449e8752e6b3c10e62b072ce599646537f084) | `sha256:4e5fe7932f66fab3ff036090f66d4a9d915e9eb6fab3ee2e7633ca1c68363cf0` | [Passed](https://github.com/nebius/nebius-physical-ai/actions/runs/38061728549/job/114241467688) |

The [machine-readable inventory](ghcr-image-refresh-20261010.json) contains
each full immutable `dev-<full-git-sha>` reference, digest, OCI revision, runtime
user, source-input comparison, and signature-verification receipt. All images
use `linux/amd64` and the non-root `ubuntu` runtime user.

FiftyOne additionally passed its real CPU bare-image, default NPA source
bootstrap, post-install, dataset/Brain, app, and cleanup qualification. The other
new digests have build and publication evidence; further functional qualification
is required before release promotion. The existing GPU coverage chart continues
to describe accepted release digests. MJLab was not built by this batch.

## Code validation

[Full Linux validation](https://github.com/nebius/nebius-physical-ai/actions/runs/38069425831/job/114267362563)
passed on the final build-repair commit `88ca5a94f94ed77ea5c33b8c8633cfd30afd5d2f`:
31 successful checks, including all six unit-test shards, browser and documentation
checks, source and dependency scanners, hostile-input regressions, native
complete-byte scan gates, and all eleven base-image vulnerability checks.
Three optional policy jobs were skipped.
