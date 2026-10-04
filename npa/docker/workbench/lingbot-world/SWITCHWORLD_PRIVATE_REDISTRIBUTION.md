# Operator-private SwitchWorld LingBot runtime

This is an `unvalidated`, operator-private validation recipe. It is not an
approved public replacement for `npa-lingbot-world`, has no public release tag,
and its OCI label explicitly prohibits publication. Its exact built digest must
have byte, secret, SBOM, target-pull, and native-workflow gates independently
recorded before any capability claim; completing a private gate never grants
public-release eligibility.

The recipe retains the upstream Apache-2.0 LingBot World source at
[`a43bec7f8091c83e9b30b16b912f6fc906236fa6`](https://github.com/Robbyant/lingbot-world/tree/a43bec7f8091c83e9b30b16b912f6fc906236fa6)
by the Robbyant Team and the Apache-2.0 Wan 2.2 source at
[`42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc).
Their retained licenses, notices, source metadata, and NPA Wan redistribution
records remain in the image. NPA copies Wan's reviewed local `EasyDict`
compatibility module and redirects the two LingBot configuration imports to it;
this avoids shipping `easydict==1.13`. The recipe also applies the current Wan
recipe's source-local attention import fallback in `wan/modules/model.py`: the
existing torch attention implementation is used at the model's
`flash_attention` call site instead of fetching or building FlashAttention.
This is a compatibility remediation after the actual native baseline asserted
that FlashAttention 2 was unavailable; it is not a performance-equivalence
claim. The modified files and original licenses remain inspectable in the
image.

SwitchWorld code and the two adapter weights are not in this image. The
workflow source-checks SwitchWorld at its pinned revision, fetches the selected
PencilHu adapter files with immutable revision/size/SHA-256 checks, and keeps
both adapters, model checkpoints, CUDA Python packages, user media, native case
tensors, credentials, runtime caches, and generated outputs outside image
layers. Their upstream terms and the operator's authorization remain separate.

The system package layer uses Debian's fixed `20260906T183022Z` snapshot so
the scanner can bind parser-like literals to the independently audited package
bytes rather than broadly excluding their paths. The image runs as `ubuntu`;
passwordless sudo is solely the existing SkyPilot bootstrap prerequisite and
does not start `sshd` by default. No EULA acceptance, credential, telemetry
consent, or private infrastructure coordinate is baked.
