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
- FiftyOne: pin GraphQL Core 3.2.11, compatible with its Strawberry 0.316 dependency.
- Lyra: install the hash-pinned cryptography 50.0.2 wheel required by current NPA.
- OpenArm: refresh unavailable deadsnakes artifacts using exact package-index
  hashes, fixed userspace headers, and the October 9 Ubuntu snapshot.
- SONIC MuJoCo: refresh the Debian snapshot to include the fixed Perl packages.
- Diffusers, LingBot World, and SAM2: derive from the verified refreshed Wan
  digest rather than the older supported Wan runtime.
- Alpamayo: exclude host bytecode recursively, disable runtime bytecode creation,
  and remove dependency test/example data and UV caches before installation
  layers commit.

## Reviewed public content

The unchanged credential detector flags expressions such as
`hf_token = credentials.hf_token`, library parameter documentation, embedded
font bytes, Unicode name tables, and cryptographic self-test constants. These
matches cannot be accepted by filename or by excluding binary files.

[The review catalog](../../../npa/scripts/image-payload-content-reviews.json)
binds each reviewed member's complete SHA-256, byte count, finding kinds and
counts to its exact public repository or package artifact. GnuTLS key-body
prefixes were compared with official known-answer self-test source; the
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
