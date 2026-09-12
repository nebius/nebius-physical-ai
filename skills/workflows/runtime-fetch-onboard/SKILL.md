---
name: runtime-fetch-onboard
description: Use when legal, license, or gated-access restrictions prevent baking source, SDKs, weights, datasets, or assets into an NPA image but the operator may obtain them at runtime; design and validate a redistributable bootstrap container without treating runtime fetch as a license bypass.
---

# Runtime-Fetch Onboarding

Keep an onboarding moving when third-party bytes cannot be redistributed in a
container. Ship the redistributable launcher and fetch restricted components
directly from the upstream provider at runtime under the operator's own access
and applicable terms.

Runtime fetch is a delivery design, not a license bypass. It can solve a
redistribution restriction. It cannot create permission to use software, offer
it as a service, violate field-of-use terms, or consume restricted outputs in a
forbidden downstream stage.

## Load First

- `skills/atomic/solution-licensing/SKILL.md` to classify all six artifact
  boundaries.
- `skills/atomic/third-party-eula-preflight/SKILL.md` when any download or use
  depends on terms or gated access.
- `skills/workflows/byof-onboard/SKILL.md` for the NPA build/run mechanics.
- `skills/atomic/secure-image-build/SKILL.md` before building or publishing.

## Default Decision

Do not reject an otherwise viable solution merely because its weights, dataset,
assets, SDK, or source may not be baked into a distributable image. Choose the
narrowest compliant shape:

| Restricted boundary | Containerize | Fetch at runtime | Result |
| --- | --- | --- | --- |
| Weights/checkpoints only | Redistributable source, dependencies, and launcher | Exact weight revision/files | Candidate image can remain public if its built bytes pass every publication gate |
| Dataset or task assets | Code and loader | Exact dataset/assets with checksums | Data stays outside the image; its use and output obligations still apply |
| Source as well as weights | Neutral bootstrap/downloader only | Exact source revision plus exact weights | Viable only when the operator is permitted to fetch and run both |
| SDK or runtime | Neutral bootstrap, or build instructions only | SDK/runtime from the vendor, or build the derivative in the operator's registry | Public image is possible only when built layers contain no restricted vendor bytes |
| Base image | Dockerfile/build recipe | Operator pulls and builds in an operator-controlled registry | Build-your-own; do not publish the derived image unless redistribution is independently allowed |
| Outputs, service use, or field of use | Only the permitted capability | Nothing can package this away | Defer or reject the prohibited capability and record the human/vendor decision needed |
| No verified right to fetch or use | Nothing dependent on the artifact | Nothing | Block and escalate; a token, private registry, or runtime download does not supply permission |

If only one capability is restricted, defer that capability rather than
discarding unrelated capabilities that can be packaged and proven safely.

## Fast Onboarding Path

1. Copy and complete
   [the runtime-fetch contract](references/onboarding-contract.md). Record the
   source, baked runtime, weights, datasets/assets, cache, and outputs
   separately, with official terms links and exact immutable identities.
2. Select one of three shapes: weights/data runtime fetch, whole-source/SDK
   runtime fetch, or build-your-own into an operator-controlled registry.
3. Build from a digest-pinned redistributable base. Bake only the downloader,
   launcher, verified open dependencies, notices, and failure checks. Run as a
   non-root user and leave artifact/cache directories empty.
4. Fetch from the upstream provider on first run. Pin the repository revision
   and the expected file set or digest; download to a unique temporary path,
   verify it, then atomically publish a ready marker into the selected cache.
5. Pass tokens and secret acceptance material only through NPA runtime secret
   plumbing. Follow the documented product policy for non-secret acceptance
   controls: for example, the Isaac default and explicit opt-out may appear in
   profiles or YAML where the canonical Isaac policy requires it. Never put
   credentials, secret acceptance material, signed URLs, restricted bytes, or
   populated caches in Docker arguments, layers, image config, YAML, source
   metadata, logs, output manifests, or PR text.
6. Prefer upstream-verifiable entitlement such as a gated-repository token or
   vendor license key. A token normally proves access only; treat it as terms
   acceptance only when the provider's gate demonstrably records acceptance of
   the exact applicable terms. Do not require a credential for a genuinely
   public, anonymous artifact; still pin and verify its immutable identity. Do
   not invent a generic `ACCEPT_TERMS=YES` variable. Where repository product
   policy requires explicit operator acceptance, fail before network access
   until that exact scoped mechanism is satisfied.
7. Use the repository model-cache surface for durable reuse. Key cache identity
   by provider, artifact, revision/digest, and format. Gated or
   non-redistributable bytes require an operator-owned, access-restricted cache
   whose reuse policy is no broader than the upstream entitlement. Default to
   node-local ephemeral caching when durable-cache permission or isolation is
   not established. Mount a completed durable cache read-only in authorized
   consumers; never copy it into another image.
8. Keep planning and image validation useful without credentials. When the
   selected artifact requires credentials, its artifact-dependent runtime path
   fails early with a specific remediation and without provisioning or partial
   downloads. Anonymous artifacts may run without credentials, subject to any
   separate documented product-specific acceptance gate.

## Required Proof

Before calling the container or capability ready, prove all of the following:

- **Byte absence:** inspect the built image's files, layers, history, OCI
  configuration, SBOM, and caches. A Dockerfile review is not evidence that
  restricted payloads are absent.
- **Applicable negative gate:** exercise the refusal that the product policy
  actually defines: missing entitlement for a gated artifact, an explicit
  opt-out from a documented default-on policy such as Isaac, or missing exact
  opt-in for a product-specific policy such as OpenPI. A genuinely anonymous
  artifact has no missing-access gate. The exact guarded command refuses before
  download or model import, names which gate refused, and leaves the cache
  empty. Mutation-test independent gates that share an exit code so one refusal
  cannot mask another.
- **Positive fetch:** using operator-authorized access, the container downloads
  the exact immutable artifact directly from upstream, verifies identity and
  checksums, and records non-secret provenance.
- **Real capability:** the fetched artifact runs the documented workload on the
  compatible target hardware and produces the declared output artifact. Import,
  CUDA visibility, and a successful download are not capability evidence.
- **Cache behavior:** a restart either reuses the verified immutable cache or
  fails safely; concurrent population cannot expose a partial artifact.
- **No secret leakage:** scan the image, logs, workflow render, artifacts, and
  diff. Preserve terms links and provenance, never credentials or private
  infrastructure identifiers.

Classify `redistribution: public` only from the bytes actually shipped and the
verified rights for those bytes. Keep the image restricted or unvalidated when
that proof is missing. Stage the exact digest in an operator-controlled private
registry until byte-level inspection and real-GPU validation pass; an official
public development push is already publication. A private registry changes
access, not licensing.

When output terms reach a downstream workflow stage, encode a fail-closed gate
or omit that stage. A documentation note is sufficient only when no automated
downstream consumer exists.

## Deliverables

- Completed runtime-fetch contract and six-boundary license decision.
- Digest-pinned bootstrap image or build-your-own recipe.
- Runtime downloader/bootstrap with immutable identity and safe caching.
- Negative refusal tests, built-image absence scan, and positive real-workload
  evidence.
- NPA workflow YAML with runtime secret references, declared input/output
  artifacts, compatible GPU resources, and no embedded restricted bytes.
- Operator guide listing official terms/access steps, cache behavior, cleanup,
  accepted capabilities, and accurately deferred claims.

## Stop Conditions

Stop and escalate when official sources conflict, operator-specific facts decide
eligibility, the provider exposes no lawful runtime delivery path, service use
is restricted, output terms prohibit the intended next stage, or a byte-level
scan cannot prove the image clean. Continue every independent packaging,
documentation, planning, and test task that does not depend on that decision.

## Verify

```bash
npa/.venv/bin/python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/workflows/runtime-fetch-onboard
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_runtime_fetch_onboard_skill.py \
  npa/tests/guardrails/test_skills_index.py -q
```
