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
3. For either runtime-fetch shape, build from a digest-pinned redistributable
   base. Bake only the downloader, launcher, verified open dependencies,
   notices, and failure checks. Run as a non-root user and leave artifact/cache
   directories empty. For the build-your-own shape, instead provide a
   digest-pinned recipe: the operator obtains the restricted base or build
   input under its own authorization and builds directly into an
   operator-controlled private registry. Never publish that derived image
   unless its redistribution rights are independently verified.
4. For either runtime-fetch shape, fetch from the upstream provider on first
   run. Pin the repository revision and the expected file set or digest;
   download to a unique temporary path, verify it, then atomically publish a
   ready marker into the selected cache. For build-your-own, verify the exact
   restricted input and resulting private image during the trusted build; do
   not claim first-run fetch or copy build credentials into the image.
5. When the artifact's exact access or product-acceptance policy requires a
   token or secret acceptance material at runtime, pass it only through NPA
   runtime secret plumbing. When build-your-own requires credentials, pass
   them only through the trusted operator build's secret mechanism and ensure
   they are absent from the resulting image. Genuinely anonymous artifacts
   need no secret reference. Follow the documented product policy for
   non-secret acceptance controls: for example, the Isaac default and explicit
   opt-out may appear in profiles or YAML where the canonical Isaac policy
   requires it. For every shape, never put credentials, secret acceptance
   material, or signed URLs in Docker arguments, layers, image config, YAML,
   source metadata, logs, output manifests, or PR text. For either
   runtime-fetch shape, also keep restricted payload bytes and populated caches
   out of those surfaces. A build-your-own private image may contain its
   declared restricted inputs in layers, but never the credentials used to
   obtain them; record only non-secret identities and terms as provenance.
6. Prefer upstream-verifiable entitlement such as a gated-repository token or
   vendor license key. A token normally proves access only; treat it as terms
   acceptance only when the provider's gate demonstrably records acceptance of
   the exact applicable terms. Do not require a credential for a genuinely
   public, anonymous artifact; still pin and verify its immutable identity. Do
   not invent a generic `ACCEPT_TERMS=YES` variable. Where repository product
   policy requires explicit operator acceptance, fail before network access
   until that exact scoped mechanism is satisfied. For gated Hugging Face
   artifacts, the operator's token and the upstream repository permission are
   the access gate; do not add an NPA-side EULA or terms-acceptance boolean.
7. For either runtime-fetch shape, use the repository model-cache surface for
   durable reuse. Key cache identity by provider, artifact, revision/digest,
   and format. Gated or non-redistributable bytes require an operator-owned,
   access-restricted cache whose reuse policy is no broader than the upstream
   entitlement. Default to node-local ephemeral caching when durable-cache
   permission or isolation is not established. Mount a completed durable cache
   read-only in authorized consumers; never copy it into another image. A
   build-your-own image does not need a runtime downloader or runtime cache
   unless its declared capability independently fetches another artifact.
8. Keep planning and image validation useful without credentials. When the
   selected artifact requires credentials, its artifact-dependent runtime path
   fails early with a specific remediation and without provisioning or partial
   downloads. Anonymous artifacts may run without credentials, subject to any
   separate documented product-specific acceptance gate.

## Required Proof

Apply the common gates, then only the subsection for the selected packaging
shape. Do not report runtime-fetch gates as failed or passed for a
build-your-own image when they are not part of its declared delivery path.

### Every packaging shape

- **Real capability:** the exact image digest runs the documented workload on
  compatible target hardware and produces the declared output artifact.
  Import, CUDA visibility, a successful download, and build success are not
  capability evidence.
- **No secret leakage:** scan the image, logs, workflow render, artifacts, and
  diff. Preserve terms links and provenance, never credentials or private
  infrastructure identifiers.
- **Rights and containment:** record official rights for every baked, fetched,
  and output boundary. Keep any image containing restricted bytes in an
  operator-controlled private registry unless redistribution is independently
  permitted.

### Runtime-fetch shapes only

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
- **Cache behavior:** a restart either reuses the verified immutable cache or
  fails safely; concurrent population cannot expose a partial artifact.

### Build-your-own shape only

- **Restricted-input provenance:** record the exact digest or checksum of each
  operator-authorized restricted base, SDK, source archive, and other build
  input, together with its official terms and the trusted-build receipt.
- **Resulting-byte inventory:** inspect the resulting image's files, layers,
  history, OCI configuration, and SBOM. Identify restricted components and
  their provenance; do not claim their absence when the point of the private
  build is to include them.
- **Build-secret absence:** inspect layers, history, OCI configuration, files,
  logs, and SBOM for credentials and secret material used by the trusted build.
- **Private-registry containment:** resolve the resulting immutable digest,
  prove pullability only from the operator-controlled private registry, and
  record that no public tag or registry copy was created. Public promotion is a
  separate gate requiring verified redistribution rights and a new complete
  publication review.

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
- For a runtime-fetch shape: a downloader/bootstrap with immutable identity and
  safe caching, applicable refusal tests, built-image restricted-byte absence,
  positive fetch, and positive real-workload evidence.
- For build-your-own: exact restricted-input provenance, resulting-image
  byte/SBOM inventory, build-secret absence, private-registry containment, and
  positive real-workload evidence. Do not require a runtime downloader, empty
  runtime cache, or cache-reuse test unless runtime fetch is also declared.
- NPA workflow YAML with runtime secret references only when the artifact's
  exact access or product-acceptance policy requires them, declared
  input/output artifacts, compatible GPU resources, and no restricted payload
  bytes embedded in YAML. Build-only credentials never appear in the workflow.
- Operator guide listing official terms/access steps, cache behavior, cleanup,
  accepted capabilities, and accurately deferred claims.

## Stop Conditions

Stop and escalate when official sources conflict, operator-specific facts decide
eligibility, the provider exposes no lawful runtime delivery path, service use
is restricted, output terms prohibit the intended next stage, or a byte-level
scan cannot prove the runtime-fetch image free of restricted payloads or the
build-your-own image fully inventoried and free of build secrets. Continue
every independent packaging, documentation, planning, and test task that does
not depend on that decision.

## Verify

```bash
npa/.venv/bin/python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/workflows/runtime-fetch-onboard
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_runtime_fetch_onboard_skill.py \
  npa/tests/guardrails/test_skills_index.py -q
```
