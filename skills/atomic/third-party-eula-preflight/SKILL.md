---
name: third-party-eula-preflight
description: Use before provisioning, building, downloading, or submitting a workload whose image, software, model, or data requires third-party EULAs or gated terms; apply the scoped Isaac ACCEPT_EULA default and explicit opt-out correctly without conflating credentials, privacy, telemetry, or redistribution rights.
---

# Third-Party EULA Preflight

Run this preflight before any costly or state-changing setup. NPA's Isaac product
policy defaults vendor acceptance on for non-interactive workflows while retaining an
explicit opt-out. For every other third-party agreement, use only the documented
upstream or product mechanism; do not invent a local opt-in.

## Operator-owned runtime fetch

For a public image that contains no gated or redistribution-restricted payload,
apply this reusable policy before asking for any additional approval:

1. Treat the operator as responsible for using their credential and fetched
   artifacts consistently with upstream terms.
2. For an exact gated Hugging Face or NGC artifact, require the operator's own
   credential and a successful usable-payload probe against the exact immutable
   revision. That probe is operationally sufficient for NPA to fetch the exact
   artifact; it is not legal acceptance or proof of compliance. Do not add an
   NPA terms checkbox or ask for another per-image attestation.
3. Preserve the exact operator statement once under one bounded manager
   task/run ID. If the operator states `noncommercial`, record exactly that and
   keep the intended activity separate. Compatible child solutions reference
   that record without another per-image question. It expires with the task/run
   and is never global or permanent. Reopen only for changed scope or a concrete
   additional fact required by exact authoritative terms.
4. Keep the result provider-, account-, artifact-, and revision-scoped. A token
   does not cover an unrelated provider or an anonymously delivered CUDA,
   cuDNN, SDK, dataset, or asset. For those, use the exact product policy,
   vendor acceptance mechanism, vendor-gated delivery, or operator-provided
   runtime.
5. Keep the public-image decision separate. A byte-scanned neutral image is
   eligible for public classification only after verifying redistribution
   rights for every baked byte and passing all `secure-image-build` publication
   gates. Runtime-fetch `Ready` grants no redistribution rights.

Concise operator-facing statement:

> You supply and control the upstream credential. By running this fetch, you
> accept responsibility for using the credential and fetched artifact under the
> provider's terms. NPA verifies access for this exact artifact and does not
> store the credential or include the artifact in the public image.

When authoritative terms expressly state that downloading or using constitutes
acceptance, do not duplicate that documented mechanism; NPA makes no conclusion
about enforceability. Do not manufacture output or service restrictions absent
from authoritative terms. Still stop when the terms require a separate
click-through, license key, paid/enterprise grant, or prohibit the declared use.

## Procedure

1. Inspect the selected image, runtime-fetch/bootstrap path, model and data
   sources, workflow renderer, and vendor documentation. Name every agreement
   that governs the exact components the workload will obtain or execute.
2. Record the vendor's official terms links and documented acceptance mechanism.
   Do not infer redistribution permission from credentials, registry access,
   prior use, or acceptance of a different agreement.
3. Apply the documented product policy. NPA defaults `ACCEPT_EULA=Y` only for
   `npa-isaac-lab`, Isaac-backed SONIC modes, and GR00T Isaac simulation. An
   absent variable succeeds non-interactively. The affirmative legacy spellings
   `Y`, `YES`, `1`, and `TRUE` are accepted case-insensitively. Empty, `N`, `NO`,
   `0`, and `FALSE`, including `--no-accept-eula`, opt out and must refuse before
   download. Any other value is invalid and must fail with an invalid-value error,
   not be described as an opt-out.
   Do not reuse this default for unrelated vendor, model, data, privacy,
   telemetry, or preview terms.
4. For runtime-fetched Hugging Face weights, do not add an NPA EULA/terms
   boolean, confirmation flag, empty placeholder, or model-check bypass. The
   operator's token and its actual upstream permissions are the only local gate
   for a gated repository. Token presence alone is not evidence. Probe every
   required repository before provisioning, using exact-revision payload bytes
   and never repository metadata; public repositories may pass anonymously. A
   successful exact probe is sufficient for NPA to proceed, but it neither
   accepts terms nor changes the artifact's redistribution classification.
5. For non-interactive Isaac execution, forward the default only to Isaac-backed
   tasks. If the operator opted out, fail before provisioning and print the exact
   resume command needed to re-enable acceptance.
6. Expose only `ACCEPT_EULA` to users. The pip-runtime launcher derives
   `OMNI_KIT_ACCEPT_EULA=YES` internally after validating the public value; do
   not add duplicate CLI, workflow, or configuration plumbing for it.
7. Never default optional privacy or telemetry consent. Keep `PRIVACY_CONSENT`
   and telemetry off by default; enable them only independently and explicitly,
   and do not derive either from EULA acceptance.
8. Keep acceptance UX separate from redistribution classification. A convenient
   default does not grant redistribution rights; the Isaac-family images remain
   public because their built layers contain no proprietary Isaac or Kit bytes.
9. Preserve a redacted run record containing agreement names, official links,
   the operator-stated scope, preflight result, and resume command. Do not store
   secret values or unnecessary personal data.

## Runtime-Fetched Model Reference

NPA must validate actual upstream access before provisioning for every gated
model that a deployment will fetch. For example, GR00T deploy validates both
the selected GR00T checkpoint and `nvidia/Cosmos-Reason2-2B`; Cosmos deploy
validates its selected checkpoint. Do not provide `--skip-model-check`, an
`ACCEPT_*` model variable, or a duplicate terms flag. A missing or unauthorized
Hugging Face token is an access failure from Hugging Face, not an NPA consent
workflow. Keep restricted weights out of image layers regardless of access.

## OpenPI pi0.5 Reference

OpenPI's public Polaris checkpoint is fetched anonymously from GCS, but it
contains Gemma-derived material governed by the Gemma Terms of Use and Gemma
Prohibited Use Policy. NPA's OpenPI product policy is explicit opt-in: require
the exact run-scoped `NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` value before an OpenPI
image build or checkpoint fetch. This is not a Hugging Face credential proxy or
an Isaac default. Forward it only as a runtime secret, never render it into YAML
or persist it in an image, repository file, credential store, checkpoint, or
dataset. A separate invalid-value workload must exit before importing the model
or starting checkpoint access, and must run before any accepted checkpoint
fetch. Acceptance covers only the two named Gemma policies.

## Isaac Reference

Isaac runtime-fetch workloads require all of these official terms:

- NVIDIA Omniverse Licence Agreement:
  `https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html`
- NVIDIA Isaac Sim Additional Software and Materials Licence:
  `https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html`
- NVIDIA Software Licence Agreement:
  `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/`

The repository mechanism defaults this value for Isaac-backed workloads:

```bash
ACCEPT_EULA=Y <resume-command>
```

An unset value becomes `Y`. Affirmative legacy values (`Y`, `YES`, `1`, `TRUE`)
normalize to `Y`; recognized negative values (empty, `N`, `NO`, `0`, `FALSE`)
make `isaac-bootstrap` exit `78` before downloading. Matching is
case-insensitive. Unrecognized values fail separately as invalid. The default
covers only the named NVIDIA terms; it does not enable unrelated privacy or
telemetry terms.

The bootstrap preserves the unset-versus-empty distinction with shell defaulting
equivalent to `${ACCEPT_EULA-Y}`. Do not replace it with `${ACCEPT_EULA:-Y}`;
that would turn an explicit empty opt-out back into acceptance.

## Guardrails

- Keep explicit opt-out refusal strict and early.
- Keep the OpenPI pi0.5 exact-value negative gate strict and runtime-only.
- For gated runtime weights, test every required upstream access probe before
  provisioning and test that no bypass or duplicate consent flag exists.
- Test both paths: unset acceptance defaults to `Y`, and explicit opt-out refuses
  before download/provision.
- Run `npa/tests/guardrails/test_third_party_eula_preflight_skill.py`,
  `npa/tests/guardrails/test_isaac_eula_plumbing.py`, and
  `npa/tests/docker/test_isaac_bootstrap.py` for Isaac-facing changes.
- Pair with `solution-licensing` when classifying what an artifact ships and
  whether it may be redistributed.
