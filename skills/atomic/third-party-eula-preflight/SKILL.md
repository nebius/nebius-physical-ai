---
name: third-party-eula-preflight
description: Use before provisioning, building, downloading, or submitting a workload whose image, software, model, or data requires third-party EULAs or gated terms; apply the scoped Isaac ACCEPT_EULA default and explicit opt-out correctly without conflating credentials, privacy, telemetry, or redistribution rights.
---

# Third-Party EULA Preflight

Run this preflight before any costly or state-changing setup. NPA's Isaac product
policy defaults vendor acceptance on for non-interactive workflows while retaining an
explicit opt-out. Other third-party agreements remain opt-in unless their product policy
documents a default.

## Procedure

1. Inspect the selected image, runtime-fetch/bootstrap path, model and data
   sources, workflow renderer, and vendor documentation. Name every agreement
   that governs the exact components the workload will obtain or execute.
2. Record the vendor's official terms links and documented acceptance mechanism
   (for example, exact environment variables and accepted values). Do not infer
   acceptance from credentials, registry access, prior use, or acceptance of a
   different agreement.
3. Apply the documented product policy. NPA defaults `ACCEPT_EULA=Y` only for
   `npa-isaac-lab`, Isaac-backed SONIC modes, and GR00T Isaac simulation. An
   absent variable succeeds non-interactively. An explicitly empty or non-`Y`
   value, including `--no-accept-eula`, opts out and must refuse before download.
   Do not reuse this default for unrelated vendor, model, data, privacy,
   telemetry, or preview terms.
4. When a product remains opt-in, ask clearly before provisioning, building,
   downloading, or submitting. Show the exact agreements, official links,
   mechanism, and the expensive action that is blocked.
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

An unset value becomes `Y`. An explicitly empty or non-`Y` value makes
`isaac-bootstrap` exit `78` before downloading. The default covers only the named
NVIDIA terms; it does not enable unrelated privacy or telemetry terms.

The bootstrap preserves the unset-versus-empty distinction with shell defaulting
equivalent to `${ACCEPT_EULA-Y}`. Do not replace it with `${ACCEPT_EULA:-Y}`;
that would turn an explicit empty opt-out back into acceptance.

## Guardrails

- Keep explicit opt-out refusal strict and early.
- Test both paths: unset acceptance defaults to `Y`, and explicit opt-out refuses
  before download/provision.
- Run `npa/tests/guardrails/test_third_party_eula_preflight_skill.py`,
  `npa/tests/guardrails/test_isaac_eula_plumbing.py`, and
  `npa/tests/docker/test_isaac_bootstrap.py` for Isaac-facing changes.
- Pair with `solution-licensing` when classifying what an artifact ships and
  whether it may be redistributed.
