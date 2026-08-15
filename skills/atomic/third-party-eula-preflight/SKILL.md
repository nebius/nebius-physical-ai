---
name: third-party-eula-preflight
description: Use before provisioning, building, downloading, or submitting a workload whose image, software, model, or data requires a third-party EULA or gated terms; discover the exact agreements and acceptance mechanism, apply product acceptance defaults, and preserve explicit opt-out behavior.
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
3. Apply the documented product policy. Isaac workloads default `ACCEPT_EULA=Y`;
   setting it to an empty or non-Y value is an explicit opt-out. Do not reuse this
   default for unrelated vendor, model, data, privacy, telemetry, or preview terms.
4. When a product remains opt-in, ask clearly before provisioning, building,
   downloading, or submitting. Show the exact agreements, official links,
   mechanism, and the expensive action that is blocked.
5. For non-interactive Isaac execution, forward the default only to Isaac-backed
   tasks. If the operator opted out, fail before provisioning and print the exact
   resume command needed to re-enable acceptance.
6. Never default optional privacy or telemetry consent. Keep EULA defaults scoped
   to runtime orchestration rather than baking proprietary vendor bytes into images.
7. Preserve a redacted run record containing agreement names, official links,
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

An unset value becomes `Y`. An explicitly empty or non-Y value makes
`isaac-bootstrap` exit `78` before downloading. The default covers only the named
NVIDIA terms; it does not enable unrelated privacy or telemetry terms.

## Guardrails

- Keep explicit opt-out refusal strict and early.
- Test both paths: unset acceptance defaults to `Y`, and explicit opt-out refuses
  before download/provision.
- Run `npa/tests/guardrails/test_third_party_eula_preflight_skill.py`,
  `npa/tests/guardrails/test_isaac_eula_plumbing.py`, and
  `npa/tests/docker/test_isaac_bootstrap.py` for Isaac-facing changes.
- Pair with `solution-licensing` when classifying what an artifact ships and
  whether it may be redistributed.
