---
name: third-party-eula-preflight
description: Use before provisioning, building, downloading, or submitting a workload whose image, software, model, or data requires a third-party EULA or gated terms; discover the exact agreements and acceptance mechanism, verify scoped operator consent, and fail fast before expensive work when consent is absent.
---

# Third-Party EULA Preflight

Run this preflight before any costly or state-changing setup. Consent is an
operator decision, never a default supplied by NPA or an agent.

## Procedure

1. Inspect the selected image, runtime-fetch/bootstrap path, model and data
   sources, workflow renderer, and vendor documentation. Name every agreement
   that governs the exact components the workload will obtain or execute.
2. Record the vendor's official terms links and documented acceptance mechanism
   (for example, exact environment variables and accepted values). Do not infer
   acceptance from credentials, registry access, prior use, or acceptance of a
   different agreement.
3. Search the current conversation and run record for an explicit operator
   statement accepting each named agreement. Treat consent as valid only for the
   named agreements, workload/run scope, and mechanism stated. Do not reuse it
   for unrelated vendor, model, data, privacy, telemetry, or preview terms.
4. If any acceptance is absent, ask clearly before provisioning, building,
   downloading, or submitting. Show the exact agreements, official links,
   mechanism, and the expensive action that is blocked. Never precheck a box,
   imply consent, or accept on the operator's behalf.
5. For non-interactive or detached execution, fail before provisioning when
   consent is absent. Exit non-zero and print: the missing agreement names, the
   exact acceptance needed, confirmation that no expensive action began, and an
   exact resume command with acceptance values supplied at launch.
6. When consent is present, forward only its documented acceptance values to
   the authorized workload. Keep acceptance unset in repository defaults,
   images, specs, and persistent global configuration.
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

The repository mechanism requires both caller-supplied values:

```bash
OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES <resume-command>
```

Absent either value, `isaac-bootstrap` must exit `78` before downloading. The
values accept only the named NVIDIA terms; they do not accept unrelated terms.

## Guardrails

- Keep the legal/runtime refusal strict and early; do not weaken it to make a
  smoke test pass.
- Test both paths: missing consent refuses before download/provision, and
  explicit caller consent is forwarded without being invented or persisted.
- Run `npa/tests/guardrails/test_third_party_eula_preflight_skill.py`,
  `npa/tests/guardrails/test_isaac_eula_plumbing.py`, and
  `npa/tests/docker/test_isaac_bootstrap.py` for Isaac-facing changes.
- Pair with `solution-licensing` when classifying what an artifact ships and
  whether it may be redistributed.
