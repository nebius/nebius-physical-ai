---
name: guardrail-failures
description: Use when a guardrail test or CI gate in npa fails and you need to map the failure to its cause and fix rather than reverse-engineering the assertion.
---

# Decoding A Guardrail Failure

The guardrail suite is ~2000 assertions over 43 files and runs in under a
minute. It is static: it reads the repo and checks that surfaces which must
agree still agree. A failure almost always means one of two things — you changed
one side of a contract and not the other, or you added something the contract
requires you to register.

Run the whole suite; it is cheap enough that narrowing is rarely worth it:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails -q
```

Narrow once you know the file:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_tool_catalog_argv.py -q
```

Most of these tests carry a module docstring explaining the incident that caused
them to exist. When the fix is not obvious, read it — it usually names the
production failure the check prevents.

## Triage Order

1. Read the assertion message. These tests are written to name the fix, not just
   report a mismatch.
2. Identify which side of the contract you changed. The guard is almost never
   wrong; it is reporting that a second file needs the same edit.
3. Only weaken a pinned exemption list if the exemption is genuinely obsolete.
   Several lists are explicitly designed to shrink and never grow.

## CLI, Catalog, Spec, And SDK Coherence

These fire when a tool's surfaces drift apart. This is the group a workbench
change hits most.

| Guardrail | Fix when it fails |
|---|---|
| `test_tool_catalog_argv` | A `toolRef` argv names a CLI option that does not exist, or passes a value the option cannot mean. Correct the argv in `catalog.py` — see `skills/atomic/toolref-argv-contract/SKILL.md`. |
| `test_module_toolref_argv` | A `python -m` toolRef no longer parses against its module's own argparse. Sync the module parser and the catalog argv. |
| `test_three_tier_contract` | A new CLI group has neither a `CapabilityContract` nor a seam entry, or a CLI parameter is unreachable from a spec without a stated reason. Add the contract or record the `spec_gap` category. |
| `test_spec_declared_outputs` | A spec's declared `outputs:` path is not what the tool writes. Fix the spec, or the tool's `result_uri_for()` helper. |
| `test_spec_paths_are_not_repo_relative` | A stage argv points at a path that exists in this checkout. Stages run in a pod; use an `s3://` URI or an in-image absolute path. |
| `test_shown_workflow_catalog` | A catalog entry no shipped spec reaches, or a spec outside the npa.workflow catalog. Add a reference spec or list the entry in `PUBLIC_REUSABLE_TOOLREFS`. |
| `test_workflow_schema_chaining` | A state depends on another but reads a schema that one does not produce. Align the producing and consuming stages. |
| `test_typer_command_calls` | A Typer command is called as a plain function without `@resolve_typer_defaults`. Add the decorator to the callee. |
| `test_internal_cli_entrypoint` | An internal argv no longer matches `python -m npa`. Build it with `internal_cli_argv()`. |
| `test_help_to_markdown` | The generated-docs converter mishandles wrapped Rich help. Fix `scripts/_help_to_markdown.py`. |

## Skills And Documentation

| Guardrail | Fix when it fails |
|---|---|
| `test_skills_index` | A `SKILL.md` exists without an `skills/index.yaml` entry (or the reverse), frontmatter `name` disagrees with the index, or a smoke does not run. Register the skill with a passing smoke. |
| `test_develop_skills` | A development skill names a repo path or guardrail that does not exist, or a new guardrail file is undocumented. Update the skill. |
| `test_antioch_skill` | Antioch references are undiscoverable, its machine-readable readiness/control/security contract drifted from the rendered two-GPU stack, or public payload and exact-cleanup invariants weakened. Update the skill resources and implementation together. |
| `test_no_dangling_workflow_references` | A doc, skill, or script names a workflow YAML that does not exist. Repoint or remove the reference. |
| `test_docs_green_path` | The documented path from README to a real submit is no longer runnable end to end. Update the quickstart with the new step. |
| `test_audit_container_docs_skill` | The public container catalog disagrees with the publish inventory. Update `docs/workbench/container-image-catalog.md`. |
| `test_solution_licensing_skill` | The licensing skill no longer covers an artifact boundary. Update the skill, not the test. |
| `test_third_party_eula_preflight_skill` | The EULA preflight skill is not discoverable from the operational skills that need it. Add the link. |
| `test_nebius_cli_compatibility` | The `nebius-cli` version drifted between packaging, `images.py`, and docs. Bump all of them together. |
| `test_paidf_image_tags_match_code` | The PAIDF guide builds tags that differ from what submit pulls. Regenerate the guide's build commands from `npa/src/npa/deploy/images.py`. |
| `test_default_cluster_fits_quickstart` | The default cluster can no longer schedule the documented quickstart. Raise the preset or lower the spec's requests. |

## Secrets, Confidentiality, And Consent

Treat every failure here as blocking. Do not add an exemption to make one pass.

| Guardrail | Fix when it fails |
|---|---|
| `test_confidentiality_scan` | The scanner, its built-in Nebius patterns, or the gitleaks wiring changed. Keep `.gitleaks.toml` and `npa/src/npa/guardrails/confidentiality.py` in sync. |
| `test_agent_secret_guard` | A secret path became tracked, `.gitignore` stopped covering agent/cursor secrets, or a literal secret or live IP landed in agent files. Remove it and rotate. |
| `test_agent_no_hardcoded_data` | Agent or insights source embeds run names, answers, or infra endpoints. Resolve them from live tool observations instead. |
| `test_access_key_list_safety` | Docs or code request secret-bearing access-key list JSON. Use a `--format jsonpath=...` projection. |
| `test_isaac_eula_plumbing` | Isaac consent uses something other than the single run-scoped mechanism. Use the scoped `ACCEPT_EULA` plumbing. |
| `test_live_tests_never_declare_a_licence` | A live test accepts vendor terms for the operator. Gate on an operator-provided variable instead. |
| `test_operator_env_covers_gated_solutions` | A solution whose smoke hits a vendor gate has no operator-env entry. Add it. |
| `test_hygiene_guards` | Several rules: no `sky launch --down/--autodown`, GPU tests skip on explicit env flags only, examples use a registry placeholder rather than a first-party ID, monolith modules must not grow, and no silent `except Exception: pass`. The message names which one. |

## Images And Packaging

| Guardrail | Fix when it fails |
|---|---|
| `test_workbench_image_k8s_prereqs` | An image lacks what SkyPilot needs on Kubernetes (python3, rsync, sudo with NOPASSWD, or the Isaac group/PATH rules). Update the Dockerfile and the shared install script together. |
| `test_unbuilt_image_records_agree` | The four files that record whether an image is built disagree. Make them agree; do not mark an image built that is not. |
| `test_trivy_policy` | `trivy.yaml` no longer matches the current nested schema. Update the config. |
| `test_public_mirror_workflows` | The public-mirror workflows stopped sharing one credential path, or the health check gained write behavior. Route through `npa/scripts/ci_source_registry_login.sh`. |
| `test_paidf_starter_asset` | The PAIDF starter asset was bundled instead of runtime-fetched, or its contract drifted. Keep the runtime-fetch model. |

## Retired Surfaces And Example Boundaries

These exist because a retired thing keeps coming back. The fix is always to use
the current surface, never to recreate the old one.

| Guardrail | Fix when it fails |
|---|---|
| `test_skypilot_catalog_retirement` | `npa/src/npa/workflows/skypilot/` reappeared, or a spec carries retired `skypilotTwin` metadata. Author an npa.workflow spec instead. |
| `test_skypilot_readme` | The retired catalog README came back. Delete it. |
| `test_burst_examples` | A burst example became multi-stage, lost its `${VAR}` placeholders, or unpinned its image. Multi-stage work belongs in an npa.workflow spec. |
| `test_byof_profiles` | A BYOF profile drifted from being a single-task resource profile. Same boundary as burst. |
| `test_nurec_examples` | The NuRec single-pod example drifted toward being a catalog entry. Keep it an example. |
| `test_workflow_image_check` | Image extraction, placeholder resolution, or the retired-catalog check broke. Use the fixtures for raw SkyPilot image tests. |

## Infrastructure, CI, And Meta

| Guardrail | Fix when it fails |
|---|---|
| `test_ci_workflows` | A workflow lacks the shared concurrency template, duplicates feature-branch runs, or makes mypy blocking. Match the existing template. |
| `test_e2e_gate_reachability` | A new `NPA_*` e2e gate has no runner mapping. Wire it into `scripts/dev-vm-daily-tests.sh` or record a reviewed manual reason. |
| `test_terraform_provisioner_shell` | Bash embedded in the agent Terraform is not syntactically valid, or an SSH wait is unbounded. Check the heredocs. |
| `test_no_syntax_warnings` | The package emits a `SyntaxWarning`, usually an invalid escape in a regex. Use a raw string. |
| `test_collection_guard` | The zero-collection guard itself broke. Something made pytest collect nothing. |

## When The Gate Is Not A Guardrail Test

`Lint / docs-drift` is not in this suite. It fails when a CLI option changed and
`docs/cli/` was not regenerated; fix it with `bash scripts/build_docs.sh`. The
full mapping from CI job to local command is in
`skills/atomic/pre-pr-validation/SKILL.md`.
