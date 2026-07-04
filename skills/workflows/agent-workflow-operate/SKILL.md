---
name: agent-workflow-operate
description: Use when verifying that a bootstrapped NPA agent VM can create, validate, plan, provision for, or run npa.workflow YAMLs.
---

# Agent Workflow Operation

## Operating Model

- **Dev/operator VM** is the source of truth for NPA lifecycle, auth, SSH keys,
  and agent records. Run `npa agent fresh-setup`, `bootstrap`, `status`, and
  `verify-live` there.
- **NPA agent VM** is the autonomous workbench execution surface. After
  bootstrap, it must have staged `~/.npa/config.yaml`, `~/.npa/credentials.yaml`,
  Nebius CLI/profile or token material, and the NPA checkout needed to validate,
  plan, provision, and run workflow YAMLs.
- Do not use the dev VM full pytest suite as the acceptance test for agent
  autonomy. The relevant check is whether the agent VM can operate workflows
  with its bootstrapped config.

## Test Framework

Run the framework by SSHing from the dev VM into the active agent VM using the
agent record SSH key. Do not print credential files or secret values.

1. On the dev VM, resolve the active agent:

   ```bash
   npa/.venv/bin/npa agent status --project <project> --name <name> --json
   ```

2. From the dev VM, SSH to the agent public IP with the recorded key and create
   a temporary `apiVersion: npa.workflow/v0.0.1` YAML. Use a known catalog
   `toolRef` such as `workbench.vlm_eval.run` and include all config tokens
   required by the tool template, including `vlm_backend`.

3. On the agent VM or through the agent API, validate and plan:

   ```bash
   npa/.venv/bin/npa workbench workflow validate-spec /tmp/spec.yaml --json
   npa/.venv/bin/npa workbench workflow plan-spec /tmp/spec.yaml --run-id agent-smoke --json
   npa/.venv/bin/npa workbench workflow run-spec /tmp/spec.yaml \
     --run-id agent-smoke --plan-only --scheduler-plan --json
   ```

   Equivalent API checks:

   ```text
   GET  /api/infra/k8s
   POST /api/infra/provision
   POST /api/workflows/validate
   POST /api/workflows/plan
   POST /api/workflows/submit
   ```

4. Check K8s provisioning readiness from the agent VM:

   ```bash
   npa/.venv/bin/npa provision-if-absent --project <project> --dry-run \
     --output-format json
   ```

5. If the operator explicitly requests live execution, let the agent use
   `POST /api/workflows/submit` or `POST /api/infra/provision` so NPA provisions
   Kubernetes as needed through normal NPA commands. Do not manually create,
   delete, or mutate cloud resources outside NPA.

## Pass Criteria

- Agent VM SSH works from the dev VM with the recorded key.
- Agent VM has non-empty bootstrapped `~/.npa/config.yaml` and
  `~/.npa/credentials.yaml`; do not print contents.
- `validate-spec` returns JSON with `status: valid`.
- `plan-spec` returns at least one planned step with the expected `tool_ref`.
- `run-spec --plan-only --scheduler-plan` returns scheduler output without
  requiring real workload execution.
- `provision-if-absent --dry-run --output-format json` resolves the project and
  reports intended or existing K8s/storage actions without credential errors.
- `GET /api/infra/k8s` reports configured and locally cached Kubernetes
  backends. If none are present, chat should say no infra is specified and offer
  options: deploy minimal infra with the agent, configure an existing backend, or
  submit with explicit project/cluster target.
- `POST /api/workflows/submit` validates YAML, provisions minimal Kubernetes
  when allowed and needed, and records a scheduler plan/run record.

