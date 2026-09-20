---
name: antioch-workbench
description: Use when connecting Antioch simulation to Workbench, configuring Antioch authentication, transferring recordings and checkpoints through Nebius S3, or operating the XR1 collection, fine-tuning, and evaluation pipeline.
---

# Antioch and Workbench

## Integration contract

Use the [XR1 cookbook](../../../docs/workbench/cookbooks/xr1-antioch.md) for the
executable procedure and pinned inputs. The operator module at
`npa/src/npa/workflows/xr1_antioch/operator.py` invokes the installed Antioch CLI
from an operator-owned Antioch project. Its commands are `assets`, `runtime`,
`source`, `collect`, `fetch`, `publish`, and `compare`. The `antioch` command
wraps native CLI arguments; `exec` launches attached remote argv through the
pinned SDK interpreter. Both resolve the NPA credential before launching.

The training stage is the `workflow.xr1.finetune` toolRef in
`workflows/partners/antioch/xr1-antioch-finetune.yaml`. It runs on Nebius and publishes
checkpoints to S3. Antioch collection, transfers, and closed-loop evaluation
are operator CLI/SDK steps around that YAML. Do not describe the entire loop as
one submitted workflow. The NPA CLI has no separate Antioch subcommand.

## Authentication

1. Inspect the installed CLI/SDK version and run the cookbook's
   `operator antioch --antioch-project <owned-directory> -- auth whoami --json`
   privately. This reports identity and credential source without printing the
   token, but organization, API, and user details still belong outside Git and
   public recordings. The tested XR1 adapter requires `antioch-sim==0.4.236`.
2. Store a personal access token as `tokens.ANTIOCH_TOKEN` in
   `~/.npa/credentials.yaml`, honoring `NPA_CONFIG_DIR` when configured. Use the
   existing atomic 0600 credential writer. A secret-injected `ANTIOCH_TOKEN`
   can be saved through `npa configure --save-env-credentials`; this imports all
   supported credentials present in the environment and reports names only.
   Never request the token in chat or put its value in command arguments,
   shell history, committed files, workflow parameters, or recordings.
3. Resolve environment `ANTIOCH_TOKEN` before the NPA file value. With neither,
   retain the native CLI browser-login fallback (`antioch auth login`). Browser
   access and refresh credentials stay in Antioch's own auth store; never copy
   them into the personal-access-token field. When a personal access token is
   supplied, native login, logout, and organization switching refuse.
   Managed Antioch workspaces use their managed credentials first. If the
   identity is wrong, resolve the intended account before operating; do not
   silently switch organizations or remove credentials.
4. Use `npa/src/npa/clients/antioch.py` to supply the resolved credential only
   to local Antioch client subprocesses. The saved token is excluded from the
   generic token map, process-wide credential export, and shared worker env.
   Bare Antioch commands do not read the NPA store; use the operator wrapper
   when that stored token is required. Never add it to training YAML or S3.
   Verify access to the intended Antioch organization and project, an owned
   `antioch.yaml`, a compatible engine, and available simulator capacity.
   A token authenticates the caller; it does not provision these prerequisites.
5. Configure Nebius and S3 separately through the selected NPA project. Follow
   `skills/atomic/health-preflight/SKILL.md` before provisioning or GPU submit,
   and the applicable model/software access procedure. An Antioch token does
   not grant Nebius access. Keep both providers' credentials on the operator;
   only the Nebius training worker needs the configured S3 credential delivery.

## Procedure and data contract

1. Use an isolated checkout and private operation directory. Confirm command
   help locally before creating sessions or submitting work:

   ```bash
   npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator --help
   npa/.venv/bin/python -m npa.cli.main workbench workflow validate-spec \
     workflows/partners/antioch/xr1-antioch-finetune.yaml
   ```

2. Seal disjoint training, validation, and test seeds before collecting data.
   Stage the pinned XR1 base model, processor, source, and split manifest in a
   new run-scoped Nebius S3 prefix. Preserve checksums and revision identities.
3. Initialize an owned Antioch project and use the cookbook's Isaac Sim 6.0.1
   engine and SDK 0.4.236. Preserve the project's generated ID. The measured
   simulator uses one RTX PRO 6000; the verified trainer uses eight Nebius RTX
   PRO 6000 GPUs. Prefer that tested configuration. B200 requires a separately
   built and verified runtime; it is not a drop-in target for the SM120 wheel.
4. Use `source` and `fetch` to bring approved inputs into Antioch, then collect
   physical demonstrations. Preserve synchronized RGB from all three cameras,
   state, issued actions, outcomes, split membership, and timestamps. Keep
   failed demonstrations as evidence and exclude them from behavior cloning.
5. Use `publish` to return demonstrations to Nebius S3. The transport passes
   signed GET/PUT requests over stdin to the Antioch worker and verifies hashes
   after complete downloads/readbacks. It does not copy static S3 keys into the
   simulator. Source archives contain allowlisted package Python files and the
   split manifest, excluding project authentication files.
6. Validate, plan, and submit the training YAML using the cookbook and
   `skills/atomic/submit-workflow/SKILL.md`. Training reads the S3 dataset,
   computes normalization from successful training episodes, fine-tunes the
   native XR1 action policy, and publishes model/optimizer state and metrics.
   Select the checkpoint using validation loss; keep test outcomes sealed.
7. Fetch the selected checkpoint and matching normalization back into Antioch.
   Evaluate base and trained policies on identical held-out scenarios, with
   physical success checks and all failures retained. Use `publish` and
   `compare` to return native videos, measured outcomes, and paired statistics
   to Nebius S3. Do not claim real-robot transfer from this simulated result.

## Long runs and evidence

The ordinary Antioch CLI execution deadline interrupted the measured evaluation.
For a full cohort, use the cookbook's `operator exec` command with the pinned
interpreter. It supplies NPA credentials to the attached SDK adapter at
`npa/src/npa/workflows/xr1_antioch/attached_exec.py`.
It removes the overall execution deadline while retaining cancellation and the
remote exit code. Do not silently upgrade the SDK: the adapter uses versioned
CLI internals. Resume only missing trials after verifying retained outputs and
unchanged inputs; preserve interrupted attempts separately.

Verify native video decoding, frame counts, simulation clocks, object sizes,
and complete SHA-256 readbacks. A manifest, terminal screenshot, or successful
upload alone does not establish robot learning. Report paired success counts,
the selected checkpoint, failures, and actual GPU types/counts. The measured
reference improved from 0/32 to 8/32; it is a narrow, low-success simulated task.
See [committed evidence](../../../docs/workbench/evidence/xr1-antioch-rtxpro.json).

Record actual transfers, execution, and native camera playback when requested.
Keep raw video paths outside Git. Distinguish live connection footage from
playback of durable results if the original session is no longer available.
Signed URLs are bearer credentials too: exclude them, raw auth output, and
private infrastructure identifiers from recordings, commits, and PR text.
Use `skills/atomic/protect-nebius-infra-details/SKILL.md` before publishing.

Publish and fully verify artifacts before releasing an Antioch session. Follow
`skills/atomic/teardown-and-cost/SKILL.md` to cancel jobs before destroying owned
Nebius compute. Retain the requested durable S3 evidence.

## Verify changes

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/clients/test_antioch.py \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_develop_skills.py -q
```

The client tests verify credential precedence, private persistence, isolation
from worker exports, CLI/SDK invocation, and collection/transfer delivery. The
registry verifies the operator/transport files and parses the training
workflow. For implementation changes, also run the XR1 tests under
`npa/tests/workflows/` and the applicable live validation; do not substitute
this documentation smoke for execution evidence.
