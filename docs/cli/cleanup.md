# `npa cleanup`

## Command Tree

```text
Usage: npa cleanup [OPTIONS]

Report (or with --yes remove) local NPA/SkyPilot residue left after teardown.

Cloud resources (agent VM, cluster, bucket, IAM) are removed only by the
commands in the printed runbook. Existing `--yes` keeps credentials/config;
`--full` removes known local credentials/state and performs a read-only
storage-IAM verification. Neither scope deletes cloud resources. Full cleanup
exits 2 when IAM is present/unverified or provider verification fails.

Options
--yes  -y  Remove the local caches (otherwise just report). Local only: this never deletes cloud resources -- see the printed runbook for
    those.
--include-sky  --keep-sky  Include machine-shared ~/.sky in the audit. It is always preserved; project teardown removes only separately isolated, affirmatively
    owned state.
    [default: include-sky]
--full  Broaden --yes to also remove locally saved HF, Token Factory, and NGC credentials, validated NPA Terraform residue, and empty
    config/known ~/.npa state. Also read-only verifies recorded storage IAM; an unverified/present account makes cleanup partial (exit
    2).
--project  <str>  Scope per-alias state and the --full read-only storage-IAM check to this NPA project alias.
--skip-jobs  Do not query the SkyPilot managed-job queue.
--attest-no-active-jobs  With --skip-jobs, explicitly attest no active jobs after exact project terminal/no-submission evidence is verified.
--sky-bin  <str>  SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.
--json  Emit a machine-readable final cleanup result.
--list-receipts  List retained non-secret teardown audit receipts and exit.
--prune-receipts  Explicitly prune only terminal teardown receipts older than --receipt-retention-days; requires --yes.
--receipt-retention-days  <int range> [x>=0]  Minimum age for explicit terminal-receipt pruning (default: 90 days). [default: 90]
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--yes` | -y  Remove the local caches (otherwise just report). Local only: this never deletes cloud resources -- see the printed runbook for those. |
| `--include-sky` | --keep-sky  Include machine-shared ~/.sky in the audit. It is always preserved; project teardown removes only separately isolated, affirmatively owned state. [default: include-sky] |
| `--full` | Broaden --yes to also remove locally saved HF, Token Factory, and NGC credentials, validated NPA Terraform residue, and empty config/known ~/.npa state. Also read-only verifies recorded storage IAM; an unverified/present account makes cleanup partial (exit 2). |
| `--project` | <str>  Scope per-alias state and the --full read-only storage-IAM check to this NPA project alias. |
| `--skip-jobs` | Do not query the SkyPilot managed-job queue. |
| `--attest-no-active-jobs` | With --skip-jobs, explicitly attest no active jobs after exact project terminal/no-submission evidence is verified. |
| `--sky-bin` | <str>  SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution. |
| `--json` | Emit a machine-readable final cleanup result. |
| `--list-receipts` | List retained non-secret teardown audit receipts and exit. |
| `--prune-receipts` | Explicitly prune only terminal teardown receipts older than --receipt-retention-days; requires --yes. |
| `--receipt-retention-days` | <int range> [x>=0]  Minimum age for explicit terminal-receipt pruning (default: 90 days). [default: 90] |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa cleanup --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cleanup`.
