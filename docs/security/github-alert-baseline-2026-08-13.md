# GitHub security-alert baseline: 2026-08-13

Authenticated GitHub REST API enumeration for `nebius/nebius-physical-ai`
found 21 open Dependabot alerts and zero open code-scanning alerts. Repository
security configuration reports secret scanning as disabled, and the
secret-scanning alert endpoint is consequently unavailable. The
GitHub-reported count of 21 is therefore exactly the open Dependabot alert
count. The authenticated REST requests used `state=open&per_page=100`; all 21
records report `pip`, `runtime`, and `direct` for ecosystem, scope, and
relationship respectively.

| Alert | Package | Advisory / CVE | Vulnerable range → first patched | Manifest path | Branch resolution |
| ---: | --- | --- | --- | --- | --- |
| 74 | transformers | GHSA-fgcw-684q-jj6r / CVE-2026-5241 | `<5.5.0` → `5.5.0` | `npa/docker/workbench/common/sim2real-genesis-requirements.txt` | Remove the unused distribution; all three consuming images uninstall the vulnerable base copy and assert it is absent. |
| 73 | transformers | GHSA-29pf-2h5f-8g72 / CVE-2026-4372 | `<5.3.0` → `5.3.0` | `npa/docker/workbench/common/sim2real-genesis-requirements.txt` | Remove the unused distribution; all three consuming images uninstall the vulnerable base copy and assert it is absent. |
| 72 | transformers | GHSA-69w3-r845-3855 / CVE-2026-1839 | `<5.0.0rc3` → `5.0.0rc3` | `npa/docker/workbench/common/sim2real-genesis-requirements.txt` | Remove the unused distribution; all three consuming images uninstall the vulnerable base copy and assert it is absent. |
| 71 | pillow | GHSA-9hw9-ch79-4vh6 / CVE-2026-59205 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 70 | pillow | GHSA-vjc4-5qp5-m44j / CVE-2026-59204 | `>=8.2.0,<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 69 | pillow | GHSA-pg7v-jwj7-p798 / CVE-2026-59203 | `>=12.0.0,<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 68 | Pillow | GHSA-jjj6-mw9f-p565 / CVE-2026-59200 | `>=5.1.0,<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 67 | Pillow | GHSA-6r8x-57c9-28j4 / CVE-2026-59199 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 66 | Pillow | GHSA-fj7v-r99m-22gq / CVE-2026-59198 | `>=5.2.0,<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 65 | Pillow | GHSA-xj96-63gp-2gmr / CVE-2026-59197 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 64 | Pillow | GHSA-4x4j-2g7c-83w6 / CVE-2026-55798 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 63 | pillow | GHSA-phj9-mv4w-65pm / CVE-2026-55380 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 62 | pillow | GHSA-45hq-cxwh-f6vc / CVE-2026-55379 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 61 | pillow | GHSA-5x94-69rx-g8h2 / CVE-2026-54060 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 60 | pillow | GHSA-8v84-f9pq-wr9x / CVE-2026-54059 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 59 | pillow | GHSA-62p4-gmf7-7g93 / CVE-2026-54058 | `<12.3.0` → `12.3.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `pillow==12.3.0`. |
| 58 | torch | GHSA-rrmf-rvhw-rf47 / CVE-2025-3000 | `<=2.12.1` → `2.13.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `torch==2.13.0` and its complete CUDA 13.0 dependency closure. |
| 57 | torch | GHSA-qfhq-4f3w-5fph / CVE-2025-3001 | `<2.10.0` → `2.10.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `torch==2.13.0` and its complete CUDA 13.0 dependency closure. |
| 56 | torch | GHSA-vgrw-7cvw-pwgx / CVE-2025-2999 | `<2.9.1` → `2.9.1` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `torch==2.13.0` and its complete CUDA 13.0 dependency closure. |
| 55 | torch | GHSA-887c-mr87-cxwp / CVE-2025-3730 | `<=2.7.1` → `2.8.0` | `npa/docker/workbench/wan2-2/runtime-requirements.txt` | Hash-lock `torch==2.13.0` and its complete CUDA 13.0 dependency closure. |
| 54 | cryptography | GHSA-g6cj-pr64-35w5 / CVE-2026-69247 | `>=44.0.0,<50.0.0` → `50.0.0` | `npa/requirements-lock.txt` | Pin `cryptography==50.0.0` in the generated application lock. |

An independent package audit also identified `PYSEC-2026-2132` in the
application lock's `click==8.3.2`. The branch pins the fixed `click==8.3.3`.
That audit finding is remediated but is not included in GitHub's 21 open-alert
count because GitHub did not return a corresponding alert record.

The table preserves GitHub's package capitalization exactly; it is not a
distinct package because Python package names are case-insensitive. Pre-merge
alerts remain open until GitHub recomputes the default-branch dependency graph
after merge. Branch validation must therefore prove that each vulnerable
version or package path is absent or replaced; it must not claim that the
dashboard has already closed the alerts.
