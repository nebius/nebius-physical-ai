# Real scanner proof: reject private-key findings before image publication

[PR #687](https://github.com/nebius/nebius-physical-ai/pull/687) repairs a severity-filtering defect in image publication checks. The real pinned Trivy scanner inspected the same retained private image archive under both severity settings.

| Measured result | Old CRITICAL filter | All severities |
| --- | ---: | ---: |
| Visible secret findings | 0 | 3 HIGH |
| CRITICAL vulnerabilities | 6 | 6 |
| Fixable CRITICAL vulnerabilities | 0 | 0 |
| Python gate decision | Accept | Reject |

The CRITICAL vulnerability identifiers were identical. The defect was that the severity filter removed the observed HIGH private-key findings before the publisher could reject them. The repaired Python gate keeps the existing vulnerability policy and rejects secret findings at every severity.

The [machine-readable report](report.json) binds the private input archive, scanner image, raw reports, executed production helper source, and verdicts by SHA-256. Root independently recomputed the counts from both reports and verified that the executed Python source is byte-identical at commits `ed24aaa02`, `1668e1286`, and `2f9032465`. The initial matched scanner comparison and the production-helper execution are separate retained experiments.

The helper experiment replaced registry transport with a local archive and removed the platform flag. It exercised the real scanner and source-generated severity arguments. It did **not** execute a public image copy, establish public key exposure, or qualify any image for release. The workflow addition has separate static/regression validation; this experiment did not run that publication workflow.

This is CPU security evidence. GPU execution and VLM scoring do not apply to this nonvisual gate. Full-suite and current CI acceptance remain separate PR requirements. Raw scanner reports, image bytes, credentials, private paths, and registry identifiers are excluded from this public bundle.

Verify the included report and this page with [SHA256SUMS](SHA256SUMS). These files are an allowlisted derivation from retained private receipts, not fabricated scanner output.
