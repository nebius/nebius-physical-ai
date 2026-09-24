# Local validation: #615 and batch two

Local validation passed; hosted CI pending. Not a merge-ready declaration.

The queue rejection of #615 was a two-minute Gitleaks job timeout, after the scanner reported no leaks. All six signed refreshes remove that short timeout while preserving required scanning, and include current main.

| PR | Source head | Full Linux suite | Hosted CI |
|---|---|---|---|
| [#615](https://github.com/nebius/nebius-physical-ai/pull/615) | `d645be7e7f7b` | 27137 passed, 135 skipped, 1 xpassed, 364 warnings in 744.12s (0:12:24) | [run](https://github.com/nebius/nebius-physical-ai/actions/runs/36069353289) |
| [#609](https://github.com/nebius/nebius-physical-ai/pull/609) | `372504146446` | 27092 passed, 135 skipped, 1 xpassed, 364 warnings in 744.10s (0:12:24) | [run](https://github.com/nebius/nebius-physical-ai/actions/runs/36069355714) |
| [#640](https://github.com/nebius/nebius-physical-ai/pull/640) | `426270eb47af` | 27085 passed, 135 skipped, 1 xpassed, 362 warnings in 744.10s (0:12:24) | [run](https://github.com/nebius/nebius-physical-ai/actions/runs/36069355723) |
| [#679](https://github.com/nebius/nebius-physical-ai/pull/679) | `469ae6d68d23` | 27089 passed, 135 skipped, 1 xpassed, 363 warnings in 744.08s (0:12:24) | [run](https://github.com/nebius/nebius-physical-ai/actions/runs/36069353841) |
| [#687](https://github.com/nebius/nebius-physical-ai/pull/687) | `761995687a8a` | 27104 passed, 135 skipped, 1 xpassed, 364 warnings in 744.08s (0:12:24) | [run](https://github.com/nebius/nebius-physical-ai/actions/runs/36069354641) |
| [#650](https://github.com/nebius/nebius-physical-ai/pull/650) | `38ae0822c366` | 27085 passed, 135 skipped, 1 xpassed, 362 warnings in 724.33s (0:12:04) | [run](https://github.com/nebius/nebius-physical-ai/actions/runs/36069355590) |

Every branch passed 859 dedicated hostile-input security tests, docs drift, lint/formatting, guardrails, confidentiality, full-range Gitleaks, and native scanner comparisons. The ordered merge simulation passed **27,172 tests with zero failures**. Native comparisons found zero new regressions and zero blocking findings.

The initial full-suite attempts each exposed one inherited-SIGINT launcher failure. This was reproduced on unchanged main and the combined candidate; normal signal handling passed both cancellation controls. Complete suites were then rerun successfully, without a test exemption or repository change.

Fresh production probes passed ten real-filesystem receipt/redaction controls for #615 and four real Kubernetes parser controls for #640. These use disposable local fixtures and make no infrastructure calls. GPU/VLM execution does not apply to this batch. #609 still has no successful real controller log-tail run; #687 retains its earlier real Trivy proof against unchanged publisher bytes, without claiming image publication or workload qualification.

[Exact results, source identities, control outcomes, and log hashes](local-validation.json). Codex performed this recovery without Cursor. No PR has been enqueued or merged by this recovery.
