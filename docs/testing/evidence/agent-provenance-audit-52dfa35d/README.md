# Agent credential-provenance audit repair

PR [#654](https://github.com/nebius/nebius-physical-ai/pull/654) hardens agent credential selection and bounded subprocess cleanup. Its healthy capability-audit test lacked the provenance that the production guard correctly requires. The repair provides an isolated, nonsecret fixture and preserves rejection of missing or unsupported provenance.

Validated and published source: `52dfa35d470642f632a845e4fb20002c0d35b218`; tree `5a42dfd87f4a39870469b925211139b33a5d7a0b`.

| Validation | Result |
|---|---:|
| Uninterrupted full Linux CPU suite | 25,140 passed; 155 skipped; 1 non-strict xpass; 0 failures/errors |
| Independently repeated fixture tests | 8 passed |
| Guard-bypass mutation | Detected by both negative cases |
| Real Linux descendant-group cleanup control | Passed |
| Actual scanner regression workload | Passed |
| Trusted-base security comparison | 609 existing findings on each side; 0 new regressions |
| Built-in confidentiality and full-branch Gitleaks | Passed |
| Published signature | GitHub verified |

The new delta changes one test file. Every production byte remains identical to the independently reviewed `0666db15f7c5a715ceba74b9dbfa57fe93054e26` producer. The cumulative PR includes production changes; it is not a test-only PR. Missing and unsupported provenance must still produce the expected error responses and withhold private diagnostics. A provider-command tripwire verifies that these fixtures do not call the provider.

The [previous scoped CPU deployment proof](https://github.com/nebius/nebius-physical-ai/blob/b83b862e1469d0bfa22ae7c8bffae11db6b1c9b3/docs/testing/evidence/agent-metadata-test-repair-09d0fa9f/README.md) remains applicable through that production-byte comparison. It demonstrated a protected fresh agent, access refresh, resource inventory and teardown. Its deployment manifest was not copied before teardown; this repair adds no deployed-Python-byte attestation or new deployment claim.

The earlier hosted failure is retained. A separate Darwin affected-suite run also remains **failed: 228 passed, 1 failed**, because the unchanged Linux reaper calls `os.waitid`, which is unavailable there. The exact unchanged descendant control and full suite passed on Linux. No assertion was weakened and no platform skip was added to obtain the pass. This does not establish macOS backend support.

GPU/VLM tests are not applicable to this nonvisual credential/process change. The full local suite retains the repository's live/GPU/end-to-end exclusions and does not replace hosted coverage or current-head CI.

[Machine-readable results and verification hashes](validation.json) · [SHA-256 manifest](SHA256SUMS)

Only allowlisted source identifiers, outcomes and hashes are public. Exact runtime identities, commands, inventories and full logs remain private.
