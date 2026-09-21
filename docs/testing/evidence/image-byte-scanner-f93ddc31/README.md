# Independent image byte scanner review

The repaired scanner completed this synthetic OCI workload **2.38× faster** than the original scanner: median **14.229 s → 5.977 s** over three alternating rounds. Every run retained the same 1,108 records and 17,007,156 scanned bytes, with byte-identical ledgers. Both arms used the real sealed Go detector and pinned native literal matcher. This is a workload-specific CPU measurement, not a production-image speedup guarantee.

The exact source-file and binary hashes are in [review.json](review.json). The reviewed source is commit [`f93ddc316ede`](https://github.com/nebius/nebius-physical-ai/tree/f93ddc316ede4951875776f6617a917407ad4408) on [PR #681](https://github.com/nebius/nebius-physical-ai/pull/681). CPU runtime validation is accepted; the PR was draft with CI pending when this review was recorded. Earlier prototype timings do not apply to these final bytes.

| Round | Original scanner | Repaired scanner |
| --- | ---: | ---: |
| 1 | 14.229 s | 5.977 s |
| 2 | 14.309 s | 5.850 s |
| 3 | 13.509 s | 6.088 s |

The PR author's separate **2.14×** experiment compares serial depth with depth 64 in the **same repaired scanner**, on a generated **64 MiB** payload. It is a different comparison from the original-source-versus-final-source **16 MiB / 2.3806×** result tabulated here; the experiments must not be combined.

The input is a deterministic, generated 16 MiB OCI payload, containing many records and one harmless synthetic literal. Timing includes actual archive traversal and detection; the fixture supplies its synthetic image-graph receipt. Three alternating rounds ran on one shared Linux host with 16 CPUs available. There is no confidentiality regex in this timing case; separate correctness cases exercise literal and regex findings. Performance depends on record sizes and available CPU capacity. An earlier production scan was cancelled and incomplete; it is not accepted evidence or the performance baseline for this review.

A separate findings-bearing OCI test produced **identical 130-line ledgers** for original and repaired scanners: 47 records, 2,118,866 scanned bytes, 3,524 total findings and six detector findings. Removing the synthetic canary reduced detector findings from six to zero and changed the ledger. Remaining literal/regex findings correctly kept raw `valid=false`; validity was never upgraded to obtain a pass. The matched ledger SHA256 is `ee80563d4e37b5bc486627ca93ae74e33d150efa62312f9488a7ceda851b58bc`.

The full native integration passed with the entire check process restricted to one CPU: 200 literal differential cases, actual OCI scans, cancellation that joins the owned helper and preserves an unrelated process, and a saturated-pipe case with caller depth 64, one helper worker and 12,000 findings. Actual macOS module import also passed.

Independent [race-enabled regression results](helper-regressions.txt) cover the three defects reproduced during review: error paths waiting for producer EOF, allocation before a byte reservation, and circular waiting between input/output pipes. The duplex case belongs to the native integration in `review.json`; the text log covers the helper failure and reservation controls. The official helper build separately reported 32 declared tests, 48 passing tests/subtests and zero failures or skips. Baseline and repaired readiness output and trusted rule configuration are byte-identical.

The byte budgets cover admitted payloads, not total process memory. One oversized record is still scanned whole. A blocked input-reading goroutine may remain until the owned helper process exits; this review does not claim every goroutine is joined. No GPU or VLM result applies to this CPU scanner change, and these tests do not approve any container image for publication.

`original_private_receipt_sha256` values identify retained original receipts. `SHA256SUMS` identifies these sanitized public review files. The full operational reports remain private; this pack contains no credentials, customer data, private endpoints, or machine identifiers.
