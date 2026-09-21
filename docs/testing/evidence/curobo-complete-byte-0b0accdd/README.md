# cuRobo complete-image content review

The terminal scans for the GPU-tested predecessor and the repaired recording image are independently reconciled. Each scanned 23 layers, 77,233 regular files and 328,653 detector records. All 265 native detections remain in each raw report, whose validity is correctly **false**. A separate exact-content policy accepts those occurrences with zero unresolved or private-literal findings.

| Image source | Regular-file bytes | All detector bytes | Native detections | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| `49fed65d` — GPU numerical producer | 8,416,813,955 | 8,489,142,910 | 265 | 0 |
| `0b0accdd` — repaired recording image | 8,416,822,650 | 8,489,149,899 | 265 | 0 |

The review streamed both full ledgers, checked record order, byte counts, layer identities, report/authorization/detector bindings and every finding's exact content hash, size and occurrence coordinates. Each image used 58 of the 65 previously reviewed catalog records across 60 finding-bearing records. No path, rule or private credential exception was added. All 28 executed scanner source files were verified against private Git objects and byte-identical published source files; the execution and published-source revisions remain distinct.

[Open the measurements and source/image/report/ledger hashes](measurements.json). [Verify these public files](SHA256SUMS).

This review completes the retained byte-scan reconciliation. It does not add GPU execution on the repaired image, qualify the visual judge, certify robot safety, or promote an image publicly. The separate actual B200/RTX numerical results and current-image CPU recording replays retain their original producer identities. All 60 prior visual-model calls remain recorded as a failed judge evaluation.

Original archives, complete raw ledgers, operational logs and customer-specific configuration remain private. This public packet contains only reviewed measurements, content hashes and scope statements.
