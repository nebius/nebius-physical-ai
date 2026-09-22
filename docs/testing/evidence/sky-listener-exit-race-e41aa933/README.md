# SkyPilot listener exit regression

When the newly started SkyPilot daemon exits between reading its network namespace and reading its TCP table, readiness used to raise `FileNotFoundError`. The fix returns false for this process-disappearance case and preserves permission and unrelated errors.

The regression starts a real child, reads its actual namespace, terminates it, and then exercises the TCP read. The predecessor fails at that read; the repaired implementation refuses readiness. No process-table response is substituted.

| Check | Result |
| --- | --- |
| Exact source | `e41aa93329ab2b60ef4b26bc7085b2ef23e9b1b6` |
| Base | `36605d46a7b67da6bb7f296bba556aa5111eee02` |
| Full Linux suite | 25,650 passed, 128 skipped, one non-strict xpass; zero failures |
| Local required gates | All ten passed |
| Native security | 607 pre-existing findings on each side; zero new or blocking findings |
| Independent input review | All 3,326 source blobs on each side matched Git bytes |
| Runtime applicability | CPU process lifecycle; GPU and VLM testing do not apply |

Full-suite log SHA-256: `ba627db9eb428a99f8559e1a9061e36c5bd856912c2e230c84c2cf6e77ee3dc3`.

The [machine-readable results](results.json) bind the exact source, file tree, test counts and retained evidence. Review the two-file implementation and regression in the associated PR. Hosted checks and required merge-queue integration remain separate gates. Codex performed this repair; Cursor's current account limit prevented its requested review.
