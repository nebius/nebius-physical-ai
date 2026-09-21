# Security regression gate

Every pull request, merge queue candidate (`merge_group`), and push to `main`
runs **Security regression / security-regression**. It is the only automatic PR
workflow and atomically owns test, lint, guardrail, gitleaks, confidentiality,
source-scanner, image-security, and hostile-input jobs. A superseding PR commit
cancels that complete gate rather than leaving work in six workflow queues.
PRs and merge-queue candidates run the same browser and focused Python
compatibility checks alongside the complete duration-balanced Python 3.12
coverage suite. This catches cross-subsystem failures before queue admission;
the queue rechecks the combined candidate against the latest main revision.
The narrow prose-only exception described in
[the contributor CI guide](../../CONTRIBUTING.md) retains smoke, documentation,
lint, guardrails, and every security gate. The test selector comes from the
trusted base commit and uses its merge-candidate policy for both events,
including when an older base still has a narrower PR policy. Missing policy
keeps full validation and invalid comparisons fail. The full three-interpreter
test audit runs daily instead of immediately after each merge. Repository-wide
job concurrency
limits validation to seven PR jobs, nine merge-candidate jobs, and three background
audit jobs. Each candidate pool has a separate slot for the short test-scope
selector, coverage, and the final required check. Shards can start without
waiting for long docs and guardrail jobs. Optional timing reports use the audit
pool. These groups bound repository demand but cannot reserve organization
runners. See the
[validation concurrency contract](../../CONTRIBUTING.md#validation-concurrency)
for queue retention, superseded-commit cancellation, and rollout limits.

The image workflow has no top-level path filter. Its two automatic jobs always
report an internal, fail-closed scope decision. Image, packaging, workflow, and
security-policy changes run complete-byte, configuration, and base-image checks;
unrelated source changes take the verified fast path. Main, scheduled, and manual
audits always run the deep checks. Seven pinned bases use one Trivy database
download and three isolated worker caches inside one runner. Deep candidates block
on fixed CRITICAL OS-package vulnerabilities and HIGH/CRITICAL configuration
findings. Their
[patched base targets and regression tests](image-reproducibility.md#cve-scanning)
are checked separately from the application-dependency scan.
Failed, skipped, or cancelled required work cannot produce a passing result.
Main, scheduled, and manual image scans additionally generate and upload SARIF
with the established per-image identities. A new finding fails the scanner job
with its file, rule or advisory, and remediation detail. Scanner, dependency
resolution, malformed report, incomplete inventory, and source parsing failures
also fail the job. There are no top-level path filters or successful error fallbacks.

NPA accepts commands and artifacts through a Python CLI and FastAPI agent,
uses browser dependencies, and runs GitHub Actions with repository access.
Those boundaries motivate three complementary maintained scanners:

| Scanner | Blocking coverage |
|---|---|
| Bandit 1.9.4 | All tracked Python, including scripts and tests; medium/high severity and confidence, including unsafe deserialization, injection, weak transport and unsafe APIs. |
| zizmor 1.30.0 | GitHub workflows and local actions; regular-persona findings with at least medium severity/confidence, including template injection and excessive permissions. |
| Trivy 0.74.0 | All advisory severities, including unfixed vulnerabilities, for exact Python requirement pins, project extra pins, npm lock dependencies including development dependencies, and the resolved NPA core/development dependency closure. |

The gate materializes regular files from the actual target commit and the
proposed merge commit, scans both with the same policy and vulnerability database,
and subtracts matching occurrences. Identities include the file, rule, and
expression or package version. Moving lines does not create a finding; adding a
second occurrence or moving vulnerable code to another file does. Existing
findings remain visible in private reports and are not silently accepted through
a committed baseline file. A fix followed by a later reintroduction fails against
the now-fixed base.

Application and CI dependencies additionally have an absolute gate: every
candidate finding blocks, including a newly published advisory that affects an
unchanged pin on both branches. This covers `npa/requirements-lock.txt`,
`npa/ci/requirements.txt`, `npa/pyproject.toml` and its resolved core/development
closure, the browser test npm lock, and `scripts/security-requirements.txt`.
Deleting either required application/CI requirements file fails scanning.
The private summary separates new regressions from all blocking findings.
Vendor and tool-runtime inventories retain differential checks and their
separate image validation requirements.

The AnyIO floor is 4.14.2 for ordinary installs and the application lock, covering
[TLS hostname verification](https://github.com/advisories/GHSA-82r6-8w77-94w6)
and [process-worker stderr hangs](https://github.com/advisories/GHSA-5p39-cfhj-2xmp).
`.github/dependabot.yml` checks application, CI, scanner, browser, and Actions
dependencies daily and proposes updates for review. After changing Python
dependency declarations, regenerate the CI pins with
`npa/.venv/bin/python npa/scripts/ci_requirements.py --update`.
Dependabot security-update enablement is a separate repository setting; the
version-update configuration does not enable it or merge its PRs automatically.

Candidate Bandit/zizmor ignore comments and configuration are disabled. Trivy
receives isolated empty configuration and ignore files, so the image scan's
OS-only settings do not hide application vulnerabilities. Python requirements
with nonstandard filenames are normalized into explicit pip inventories, including
pins with inline hashes and whitespace-separated comments. npm locks must retain
resolved direct packages and agree with exact direct version pins. The
core/development dependency closure is resolved with pinned uv, Python 3.12 and
public PyPI metadata, with source builds disabled. Candidate application code,
build hooks, npm scripts and dependencies are never installed or executed by
the scanner job. Its tools live in a separate environment.

The job reads scanner code and pins from the target revision. The installing PR
bootstraps them from its own revision because the base has no scanner yet.
Changes to scanner policy take effect on subsequent PRs after merging; validate
the proposed policy locally as well. Changes to the workflow itself still need
trusted review: a required status name cannot establish the integrity of the
workflow that produces that status.

## Reproduce locally

Use a fresh private directory outside the repository for each run. Install the
scanner requirements into a separate virtual environment and put its `bin`
directory on `PATH`; keep `npa/.venv/bin/python` for repository validation.
The installer supports Linux x86-64 and verifies the Trivy archive's SHA-256.

```bash
umask 077
scan_tools="$(mktemp -d)"
npa/.venv/bin/python -m venv "$scan_tools/venv"
"$scan_tools/venv/bin/python" -m pip install -r scripts/security-requirements.txt
bash scripts/security_install.sh "$scan_tools/bin"
export PATH="$scan_tools/venv/bin:$scan_tools/bin:$PATH"
npa/.venv/bin/python scripts/security_regression_checks.py --output-dir "$scan_tools/regressions"
npa/.venv/bin/python scripts/security_gate.py --base origin/main --output-dir "$scan_tools/report"
```

Without `--head`, the gate scans tracked working files and unignored additions,
which allows validation before committing. For the exact CI comparison, supply
`--base <target-sha> --head <merge-sha>`. The output directory must not exist yet.
Raw reports can contain source and dependency locations; keep them private.
CI prints only actionable finding summaries and does not upload raw artifacts.
The real regression workload generates inert Python, workflow and vulnerable
dependency fixtures, scans them with the actual binaries/database, and verifies
rejection. It never launches the fixtures.
The workload also verifies that an unchanged vulnerable application pin fails
and that the patched AnyIO pin passes. Unit regressions cover each protected
manifest, duplicate findings, removals, required inventories, and gate exit codes.

The customer confidentiality scan retains every raw redacted finding and reports
raw, dispositioned, and unresolved counts separately. One NCore-specific source
correction recognizes only lines 633 and 640 of the exact regular Git `100644`
file at
`npa/docker/workbench/ncore/notices/cpython/LICENSE.third-party`. Before those two
locations can be dispositioned, the scanner verifies the complete notice bytes
against fixed hashes and sizes for both the official Python 3.12.14 archive and
the official archive of its pinned CPython commit. Each archive must contain one
exact regular `Doc/license.rst` member whose complete bytes equal the notice.
Other paths, lines, policies, stdin patches, modified notices, and failed or
missing proof remain unresolved. The private proof directory is caller-owned and
must have mode `0700`; CI creates it under the runner's temporary directory.
The shared `npa.guardrails.ncore_attribution.verify_public_notice` API accepts
only the complete notice bytes and that private directory; it returns public
URL/member/hash/size provenance and never returns cache paths.

To exercise that same source proof locally, use an ephemeral private directory:

```bash
umask 077
proof_directory="$(mktemp -d)"
npa/.venv/bin/python -m npa.guardrails.confidentiality \
  --repo-root . --tree --pattern-env CUSTOMER_DENYLIST \
  --ncore-attribution-proof-directory "${proof_directory}"
```

## Merge enforcement and limits

Require the **security-regression**, **gitleaks**, and **scan** (confidentiality)
contexts from GitHub Actions on `main`. Use the merge queue so GitHub validates
the exact latest-main candidate without requiring every open PR branch to be
bulk-refreshed. Preserve other required checks and branch protections. Workflow
configuration alone does not enable merge enforcement; administrators must
verify the required contexts and active merge-queue rule.
Keep the repository's **Allow auto-merge** setting enabled as well so contributors
can request automatic merging through clients that use that setting. It does
not replace or bypass queue validation. See the contributor guide for
[mobile recovery and pending-check diagnosis](../../CONTRIBUTING.md#auto-merge-and-the-merge-queue).
Repository administrators and configured bypass actors may still bypass checks.
Keep the component workflows enabled as reusable candidate gates and
superseding main audits. Secret and confidentiality checks also run for merge
queue candidates.

These checks reduce known risks; they do not prove the absence of vulnerabilities.
Bandit is Python pattern analysis, not application-wide dataflow or JavaScript
source analysis. zizmor runs offline and cannot audit remote action internals.
Trivy covers recorded exact versions and the Linux/Python 3.12 core/development
resolution; unpinned external tool requirements, other optional extra closures,
Git dependencies, runtime downloads and packages installed inside images need
their own workload/image validation. npm lockfiles must be version 2 or 3.
Dependency versions and advisory data can change between runs. Existing Trivy
image/config scans remain responsible for their configured container and IaC
coverage; no model or GPU execution is necessary for this security workload.

Tool reference: [Bandit](https://bandit.readthedocs.io/en/latest/start.html),
[zizmor](https://docs.zizmor.sh/usage/),
[Trivy Python coverage](https://trivy.dev/docs/latest/coverage/language/python/),
[GitHub required checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).
