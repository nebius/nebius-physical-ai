---
name: protect-nebius-infra-details
description: Use when preparing commits, documentation, reports, examples, tests, pull-request bodies, issues, or live-validation handoffs involving Nebius, to prevent concrete live infrastructure details or credentials from reaching Git or public collaboration surfaces.
---

# Protect Nebius Infra Details

Keep concrete live resource identifiers and names, account or tenant details,
private endpoints and object/registry URIs, and credentials out of Git and public
collaboration surfaces. Apply the rule even when a value is not itself a secret.

## Publication workflow

1. Treat commits, docs, reports, examples, fixtures, tests, PR or issue text, and
   live-validation handoffs as publication surfaces.
2. Store exact operational evidence only in access-controlled external
   artifacts. In repository summaries, use generic resource roles plus
   non-identifying counts, timings, measurements, and cryptographic hashes.
3. Do not copy task-scoped consent, resource inventories, private locations, or
   teardown identifiers into reusable configs. State the requirement and point
   to externally retained evidence without naming its location.
4. Scan staged content before committing:

   ```bash
   git diff --cached --no-ext-diff | \
     npa/.venv/bin/python -m npa.guardrails.confidentiality \
       --built-in-nebius-infra --stdin-source staged-diff --stdin-is-diff
   ```

5. Run gitleaks with `.gitleaks.toml` for credentials and secret-shaped values.
   Scan proposed PR or issue prose through the same built-in scanner before
   publication; pass the text on stdin and use a descriptive `--stdin-source`.
6. Inspect the complete aggregate diff, not only the newest edit. If sensitive
   values entered reachable branch history, rewrite only the authorized task
   branch and verify the offending commit is no longer an ancestor.
7. Stop publication when sanitization cannot be proven. Do not replace a live
   identifier with an invented identifier or fake private URI.

Placeholders in ordinary documentation and clearly synthetic unit fixtures are
allowed. Prefer environment-variable names or generic roles over realistic
opaque identifiers.
