# Security Policy

## Reporting Security Issues

The Nebius team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

To report a security issue, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/nebius/nebius-physical-ai/security/advisories/new) tab.

The Nebius team will send a response indicating the next steps in handling your report. After the initial reply to your report, the Nebius team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

## Supported Versions

Security updates are provided for the following versions of Nebius Physical AI Solutions:

| Version | Status           |
|---------|------------------|
| Latest  | Supported        |
| Older   | Community-driven |

We recommend keeping your installation up to date to ensure you have the latest security patches and improvements.

## General Security Practices

When working with Nebius Physical AI Solutions:

- Keep your dependencies updated
- Follow the principle of least privilege when configuring access
- Review security advisories regularly
- Report vulnerabilities responsibly without public disclosure until a patch is available

## Credential Handling

This repository does not contain real credentials. If you find any hardcoded
credentials, tokens, passwords, private keys, or live infrastructure IDs that
should not be public, please report them immediately through the advisory flow
above.

Required secrets are configured outside the repository in
`~/.npa/credentials.yaml`. Use `docs/credentials.yaml.example` as the template
and keep the real credentials file out of git. Non-secret resource identifiers
such as `NEBIUS_PROJECT_ID`, `NEBIUS_TENANT_ID`, `NPA_REGISTRY`, and
`NPA_S3_BUCKET` are documented in `docs/workbench/getting-started.md`.

Do not commit live infrastructure identifiers. Parameterize or redact:

- Nebius project and tenant IDs
- Nebius compute instance, managed Kubernetes node group, and container
  registry IDs
- Concrete `cr.eu-north1.nebius.cloud/<registry-id>/...` image references
- Concrete S3 bucket names and `s3://<bucket>/...` paths from validation runs
- Public VM IP addresses or endpoints from live Nebius workloads

## Historical leak handling

Earlier validation runs in W10-W11 introduced operational identifiers
(public VM IPs and working bucket names) into commits that have since
been removed via the W14 history condensation (2026-05-21). The current
tree contains no real Nebius identifiers; placeholders and RFC 5737
test IPs are used throughout.

Future commits are gated by `.github/workflows/gitleaks.yml` against
the patterns in `.gitleaks.toml`. The gitleaks workflow scans PR diffs
and main pushes; historical commits prior to W14 are no longer
reachable from `main`.

If pre-W14 history needs to be referenced for any reason, it is
preserved out-of-band at the local archive tag created during W14.

## LeIsaac agent TLS trust model

The LeIsaac reverse-client path (`npa/src/npa/workbench/leisaac/reverse_client.py`)
and the agent CLI (`npa/src/npa/cli/workbench/leisaac.py`) connect to agent
HTTPS endpoints with `ssl.CERT_NONE` and hostname checking disabled, and instead
pin the peer by SHA-256 certificate fingerprint:

- Every connection fetches the peer certificate in binary form and compares
  `hashlib.sha256(certificate).hexdigest()` against the expected fingerprint
  (`reverse_client.py` `connect()`; `cli/workbench/leisaac.py`
  `_select_agent_leisaac_run()` uses `secrets.compare_digest` and fails the
  connection on mismatch).
- The expected fingerprint is captured at agent-deploy time by
  `_agent_certificate_sha256()`, which records the fingerprint of whatever
  certificate the fresh agent endpoint presents.

This is a trust-on-first-use (TOFU) model. Its assumptions and operator
guidance:

- **First capture is the trust decision.** The deploy-time capture itself is
  unauthenticated: a man-in-the-middle present during the very first fetch
  would pin the attacker's certificate instead. Perform initial deployment
  over a trusted network, and independently verify the fingerprint out of
  band when the deployment is security-sensitive.
- **A fingerprint change is treated as an attack.** After pinning, any
  certificate change fails the connection loudly; there is no silent
  re-pinning. If you legitimately rotate the agent certificate, update the
  stored fingerprint deliberately — never automate re-capture on mismatch.
- **Protect the stored fingerprint.** The fingerprint is written into the
  per-run Kubernetes Secret (`relay_client_secret_manifest`) alongside the
  relay credentials; anyone who can modify it can redirect trust. It is
  protected by Kubernetes Secret RBAC in the agent namespace — not by
  local file permissions — so restrict `get`/`update` on that Secret to
  the deployer role. Never commit fingerprints to the repo.
- **TLS 1.2+ is still enforced** (`context.minimum_version =
  ssl.TLSVersion.TLSv1_2`); only CA-chain validation is replaced by pinning.

## Learning More About Security in Nebius

To learn more about security in Nebius, please see the [Nebius Security Documentation](https://nebius.ai/docs/security).

## Code of Conduct

We expect all contributors and reporters to follow our community standards. Please report any code of conduct violations to the Nebius team.
