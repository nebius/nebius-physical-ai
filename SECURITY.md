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
`NPA_S3_BUCKET` are documented in `docs/getting-started.md`.

## Learning More About Security in Nebius

To learn more about security in Nebius, please see the [Nebius Security Documentation](https://nebius.ai/docs/security).

## Code of Conduct

We expect all contributors and reporters to follow our community standards. Please report any code of conduct violations to the Nebius team.
