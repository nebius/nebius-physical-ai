---
name: team-identity
description: Configure or use optional Workbench team identities, including local access keys, explicit generic JWT links, and native Nebius human identity verification.
---

# Workbench team identity

Use this skill for the optional `npa workbench team` identity boundary. It is
not a chat-agent routing feature and does not provision a cluster, gateway, or
browser login flow.

Local Workbench accounts are the stable ownership domain. Create an
`account_namespace` once, then use `Accounts` for local people and
`issue_key_file` for a personal mode-0600 key. `TeamClient` is the supported
agent-callable API for an already-authorized service endpoint. Obtain credentials
through the private saved-session provider or the explicit `load_bearer_token`
file/environment path. Never print bearer credentials or copy them into reports.

For end users, start with `npa login --endpoint <team-HTTPS-endpoint>` and choose
a privately delivered personal key or `--nebius`. Successful login saves a
private named connection and unambiguous placement defaults. Subsequent team
commands can omit endpoint and credential flags; agents can use
`npa.sdk.workbench.team.connect()`. A native session asks the official Nebius CLI
for a token on each command without copying IAM tokens into Workbench storage.
If no human profile exists, create an isolated CLI configuration, preserving
the machine's default profile. `npa logout` forgets the local connection; it
does not revoke the server key, cloud session, or running work.

The CLI retains automatic retry identities before submission. Repeating an
unchanged submission returns its original run. Use `--new-run` only after an
acknowledged submission when another run is intended; reconcile an unconfirmed
submission first. Explicit idempotency keys remain supported.

On a remote operator VM, `--no-browser --ssh-host <alias>` emits the verified
official browser URL and scoped loopback callback forward. An authorized agent
may open that URL and establish the tunnel. Let the human complete any required
password or MFA; never request the callback URL or token in chat. Verify actual
human login and a real workload before labeling a demo SSO-verified.

An external identity is only a credential for an already-created local account:

- Generic `identity` uses the configured JWT issuer and its verified subject.
- Native `nebius_identity` selects one tenant and optionally one federation. It
  requires `account_namespace`, cannot coexist with generic `identity`, and
  verifies each human token at Nebius IAM's fixed ProfileService endpoint.
- Use `link_identity` only with the exact configured issuer and the
  provider-verified immutable subject. There is no auto-enrolment, email match,
  cloud-group import, or cloud-role-to-Workbench-grant mapping.

Use `account unlink` with the exact local user, old issuer, and old subject
before replacing an external provider link. It works for a retired issuer or
disabled account, preserves local ownership and keys, and revokes that login
on the next request. It does not cancel already admitted work.

Native Nebius browser SSO happens in the person's Nebius CLI/identity provider.
Workbench only verifies the resulting short-lived IAM bearer token. A linked
account's local groups, grants, allocations, ownership, and disabled state stay
authoritative; local keys remain usable. See
[`docs/workbench/team-identity.md`](../../../docs/workbench/team-identity.md)
for the private token-file and linking contract.
