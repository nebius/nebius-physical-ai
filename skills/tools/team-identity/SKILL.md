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
agent-callable API for an already-authorized service endpoint; its token comes
from the existing `load_bearer_token` file/environment path. Do not read, print,
or persist the bearer token outside that path.

An external identity is only a credential for an already-created local account:

- Generic `identity` uses the configured JWT issuer and its verified subject.
- Native `nebius_identity` selects one tenant and optionally one federation. It
  requires `account_namespace`, cannot coexist with generic `identity`, and
  verifies each human token at Nebius IAM's fixed ProfileService endpoint.
- Use `link_identity` only with the exact configured issuer and the
  provider-verified immutable subject. There is no auto-enrolment, email match,
  cloud-group import, or cloud-role-to-Workbench-grant mapping.

Native Nebius browser SSO happens in the person's Nebius CLI/identity provider.
Workbench only verifies the resulting short-lived IAM bearer token. A linked
account's local groups, grants, allocations, ownership, and disabled state stay
authoritative; local keys remain usable. See
[`docs/workbench/team-identity.md`](../../../docs/workbench/team-identity.md)
for the private token-file and linking contract.
