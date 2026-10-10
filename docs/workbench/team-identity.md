# Local team accounts and personal access keys

Workbench team mode starts with local, administrator-managed accounts. A person
receives a persistent Workbench user ID, local group membership, and one or more
independently revocable personal access keys. They use that key file with the
existing CLI, HTTP API, and SDK; there is no Workbench browser-login flow.

The account database lives with the team service's private persistent state. It
is not a password database, cloud directory, or self-service identity product.
Neither a personal Nebius account nor an external identity provider is required.

## Account model

`account_namespace` is a UUID selected once per installation. It defines stable
local ownership IDs. Back up the account database with the team run ledger and
retain the same namespace during restores; changing it would make the existing
accounts incompatible with their recorded runs and allocations.

| Concept | Administrator controls | Why it is separate |
| --- | --- | --- |
| Local user | Stable ID and administration name | Credentials can change without moving ownership |
| Local group | Membership and workspace role grants | A group is permission, not capacity |
| Allocation | Per-user cluster GPU cap and storage scope | Capacity and data access remain explicit |
| Personal key | A delivered file and credential ID | A lost key can be revoked without disabling the user |

The service retains only a digest of each access key. `issue-key` writes the
plaintext exactly once to a new mode-0600 file and does not print it.

## Create users, groups, and keys

Run account administration against the service's authoritative private
configuration. Repeat `--group` to assign multiple local groups. The generated
`id` in the `create` response is the value to use in allocation `subject`
entries in the team policy.

```bash
npa workbench team account create --config /private/npa-team/team.yaml \
  --name researcher-a --group researchers
npa workbench team account create --config /private/npa-team/team.yaml \
  --name reviewer-a --group reviewers
npa workbench team account list --config /private/npa-team/team.yaml

npa workbench team account issue-key --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --output-file /private/key-delivery/researcher-a.key
```

Deliver the key file through an approved private channel. It is a bearer
credential: never commit it, include it in a URL, paste it into a workflow, or
share it between people. The team CLI reads the mode-0600 file directly with
`--token-file`, so the key does not need to be exported into the shell
environment:

```bash
export NPA_TEAM_ENDPOINT=https://team.example.invalid
TEAM_KEY_FILE="$HOME/.config/npa/team.key"
npa workbench team whoami --token-file "$TEAM_KEY_FILE"
npa workbench team list --workspace robotics --token-file "$TEAM_KEY_FILE"
```

See [team access](team-access.md#user-submit-and-inspect-work-from-a-key-file)
for submit, status, logs, artifacts, cancellation, recovery, HTTP, and SDK
examples.

## Change membership or revoke access

Replace group membership with one or more `--group` options, remove every group
with `--clear-groups`, or disable the whole account. A disabled account cannot
use any of its keys or a linked external identity.

```bash
npa workbench team account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --group reviewers
npa workbench team account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --clear-groups
npa workbench team account revoke-key --config /private/npa-team/team.yaml \
  --key-id "$WORKBENCH_KEY_ID"
npa workbench team account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --disabled
```

Revocation blocks that key on the next request. Disabling blocks all local keys
and linked identities. Neither operation automatically cancels an already
admitted run nor rotates the allocation's storage credentials; cancel a known
run with `npa workbench team stop-run` and revoke workload storage credentials
through the storage system as separate actions.

## Optionally link one external identity later

An installation may use either a trusted generic JWT issuer or native Nebius
human identity verification. They are deliberately mutually exclusive. Both
modes require an explicit one-to-one link to a pre-existing local account; they
never create an account, match an email address, or import groups or cloud
roles.

For a generic JWT issuer, the administrator links its authenticated immutable
subject under the configured issuer:

```bash
npa workbench team account link --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --issuer "$EXTERNAL_ISSUER" --subject "$EXTERNAL_SUBJECT"
```

This preserves the local user ID, allocations, namespaces, runs, and storage
scope. It does not match email addresses, import groups, synchronize a
directory, create new accounts, or establish a browser-login path. Unlinked
external identities are denied, and local group and allocation policy remains
the authorization source.

### Native Nebius human identity

Native Nebius verification is optional and must retain `account_namespace` so
that Workbench ownership stays local and stable. Configure the selected tenant
and, when needed, an exact federation restriction in the private team policy:

```yaml
account_namespace: "<stable-workbench-UUID>"
nebius_identity:
  tenant_id: "<Nebius-tenant-ID>"
  # Omit this field unless this installation must require one federation.
  federation_id: "<Nebius-federation-ID>"
```

This is not generic JWT mode: do not set `identity` alongside
`nebius_identity`. Changing either identity configuration while the service is
running requires a server restart.

The person completes browser SSO through their existing Nebius CLI and identity
provider. Workbench does not host an OAuth callback, Keycloak UI, or browser
login page. They place their short-lived IAM token in a private, regular
mode-0600 file and use the existing team CLI and SDK token-file path:

```bash
NEBIUS_IAM_TOKEN_FILE="$HOME/.config/npa/nebius-iam-token"
chmod 600 "$NEBIUS_IAM_TOKEN_FILE"
npa workbench team whoami --token-file "$NEBIUS_IAM_TOKEN_FILE"
```

For every such bearer request, Workbench calls the fixed HTTPS Nebius IAM
ProfileService endpoint (`GET https://api.nebius.cloud/iam/v1/profiles`) with
normal TLS verification, no redirects, and a bounded timeout. It accepts only a
human `userProfile` whose global account and exactly one membership in the
configured tenant are `ACTIVE`; service accounts, anonymous profiles, malformed
responses, unavailable providers, and federation mismatches are denied. The
provider-verified `tenantUserAccountId` is the external subject and the fixed
issuer is `https://api.nebius.cloud`.

Link that exact provider-verified subject to the existing Workbench user. Do not
substitute an email address, display name, project role, or cloud group:

```bash
npa workbench team account link --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --issuer https://api.nebius.cloud \
  --subject "$NEBIUS_TENANT_USER_ACCOUNT_ID"
```

Configured Nebius tenant access is identity proof, not Workbench authorization.
Workbench groups, workspace grants, allocations, run ownership, and disablement
remain those of the linked local account. Local keys continue to work, and a
disabled local account denies both its keys and its linked Nebius identity. This
implementation verifies the resulting IAM identity only; it does not prove a
specific browser SSO flow, which must be qualified separately for an
installation.
