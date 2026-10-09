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
share it between people. A user loads it locally before invoking the familiar
team commands:

```bash
export NPA_TEAM_ENDPOINT=https://team.example.invalid
export NPA_TEAM_TOKEN="$(<"$HOME/.config/npa/team.key")"
npa workbench team whoami
npa workbench team list --workspace robotics
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

## Optionally link an external identity later

If an administrator configures a trusted external JWT issuer, its authenticated
subject can be linked to a pre-existing local account. The subject must be an
exact immutable provider identifier and the issuer must match the configured
provider.

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

External identity verification is optional. Keep local key delivery and
revocation as the dependable baseline until that separate API authentication
path has been verified for the installation.
