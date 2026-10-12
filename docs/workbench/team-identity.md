# Local Workbench accounts and personal access keys

Workbench access starts with local, administrator-managed accounts. A person
receives a persistent Workbench user ID, local group membership, and one or more
independently revocable personal access keys. They use that key file with the
existing CLI, HTTP API, and SDK; there is no Workbench browser-login flow.

## Sign in once with `npa login`

Your administrator supplies the Workbench HTTPS endpoint and either a personal key
or an explicit link from your Nebius identity to your Workbench account. Once
`npa` is installed, choose one path:

```bash
# Without a Nebius account: use the privately delivered personal key.
npa login --endpoint https://team.example.invalid --token-file ./workbench.key

# With a human Nebius account: complete the official browser sign-in.
npa login --endpoint https://team.example.invalid --nebius
```

Running `npa login` in a terminal guides you through missing choices. Login
verifies your access before saving anything, displays your workspace, cluster,
and GPU allowance, and chooses placement defaults when there is exactly one
choice. For multiple choices, add `--workspace` and `--cluster`. A reader can
sign in without an execution allocation.

Then both authentication methods use the same commands:

`submit` starts execution and returns an ID. Set `RUN_ID` to that returned ID;
`runs` lists your executions, while `run RUN_ID` inspects one existing execution.

```bash
npa workbench whoami
npa workbench submit --spec workflow.yaml
npa workbench runs
npa workbench run "$RUN_ID" --action status
npa workbench run "$RUN_ID" --action logs
npa workbench run "$RUN_ID" --action artifacts
```

The CLI saves private connection profiles under `$NPA_CONFIG_DIR/team` (default
`~/.npa/team`). Personal keys are copied only after successful verification, to
mode-0600 files inside a mode-0700 directory. Nebius sessions retain the selected
CLI profile, never a copied IAM token. Each subsequent command asks the official
Nebius CLI for a current token; its refresh credential remains managed by that
CLI. If browser authentication is needed again, the command asks you to rerun
`npa login` without attempting the workload.

Use `--profile research` to name a connection; successful login makes it active.
Use `--profile research` on a Workbench command to select it explicitly. Saved
credentials are never sent to a different `--endpoint` without an explicit new
credential. For a private service certificate, supply your administrator's
`--ca-file` at login; TLS verification remains enabled.

`npa logout` removes the active local connection and its saved personal key. It
does not revoke the server key, sign out the Nebius CLI, or cancel running jobs.
Other named connections remain available by explicit selection. Both `npa login`
and `npa logout` support `--output-format json` for agents and scripts. The older
`npa workbench team` commands remain compatible with the same saved connections.

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

These commands use the database under the configuration's `state_dir`. Run them
in the operator environment that mounts the service's authoritative persistent
state; passing a local config file does not remotely administer the service.
See [initial account setup](team-access.md#administrator-establish-local-accounts-and-policy)
for the bootstrap order and runtime paths.

```bash
npa workbench access account create --config /private/npa-team/team.yaml \
  --name researcher-a --group researchers
npa workbench access account create --config /private/npa-team/team.yaml \
  --name reviewer-a --group reviewers
npa workbench access account list --config /private/npa-team/team.yaml

npa workbench access account issue-key --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --output-file /private/key-delivery/researcher-a.key
```

Deliver the key file through an approved private channel. It is a bearer
credential: never commit it, include it in a URL, paste it into a workflow, or
share it between people. The team CLI reads the mode-0600 file directly with
`--token-file`, so the key does not need to be exported into the shell
environment:

```bash
TEAM_KEY_FILE="$HOME/.config/npa/team.key"
npa login --endpoint https://team.example.invalid --token-file "$TEAM_KEY_FILE"
npa workbench whoami
npa workbench runs
```

See [team access](team-access.md#user-sign-in-submit-and-inspect-work)
for submit, status, logs, artifacts, cancellation, recovery, HTTP, and SDK
examples.

## Change membership or revoke access

Replace group membership with one or more `--group` options, remove every group
with `--clear-groups`, or disable the whole account. A disabled account cannot
use any of its keys or a linked external identity.

```bash
npa workbench access account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --group reviewers
npa workbench access account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --clear-groups
npa workbench access account revoke-key --config /private/npa-team/team.yaml \
  --key-id "$WORKBENCH_KEY_ID"
npa workbench access account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --disabled
```

Revocation blocks that key on the next request. Disabling blocks all local keys
and linked identities. Neither operation automatically cancels an already
admitted run nor rotates the allocation's storage credentials; cancel a known
run with `npa workbench access stop-run` and revoke workload storage credentials
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
npa workbench access account link --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --issuer "$EXTERNAL_ISSUER" --subject "$EXTERNAL_SUBJECT"
```

This preserves the local user ID, allocations, namespaces, runs, and storage
scope. It does not match email addresses, import groups, synchronize a
directory, create new accounts, or establish a browser-login path. Unlinked
external identities are denied, and local group and allocation policy remains
the authorization source.

To change providers, remove the exact old link and then link the new verified
identity. `unlink` also accepts a retired issuer after the service configuration
has changed. It preserves local keys, grants, namespace placement, and run
ownership; the removed identity fails authentication on its next request.
Already admitted work continues unless separately cancelled.

```bash
npa workbench access account unlink --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --issuer "$OLD_ISSUER" --subject "$OLD_EXTERNAL_SUBJECT"
```

Repeating the exact removal returns `unlinked: false`. A mismatched account,
issuer, or subject never removes another link. Disabling the account first is
optional; administrators can also unlink a disabled account during offboarding.

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

`npa login --nebius` uses the official Nebius CLI and identity provider to
complete browser sign-in. Select an existing human profile with
`--nebius-profile`; a single human profile is selected automatically. If no
human profile exists, NPA creates an isolated official CLI configuration for
Workbench. Neither path changes the global Nebius CLI default, so an operator
VM can keep its service-account profile. Workbench does not host an OAuth
callback, Keycloak UI, or browser login page.

```bash
npa login --endpoint https://team.example.invalid --nebius
npa workbench whoami
```

The official CLI opens the local browser by default. On a remote machine, use
`--no-browser --ssh-host YOUR_VM_ALIAS`: NPA prints the official sign-in URL and
the exact loopback callback-forward command for the laptop. An agent can open
the URL and establish that scoped tunnel; the person completes any password,
account selection, or MFA required by their identity provider. Tokens and
returned callback URLs must not be pasted into chat. An authenticated Nebius
account still needs the administrator's explicit Workbench link below.

For every such bearer request, Workbench calls the fixed HTTPS Nebius IAM
ProfileService endpoint (`GET https://api.nebius.cloud/iam/v1/profiles`) with
normal TLS verification, no redirects, and a bounded timeout. It accepts only a
human `userProfile` whose global account and exactly one membership in the
configured tenant are `ACTIVE`; service accounts, anonymous profiles, malformed
responses, unavailable providers, and federation mismatches are denied. The
provider-verified `tenantUserAccountId` is the external subject and the fixed
issuer is `https://api.nebius.cloud`.

To obtain the subject for that link, use the same human CLI profile and
configuration that completed browser sign-in. Set `NEBIUS_HUMAN_PROFILE` and
`NEBIUS_HUMAN_CONFIG` to that profile name and configuration path, then request
the current user's profile without exporting an IAM token:

```bash
umask 077
install -d -m 700 "$HOME/.config/npa-private"
nebius --config "$NEBIUS_HUMAN_CONFIG" --profile "$NEBIUS_HUMAN_PROFILE" \
  --format json iam profile get > "$HOME/.config/npa-private/nebius-human-profile.json"
```

The CLI's JSON uses snake-case fields: require `user_profile.user_account_state`
to be `ACTIVE`, then find exactly one entry in `user_profile.tenants` whose
`tenant_id` matches the installation. Its `tenant_user_account_state` must also
be `ACTIVE`; its `tenant_user_account_id` is the value for
`NEBIUS_TENANT_USER_ACCOUNT_ID`. Check any configured federation restriction as
well. If that tenant membership is missing or inactive, the tenant
administrator must establish it before Workbench SSO can work. Keep this
identity report private; the service independently verifies the provider on
every bearer request.

Link that exact provider-verified subject to the existing Workbench user. Do not
substitute an email address, display name, project role, or cloud group:

```bash
npa workbench access account link --config /private/npa-team/team.yaml \
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
