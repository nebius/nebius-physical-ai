---
name: nebius-service-account-auth
description: Set up or verify unattended Nebius CLI authentication on a VM or CI runner using a project-scoped service account, an attached VM identity, or an authorized-key profile. Use human OAuth skills when the operator needs their personal identity.
---

# Nebius service-account authentication

A service account gives the VM its own identity and permissions. Initial setup
requires an authenticated administrator for the target project; subsequent CLI
authentication needs no browser or copied human OAuth cache. Nebius documents
the [authorized-key flow](https://docs.nebius.com/cli/no-browser) and
[VM attachment](https://docs.nebius.com/cli/compute-vm).

## Choose the identity before changing IAM

Use this route for unattended workloads or a requested service-account setup.
For interactive work as the human operator, prefer
[vm-nebius-auth](../vm-nebius-auth/SKILL.md) when SSH forwarding is available,
or [nebius-headless-oauth](../nebius-headless-oauth/SKILL.md) when a private
terminal is available but callback forwarding is not. Human OAuth needs no
new IAM identity and retains the operator's existing permissions; a service
account has its own grants and credential lifecycle.

For a readiness-only request, run the selected profile's Nebius preflight and
report its result without changing credentials or IAM. Perform authorized IAM
setup on the already authenticated operator machine. Resolve the exact project
and operator profile privately, verify the project's
parent, and run `npa workbench health preflight --checks nebius --json` with
`NPA_NEBIUS_PROFILE` selecting that operator profile before creating resources.
Reuse an explicitly selected, verified account when appropriate. Record the
provider-created IDs and their ownership before the next mutation. Capture CLI
responses privately; keep identities, keys, tokens, and infrastructure details
out of chat, Git, logs, and public validation reports.

## Choose project permissions

| Project role | Intended use | IAM distinction |
| --- | --- | --- |
| `viewer` | Read resource information | Cannot create resources or manage permissions |
| `editor` | Create and manage workload resources | Can create service accounts; cannot create authorized keys or manage groups and grants |
| `admin` | Automate project setup, including IAM | Can manage project groups, grants, memberships, and authorized keys |

Use narrower service-specific roles where they cover the workload. See the
[role matrix](https://docs.nebius.com/iam/authorization/roles). S3 access keys
are separate from the authorized key used for Nebius CLI authentication.

The service account does not need `admin` merely to authenticate. A project
administrator is needed for initial group/permission and authorized-key setup.
Choose the service account's runtime role for its actual work; use `admin`
when project IAM administration is requested, not as a login workaround.

Create an account and a custom group in the selected project when needed.
Replace `<role>` with the chosen project role, then add the account to the group:

```bash
nebius --profile "<operator-profile>" iam service-account create \
  --parent-id "<project-id>" --name "<service-account-name>"
nebius --profile "<operator-profile>" iam group create \
  --parent-id "<project-id>" --name "<group-name>"
nebius --profile "<operator-profile>" iam access-permit create \
  --parent-id "<group-id>" --resource-id "<project-id>" --role "<role>"
nebius --profile "<operator-profile>" iam group-membership create \
  --parent-id "<group-id>" --member-id "<service-account-id>"
```

Capture returned IDs privately and verify each object's parent, the permit's
resource and role, and the membership's account. An administrator of that
project can perform this setup; tenant-wide admin is
unnecessary. The default `admins` and `editors` groups have tenant-wide scope.
Follow the [custom-group guide](https://docs.nebius.com/iam/authorization/groups/manage)
for project-scoped bindings. Use `editor` instead when IAM administration is
outside the workload's scope.

## Configure the VM identity

For a new Nebius VM, specify its service account at creation. The CLI can use
the token supplied at `/mnt/cloud-metadata/token`. Preserve an existing
npa-agent VM's attached identity and profile. This avoids distributing a
private authorized key. It does not authorize creating, replacing, or
reattaching a VM solely to complete authentication.

If a dedicated CLI profile is needed for an already attached identity, check
the installed CLI's help for `profile create "<service-account-profile>"`
with `--endpoint api.nebius.cloud --parent-id "<project-id>"`
and `--token-file /mnt/cloud-metadata/token --skip-auth`. Verify the returned
identity matches the attached account; token-file presence alone does not
establish authentication or permissions.

For an existing development VM without an attached service account:

1. An administrator creates the service account, project group, role permit,
   and membership, then generates an authorized key:

   ```bash
   umask 077
   nebius --profile "<operator-profile>" iam auth-public-key generate \
     --parent-id "<project-id>" --service-account-id "<service-account-id>" \
     --output "<private-credentials-file>"
   ```

2. Deliver only that credentials file through a private SSH channel or secret
   manager. Keep it readable only by its intended user and out of Git, chat,
   command arguments, logs, and images. Create a dedicated profile on the VM:

   ```bash
   nebius --no-browser --no-check-update profile create "<service-account-profile>" \
     --endpoint api.nebius.cloud --parent-id "<project-id>" \
     --service-account-file "<private-credentials-file>" --skip-auth
   ```

Profile creation activates the new profile. When adding a profile alongside an
existing operator profile, preserve the previous active selection and restore
it with `nebius profile activate "<previous-profile>"`; explicitly select the
service-account profile for its workload. Keep the key available for future
token exchanges. Rotate it through the administrator's approved credential
lifecycle rather than copying the operator's human CLI cache.

## Verify authentication and the intended permission

On the target machine, verify that the explicit profile's `iam whoami` result
matches the selected service account; capture the result privately. Then run:

```bash
env -u NEBIUS_IAM_TOKEN -u NEBIUS_IAM_TOKEN_FILE \
  -u NPA_NEBIUS_IAM_TOKEN \
  NPA_NEBIUS_PROFILE="<service-account-profile>" \
  npa/.venv/bin/npa workbench health preflight --checks nebius --json
```

PASS requires identity verification and IAM token minting. The CLI uses the
private key or attached identity without human OAuth. If it starts an OAuth
prompt, stop and correct the profile selection or auth type. Do not compensate
by transferring a human token. Separately verify the requested project
operation: a resource read establishes read access only, and a project-admin
claim needs an authorized IAM operation on owned test objects. Authentication
does not establish quota, capacity, S3 access, or workload success.

## Cleanup for disposable validation

For disposable tests, record provider-created IDs privately, delete only the
owned keys, memberships, permits, groups, and accounts, and verify their absence.
Remove temporary credential files and profiles from both machines and check
for separately cached tokens. CLI 0.12.211 also caches service-account tokens
outside a temporary `--config` directory. Remove only entries proven to belong
to the exact test account; preserve unrelated entries. Deleting a profile or
its authorized-key file alone does not clear that cache.

## Live verification

On 2026-09-14, the authorized-key profile was exercised on a real development
VM with CLI 0.12.211, using CLI 0.12.254 on the operator machine for setup.
Identity matched the new service account, IAM token minting succeeded, and
the current Workbench Nebius preflight passed without browser interaction.

With project `editor`, compute listing and service-account creation succeeded;
group creation and authorized-key creation returned permission denial. With
project `admin`, group, permit, membership, and authorized-key creation
succeeded; tenant-level group listing remained denied. This verifies these
specific permissions and the tested scope boundary. VM attachment and workload
provisioning were not exercised in this test. All 11 created IAM objects were
verified absent afterward; temporary key files, profiles, and the exact test
account's cached token were removed. Unrelated cache entries were preserved
during cleanup.
