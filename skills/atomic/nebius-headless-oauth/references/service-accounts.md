# Service accounts for unattended authentication

A service account gives the VM its own identity and permissions. Initial setup
requires an authenticated administrator for the target project; subsequent CLI
authentication needs no browser or copied human OAuth cache. Nebius documents
the [authorized-key flow](https://docs.nebius.com/cli/no-browser) and
[VM attachment](https://docs.nebius.com/cli/compute-vm).

## Choose project permissions

| Project role | Intended use | IAM distinction |
| --- | --- | --- |
| `viewer` | Read resource information | Cannot create resources or manage permissions |
| `editor` | Create and manage workload resources | Can create service accounts; cannot create authorized keys or manage groups and grants |
| `admin` | Automate project setup, including IAM | Can manage project groups, grants, memberships, and authorized keys |

Use narrower service-specific roles where they cover the workload. See the
[role matrix](https://docs.nebius.com/iam/authorization/roles). S3 access keys
are separate from the authorized key used for Nebius CLI authentication.

For project administration, create the service account and a custom group in
the selected project, then bind the group to that exact project with `admin`:

```bash
nebius iam group create --parent-id "<project-id>" --name "<group-name>"
nebius iam access-permit create \
  --parent-id "<group-id>" --resource-id "<project-id>" --role admin
nebius iam group-membership create \
  --parent-id "<group-id>" --member-id "<service-account-id>"
```

Run these with the explicit operator profile. Capture returned IDs privately
and verify each object's parent and the permit's resource and role. An
administrator of that project can perform this setup; tenant-wide admin is
unnecessary. The default `admins` and `editors` groups have tenant-wide scope.
Follow the [custom-group guide](https://docs.nebius.com/iam/authorization/groups/manage)
for project-scoped bindings. Use `editor` instead when IAM administration is
outside the workload's scope.

## Configure the VM identity

For a new Nebius VM, specify its service account at creation. The CLI can use
the token supplied at `/mnt/cloud-metadata/token`. Preserve an existing
npa-agent VM's attached identity and profile.

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

3. Run the profile-scoped Workbench Nebius preflight from the parent skill.
   The CLI uses the private key to obtain IAM access tokens without OAuth.
   Separately verify the actual operations the workload requires.

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
the PR's Workbench Nebius preflight passed without browser interaction.

With project `editor`, compute listing and service-account creation succeeded;
group creation and authorized-key creation returned permission denial. With
project `admin`, group, permit, membership, and authorized-key creation
succeeded; tenant-level group listing remained denied. This verifies these
specific permissions and the tested scope boundary. VM attachment and workload
provisioning were not exercised in this test. All 11 created IAM objects were
verified absent afterward; temporary key files, profiles, and the exact test
account's cached token were removed. Unrelated cache entries were preserved
during cleanup.
