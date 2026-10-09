# Workbench access without personal Nebius accounts

Start with administrator-managed Workbench accounts and personal access keys.
Neither Nebius user accounts nor SSO nor Keycloak is required. The existing
gateway holds the account database on its persistent volume; there is no extra
identity server or VM. Administrators connect clusters and provision each
person's scoped workload storage credentials.

## Start with local accounts

Generate one UUID for `account_namespace` and retain it for the lifetime of the
installation. Add these settings to the private configuration from the
[team deployment guide](team-access.md):

```yaml
account_namespace: 3e1f832d-b0e5-426b-97aa-7535f1c6bbfa # example; generate your own once
browser_login:
  public_url: https://workbench.example.com
```

Omit `identity` and `browser_login.client_id`. The portal offers an access-key
sign-in form. Omitting `browser_login` retains API/CLI/SDK authentication without
serving a login page. These are persistent accounts, not a persona selector.

Run account administration on the server against its authoritative state:

```bash
npa workbench team account create --config /private/team.yaml --name alice --group researchers
npa workbench team account issue-key --config /private/team.yaml --user "$WORKBENCH_USER_ID" --output-file /private/alice-key
npa workbench team account list --config /private/team.yaml
```

The create response contains the permanent user ID. Use it in the workspace's
`subject` grants and allocations; group grants refer to local groups.
Membership does not allocate compute or storage. The key command creates a
new mode-0600 file and prints only its credential ID and path. Deliver the key
privately to that person. API/CLI callers use it as a bearer credential
(`NPA_TEAM_TOKEN`); SDK callers pass it to `TeamClient`.

Browser users enter the key once to obtain a Secure, HttpOnly session cookie.
The key never enters a URL or browser storage and is not retained in the
browser session. The database stores only SHA-256 hashes of randomly generated
256-bit keys, not passwords or recoverable key values.

```bash
npa workbench team account revoke-key --config /private/team.yaml --key-id "$WORKBENCH_KEY_ID"
npa workbench team account update --config /private/team.yaml --user "$WORKBENCH_USER_ID" --disabled
npa workbench team account update --config /private/team.yaml --user "$WORKBENCH_USER_ID" --group reviewers
```

Revocation rejects the key and its browser sessions on the next request.
Disabling the person also rejects linked SSO logins and stops new workflow
waves. Local group changes apply on the next request or workflow wave.
Revoking one key does not disable the person or stop already admitted jobs.
Job cancellation and storage-key revocation remain separate offboarding steps.
There is no self-service signup, email recovery or password database; the
operator issues a replacement key when needed. Back up `accounts.sqlite3`
together with `team.sqlite3`, private configuration and scheduler state.

## Add SSO later without changing ownership

Keep `account_namespace`, the account database, allocations and user IDs.
Configure the optional external provider below and restart the gateway. After
verifying the person's exact issuer and immutable provider subject, link it:

```bash
npa workbench team account link --config /private/team.yaml --user "$WORKBENCH_USER_ID" --issuer "$OIDC_ISSUER" --subject "$OIDC_SUBJECT"
```

Both login methods now resolve to the same person, namespaces, jobs and data.
Unlinked SSO users are denied. Matching email addresses or names never create
a link. Local groups and workspace grants remain authoritative: enabling SSO
does not import external roles or group membership. Retain local keys or
explicitly revoke them after verifying SSO access. The SDK exposes the same
administration through `Accounts`, `issue_key_file` and `link_identity` in
`npa.sdk.workbench.team`.

Legacy installations configured only with `identity` continue to use external
issuer/subject ownership and external group claims. Adding `account_namespace`
to an already-used legacy installation is not an automatic migration: do not
change its ownership domain while it has existing runs or allocations.

## Connect an optional identity provider

Register an OpenID Connect application with authorization-code flow and S256
PKCE. Set its exact callback to `https://<workbench-host>/auth/callback` and
restrict redirects to that address. Use the application client ID as Workbench's
configured audience. The provider must expose HTTPS discovery and JWKS endpoints
and issue signed ID tokens containing `iss`, `aud`, `sub`, `iat`, `exp`, and
`nonce`. A groups claim, when present, must be a list of strings. Individual
subject grants also work without groups.

Add the following to private team configuration, replacing example values:

```yaml
identity:
  issuer: https://identity.example.com/realms/research
  audience: workbench
  jwks_url: https://identity.example.com/realms/research/protocol/openid-connect/certs
  groups_claim: groups
  algorithms: [RS256]
browser_login:
  public_url: https://workbench.example.com
  client_id: workbench
  scopes: [openid, profile, email]
```

`browser_login` is optional and disabled when omitted. `public_url` must be the
HTTPS origin without a path, query, or credentials. `client_id` must match
`identity.audience` when an external provider is enabled. The default scopes are `openid`, `profile`, `email`, and
`groups`; override them with the provider's supported scopes. For a confidential
client, set `browser_login.client_secret_file` to an absolute path to a mode-0600
file containing the client secret. Public clients use PKCE without that file.
Discovery must match the configured issuer and JWKS URL. Authentication settings
require a service restart; workspace grants remain reloadable.

Serve the gateway behind HTTPS as described in [team deployment](team-access.md).
Opening its root URL shows the live access portal. Sign-in redirects to the
identity provider; the callback verifies state, nonce, signature, issuer,
audience, expiration, and authorized party. Tokens stay on the server. The
browser receives only an opaque Secure, HttpOnly, SameSite cookie. Writes with
that cookie require the configured Origin and `X-Workbench-CSRF` proof from
`GET /v1/me`. Existing bearer-token API, CLI, and SDK clients continue to work.

The portal shows verified groups, permitted workspaces, allocation status, and
personal run access. It has no impersonation selector or administrator controls.
It does not launch GPU jobs. Use the existing team submit API, CLI, or SDK for
execution once the administrator has enrolled the person's allocation.

## Optional Keycloak setup

Using the [Keycloak account setup guide](https://www.keycloak.org/getting-started/getting-started-zip):

1. Create an application realm, separate from the administrative master realm.
2. Create users with their own credentials. Create the external groups that
   should receive Workbench access, and add the intended people to those groups.
3. Register the Workbench OpenID Connect client with standard flow enabled,
   S256 PKCE required, and the exact HTTPS callback. Password grants and service
   account authentication are not needed for researchers' browser login.
4. Add a **Group Membership** protocol mapper to the client or its default
   client scope. Emit `groups` into ID tokens. Choose full group paths or short
   names deliberately, and use exactly those values in Workbench grants.
5. Configure the issuer, audience, keys, browser origin, and supported scopes
   above. Never infer a person's identity from an email or a display name; use
   the signed issuer plus immutable subject.

For example, a Workbench workspace grant can select `researchers` as `runner`
and `reviewers` as `reader`. These names must exist in the selected provider's
verified claims. Creating a group alone grants neither cloud access nor cluster
capacity. The administrator separately assigns the exact subject a personal
allocation and runs `npa workbench team enroll`.

Login can be qualified before allocating compute: use `clusters: {}` and
workspaces with `gpu_limit: 0`, `gpu_limits: {}`, and `allocations: []`. Membership,
permission checks, and private empty run lists are real; submissions are denied
until an allocation exists. This does not claim GPU, storage, or network
isolation has been tested.

## Session and offboarding behavior

Local access-key browser sessions expire after eight hours; revocation or user
disablement can invalidate them earlier. External browser sessions expire with
the identity provider's ID token. There is no
automatic refresh; sign in again to receive fresh group claims. Restarting the
single gateway process ends browser sessions but preserves the run ledger.
Signing out invalidates that Workbench session, including copied cookies. It
does not end the identity provider's other sessions or stop admitted jobs.

Removing a group in the identity provider takes effect when a fresh token is
used. For immediate Workbench denial, put the exact subject in
`disabled_subjects` or remove its local workspace grants. Terminating existing
jobs and revoking workload storage keys remain separate administrator actions,
as described in the [offboarding guide](team-access.md#submit-recover-and-offboard).

## Live browser qualification

For local accounts, `npa/tests/browser/team_local_live.test.cjs` signs in with
real access keys against a running HTTPS gateway, with no identity provider.
Set `NPA_TEAM_LOCAL_LIVE_CONFIG` to a private JSON file containing `endpoint`
and at least two `accounts`. Each account supplies `name`, `access_key`,
permanent `subject`, `groups` and `workspaces` (workspace names mapped to roles).
Optional `runs` verifies that real recorded run IDs appear in the account's
own listing. Optional `certificate` pins only the selected test certificate's
public key in Chrome; optional `screenshot` writes a private screenshot.
For a private endpoint, optional `proxy` selects an operator-provided HTTP
CONNECT proxy. The test still uses the gateway's HTTPS identity and origin.
Keep all of these files outside Git.

```bash
cd npa/tests/browser
NPA_TEAM_LOCAL_LIVE_CONFIG=/private/local-browser.json npm run team:local-live
```

The test checks wrong-key rejection, real permissions, secure cookies, CSRF
rejection, logout and mobile layout. It does not simulate an identity or job.

`npa/tests/browser/team_identity_live.test.cjs` performs real browser sign-ins
against running HTTPS services. Provide a private JSON file through
`NPA_TEAM_IDENTITY_LIVE_CONFIG` containing `endpoint` and an `accounts` list.
Each account supplies `username`, `password`, expected immutable `subject`,
`groups`, and a `workspaces` object mapping names to roles. It verifies actual
claims, workspace allow/deny results, CSRF rejection, secure cookies, logout,
and mobile layout. No requests or identities are mocked.

```bash
cd npa/tests/browser
NPA_TEAM_IDENTITY_LIVE_CONFIG=/private/browser-live.json npm run team:identity-live
```

Optional `local_certificate` pins one local test certificate in the test
browser and requires a `localhost` Workbench endpoint. It does not disable TLS
verification in the gateway or change OS trust. Optional `screenshot` writes a
private screenshot after login. Keep this configuration, passwords, identity
records, and screenshots outside Git. An omitted live configuration skips the
test and supplies no live evidence.

Local Keycloak development mode and its embedded database support a private
demonstration. A production deployment needs the identity provider's supported
database, certificates, operational controls, and availability configuration.
Workbench's existing single-process deployment and live cluster qualification
requirements still apply.
