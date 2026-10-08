# Workbench access without personal Nebius accounts

People sign in with the organization's identity provider. Workbench maps their
verified subject or group membership to workspace permissions. The administrator
connects the clusters and provisions scoped workload storage credentials. The
researchers do not need Nebius user accounts, Nebius group membership, or cloud
administrator credentials.

For teams without existing company SSO, a separately operated Keycloak instance
can hold real accounts and groups. Workbench remains an application using those
identities. This is the same login protocol used when connecting company SSO;
the identity provider is not bundled into the Workbench service.

## Connect an identity provider

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
`identity.audience`. The default scopes are `openid`, `profile`, `email`, and
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

## Keycloak account and group setup

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

Browser sessions expire with the identity provider's ID token. There is no
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
