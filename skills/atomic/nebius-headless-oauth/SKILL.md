---
name: nebius-headless-oauth
description: Complete human Nebius CLI OAuth on a headless operator/dev machine when the browser is elsewhere and SSH port forwarding is unavailable, using a validated manual loopback callback.
license: MIT
---

# Nebius headless OAuth

Use the operator's signed-in browser to authenticate a Nebius CLI profile on
another machine. Keep one CLI login process alive, open its authorization link,
then copy the browser's final loopback URL into a private terminal on the CLI
machine. The bundled helper delivers that callback to the waiting CLI.

This adapts the operator-contributed `nebius-headless-oauth` skill (MIT).
[Nebius documents](https://docs.nebius.com/cli/no-browser) `--no-browser`, the
dynamic loopback redirect, and SSH forwarding. Manual callback delivery is this
skill's adaptation, not an upstream CLI command or an NPA agent chat feature.

## Choose the authentication path

- For a readiness-only request, run
  `npa workbench health preflight --checks nebius --json` and stop. A failure
  alone does not establish an expired login; distinguish missing configuration,
  denied access, and network failures before starting OAuth.
- When SSH forwarding is available, use
  [vm-nebius-auth](../vm-nebius-auth/SKILL.md). It returns the authorization link
  and exact tunnel command without transferring the callback code manually.
- For unattended VM or CI authentication, use a
  [project-scoped service account](../nebius-service-account-auth/SKILL.md).
  After initial setup, an attached VM identity or an authorized-key profile works
  without browser interaction. Choose its role for the intended operations.
  For interactive work that needs the operator's own identity, human OAuth
  avoids creating a separate account and maintaining an authorized key.
- Use this manual method for an authorized human login when forwarding is
  unavailable and the operator can enter hidden input in a terminal on the CLI
  machine, for example an existing private browser terminal. Chat alone does not
  provide that input channel. Never ask for a callback URL, authorization code,
  access token, or refresh token in chat or a tool-call argument.
- Preserve an npa-agent VM's attached-service-account metadata profile. This
  skill does not configure CI credentials, S3 keys, or Token Factory keys.

## Start one login attempt

1. Check `command -v nebius` and the installed CLI's `profile --help` and
   `iam whoami --help`. Establish the intended human profile from non-secret
   metadata; do not dump CLI configuration. Use the same explicit profile in
   every command. Login authorization already given in this task is sufficient.
2. On the CLI machine, run the following in a persistent terminal. Replace the
   quoted profile placeholder. Ambient IAM overrides are removed only for this
   process so they cannot impersonate a successful profile login:

   ```bash
   env -u NEBIUS_IAM_TOKEN -u NEBIUS_IAM_TOKEN_FILE \
     -u NPA_NEBIUS_IAM_TOKEN \
     nebius --profile "<profile>" --no-browser --no-check-update \
     iam whoami --format json > /dev/null
   ```

   The identity result goes to `/dev/null`; the CLI's authorization prompt
   remains visible on stderr. If it exits successfully without prompting, the
   profile is already authenticated. Continue to verification without starting
   another login. Do not use an unsuppressed `iam get-access-token` command to
   trigger authentication, or enable debug output or terminal recording.
3. If the profile is established to be absent, create the selected human profile
   with `nebius --no-browser --no-check-update profile create "<profile>"` in a
   persistent terminal, following the official profile prompts. Do not overwrite
   or delete an existing profile. Some CLI versions defer OAuth until first use;
   run the profile-scoped identity command above if creation did not start it.
4. Keep exactly one waiting OAuth process. The current supported link is HTTPS
   on `auth.nebius.com`, with an S256 PKCE challenge, `state`, and a
   `redirect_uri` of `http://127.0.0.1:<port>` (an optional trailing slash is
   accepted). Preserve this original link for the helper; do not construct one
   or reuse a previous attempt. If this CLI emits another shape, stop and review
   its documented flow instead of relaxing the validator.

## Open, copy, and complete

Give the operator only the authorization link from the active CLI process, in
the private session, with these instructions:

> Open this link in the browser where you are signed in to Nebius. Complete any
> account or SSO prompt. If the browser cannot connect to `127.0.0.1`, copy the
> entire final URL from its address bar. Paste it into the hidden callback
> prompt on the CLI machine. It contains a one-time login code; keep it out of
> chat, screenshots, shell commands, and saved notes.

Leave the CLI process running. In a second private terminal on that same
machine and in the same network namespace, run from the repository root:

```bash
npa/.venv/bin/python \
  skills/atomic/nebius-headless-oauth/scripts/relay_callback.py
```

The operator first pastes the original authorization link, then the final
callback URL. Both prompts disable echo and refuse a non-terminal input stream.
The helper checks the original redirect, matching port and state, and exactly
one nonempty authorization code. It sends one HTTP GET directly to the pinned
IPv4 loopback listener, ignoring proxy settings and following no redirects.
Only a generic delivery status is printed; URLs, response bodies, and exceptions
containing credentials are not printed or written to disk.

A successful delivery only proves that the callback listener returned HTTP 200.
Wait for the original CLI process to exit with status zero, then verify the
same profile:

```bash
env -u NEBIUS_IAM_TOKEN -u NEBIUS_IAM_TOKEN_FILE \
  -u NPA_NEBIUS_IAM_TOKEN \
  NPA_NEBIUS_PROFILE="<profile>" \
  npa/.venv/bin/npa workbench health preflight --checks nebius --json
```

Require both identity verification and IAM token minting to pass. Report those
results without account identifiers, email addresses, profile names, or tokens.
Authentication does not prove project permissions, quota, capacity, or model
access. Human OAuth credentials stay in the Nebius CLI's private profile/cache;
do not copy them into `~/.npa/credentials.yaml`.

## Failure and cleanup

If the operator cancels, the CLI expires, callback validation/delivery fails,
or verification starts another login, interrupt and reap the process belonging
to this attempt. The helper does not own or stop the CLI process. Stop the
helper too if it is still waiting. Do not use a broad process-name kill.

For disposable live tests, audit the CLI's credential cache separately from its
profile configuration. On CLI 0.12.211, a temporary `--config` and subsequent
`profile delete` still left the test credential in `~/.nebius/credentials.yaml`.
Use a unique test profile, inspect cache metadata privately, and remove only the
entry belonging to that test after reaping its processes. Verify that unrelated
profiles and cached credentials remain unchanged; never dump or delete the
shared credential store. Use a current Workbench checkout for the final Nebius
preflight check; older checkouts may not recognize `--checks nebius`.

Discard the URLs and clear the clipboard and any saved callback in browser
history. Hidden input prevents terminal echo; it does not disable session
recording or clear browser/clipboard history for the operator. A retry uses a
fresh process, link, port, state, and code. Do not retry a delivered callback or interpret a browser
error page as proof of authentication. Leave the CLI's own authentication
timeout unchanged unless the operator requests a different value.

For development validation, run
`npa/.venv/bin/python -m pytest npa/tests/unit/test_headless_oauth_relay.py -q`.
These tests run the real helper in a private pseudo-terminal against a local
HTTP listener, including success, rejected callbacks, server errors, and
cancellation. They check echo, output, local files, and child-process cleanup.
They do not prove a live Nebius SSO login; that requires an operator to complete
the browser step.
