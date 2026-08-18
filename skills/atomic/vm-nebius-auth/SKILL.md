---
name: vm-nebius-auth
description: Start, recover, or verify human Nebius CLI authentication on a remote operator/dev VM with no browser, a safe loopback callback tunnel, and secret-free identity/IAM verification. Use for remote CLI profile setup; do not replace an npa-agent VM's attached-service-account metadata profile.
---

# VM Nebius Authentication

Run on the operator/dev VM:

```bash
npa/.venv/bin/npa agent auth-profile \
  --ssh-host <operator-vm-host> --ssh-user <operator-vm-user> \
  --profile <profile>
```

The command strips ambient IAM-token variables, returns immediately for an
already-authenticated profile, or starts `nebius --no-browser` profile creation
and its first profile-scoped IAM probe in one PTY (CLI versions may defer OAuth
until first use). When the
CLI advertises exactly one official HTTPS browser URL and one loopback callback,
it prints the URL and an exact local-machine `ssh -N -L` command using the
runtime-selected port. Run the tunnel locally before opening the URL. Unsafe,
ambiguous, or malformed callbacks fail closed; cancellation and timeout stop the
child flow.

The browser callback completes the CLI profile. NPA then verifies `iam whoami`
and separately verifies that IAM can mint an access token, with command output
discarded. Never request, paste, print, log, return, or chat the IAM token.

Human interactive authentication is for an operator/dev VM or explicit
recovery. Keep the npa-agent VM on its attached service account and
`cursor-sa` metadata-token profile by default.

Verify changes with:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/unit/test_nebius_vm_auth.py \
  npa/tests/cli/test_agent_auth_profile.py -q
```
