# `npa login`

## Command Tree

```text
Usage: npa login [OPTIONS]

Verify access and remember a Workbench connection for subsequent commands.

Args:
endpoint, profile: Team HTTPS endpoint and saved connection name.
token_file, nebius, nebius_profile, nebius_config: Personal key or Nebius sign-in.
workspace, cluster: Optional placement defaults, verified against access.
ca_file: Optional trusted private CA; TLS verification remains enabled.
no_browser, ssh_host: Local-browser or remote callback instructions.
output_format: Human text or credential-free JSON.
Returns:
None; prints verified access after saving the session.
Raises:
TeamError: Login, placement, configuration, or persistence fails.

Options
--endpoint  <str>  [env var: NPA_TEAM_ENDPOINT]
--token-file  <path>  Personal key supplied by your administrator.
--nebius  Sign in with your human Nebius account.
--nebius-profile  <str>  Human Nebius CLI profile; preserves its global default.
--nebius-config  <path>  Optional isolated Nebius CLI configuration.
--profile  <str>  Saved connection name; defaults to the active connection.
--workspace  <str>  Preferred authorized workspace.
--cluster  <str>  Preferred allocated cluster.
--ca-file  <path>  Administrator-provided CA for a private HTTPS service.
--no-browser  Print the official sign-in link instead of opening a browser.
--ssh-host  <str>  When logging in on a VM, print its loopback callback forward.
--output-format  <str>  text or json; credentials are never printed. [default: text]
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--endpoint` | <str>  [env var: NPA_TEAM_ENDPOINT] |
| `--token-file` | <path>  Personal key supplied by your administrator. |
| `--nebius` | Sign in with your human Nebius account. |
| `--nebius-profile` | <str>  Human Nebius CLI profile; preserves its global default. |
| `--nebius-config` | <path>  Optional isolated Nebius CLI configuration. |
| `--profile` | <str>  Saved connection name; defaults to the active connection. |
| `--workspace` | <str>  Preferred authorized workspace. |
| `--cluster` | <str>  Preferred allocated cluster. |
| `--ca-file` | <path>  Administrator-provided CA for a private HTTPS service. |
| `--no-browser` | Print the official sign-in link instead of opening a browser. |
| `--ssh-host` | <str>  When logging in on a VM, print its loopback callback forward. |
| `--output-format` | <str>  text or json; credentials are never printed. [default: text] |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa login --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `login`.
