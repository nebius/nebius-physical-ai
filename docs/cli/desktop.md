# `npa tools desktop`

## Command Tree

```text
Usage: npa tools desktop [OPTIONS] COMMAND [ARGS]...

Persistent Ubuntu desktop with VS Code and Codex on an existing VM.

Options
--help  Show this message and exit.
Commands
setup  Install a private desktop, preserving existing credentials and sessions.
display  Change desktop density without restarting applications.
status  Inspect desktop services and recovery records without disclosing secrets.
optimize  Reduce desktop input latency without changing resolution or restarting apps.
public-access  Enable HTTPS with a trusted IP certificate and a separate strong login.
open  Open the desktop using public HTTPS or a private SSH tunnel.
chat-setup  Install authenticated mobile chat on the existing desktop HTTPS gateway.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `setup` | Install a private desktop, preserving existing credentials and sessions. |
| `display` | Change desktop density without restarting applications. |
| `status` | Inspect desktop services and recovery records without disclosing secrets. |
| `optimize` | Reduce desktop input latency without changing resolution or restarting apps. |
| `public-access` | Enable HTTPS with a trusted IP certificate and a separate strong login. |
| `open` | Open the desktop using public HTTPS or a private SSH tunnel. |
| `chat-setup` | Install authenticated mobile chat on the existing desktop HTTPS gateway. |

## Examples

```bash
npa tools desktop --help
npa tools desktop setup --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `desktop`.
