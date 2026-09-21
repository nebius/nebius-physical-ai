# Persistent development desktop

[Operator tools](README.md) · [CLI reference](../cli/tools.md)

Run full VS Code, Codex, terminals, and CPU development work on an existing
Ubuntu AMD64 VM. The laptop displays the desktop; builds and tests run on the
VM. This capability lives under `npa tools desktop` and has a Python surface in
`npa.tools.desktop`. GPU workloads continue to use their own execution paths.

## Set up an existing VM

Start with SSH key access, passwordless sudo, and an existing Git checkout on
the VM. Inventory its free memory and disk and preserve existing work. Setup
installs XFCE, TigerVNC, a pinned noVNC viewer, Microsoft VS Code, and the Codex,
Python, and Ruff extensions. It keeps the desktop running across SSH disconnects
and reuses existing desktop and keyring passwords.

```bash
export DESKTOP_SSH_HOST='<existing-ssh-alias>'
npa tools desktop setup --ssh-host "$DESKTOP_SSH_HOST" --dry-run --json
npa tools desktop setup --ssh-host "$DESKTOP_SSH_HOST" \
  --repository-path '~/nebius-physical-ai' --dpi 192 --geometry 2880x1800 --json
npa tools desktop open --ssh-host "$DESKTOP_SSH_HOST"
npa tools desktop status --ssh-host "$DESKTOP_SSH_HOST" --json
```

The installer uses the publisher's signed Ubuntu repository for VS Code and
fetches noVNC 1.6.0 from its upstream repository. VS Code/Codex account login,
the NPA environment, Docker, and cloud credentials remain operator setup steps.
Setup preserves existing applications and does not restart a running desktop.
Refreshing its browser connection can be necessary after a viewer upgrade.

Without public access, `open` establishes a loopback-only SSH tunnel. Copy the
VNC password privately; on macOS:

```bash
ssh "$DESKTOP_SSH_HOST" 'cat ~/.local/share/nebius-desktop/password' | pbcopy
```

Paste it into the desktop password dialog. Linux application shortcuts usually
use Ctrl rather than Command. Closing the tab leaves the remote session running.
Use **Clipboard** to exchange text between the laptop and desktop.

## Sharp text at a comfortable size

Use the viewer's **Workspace** selector: **Comfortable**, **More space**, or
**Larger text**. This changes framebuffer size while preserving device-pixel
rendering and lossless Tight compression. **Full screen** gives the desktop more
room. The choice is stored in that browser; passwords are not stored there.

Keep one viewer connected at a time. All viewers share one Linux desktop, and
an older noVNC tab can shrink its framebuffer using CSS pixels while the desktop
still uses larger Retina fonts. That combination makes the interface look
zoomed in. Close older viewer tabs after upgrading.

Desktop font density is separate from browser workspace scale:

```bash
npa tools desktop display --ssh-host "$DESKTOP_SSH_HOST" --dpi 192
```

Use 192 DPI for a device-pixel-ratio-2 Retina display, or 96 for standard density.
The density command updates the active XFCE session without restarting VS Code.
At a 1512 × 900 CSS viewport, Retina mode needs roughly twice each dimension in
framebuffer pixels; the viewer toolbar takes part of the height. Raising JPEG
quality alone cannot correct a low-resolution framebuffer.

## Responsive clicking and typing

Setup disables XFCE compositing and GTK animations to reduce redundant rendering
on the CPU desktop. Apply the same persistent preferences to a running desktop:

```bash
npa tools desktop optimize --ssh-host "$DESKTOP_SSH_HOST" --dry-run --json
npa tools desktop optimize --ssh-host "$DESKTOP_SSH_HOST" --json
```

The change takes effect immediately, preserving applications, resolution, font
density, and lossless text rendering. `status` reports the applied performance
preferences. Network round-trip time still contributes to input latency.
Desktop effects can be re-enabled in XFCE's Settings Editor; running `setup` or
`optimize` reapplies the responsive defaults.

## Direct access through an external IP

Select a provider-verified static external IPv4 address already assigned to the
VM. An address reachable from a corporate or operator network may still be
unreachable from the public Internet. Check the assigned pool and external
routing before requesting a certificate. This command does not allocate or move
cloud addresses or modify security groups.

Allow inbound TCP 80 for certificate validation and the selected HTTPS port for
the desktop. Port 80 serves only ACME challenge files and returns 404 elsewhere.
The default HTTPS port is 8443, preserving unrelated services on port 443.

```bash
export DESKTOP_PUBLIC_IP='<verified-vm-external-ip>'
npa tools desktop public-access --ssh-host "$DESKTOP_SSH_HOST" \
  --public-ip "$DESKTOP_PUBLIC_IP" --https-port 8443 --dry-run --json
npa tools desktop public-access --ssh-host "$DESKTOP_SSH_HOST" \
  --public-ip "$DESKTOP_PUBLIC_IP" --https-port 8443 --json
npa tools desktop open --ssh-host "$DESKTOP_SSH_HOST"
```

Public access uses a dedicated nginx gateway with a strong generated login and
Certbot 5.8.0. It obtains a publicly trusted, short-lived IP certificate and
checks renewal twice daily, reloading the gateway after renewal. Public access
requires successful ACME validation; it never falls back to plaintext or an
untrusted certificate. See [Let's Encrypt's IP certificate guidance](https://letsencrypt.org/2026/03/11/shorter-certs-certbot/).
An existing nginx installation not owned by this tool is left for its operator
to configure separately.

The HTTPS username is `developer`. Copy its separate strong password:

```bash
ssh "$DESKTOP_SSH_HOST" 'cat ~/.local/share/nebius-desktop/public-password' | pbcopy
```

After HTTPS login, the viewer obtains the VNC credential through the same
protected origin. The page, WebSocket, and credential endpoint all require
authentication. Credentials never appear in URLs or ordinary status output.
Foreign browser origins are rejected, and the gateway disables response caching.
VNC and websockify remain bound to the VM's loopback interface.

## Mobile Codex chat

After `public-access` is configured, add a phone-friendly chat interface to the
same HTTPS gateway:

```bash
npa tools desktop chat-setup --ssh-host "$DESKTOP_SSH_HOST" \
  --connect-vscode --dry-run --json
npa tools desktop chat-setup --ssh-host "$DESKTOP_SSH_HOST" \
  --connect-vscode --json
npa tools desktop open --ssh-host "$DESKTOP_SSH_HOST" --chat
```

Open `/chat/` on the desktop's HTTPS origin and sign in with the same desktop
username and password. No VPN is required when the gateway uses a verified
external address. The layout supports iPhone-sized screens, a session search,
archived sessions, paginated history, and a composer that stays above the mobile
keyboard. It sends prompts, streams replies and tool activity, steers a running
turn, stops a turn, and presents command/file approvals and questions. Specialized
MCP and dynamic-tool requests remain in VS Code.

Choose **Model** and **Reasoning** above the composer. Choices come from the
signed-in Codex runtime and include only reasoning levels supported by that
model. Changes apply to the next turn of the same conversation and update the
connected VS Code picker. Changes made in VS Code update the mobile controls too.
The browser tab, favicon, chat header, and session list show when Codex is working;
an outstanding approval or question changes the tab title to **Needs input**.

Open the same conversation in either client to see prompts and replies sent
from the other. Mobile does not create a separate copy of the chat. The browser
provides these Codex conversation controls; the full editor, extensions, file
explorer, and terminal remain available through the desktop's VS Code window.

The browser and VS Code share one Codex app-server through a private Unix
socket. `--connect-vscode` backs up the existing VS Code settings and selects a
local transport adapter using `chatgpt.cliExecutable`. Reload the VS Code window
after its current work finishes. The adapter uses the installed extension's
Codex binary, so no second account login or API key is copied to the browser.
This integration uses Codex's experimental app-server API and development
executable setting; revalidate both clients after extension upgrades.

Existing sessions open in an older, independent Codex process are readable in
the mobile UI. Close the session in that client before reopening it for mobile
control. The service detects those writers rather than running two agents
against the same session log. New sessions and sessions opened in the connected
VDI VS Code window share live controls. Other SSH-based IDE windows are left
unchanged.

The `npa-codex-server.service` and `npa-codex-chat.service` user services survive
browser disconnects. Chat binds only to loopback, requires authentication on
every route, and rejects cross-origin mutations. The gateway protects the page,
assets, session history, and control endpoints. Prompts render as text; returned
HTML cannot execute scripts. No credentials are embedded in the web assets or
URLs. Chat history stays in the existing Codex home and its encrypted backup.

Repeated setup preserves the running Codex engine; it refreshes the lightweight
web service. Refresh browser tabs after an update. To undo IDE integration,
restore the saved `chatgpt.cliExecutable` setting from the private chat state and
reload the window after work finishes. Stop the shared engine only after its
active conversations have finished.

## Recovery

| Failure | Recovery |
| --- | --- |
| Browser or SSH disconnect | Reopen the same running desktop. |
| VM or desktop restart with disk intact | Enabled services restart; saved files and session history remain. Running processes and RAM do not. |
| Disk loss or corruption | Restore a READY disk snapshot and newer files from an independently retained encrypted backup. |

`setup` preserves recovery services and `status` reports existing backup and
snapshot verification records. It does not provision a backup repository or
cloud storage credentials. Configure these for the exact project using its
private storage settings, and keep the encryption key outside the VM.

A useful hourly restic backup covers existing registered Git worktrees' tracked
and nonignored untracked files, their Git metadata, Codex sessions/configuration,
and VS Code/desktop settings. Ignored datasets, environments, caches, and model
weights need their own artifact policy. Include authentication data only in the
encrypted private repository. Use SQLite's online backup API for Codex databases,
check the copies, and record their original paths; copying a live database
without its WAL can omit committed state.

Require a successful backup exit before updating the last-success record. Use a
persistent systemd timer and a lock to prevent overlapping backups. Retain
backups until an operator chooses a pruning policy. Hourly recovery can lose
changes since the last successful run; failures can make that interval longer.

Capture the exact boot disk with `nebius compute disk-snapshot create`, using
private runtime IDs. Reconcile asynchronous creation and wait for `READY`;
creation returning an operation ID is not readiness. An online snapshot is
crash-consistent, so databases can require additional application-level backups.

Prove recovery from a separate directory using independently retained keys.
Restore a probe, a tracked file, and an uncommitted file and compare hashes.
Check restored Codex databases, and stop Codex before replacing its databases
and session files. Avoid mixing restored databases with stale WAL/SHM files.
A selected-file restore is separate from a complete replacement-VM boot test.

For a lost VM, reuse its retained disk. For a lost disk, create a new disk from a
READY snapshot, boot a replacement VM, restore newer development files, and
recreate cloud identity/permissions. Never overwrite the only surviving work
copy during a recovery rehearsal.

## Operations and cleanup

Setup diagnostics stay on the VM in `~/.local/share/nebius-desktop/setup.log`.
The desktop and viewer use `nebius-desktop.service` and
`nebius-desktop-web.service` in the user's systemd manager. The optional system
gateway and certificate timer are `npa-desktop-gateway.service` and
`npa-desktop-cert-renew.timer`. Passwords and operational records remain under
the private desktop state directory.

To retire public access, stop and disable the gateway and its renewal timer,
then remove only their owned configuration and corresponding ingress rules.
Stop the desktop's user services only when ready to end its running session.
Preserve recovery keys and backups until explicitly retired. VM shutdown or
removal can interrupt unrelated work; use the
[teardown procedure](../../skills/atomic/teardown-and-cost/SKILL.md).
