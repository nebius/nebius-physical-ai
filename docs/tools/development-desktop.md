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
the desktop. Before HTTPS is configured, port 80 serves only ACME challenge
files and returns 404 elsewhere. Once HTTPS is ready, other HTTP requests
redirect to the configured secure address and port, preserving their path.
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
keyboard. Session-list titles and previews are shortened for fast refreshes;
opening a conversation retains its full messages. It sends prompts, streams
replies and tool activity, steers or stops a running turn, and presents command/file
approvals and questions. Specialized MCP and dynamic-tool requests remain in VS Code.

Tap **…** beside a chat, or in the open chat's header, to **Save name** or
**Archive**. Archiving preserves messages and browser drafts. Choose **Archived**
in the session list and use **… → Restore** to continue the same conversation.
Archived chats remain read-only until restored. A running turn must finish or
be stopped before archiving; renaming does not start or interrupt a turn.

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

## Local Mac sessions

Use the same mobile interface with Codex running on your Mac:

```sh
npa tools desktop chat-setup --local --dry-run --json
npa tools desktop chat-setup --local
npa tools desktop status --local --json
npa tools desktop open --local --chat
```

Local mode requires macOS, Node.js 22.13 or newer, npm, and an existing signed-in
Codex installation. It follows the native VS Code conversation owner and keeps
the same session IDs, so existing open chats can receive prompts, steering,
interruptions, model changes, and supported approval responses. It does not
replace the VS Code executable or require reloading an active window. Local
mode uses a private adapter for the installed extension's IPC protocol; verify
compatibility after extension upgrades. New or unowned chats use a separate
Codex app-server that belongs to the local service.

Archive controls preserve the native writer lock: close a chat in its other
Codex client before archiving it from mobile when that client still owns it.
Saved chats use Codex's own rename/archive APIs; the adapter never edits the
session database or moves history files itself.

For access outside the Mac, select an existing managed HTTPS gateway:

```sh
npa tools desktop chat-setup --local --gateway-ssh-host "$CHAT_GATEWAY_HOST"
```

The gateway must already be an authenticated `npa tools desktop public-access`
deployment or the standalone Codex Mobile deployment being migrated. This command
does not allocate cloud resources, change the public IP, or install a Linux
desktop on the Mac. It preserves the gateway's existing applications and login.
The standalone gateway's saved private access file is required for migration.
An existing signed mobile login cookie continues to work on its new chat route.

The Mac opens an outbound SSH connection using the selected alias and your
existing SSH authentication. Both ends of the forwarded listener bind to
loopback. Setup removes the tunnel if gateway verification fails, including
when the SSH server forces a public listener; correct the gateway and rerun setup. Phones use the public HTTPS URL and need no VPN or SSH client. A
desktop gateway retains its Linux chat at `/chat/` and adds the Mac at
`/local-chat/`; the standalone mobile gateway adds `/chat/` and keeps its original
root interface. `status --local` reports the selected URL. `--local-port` and
`--gateway-port` default to 6091 and can select unused unprivileged ports for the
first installation. Repeated setup preserves the saved configuration. A local
installation can adopt a gateway on a later setup; switching an existing gateway
requires an explicit configuration migration.

The Mac engine, web service, and tunnel are independent LaunchAgents under the signed-in macOS user.
Keep that user logged in and the Mac awake. VS Code must remain open for turns
it owns. The password file, runtime configuration, private logs, versioned
runtime assets, and send journal live under
`~/.local/share/nebius-desktop/local-chat/`; never commit that directory.
Setup reports the username and password-file location without printing the
password. To return to the prior standalone interface, use its unchanged root
URL. Stop the local LaunchAgents before removing their private runtime directory.

Both runtime modes share model/reasoning controls, account-supported speed and
Plan settings, image input, working indicators, and browser-local text drafts.
Model and mode changes apply to the next turn. If the Linux runtime has not reported its current mode or speed, the
picker says **Choose mode** or **Choose speed**; it updates when you make a
selection or receive a settings notification. The Mac adapter reads these
settings from the native owner. Browser sends retain a stable
identity, and a private SQLite journal returns saved outcomes on retries even
after the web service restarts. If the process dies after forwarding a send but
before recording its result, the UI reports an uncertain delivery and requires
checking the conversation before discarding that pending send. It does not
automatically replay the prompt. Pending approvals follow the active runtime;
specialized requests must still be answered in VS Code.

Local runtime updates are versioned and refuse to restart while a mobile-owned
turn is active. VS Code-owned turns run independently. A web-service restart preserves mobile-owned turns. An engine restart can
interrupt a mobile-owned turn; it preserves saved history but cannot restore
RAM or resume shell actions automatically. Back up the Mac's Codex state and
private service configuration separately from the gateway disk. The recovery
procedures below describe backup requirements; setup does not create a backup
repository or assert that recovery has been rehearsed.

## Recovery

| Failure | Recovery |
| --- | --- |
| Browser disconnect | Reopen the same conversation; the engine keeps running. |
| Mac web service or SSH tunnel exits | Its LaunchAgent restarts it; reconnect to the saved URL. The independent engine keeps its active turns. |
| Mac sleeps, logs out, or loses its network | Wake it, sign in, and restore connectivity. The gateway cannot execute Mac work while the Mac is unavailable. |
| Mac engine exits | Saved history remains; inspect interrupted work before sending again. An uncertain send is never replayed automatically. |
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
