# A persistent Linux development desktop on Nebius

[Cookbook index](README.md) · [Workbench setup](../getting-started.md)

Run VS Code, Codex, terminals, and CPU tests on a persistent Nebius VM while a
laptop displays the desktop through an SSH tunnel. This operator recipe uses
Ubuntu, XFCE, TigerVNC, and noVNC. It is not a new `npa workbench` tool or a
container image. GPU training and rendering still use the workbench's GPU
execution paths.

An existing CPU development VM with sufficient free memory and disk is a good
starting point. Inventory it before installing anything, preserve existing Git
changes, and keep its current workloads running. This recipe does not require a
VM reboot or a new GPU allocation.

## Prerequisites

Use an operator-owned Ubuntu VM with SSH key access, passwordless sudo for
installation, a working NPA checkout, and its development virtualenv. Set the SSH
alias on the laptop and the checkout path in the VM shell; never commit live
infrastructure details or credentials:

```bash
export DESKTOP_SSH_HOST='<existing-ssh-config-alias>'
```

On the VM:

```bash
export WORKBENCH_REPO="$HOME/nebius-physical-ai"
```

On the VM, run the credential preflight supported by the installed NPA version:

```bash
cd "$WORKBENCH_REPO"
npa/.venv/bin/python -m npa workbench health preflight --checks nebius --json
```

Older NPA installations might not expose that check. Verify the selected Nebius
CLI profile and exact project independently; do not mistake an unsupported CLI
option for an authentication failure. A profile using a missing
`/mnt/cloud-metadata/token` can instead use the attached service account's JSON
metadata endpoint, `http://metadata.nebius.internal/v1/iam/sa/token`, through
Nebius CLI's `token-endpoint` setting. The `/access_token` child returns plain
text and is not the JSON endpoint that setting expects. Verify a private copy
of the profile configuration before replacing the token-file setting. Preserve
a private backup and do not remount a metadata filesystem that was deliberately
disabled during host recovery.

## Install the desktop applications

Install the session and transport packages on the VM. Listing restart candidates
rather than restarting unrelated services is useful on a shared development VM.

```bash
sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get update
sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y \
  --no-install-recommends xfce4 xfce4-terminal tigervnc-standalone-server \
  tigervnc-tools websockify dbus-x11 x11-xserver-utils fonts-dejavu \
  fonts-liberation gnome-keyring libsecret-1-0 git curl restic
```

Install full VS Code from the [official signed Ubuntu repository](https://code.visualstudio.com/docs/setup/linux),
then install the [Codex extension](https://learn.chatgpt.com/docs/codex/ide):

```bash
code --install-extension openai.chatgpt
code --install-extension ms-python.python
code --install-extension charliermarsh.ruff
```

Install a supported browser from its publisher if browser work is needed inside
the desktop. Docker, the Nebius CLI, and the NPA environment are separate
prerequisites; a graphical desktop does not create their credentials or configure
a GPU target.

## Keep the session private and persistent

Use a private state directory and a VNC password. Classic VNC passwords use only
eight characters; SSH provides the encrypted, key-authenticated transport.
Never put the password in the browser URL or publish either service port.

```bash
install -d -m 700 "$HOME/.vnc" "$HOME/.local/share/nebius-desktop" \
  "$HOME/.local/bin" "$HOME/.config/systemd/user"
umask 077
openssl rand -base64 6 > "$HOME/.local/share/nebius-desktop/password"
tigervncpasswd -f < "$HOME/.local/share/nebius-desktop/password" > "$HOME/.vnc/passwd"
openssl rand -hex 32 | tr -d '\n' > "$HOME/.local/share/nebius-desktop/keyring-password"
```

Create these passwords once. Re-running password generation on an established
desktop rotates VNC access and prevents an existing keyring from unlocking.

Create `~/.vnc/xstartup` with executable permissions:

```sh
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XDG_CURRENT_DESKTOP=XFCE
export XDG_SESSION_DESKTOP=xfce
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
xset s off
xset -dpms
exec dbus-run-session -- "$HOME/.local/bin/nebius-desktop-session"
```

Create `~/.local/bin/nebius-desktop-session`, also executable:

```sh
#!/bin/sh
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
dbus-update-activation-environment --systemd PATH DISPLAY XAUTHORITY
gnome-keyring-daemon --unlock --components=secrets \
  < "$HOME/.local/share/nebius-desktop/keyring-password" >/dev/null
exec startxfce4
```

Use one keyring daemon in the desktop's D-Bus session. Starting a separate daemon
before `--unlock` can leave applications attached to a locked instance and show
a password dialog after restart. The automatic unlock file is readable only by
the desktop user; it is an explicit unattended-session convenience, not a second
independent security boundary. Preserve it privately for recovery.

Create `~/.config/systemd/user/nebius-desktop.service`:

```ini
[Unit]
Description=Persistent development desktop through SSH
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/tigervncserver :10 -fg -desktop "Development Desktop" -localhost yes -geometry 3024x1800 -depth 24 -SecurityTypes VncAuth -PasswordFile %h/.vnc/passwd -xstartup %h/.vnc/xstartup
ExecStop=/usr/bin/tigervncserver -kill :10
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

Enable user services to survive an SSH disconnect:

```bash
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now nebius-desktop.service
```

Optionally add a VS Code `.desktop` launcher in `~/.config/autostart/` pointing at
the operator's checkout. Use the desktop session's environment so extensions can
find tools installed in `~/.local/bin`. A CLI started from an unrelated SSH shell
may inherit a different `PATH`.

## Render text at Retina resolution

A high-resolution initial geometry is insufficient: ordinary browser remote
resize requests use CSS pixels. On a display with device pixel ratio 2, a
1512-pixel-wide browser should receive 3024 framebuffer pixels. Simply increasing
JPEG quality does not fix a framebuffer that is half the required width.

Pin the viewer to noVNC 1.6.0. The following small, version-checked adaptation
multiplies resize requests by the browser pixel ratio, scales the resulting
framebuffer into the browser, and omits JPEG-quality negotiation so Tight uses
lossless compression. These are upstream implementation details, so repeat the
browser checks below before upgrading noVNC. Keep the pristine checkout intact.

```bash
git clone --depth 1 --branch v1.6.0 https://github.com/novnc/noVNC.git \
  "$HOME/.local/share/nebius-desktop/novnc-1.6.0"
cd "$WORKBENCH_REPO"
npa/.venv/bin/python <<'PY'
from pathlib import Path
import json
import shutil

source = Path.home() / '.local/share/nebius-desktop/novnc-1.6.0'
target = Path.home() / '.local/share/nebius-desktop/novnc-retina'
assert json.loads((source / 'package.json').read_text())['version'] == '1.6.0'
shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', 'node_modules'))
rfb = target / 'core/rfb.js'
text = rfb.read_text()
old = '        const size = this._screenSize();\n\n        // Do we actually change anything?'
new = '''        const size = this._screenSize();
        const pixelRatio = window.devicePixelRatio || 1;
        size.w = Math.floor(size.w * pixelRatio);
        size.h = Math.floor(size.h * pixelRatio);

        // Do we actually change anything?'''
assert text.count(old) == 1
text = text.replace(old, new)
quality = '        encs.push(encodings.pseudoEncodingQualityLevel0 + this._qualityLevel);'
assert text.count(quality) == 1
text = text.replace(quality, '        // Omit JPEG quality negotiation for lossless Tight.')
rfb.write_text(text)
ui = target / 'app/ui.js'
text = ui.read_text()
old = "UI.rfb.scaleViewport = UI.getSetting('resize') === 'scale';"
new = "UI.rfb.scaleViewport = ['scale', 'remote'].includes(UI.getSetting('resize'));"
assert text.count(old) == 2
ui.write_text(text.replace(old, new))
PY
```

Create `~/.config/systemd/user/nebius-desktop-web.service`:

```ini
[Unit]
Description=Browser access to development desktop through SSH
After=nebius-desktop.service
Requires=nebius-desktop.service

[Service]
Type=simple
ExecStart=/usr/bin/websockify --web=%h/.local/share/nebius-desktop/novnc-retina 127.0.0.1:6080 127.0.0.1:5910
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now nebius-desktop-web.service
```

Inside the graphical desktop terminal, use 192 DPI for readable application text
on a device-pixel-ratio-2 display:

```bash
xfconf-query -c xsettings -p /Xft/DPI -n -t int -s 192
```

Use 96 DPI for a standard-density display. The viewer follows the browser's pixel
ratio, while the desktop font preference is shared by everyone viewing that
session. When moving between monitors with different density, adjust it as
needed. Concurrent viewers share one desktop geometry.

## Connect from the laptop

Bind local forwarded ports explicitly to loopback:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:16080:127.0.0.1:6080 "$DESKTOP_SSH_HOST"
```

Open [the local desktop](http://127.0.0.1:16080/vnc.html?autoconnect=true&resize=remote&compression=2)
in the browser. On macOS, copy the password without printing it:

```bash
ssh "$DESKTOP_SSH_HOST" 'cat ~/.local/share/nebius-desktop/password' | pbcopy
```

Paste it into the viewer's password field. After a viewer upgrade, refresh old
tabs so an older client cannot request a lower-resolution desktop. Use the
viewer's clipboard panel when transferring text into Linux applications.
Closing the tab disconnects the display; it does not terminate work on the VM.

## Recovery has three different failure cases

| Failure | What recovers | What does not |
| --- | --- | --- |
| Browser or SSH disconnect | The same running desktop after reconnect | The disconnected browser connection |
| VM or desktop process restart, persistent disk intact | Saved files, Git state, desktop configuration, Codex session files; enabled desktop services start again | RAM, terminal processes, uncheckpointed training, guarantees about unsaved editor buffers |
| Disk loss, deletion, or corruption | A retained disk snapshot and independently stored development backups | Changes newer than the last successful backup |

Enable VS Code autosave if that matches the operator's editing preference. A
short autosave delay reduces exposure but does not promise zero data loss.
Codex resume uses saved session history; a running agent process is not restored
from RAM. Use workload checkpoints for GPU jobs.

### Capture a disk recovery point

Select the exact boot disk from provider inventory. Keep its identity and the
returned snapshot identity in private recovery records. Flush filesystem writes
before taking an online snapshot:

```bash
ssh "$DESKTOP_SSH_HOST" sync
nebius compute disk-snapshot create \
  --parent-id "$NPA_PROJECT_ID" \
  --source-disk-id "$DESKTOP_BOOT_DISK_ID" \
  --name "$DESKTOP_SNAPSHOT_NAME" \
  --description 'Online development desktop recovery point' --async
```

An asynchronous create response is not proof of a usable snapshot. Reconcile it
through provider inventory, require the exact source disk, and wait for `READY`.
The CLI version used during validation returned a plain operation ID for async
creation even with JSON output selected. Do not blindly parse that response as
a resource JSON object or retry a successful create because parsing failed.
An online snapshot is crash-consistent; databases may need application-level
backup or quiescing for stronger consistency.

### Back up development work independently

Use an encrypted restic repository under a dedicated prefix in the exact
project's private object storage. Verify list, write, read, and delete access
with a disposable probe under that prefix before enabling a schedule. Do not
silently fall back to another project's ambient S3 keys.

Store `RESTIC_REPOSITORY`, `RESTIC_PASSWORD_FILE`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and the storage region in a private environment file.
Generate a strong restic password once. Copy the password and recovery settings
to an operator-controlled location outside the VM, with mode 0600 inside a mode
0700 directory. Never include them in Git or PR evidence.

The backup selection should cover:

- Each existing registered Git worktree's tracked and nonignored untracked files,
  its `.git` entry, and the common Git directory. This preserves uncommitted work
  and branch history. Ignored datasets, build products, caches, and environments
  need their own artifact policy; GitHub pushes alone do not cover local edits.
- Codex session and archive files, configuration, and the session index.
- VS Code user settings, VNC/session configuration, keyring state, and desktop
  service definitions. Authentication files, if included, belong only in the
  encrypted private backup.
- A small recovery probe with known contents for a restore check.

Use a NUL-delimited file list with `restic backup --files-from-raw` to preserve
spaces in filenames. Check the installed Git version: older Ubuntu Git versions
support `worktree list --porcelain` but not its newer `-z` option. Fail visibly on
unsupported path encodings rather than silently omitting a worktree.

Initialize the repository once, then run the backup manually before scheduling
it. Require a successful exit and record the restic snapshot ID and completion
time. A failure or partial-read exit must not update the last-success record.
Use `flock` to prevent overlapping scheduled and manual runs.

For an operator-approved hourly cadence, pair the backup's oneshot user service
with this timer:

```ini
[Unit]
Description=Hourly encrypted development backup

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

Retain backups until the operator chooses a retention policy; do not silently
add pruning. An hourly schedule means recovery can lose changes since the last
successful run, and a failed timer can make that interval longer. Check both the
timer and its service's last result, plus the private success record.

### Prove recovery before relying on it

From a separate directory, use the independently retained recovery credentials
to list snapshots and restore selected files. Compare hashes of the restored
probe, a tracked source file, and an uncommitted file where available. A snapshot
listing or a successful upload is not a restore test.

For a lost VM with its disk intact, attach the retained disk to a replacement
VM. For a lost disk, create a **new** disk from the retained snapshot, boot a
replacement VM, then restore newer development files from restic. Restore into
a separate directory first, inspect it, and only then choose what to replace.
Never overwrite the only surviving working copy during a recovery rehearsal.
Provisioning a replacement VM, recreating IAM access, and restoring a complete
boot disk are distinct checks from restoring selected backup files.

## Validation and cleanup

Before handoff, verify authenticated VNC access; loopback-only listeners; desktop
and keyring startup after a session restart; VS Code and Codex account readiness;
workbench credential checks; and a real CPU test run. Avoid restarting a desktop
that the operator is actively using just to repeat an already completed check.

For Retina validation, use a browser with `deviceScaleFactor: 2` and a known CSS
viewport. Assert that the VNC canvas is twice as wide and twice as high as its
CSS display size. A validated 1512 × 900 viewport produced a 3024 × 1800 canvas.
Verify the lossless encoding configuration and inspect the decoded desktop, not
only an HTTP 200 from the viewer. Confirm reconnecting leaves the desktop running.

Separately record the disk snapshot state, the automatic backup schedule, the
last successful backup, and the restore hashes. Keep exact resource identities,
private object paths, credentials, and screenshots of private work out of public
validation notes.

To stop displaying the desktop, close the browser and its SSH tunnel. To remove
only this desktop installation, stop and disable its user services and timer,
then remove its specific user configuration. Inspect shared user-service use
before disabling lingering. VM shutdown or deletion can interrupt other work;
use the [teardown procedure](../../../skills/atomic/teardown-and-cost/SKILL.md)
from the repository root, and retain recovery artifacts until explicitly retired.
