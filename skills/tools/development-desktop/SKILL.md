---
name: development-desktop
description: Set up and operate a persistent Ubuntu development desktop with VS Code and Codex through npa tools desktop, including readable Retina rendering and authenticated public HTTPS access.
---

# Development desktop

Use `npa tools desktop` for an operator's remote development environment.
Implementation belongs in `npa/src/npa/tools/desktop`, independently of Workbench.

1. Inventory the exact existing VM and its SSH access. Preserve its checkout,
   running applications, credentials, and unrelated services. The current
   installer supports Ubuntu AMD64 and requires passwordless sudo.
2. Run `npa tools desktop setup --ssh-host <alias> --dry-run --json`, then the
   same command without `--dry-run`. This installs XFCE, TigerVNC, the pinned
   noVNC viewer, full VS Code, and the Codex extension. Reuse existing VNC and
   keyring passwords. The desktop stays on loopback unless public access is
   explicitly requested.
3. Open with `npa tools desktop open --ssh-host <alias>`. Prefer one active
   viewer; old noVNC tabs can resize the shared desktop using CSS pixels and
   make Retina fonts look enormous. The viewer's **More space** control changes
   workspace scale while retaining lossless rendering and device-pixel density.
   `display --dpi 192` suits a Retina display; use 96 for standard density.
   For delayed clicks or typing, measure VM load and network latency, then use
   `optimize --ssh-host <alias>` to disable compositing and GTK animations live.
   Preserve resolution and running applications; compare actual input-to-frame
   latency before claiming an improvement. Setup uses these same defaults.
4. For authorized public access, verify the VM's external IPv4 address and
   ingress for TCP 80 and the selected HTTPS port. Then use
   `public-access --ssh-host <alias> --public-ip <address> --https-port 8443`.
   Preserve existing port 443 services. This uses Certbot's short-lived IP
   certificates, automatic renewal, and a separate generated strong login.
   Passwords stay in private VM files, never URLs, Git, or PR evidence.
5. Prove trusted TLS, unauthenticated 401s for the page, WebSocket and credential
   endpoint, authenticated desktop rendering, and rejection of foreign origins.
   If ACME validation fails, investigate public routing before retrying; an
   address reachable from an operator network is not proof of Internet reachability.
6. `status --json` reports existing backup and snapshot records. Setup preserves
   recovery services but does not provision a backup repository or storage keys.
   Use encrypted backups with independent recovery keys, consistent online
   SQLite copies for Codex databases, and actual file/database restore checks.
7. For mobile control, run `chat-setup --ssh-host <alias> --connect-vscode`
   after authenticated public access works. The mobile UI lives at `/chat/`
   on the same origin and uses the same desktop login. Show VS Code-origin
   history in Recent, Archived, and search; exclude CLI, execution, subagent,
   and unknown-origin chats without deleting them. Filter before pagination.
   It shares a private Codex runtime with the VDI IDE; reload the IDE only when
   its work is idle.
   Keep sessions held by older independent Codex processes read-only until
   their owner releases them. Test real prompts and replies in both directions
   using the actual VS Code window and a phone-sized browser, plus reconnects,
   active-turn steering/stopping, approvals, and working tab/session indicators.
   Verify shared model/reasoning controls against the live runtime catalog;
   restore test selections afterward. Preserve
   the existing account and model configuration. Never copy account tokens to
   the browser or restart a shared engine with active work during setup.
8. For existing Mac sessions, use `chat-setup --local` and optionally
   `--gateway-ssh-host <existing-managed-gateway>`. Local mode follows the native
   VS Code owner, preserving active chats without changing `chatgpt.cliExecutable`.
   Require macOS, Node.js 22.13+, npm, and a signed-in Codex installation. Keep
   credentials in private runtime files. Preserve the existing HTTPS origin,
   credentials, desktop, and remote chat route. On a standalone mobile gateway,
   route old root bookmarks and Home Screen entry points to `/chat/`; preserve
   the original UI at `/legacy/` and its running services. Verify entry redirects,
   chat fragments, existing login cookies, and unsent-draft migration. Validate local/remote CLI exclusivity,
   same-thread messages, images, Plan/speed settings, draft retention, reconnects,
   repeated setup, and lost-response delivery without duplicate execution.
   Use `status --local --json` and `open --local --chat` for the saved setup.
   Recheck extension IPC compatibility after upgrades. Do not claim a backup
   exists from setup alone, or restart a mobile-owned active turn during updates.

See [the operator guide](../../../docs/tools/development-desktop.md) for login,
recovery, public-access prerequisites, and cleanup. Use the contribution,
testing, and confidentiality skills before pushing changes. Keep live addresses,
infrastructure identities, credentials, and screenshots in private evidence.
