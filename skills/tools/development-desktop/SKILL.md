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

See [the operator guide](../../../docs/tools/development-desktop.md) for login,
recovery, public-access prerequisites, and cleanup. Use the contribution,
testing, and confidentiality skills before pushing changes. Keep live addresses,
infrastructure identities, credentials, and screenshots in private evidence.
