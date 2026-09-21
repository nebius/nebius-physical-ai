"""Apply desktop configuration on an existing Ubuntu host through private SSH stdin."""

from contextlib import redirect_stderr
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import traceback

_HOME = Path.home()
_STATE = _HOME / ".local/share/nebius-desktop"
_UNITS = _HOME / ".config/systemd/user"
_BIN = _HOME / ".local/bin"


def _command(argv, *, data=None, capture=False, environment=None):
    with (_STATE / "setup.log").open("a") as log:
        result = subprocess.run(
            argv,
            input=data,
            text=True,
            check=True,
            stdout=subprocess.PIPE if capture else log,
            stderr=log,
            env=environment,
        )
    return result.stdout if capture else None


def _write(path, content, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".npa-tmp")
    temporary.write_text(content)
    temporary.chmod(mode)
    temporary.replace(path)


def _root_file(path, content, mode="600", group="root"):
    temporary = _STATE / "root-file.tmp"
    _write(temporary, content)
    try:
        _command(
            [
                "sudo",
                "-n",
                "install",
                "-D",
                "-m",
                mode,
                "-o",
                "root",
                "-g",
                group,
                str(temporary),
                path,
            ]
        )
    finally:
        temporary.unlink()


def _packages(packages):
    prefix = [
        "sudo",
        "-n",
        "env",
        "DEBIAN_FRONTEND=noninteractive",
        "NEEDRESTART_MODE=l",
        "apt-get",
    ]
    _command(prefix + ["update"])
    _command(prefix + ["install", "-y", "--no-install-recommends", *packages])


def _credentials():
    for name, value in [
        ("password", secrets.token_urlsafe(6)),
        ("keyring-password", secrets.token_hex(32)),
    ]:
        path = _STATE / name
        if not path.exists():
            _write(path, value)
    vnc = _HOME / ".vnc/passwd"
    if not vnc.exists():
        result = subprocess.run(
            ["tigervncpasswd", "-f"],
            input=(_STATE / "password").read_bytes(),
            capture_output=True,
            check=True,
        )
        vnc.parent.mkdir(mode=0o700, exist_ok=True)
        vnc.write_bytes(result.stdout)
        vnc.chmod(0o600)


def _desktop_services(config):
    _write(_HOME / ".vnc/xstartup", _XSTARTUP, 0o700)
    _write(_BIN / "nebius-desktop-session", _SESSION, 0o700)
    _write(
        _UNITS / "nebius-desktop.service",
        _DESKTOP_UNIT.replace("@GEOMETRY@", config["geometry"]),
    )
    _write(_UNITS / "nebius-desktop-web.service", _WEB_UNIT)
    _command(["sudo", "-n", "loginctl", "enable-linger", str(os.getuid())])
    _command(["systemctl", "--user", "daemon-reload"])
    _command(["systemctl", "--user", "enable", "--now", "nebius-desktop.service"])
    _command(["systemctl", "--user", "enable", "nebius-desktop-web.service"])
    _command(["systemctl", "--user", "restart", "nebius-desktop-web.service"])


def _viewer():
    source = _STATE / "novnc-1.6.0"
    if not source.exists():
        _command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v1.6.0",
                "https://github.com/novnc/noVNC.git",
                str(source),
            ]
        )
    if json.loads((source / "package.json").read_text())["version"] != "1.6.0":
        raise RuntimeError("The viewer requires noVNC 1.6.0.")
    target = _STATE / "novnc-tools-v1"
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", "node_modules"),
    )
    _patch_viewer(target)


def _patch_viewer(target):
    path = target / "core/rfb.js"
    content = path.read_text()
    old = "        const size = this._screenSize();\n\n        // Do we actually change anything?"
    new = "        const size = this._screenSize();\n        const ratio = (window.devicePixelRatio || 1) / (window.npaDesktopScale || 1);\n        size.w = Math.round(size.w * ratio);\n        size.h = Math.round(size.h * ratio);\n\n        // Do we actually change anything?"
    quality = (
        "        encs.push(encodings.pseudoEncodingQualityLevel0 + this._qualityLevel);"
    )
    if content.count(old) != 1 or content.count(quality) != 1:
        raise RuntimeError("Unexpected noVNC resize or encoding implementation.")
    path.write_text(
        content.replace(old, new).replace(
            quality, "        // Leave Tight in lossless mode."
        )
    )
    _write(target / "desktop.html", _VIEWER, 0o644)
    _write(
        target / "index.html",
        '<meta http-equiv="refresh" content="0;url=desktop.html">',
        0o644,
    )


def _install_vscode():
    if not shutil.which("code"):
        _command(
            [
                "curl",
                "-fsSL",
                "https://packages.microsoft.com/keys/microsoft.asc",
                "-o",
                str(_STATE / "microsoft.asc"),
            ]
        )
        _command(
            [
                "gpg",
                "--batch",
                "--yes",
                "--dearmor",
                "-o",
                str(_STATE / "microsoft.gpg"),
                str(_STATE / "microsoft.asc"),
            ]
        )
        _command(
            [
                "sudo",
                "-n",
                "install",
                "-m",
                "644",
                str(_STATE / "microsoft.gpg"),
                "/usr/share/keyrings/npa-microsoft.gpg",
            ]
        )
        _root_file(
            "/etc/apt/sources.list.d/npa-vscode.list",
            "deb [arch=amd64 signed-by=/usr/share/keyrings/npa-microsoft.gpg] https://packages.microsoft.com/repos/code stable main\n",
            "644",
        )
        _packages(["code"])


def _vscode(repository):
    _install_vscode()
    for extension in ("openai.chatgpt", "ms-python.python", "charliermarsh.ruff"):
        _command(["code", "--install-extension", extension])
    settings = _HOME / ".config/Code/User/settings.json"
    if not settings.exists():
        _write(
            settings,
            json.dumps(
                {
                    "files.autoSave": "afterDelay",
                    "files.autoSaveDelay": 1000,
                    "window.zoomLevel": 0,
                },
                indent=2,
            ),
        )
    # Desktop Exec has its own quoting rules, so reject control/expansion syntax.
    if any(character in repository for character in '\n\r"\\%`$'):
        raise ValueError(
            "Repository path contains unsupported desktop-launcher characters."
        )
    _write(
        _HOME / ".config/autostart/nebius-workbench.desktop",
        '[Desktop Entry]\nType=Application\nName=VS Code\nExec=code --new-window "'
        + repository
        + '"\n',
        0o600,
    )


def _desktop_environment():
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-x", "xfce4-session"],
        capture_output=True,
        text=True,
        check=True,
    )
    sessions = result.stdout.split()
    for process in sessions:
        environment = dict(
            item.split("=", 1)
            for item in (Path("/proc") / process / "environ").read_text().split("\0")
            if "=" in item
        )
        if environment.get("DISPLAY") == ":10":
            return {**os.environ, **environment}
    raise RuntimeError(
        "Desktop session :10 is not ready; retry the display command after startup."
    )


def _display(dpi):
    environment = _desktop_environment()
    _command(
        [
            "xfconf-query",
            "-c",
            "xsettings",
            "-p",
            "/Xft/DPI",
            "-n",
            "-t",
            "int",
            "-s",
            str(dpi),
        ],
        environment=environment,
    )
    _write(_STATE / "display.json", json.dumps({"dpi": dpi}))


def _optimize(environment=None):
    prefix = [] if environment is not None else ["dbus-run-session", "--"]
    for channel, key in [
        ("xfwm4", "/general/use_compositing"),
        ("xsettings", "/Gtk/EnableAnimations"),
    ]:
        _command(
            prefix + [
                "xfconf-query", "-c", channel, "-p", key,
                "-n", "-t", "bool", "-s", "false",
            ],
            environment=environment,
        )
    _write(
        _STATE / "performance.json",
        json.dumps({"compositing": False, "gtk_animations": False}),
    )


def _setup(config):
    repository = str(Path(config["repository_path"]).expanduser())
    if not (Path(repository) / ".git").exists():
        raise ValueError("Repository path must identify an existing Git checkout.")
    if any(character in repository for character in '\n\r"\\%`$'):
        raise ValueError(
            "Repository path contains unsupported desktop-launcher characters."
        )
    if os.uname().machine != "x86_64":
        raise ValueError("The current desktop installer supports Ubuntu AMD64.")
    _packages(_DESKTOP_PACKAGES)
    _credentials()
    _viewer()
    _vscode(repository)
    _initial_density(config["dpi"])
    _desktop_services(config)


def _initial_density(dpi):
    probe = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-x", "xfce4-session"], capture_output=True
    )
    if probe.returncode == 0:
        _display(dpi)
        _optimize(_desktop_environment())
        return
    _command(
        [
            "dbus-run-session",
            "--",
            "xfconf-query",
            "-c",
            "xsettings",
            "-p",
            "/Xft/DPI",
            "-n",
            "-t",
            "int",
            "-s",
            str(dpi),
        ]
    )
    _write(_STATE / "display.json", json.dumps({"dpi": dpi}))
    _optimize()


_DESKTOP_PACKAGES = [
    "xfce4",
    "xfce4-terminal",
    "tigervnc-standalone-server",
    "tigervnc-tools",
    "websockify",
    "dbus-x11",
    "x11-xserver-utils",
    "fonts-dejavu",
    "fonts-liberation",
    "gnome-keyring",
    "libsecret-1-0",
    "git",
    "curl",
    "gnupg",
]


def _status():
    services = {}
    for name in (
        "nebius-desktop",
        "nebius-desktop-web",
        "nebius-desktop-backup.timer",
        "nebius-desktop-snapshot-watch",
        "npa-codex-server",
        "npa-codex-chat",
    ):
        result = subprocess.run(
            ["systemctl", "--user", "is-active", name],
            capture_output=True,
            text=True,
            check=False,
        )
        services[name] = result.stdout.strip()
    result = {
        "services": services,
        "viewer": "/desktop.html",
        "password_file": str(_STATE / "password"),
    }
    for name, key in (
        ("public-access.json", "public_access"),
        ("display.json", "display"),
        ("performance.json", "performance"),
        ("backup-last-success.json", "last_backup"),
        ("snapshot-verification.json", "snapshot"),
        ("chat.json", "chat"),
    ):
        path = _STATE / name
        if path.exists():
            result[key] = json.loads(path.read_text())
    if "public_access" in result:
        result["public_url"] = result["public_access"]["url"]
    return result


def _gateway_credentials():
    password = _STATE / "public-password"
    if not password.exists():
        _write(password, secrets.token_urlsafe(24))
    entry = _command(
        ["htpasswd", "-niB", "developer"],
        data=password.read_text() + "\n",
        capture=True,
    )
    _root_file("/etc/npa-desktop/htpasswd", entry, "640", "www-data")
    _root_file(
        "/etc/npa-desktop/vnc-credentials.json",
        json.dumps({"password": (_STATE / "password").read_text().strip()}),
        "640",
        "www-data",
    )


def _gateway_config(config, *, tls):
    text = _NGINX_HTTP
    if tls:
        text += _NGINX_TLS.replace("@PORT@", str(config["https_port"])).replace(
            "@PUBLIC_IP@", config["public_ip"]
        )
    text = text.replace(
        "@CHAT_ROUTE@", _NGINX_CHAT if (_STATE / "chat.json").exists() else ""
    )
    return (
        "user www-data;\nworker_processes auto;\npid /run/npa-desktop-gateway.pid;\nerror_log /var/log/nginx/npa-desktop-error.log;\nevents {}\nhttp {\ninclude /etc/nginx/mime.types;\n"
        + text
        + "\n}\n"
    )


def _configure_gateway(config, *, tls):
    content = _gateway_config(config, tls=tls)
    _root_file("/etc/npa-desktop/nginx.candidate.conf", content)
    _command(
        ["sudo", "-n", "nginx", "-t", "-c", "/etc/npa-desktop/nginx.candidate.conf"]
    )
    _root_file("/etc/npa-desktop/nginx.conf", content)
    _root_file("/etc/systemd/system/npa-desktop-gateway.service", _GATEWAY_UNIT, "644")
    _command(["sudo", "-n", "systemctl", "daemon-reload"])
    _command(["sudo", "-n", "systemctl", "enable", "--now", "npa-desktop-gateway"])
    _command(["sudo", "-n", "systemctl", "reload", "npa-desktop-gateway"])


def _public_access(config):
    if not (_STATE / "novnc-tools-v1/desktop.html").exists():
        raise RuntimeError("Run desktop setup before enabling public access.")
    _gateway_packages(config)
    _gateway_credentials()
    _command(["sudo", "-n", "mkdir", "-p", "/var/lib/npa-desktop/acme"])
    certificate = Path("/etc/letsencrypt/live/npa-desktop/fullchain.pem")
    has_certificate = (
        subprocess.run(
            ["sudo", "-n", "test", "-f", str(certificate)], capture_output=True
        ).returncode
        == 0
    )
    _configure_gateway(config, tls=has_certificate)
    _certificate(config)
    _configure_gateway(config, tls=True)
    _verify_gateway(config)
    _renewal()
    _write(
        _STATE / "public-access.json",
        json.dumps(
            {
                "url": f"https://{config['public_ip']}:{config['https_port']}/desktop.html",
                "username": "developer",
                "password_file": str(_STATE / "public-password"),
            }
        ),
    )


def _gateway_packages(config):
    owned = (_STATE / "gateway-owned").exists()
    if owned:
        return
    if shutil.which("nginx"):
        raise RuntimeError(
            "Existing nginx is not owned by this desktop; configure its gateway separately."
        )
    for port in (80, config["https_port"]):
        listeners = _command(
            ["ss", "-H", "-ltn", "sport", "=", f":{port}"], capture=True
        )
        if listeners.strip():
            raise RuntimeError("A required gateway port already has a listener.")
    _packages(["nginx", "apache2-utils", "python3-venv"])
    _command(["sudo", "-n", "systemctl", "disable", "--now", "nginx"])
    _write(_STATE / "gateway-owned", "Dedicated desktop gateway\n")


def _certificate(config):
    _command(
        ["sudo", "-n", "/usr/bin/python3", "-m", "venv", "/opt/npa-desktop-certbot"]
    )
    _command(
        ["sudo", "-n", "/opt/npa-desktop-certbot/bin/pip", "install", "certbot==5.8.0"]
    )
    _command(
        [
            "sudo",
            "-n",
            "/opt/npa-desktop-certbot/bin/certbot",
            "certonly",
            "--non-interactive",
            "--agree-tos",
            "--register-unsafely-without-email",
            "--preferred-profile",
            "shortlived",
            "--webroot",
            "--webroot-path",
            "/var/lib/npa-desktop/acme",
            "--ip-address",
            config["public_ip"],
            "--cert-name",
            "npa-desktop",
            "--keep-until-expiring",
        ]
    )


def _renewal():
    _root_file(
        "/etc/systemd/system/npa-desktop-cert-renew.service", _RENEW_SERVICE, "644"
    )
    _root_file("/etc/systemd/system/npa-desktop-cert-renew.timer", _RENEW_TIMER, "644")
    _command(["sudo", "-n", "systemctl", "daemon-reload"])
    _command(
        ["sudo", "-n", "systemctl", "enable", "--now", "npa-desktop-cert-renew.timer"]
    )


def _verify_gateway(config):
    address, port = config["public_ip"], config["https_port"]
    base = [
        "curl",
        "--silent",
        "--show-error",
        "--output",
        "/dev/null",
        "--write-out",
        "%{http_code}",
        "--connect-to",
        f"{address}:{port}:127.0.0.1:{port}",
    ]
    for path in ("desktop.html", "websockify", "desktop-credentials.json"):
        status = _command(base + [f"https://{address}:{port}/{path}"], capture=True)
        if status != "401":
            raise RuntimeError(
                "Gateway did not require authentication on every protected route."
            )
    credentials = (
        'user = "developer:' + (_STATE / "public-password").read_text().strip() + '"\n'
    )
    status = _command(
        base + ["--config", "-", f"https://{address}:{port}/desktop.html"],
        data=credentials,
        capture=True,
    )
    if status != "200":
        raise RuntimeError("Authenticated gateway verification failed.")


def _chat_setup(config):
    from urllib.parse import urlsplit

    public = _STATE / "public-access.json"
    if not public.exists():
        raise RuntimeError("Configure authenticated public-access before mobile chat.")
    names = {
        "chat_setup.py",
        "chat_server.py",
        "chat_rpc.py",
        "chat_proxy.py",
        "chat_history.py",
        "chat.html",
        "chat.css",
        "chat.js",
    }
    if set(config["chat_assets"]) != names:
        raise ValueError("Unexpected mobile chat assets.")
    root = _STATE / "codex-chat"
    root.mkdir(mode=0o700, exist_ok=True)
    for name, source in config["chat_assets"].items():
        _write(root / name, source)
    _command(
        ["/usr/bin/python3", str(root / "chat_setup.py")],
        data=json.dumps({"connect_vscode": config.get("connect_vscode", False)}),
    )
    url = urlsplit(json.loads(public.read_text())["url"])
    _configure_gateway(
        {"public_ip": url.hostname, "https_port": url.port or 443}, tls=True
    )


def _run(config):
    os.umask(0o077)
    _STATE.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        with (_STATE / "operation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if config["action"] == "setup":
                _setup(config)
            elif config["action"] == "display":
                _display(config["dpi"])
            elif config["action"] == "public-access":
                _public_access(config)
            elif config["action"] == "chat-setup":
                _chat_setup(config)
            elif config["action"] == "optimize":
                _optimize(_desktop_environment())
        print(json.dumps(_status()))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        with (_STATE / "setup.log").open("a") as log, redirect_stderr(log):
            traceback.print_exc()
        raise SystemExit(1)


_XSTARTUP = """#!/bin/sh
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS
export XDG_CURRENT_DESKTOP=XFCE XDG_SESSION_DESKTOP=xfce
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
xset s off
xset -dpms
exec dbus-run-session -- "$HOME/.local/bin/nebius-desktop-session"
"""
_SESSION = """#!/bin/sh
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
dbus-update-activation-environment --systemd PATH DISPLAY XAUTHORITY
gnome-keyring-daemon --unlock --components=secrets < "$HOME/.local/share/nebius-desktop/keyring-password" >/dev/null
exec startxfce4
"""
_DESKTOP_UNIT = """[Unit]
Description=Persistent development desktop through SSH
After=network.target
[Service]
Type=simple
ExecStart=/usr/bin/tigervncserver :10 -fg -desktop "Development Desktop" -localhost yes -geometry @GEOMETRY@ -depth 24 -SecurityTypes VncAuth -PasswordFile %h/.vnc/passwd -xstartup %h/.vnc/xstartup
ExecStop=/usr/bin/tigervncserver -kill :10
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
"""
_WEB_UNIT = """[Unit]
Description=Browser development desktop through SSH
After=nebius-desktop.service
Requires=nebius-desktop.service
[Service]
Type=simple
ExecStart=/usr/bin/websockify --web=%h/.local/share/nebius-desktop/novnc-tools-v1 127.0.0.1:6080 127.0.0.1:5910
Restart=on-failure
RestartSec=3
[Install]
WantedBy=default.target
"""
_NGINX_HTTP = """server {
    listen 80;
    server_name _;
    location /.well-known/acme-challenge/ { root /var/lib/npa-desktop/acme; }
    location / { return 404; }
}
"""
_NGINX_TLS = """map $http_origin $npa_desktop_origin_allowed {
    default 0;
    "" 1;
    "https://@PUBLIC_IP@:@PORT@" 1;
}
server {
    listen @PORT@ ssl;
    server_name _;
    ssl_certificate /etc/letsencrypt/live/npa-desktop/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/npa-desktop/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    if ($npa_desktop_origin_allowed = 0) { return 403; }
    auth_basic "Development desktop";
    auth_basic_user_file /etc/npa-desktop/htpasswd;
    add_header Cache-Control "no-store" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    location = /desktop-credentials.json { alias /etc/npa-desktop/vnc-credentials.json; default_type application/json; }
@CHAT_ROUTE@
    location / {
        proxy_pass http://127.0.0.1:6080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
        proxy_buffering off;
    }
}
"""
_NGINX_CHAT = """    location = /chat { return 308 /chat/; }
    location /chat/ {
        proxy_pass http://127.0.0.1:6090;
        proxy_http_version 1.1;
        proxy_set_header Host $http_host;
        proxy_set_header Authorization $http_authorization;
        proxy_set_header Connection "";
        proxy_read_timeout 65s;
        proxy_buffering off;
        client_max_body_size 2m;
    }
"""
_GATEWAY_UNIT = """[Unit]
Description=NPA development desktop authenticated HTTPS gateway
After=network-online.target
[Service]
Type=simple
ExecStart=/usr/sbin/nginx -c /etc/npa-desktop/nginx.conf -g "daemon off;"
ExecReload=/usr/sbin/nginx -c /etc/npa-desktop/nginx.conf -s reload
Restart=on-failure
[Install]
WantedBy=multi-user.target
"""
_RENEW_SERVICE = """[Unit]
Description=Renew the development desktop IP certificate
[Service]
Type=oneshot
ExecStart=/opt/npa-desktop-certbot/bin/certbot renew --cert-name npa-desktop --quiet --deploy-hook "systemctl reload npa-desktop-gateway.service"
"""
_RENEW_TIMER = """[Unit]
Description=Check development desktop certificate renewal twice daily
[Timer]
OnCalendar=*-*-* 00,12:00:00
Persistent=true
[Install]
WantedBy=timers.target
"""
_VIEWER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NPA Desktop</title><style>
*{box-sizing:border-box}body{margin:0;background:#151719;color:#eee;font:13px system-ui;height:100vh;display:flex;flex-direction:column}
header{height:36px;flex:none;display:flex;align-items:center;gap:14px;padding:0 12px;background:#24282b}
header strong{margin-right:auto}button,select,input{font:inherit;padding:4px 8px;border-radius:5px;border:1px solid #626970;background:#24282b;color:#eee}
#screen{flex:1;min-height:0;overflow:hidden}dialog{color:#eee;background:#24282b;border:1px solid #626970;border-radius:12px;padding:24px}dialog input{display:block;margin:14px 0;width:260px}textarea{display:block;width:min(70vw,600px);height:200px;margin:14px 0;background:#151719;color:#eee}
</style></head><body><header><strong>NPA Desktop</strong><span id="status">Connecting…</span><label>Workspace <select id="scale"><option value="1">Comfortable</option><option value="0.85">More space</option><option value="1.15">Larger text</option></select></label><button id="fullscreen">Full screen</button></header>
<main id="screen"></main><dialog id="login"><form method="dialog"><label>Desktop password<input id="password" type="password" autocomplete="current-password" required autofocus></label><button>Connect</button></form></dialog>
<dialog id="clipboard"><label>Clipboard<textarea id="clipboard-text" spellcheck="false"></textarea></label><button id="send-clipboard">Send to desktop</button> <button id="close-clipboard">Close</button><p>Paste laptop text here to send it. Copy desktop text from this box.</p></dialog>
<script type="module">
import RFB from './core/rfb.js?npa-desktop=1';
const status=document.querySelector('#status'),scale=document.querySelector('#scale'),login=document.querySelector('#login');
window.npaDesktopScale=Number(localStorage.getItem('npaDesktopScale')||1);
if(![0.85,1,1.15].includes(window.npaDesktopScale))window.npaDesktopScale=1;
scale.value=String(window.npaDesktopScale);
const url=new URL('websockify',location.href);url.protocol=location.protocol==='https:'?'wss:':'ws:';
const rfb=new RFB(document.querySelector('#screen'),url.href);window.desktopRfb=rfb;
rfb.scaleViewport=true;rfb.resizeSession=true;rfb.compressionLevel=2;
const clipboard=document.querySelector('#clipboard'),clipboardText=document.querySelector('#clipboard-text');
const clipboardButton=document.createElement('button');clipboardButton.textContent='Clipboard';
document.querySelector('header').insertBefore(clipboardButton,document.querySelector('#fullscreen'));
clipboardButton.addEventListener('click',()=>{clipboard.showModal();clipboardText.focus();clipboardText.select();});
rfb.addEventListener('clipboard',event=>{clipboardText.value=event.detail.text;});
document.querySelector('#send-clipboard').addEventListener('click',()=>{rfb.clipboardPasteFrom(clipboardText.value);clipboard.close();rfb.focus();});
document.querySelector('#close-clipboard').addEventListener('click',()=>{clipboard.close();rfb.focus();});
rfb.addEventListener('connect',()=>{status.textContent='Connected';rfb.focus();});
rfb.addEventListener('disconnect',()=>{status.textContent='Disconnected — refresh to reconnect';});
rfb.addEventListener('credentialsrequired',async()=>{
  const response=await fetch('desktop-credentials.json',{cache:'no-store'});
  if(response.ok){rfb.sendCredentials(await response.json());return;}
  login.showModal();
});
login.addEventListener('close',()=>{rfb.sendCredentials({password:document.querySelector('#password').value});document.querySelector('#password').value='';});
scale.addEventListener('change',()=>{localStorage.setItem('npaDesktopScale',scale.value);location.reload();});
document.querySelector('#fullscreen').addEventListener('click',()=>{document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();});
</script></body></html>
"""
