"""Public login helpers for the self-signed NPA agent UI."""

from __future__ import annotations


def agent_strip_url_credentials_js() -> str:
    """JS to strip user:pass@ from the URL bar while keeping the Basic auth session."""
    return """    <script>
    (function stripUrlCredentials() {
      try {
        if (location.username || location.password) {
          const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
          history.replaceState(null, "", clean);
        }
      } catch (_err) { /* best-effort */ }
    })();
    </script>"""


def agent_mobile_login_help_html() -> str:
    """Mobile certificate + sign-in troubleshooting for public pages."""
    return """    <details class="mobile-help" style="margin:20px 0;padding:12px 16px;border:1px solid #e0e0e0;border-radius:8px;background:#fffbeb;">
      <summary style="font-weight:600;cursor:pointer;">Phone / tablet login help</summary>
      <ol style="margin:12px 0 0;padding-left:20px;line-height:1.55;">
        <li><strong>Accept the certificate first.</strong> Open <a href="/healthz">/healthz</a> (no login). If Safari/Chrome warns the connection is not private, tap <em>Show Details</em> → <em>visit this website</em> / <em>Proceed</em>.</li>
        <li>Return here and use the sign-in form (mobile browsers block password-in-URL redirects).</li>
        <li>If sign-in still fails, try <strong>Chrome on Android</strong> or use a desktop browser.</li>
        <li>Username is prefilled; password is in your operator <code>auth.env</code> file.</li>
      </ol>
    </details>"""


def agent_public_login_form_html(auth_user: str) -> str:
    """Shared sign-in form for public welcome/login-help pages."""
    return f"""    <section class="sign-in-panel" aria-labelledby="sign-in-heading">
      <h2 id="sign-in-heading">Sign in</h2>
      <p class="muted">Use the form if your browser does not show an HTTP Basic Auth dialog.</p>
      <form id="npa-sign-in" class="sign-in" autocomplete="on">
        <label for="npa-user">Username</label>
        <input id="npa-user" name="username" type="text" value="{auth_user}" autocomplete="username" required>
        <label for="npa-pass">Password</label>
        <input id="npa-pass" name="password" type="password" autocomplete="current-password" required>
        <button type="submit" id="npa-sign-in-btn">Sign in</button>
        <p id="npa-sign-in-status" class="muted" role="status" aria-live="polite"></p>
      </form>
      <p class="muted note">Credentials are not left in the address bar after sign-in.</p>
    </section>
    <script>
    (function () {{
      try {{
        if (location.username || location.password) {{
          const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
          history.replaceState(null, "", clean);
        }}
      }} catch (_err) {{ /* best-effort */ }}
      var form = document.getElementById("npa-sign-in");
      var statusEl = document.getElementById("npa-sign-in-status");
      var btn = document.getElementById("npa-sign-in-btn");
      if (!form) return;

      function setStatus(msg, isError) {{
        if (!statusEl) return;
        statusEl.textContent = msg || "";
        statusEl.style.color = isError ? "#991b1b" : "#5f6573";
      }}

      function isMobileUa() {{
        return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
      }}

      function destPath() {{
        var rawPath = String(location.pathname || "/");
        var normalizedPath = rawPath.length > 1 && rawPath.endsWith("/") ? rawPath.slice(0, -1) : rawPath;
        return (normalizedPath === "/login-help.html" || normalizedPath === "/welcome") ? "/" : normalizedPath;
      }}

      function basicAuthHeader(user, pass) {{
        return "Basic " + btoa(unescape(encodeURIComponent(user + ":" + pass)));
      }}

      function persistBasicAuth(user, pass) {{
        try {{
          sessionStorage.setItem("npa_agent_basic_auth", basicAuthHeader(user, pass));
        }} catch (_err) {{ /* sessionStorage may be unavailable */ }}
      }}

      function xhrSignIn(user, pass, dest) {{
        return new Promise(function (resolve, reject) {{
          var xhr = new XMLHttpRequest();
          xhr.open("GET", dest, true, user, pass);
          xhr.onload = function () {{
            if (xhr.status >= 200 && xhr.status < 400) {{
              resolve();
              return;
            }}
            if (xhr.status === 401) {{
              reject(new Error("Invalid username or password."));
              return;
            }}
            reject(new Error("Sign-in failed (HTTP " + xhr.status + ")."));
          }};
          xhr.onerror = function () {{
            reject(new Error("Network error — open /healthz first and accept the certificate warning."));
          }};
          xhr.send();
        }});
      }}

      function fetchSignIn(user, pass, dest) {{
        return fetch(dest, {{
          method: "GET",
          headers: {{ "Authorization": basicAuthHeader(user, pass) }},
          credentials: "omit",
          cache: "no-store",
        }}).then(function (resp) {{
          if (!resp.ok) {{
            throw new Error(resp.status === 401 ? "Invalid username or password." : "Sign-in failed (HTTP " + resp.status + ").");
          }}
        }});
      }}

      function urlEmbedSignIn(user, pass, dest) {{
        var u = encodeURIComponent(user);
        var p = encodeURIComponent(pass);
        location.href = location.protocol + "//" + u + ":" + p + "@" + location.host + dest;
      }}

      form.addEventListener("submit", function (ev) {{
        ev.preventDefault();
        var user = document.getElementById("npa-user").value;
        var pass = document.getElementById("npa-pass").value;
        var dest = destPath();
        setStatus("Signing in…", false);
        if (btn) btn.disabled = true;

        xhrSignIn(user, pass, dest)
          .catch(function () {{ return fetchSignIn(user, pass, dest); }})
          .then(function () {{
            persistBasicAuth(user, pass);
            window.location.href = dest;
          }})
          .catch(function (err) {{
            if (!isMobileUa()) {{
              persistBasicAuth(user, pass);
              urlEmbedSignIn(user, pass, dest);
              return;
            }}
            setStatus((err && err.message) ? err.message : "Sign-in failed on this device.", true);
            if (btn) btn.disabled = false;
          }});
      }});
    }})();
    </script>"""
