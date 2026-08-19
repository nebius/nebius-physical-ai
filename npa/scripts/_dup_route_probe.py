import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_agent_capabilities import load_backend_app, render_backend_body  # noqa: E402

body = render_backend_body()
targets = ("/artifacts/file/{filename}", "/artifacts/download", "/artifacts/content")

print("== decorator order in rendered backend ==")
for needle in targets:
    offsets = []
    start = 0
    while True:
        i = body.find('"' + needle + '"', start)
        if i < 0:
            break
        window = body[max(0, i - 60) : i]
        if "@app." in window:
            kind = window[window.rfind("@app.") :].strip()
            offsets.append((i, kind))
        start = i + 1
    print(f"{needle:32s} -> {offsets}")

print()
print("== routes the app actually registers ==")
with tempfile.TemporaryDirectory() as tmp:
    app, _ = load_backend_app(body, Path(tmp))
    for route in app.routes:
        path = getattr(route, "path", "")
        if path in targets:
            fn = getattr(route, "endpoint", None)
            methods = sorted(getattr(route, "methods", []) or [])
            code = getattr(fn, "__code__", None)
            line = code.co_firstlineno if code else "?"
            sig = sorted((code.co_varnames[: code.co_argcount] if code else ()))
            print(f"{path:32s} methods={methods} def_line={line} args={sig}")
