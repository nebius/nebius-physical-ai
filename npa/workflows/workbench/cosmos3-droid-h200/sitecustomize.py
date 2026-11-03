"""Install the DROID video reader in native Python data-loader subprocesses."""

import json
import os
from pathlib import Path

if os.environ.get("NPA_DROID_SETTINGS"):
    try:
        import native_video

        native_video.install(
            json.loads(Path(os.environ["NPA_DROID_SETTINGS"]).read_text())
        )
    except Exception:
        os.write(2, b"DROID worker adapter initialization failed\n")
        os._exit(2)
