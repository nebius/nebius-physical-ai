"""GR00T runtime compatibility and environment command renderers."""

from __future__ import annotations

import json
import shlex
from typing import Any, Callable, Mapping


def build_status_result(
    data: Mapping[str, Any],
    *,
    endpoint_url: str,
    hf_token_present: bool,
    default_model: str,
) -> dict[str, Any]:
    """Build the public GR00T health/readiness response from server facts."""

    loaded = bool(data.get("loaded"))
    ngc_ok = bool(data.get("ngc_credentials_configured"))
    # A model that is actually loaded and serving is ready. NGC and HF
    # credentials only matter for downloading future checkpoints.
    readiness: dict[str, Any] = {
        "hf_token_present": hf_token_present,
        "ngc_credentials_configured": ngc_ok,
        "model_loaded": loaded,
        "ready": loaded,
        "blockers": [],
        "notes": [],
    }
    if not loaded:
        readiness["blockers"].append(
            f"Model {data.get('model') or default_model} not loaded"
        )
        if not hf_token_present:
            readiness["blockers"].append(
                "HF_TOKEN not configured - gated model downloads will fail"
            )
        if not ngc_ok:
            readiness["blockers"].append(
                "NGC credentials not configured - required only for NGC-hosted checkpoints"
            )
    else:
        if not hf_token_present:
            readiness["notes"].append(
                "HF_TOKEN not configured - only affects future gated HF downloads"
            )
        if not ngc_ok:
            readiness["notes"].append(
                "NGC credentials not configured - only needed for NGC-hosted checkpoints"
            )
    result: dict[str, Any] = {
        "endpoint": endpoint_url,
        "app_status": "healthy" if loaded else "degraded",
        "server": "up",
        **data,
        "readiness": readiness,
    }
    if not loaded:
        result["reason"] = "model not loaded"
    return result


def build_runtime_pin_patch_command(
    repo_path: str, cosmos_model_name: str, cosmos_model_revision: str
) -> str:
    """Patch the pinned checkout for revision and action-mode parity."""

    return f"""\
python3.10 - <<'PY'
from pathlib import Path

repo = Path({repo_path!r})
cosmos_model = {cosmos_model_name!r}
cosmos_revision = {cosmos_model_revision!r}


def replace_once(rel: str, old: str, new: str) -> None:
    path = repo / rel
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        return
    if new in text:
        return
    raise RuntimeError("Could not apply GR00T runtime pin patch to " + rel)


replace_once(
    "gr00t/configs/model/gr00t_n1d7.py",
    "    model_revision: str | None = None\\n",
    "    model_revision: str | None = \\"" + cosmos_revision + "\\"\\n",
)
replace_once(
    "gr00t/experiment/launch_finetune.py",
    "    config.model.model_name = \\"" + cosmos_model + "\\"\\n",
    "    config.model.model_name = \\"" + cosmos_model + "\\"\\n"
    "    config.model.model_revision = \\"" + cosmos_revision + "\\"\\n",
)
replace_once(
    "gr00t/configs/finetune_config.py",
    "    tune_diffusion_model: bool = True\\n"
    "    \\"\\"\\"If True, fine-tune the diffusion-based action decoder (if present in the model).\\"\\"\\"\\n\\n",
    "    tune_diffusion_model: bool = True\\n"
    "    \\"\\"\\"If True, fine-tune the diffusion-based action decoder (if present in the model).\\"\\"\\"\\n\\n"
    "    use_relative_action: bool = True\\n"
    "    \\"\\"\\"Convert action groups declared RELATIVE when training and inference.\\"\\"\\"\\n\\n",
)
replace_once(
    "gr00t/experiment/launch_finetune.py",
    "    config.model.use_relative_action = True\\n",
    "    config.model.use_relative_action = ft_config.use_relative_action\\n",
)
replace_once(
    "gr00t/configs/finetune_config.py",
    "    save_steps: int = 1000\\n"
    "    \\\"\\\"\\\"Frequency (in training steps) at which to save checkpoints.\\\"\\\"\\\"\\n",
    "    logging_steps: int = 10\\n"
    "    \\\"\\\"\\\"Frequency (in optimizer steps) at which to record real trainer losses.\\\"\\\"\\\"\\n\\n"
    "    save_steps: int = 1000\\n"
    "    \\\"\\\"\\\"Frequency (in training steps) at which to save checkpoints.\\\"\\\"\\\"\\n",
)
replace_once(
    "gr00t/experiment/launch_finetune.py",
    "    config.training.output_dir = ft_config.output_dir\\n"
    "    config.training.save_steps = ft_config.save_steps\\n",
    "    config.training.output_dir = ft_config.output_dir\\n"
    "    config.training.logging_steps = ft_config.logging_steps\\n"
    "    config.training.save_steps = ft_config.save_steps\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
    "        super().__init__(config)\\n"
    "        self.config = config\\n\\n"
    "        backbone_cls = get_backbone_cls(config)\\n",
    "        super().__init__(config)\\n"
    "        self.config = config\\n"
    "        if (\\n"
    "            getattr(config, \\"model_name\\", \\"\\") == \\"" + cosmos_model + "\\"\\n"
    "            and not getattr(config, \\"model_revision\\", None)\\n"
    "        ):\\n"
    "            config.model_revision = \\"" + cosmos_revision + "\\"\\n"
    "        if getattr(config, \\"model_revision\\", None) and \\"revision\\" not in transformers_loading_kwargs:\\n"
    "            transformers_loading_kwargs = {{\\n"
    "                **transformers_loading_kwargs,\\n"
    "                \\"revision\\": config.model_revision,\\n"
    "            }}\\n\\n"
    "        backbone_cls = get_backbone_cls(config)\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        model_name: str = \\"" + cosmos_model + "\\",\\n"
    "        model_type: str = \\"qwen\\",\\n",
    "        model_name: str = \\"" + cosmos_model + "\\",\\n"
    "        model_revision: str | None = \\"" + cosmos_revision + "\\",\\n"
    "        model_type: str = \\"qwen\\",\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        self.model_name = model_name\\n"
    "        self.model_type = model_type\\n\\n",
    "        self.model_name = model_name\\n"
    "        self.model_revision = model_revision\\n"
    "        self.model_type = model_type\\n"
    "        if model_revision and \\"revision\\" not in transformers_loading_kwargs:\\n"
    "            transformers_loading_kwargs = {{\\n"
    "                **transformers_loading_kwargs,\\n"
    "                \\"revision\\": model_revision,\\n"
    "            }}\\n\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        processor_kwargs.setdefault(\\"model_name\\", \\"" + cosmos_model + "\\")\\n",
    "        processor_kwargs.setdefault(\\"model_name\\", \\"" + cosmos_model + "\\")\\n"
    "        processor_kwargs.setdefault(\\"model_revision\\", \\"" + cosmos_revision + "\\")\\n",
)
print("GROOT_RUNTIME_PIN_PATCH_OK " + cosmos_revision)
PY
"""


def build_reload_env_command(
    env_names: tuple[str, ...],
    *,
    port: int,
    restart: bool,
    service_name: str,
    container_name: str,
    server_env_path: str,
    container_env_path: str,
    remote_bash: Callable[[str], str],
) -> str:
    names_json = json.dumps(list(env_names))
    env_assignments = " ".join(name + '="$' + "{" + name + ':-}"' for name in env_names)
    restart_block = ""
    if restart:
        restart_block = f"""
if [ "$mode" = "systemd" ]; then
  sudo systemctl restart {service_name}
elif [ "$mode" = "container" ]; then
  sudo docker restart {container_name} >/dev/null
fi
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:{port}/health" >/dev/null 2>/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:{port}/health" >/dev/null
"""
    script = f"""\
set -euo pipefail
server_env={server_env_path}
container_env={container_env_path}
env_path=""
mode=""
if sudo test -f "$server_env"; then
  env_path="$server_env"
  mode="systemd"
elif sudo test -f "$container_env"; then
  env_path="$container_env"
  mode="container"
else
  echo "No GR00T service env file found" >&2
  exit 2
fi
sudo env {env_assignments} python3 - "$env_path" {shlex.quote(names_json)} <<'PY'
from pathlib import Path
import json
import os
import sys

path = Path(sys.argv[1])
env_names = json.loads(sys.argv[2])
updates = {{name: os.environ.get(name, "") for name in env_names if os.environ.get(name, "")}}
if not updates:
    raise SystemExit("No credential values were supplied")

lines = path.read_text().splitlines() if path.exists() else []
seen = set()
out = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line else ""
    if key in updates:
        if key not in seen:
            out.append(f"{{key}}={{updates[key]}}")
            seen.add(key)
        continue
    out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f"{{key}}={{value}}")
path.write_text("\\n".join(out).rstrip() + "\\n")
path.chmod(0o600)
print("updated_keys=" + ",".join(sorted(updates)))
PY
{restart_block}
echo "NPA_GROOT_RELOAD_ENV_COMPLETE env_path=$env_path mode=$mode"
    """
    return remote_bash(script)


def build_read_env_command(
    *,
    server_env_path: str,
    container_env_path: str,
    remote_bash: Callable[[str], str],
) -> str:
    script = f"""\
set -euo pipefail
server_env={server_env_path}
container_env={container_env_path}
env_path=""
mode=""
if sudo test -f "$server_env"; then
  env_path="$server_env"
  mode="systemd"
elif sudo test -f "$container_env"; then
  env_path="$container_env"
  mode="container"
else
  echo "NPA_GROOT_ENV_READ env_path= mode=missing"
  exit 0
fi
echo "NPA_GROOT_ENV_READ env_path=$env_path mode=$mode"
sudo cat "$env_path" || true
"""
    return remote_bash(script)


def parse_env_read(stdout: str) -> tuple[str, str, str]:
    """Split the env-file marker from the remote file body."""

    env_path = ""
    mode = ""
    body: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("NPA_GROOT_ENV_READ "):
            parts = dict(
                item.split("=", 1)
                for item in line.removeprefix("NPA_GROOT_ENV_READ ").split()
                if "=" in item
            )
            env_path = parts.get("env_path", "")
            mode = parts.get("mode", "")
        else:
            body.append(line)
    return env_path, mode, "\n".join(body) + ("\n" if body else "")
