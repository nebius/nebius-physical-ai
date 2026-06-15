#!/usr/bin/env bash
# Default BYO trainer: LeRobot policy container vlm-signal-step → NPA_SIM2REAL_OUTPUT_JSON.
# Used when NPA_SIM2REAL_USE_LEROBOT_TRAINER=1 (production runs — no in-process reference trainer).

lerobot_byo_trainer_command() {
  printf '%s' 'python3 -c "import json,os,subprocess,sys;from pathlib import Path;s=os.environ[\"NPA_SIM2REAL_SIGNAL_JSON\"];o=Path(os.environ[\"NPA_SIM2REAL_OUTPUT_JSON\"]);o.parent.mkdir(parents=True,exist_ok=True);d=o.parent/\"vlm-signal-step\";d.mkdir(parents=True,exist_ok=True);p=subprocess.run([sys.executable,\"-m\",\"npa.workbench.lerobot.policy_container\",\"vlm-signal-step\",\"--signal-json\",s,\"--output-dir\",str(d),\"--learning-rate\",os.environ.get(\"NPA_SIM2REAL_LEARNING_RATE\",\"0.05\"),\"--signal-loss-weight\",os.environ.get(\"NPA_SIM2REAL_SIGNAL_LOSS_WEIGHT\",\"1.0\")],capture_output=True,text=True);p.returncode!=0 and (sys.stderr.write(p.stderr or p.stdout or \"vlm-signal-step failed\\n\"),sys.exit(p.returncode));o.write_text(json.dumps(json.loads(p.stdout),indent=2,sort_keys=True)+\"\\n\",encoding=\"utf-8\")"'
}

lerobot_prod_defaults_apply() {
  if [[ "${NPA_SIM2REAL_USE_LEROBOT_POLICY:-0}" == "1" ]] && [[ -z "${POLICY_IMAGE:-}" ]]; then
    export POLICY_IMAGE="${TRAINER_IMAGE:-}"
  fi
  if [[ "${NPA_SIM2REAL_USE_LEROBOT_TRAINER:-0}" == "1" ]] && [[ -z "${BYO_TRAINER_COMMAND:-}" ]]; then
    export BYO_TRAINER_COMMAND
    BYO_TRAINER_COMMAND="$(lerobot_byo_trainer_command)"
  fi
}
