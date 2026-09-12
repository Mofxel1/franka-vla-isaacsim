#!/bin/bash
# REST'siz modelin KAPALI DONGU algisi. Asil test: eval adim 0'da tam REST
# penceresinde basliyor; hipotez dogruysa canli taban 61 mm'den dusmeli.
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
run () {  # $1=ckpt  $2=etiket
  conda activate lerobot
  nohup python -u scripts/policy_server.py --ckpt "$1" --device cuda \
    --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
    > "$R/server_$2.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_$2.log" 2>/dev/null && break; sleep 10; done
  conda activate isaaclab
  timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
    --num_episodes 20 --env_spacing 25.0 --headless \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out_gif "$R/eval_$2.gif" > "$R/eval_$2.log" 2>&1
  kill $SRV 2>/dev/null; sleep 8
  log "algi olcumu bitti: $2"
}
run ~/franka_runs/train_rest/checkpoints/004000/pretrained_model restalgi
log "REST ALGI OLCUMU HAZIR"
