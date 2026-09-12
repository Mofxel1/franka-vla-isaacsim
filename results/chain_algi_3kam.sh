#!/bin/bash
# 3 KAMERA kapali dongu. DIKKAT: eval'e --side_cam GECMEK SART; yoksa yan
# kamera sahnede olusmaz, sunucuya gitmez ve model egitildiginden EKSIK
# goruntuyle calisir (sessizce bozulur).
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
CKPT=~/franka_runs/train_3kam/checkpoints/004000/pretrained_model
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CKPT" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_3kam.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_3kam.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless --side_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_3kam.gif" > "$R/eval_3kam.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "3KAM KAPALI DONGU HAZIR"
