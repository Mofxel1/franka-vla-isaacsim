#!/bin/bash
# Yan kameranin CANLI goruntusunu dok ki egitim goruntusuyle karsilastirilabilsin.
# Gerekce: 3 kameralı modelin canli x korelasyonu 0.794 -> 0.285 cokuyor.
# On kamerada ayni imza 2026-09-04'te gercek bir geometri uyusmazligi cikmisti.
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
CKPT=~/franka_runs/train_3kam/checkpoints/004000/pretrained_model
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CKPT" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_yandump.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_yandump.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 1800 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless --side_cam \
  --dump_obs "$R/live_obs_3kam.npz" --dump_step 2 \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_yandump.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "YAN KAMERA CANLI DOKUM HAZIR"
