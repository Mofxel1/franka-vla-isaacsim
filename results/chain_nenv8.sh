#!/bin/bash
R=~/Projects/franka_vla_data/results
CK=~/franka_runs/train_objc/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_nenv8.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_nenv8.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py \
  --num_envs 8 --num_episodes 45 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --dump_obs "$R/live_obs_nenv8.npz" --out_gif "$R/eval_nenv8.gif" \
  > "$R/eval_nenv8.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "num_envs=8 canli gozlemler kaydedildi"
conda activate lerobot
K=16 FRAME=2 OBJ_CENTRIC=1 python scripts/diagnostics/live_vs_dataset.py \
  "$CK" "$R/live_obs_nenv8.npz" > "$R/sonuc_nenv8.log" 2>&1
log "NUM_ENVS=8 UCURUM OLCULDU"
