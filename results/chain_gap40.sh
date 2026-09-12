#!/bin/bash
R=~/Projects/franka_vla_data/results
CK=~/franka_runs/train_mixed/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CK" \
  --device cuda --relative_actions --n_action_steps 25 --port 8765 \
  > "$R/server_gap40.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_gap40.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 3600 python -u scripts/eval_policy_isaacsim.py \
  --num_episodes 45 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --dump_obs "$R/live_obs_45.npz" --out_gif "$R/eval_gap40.gif" \
  > "$R/eval_gap40.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "45 canli gozlem kaydedildi"

conda activate lerobot
K=16 python scripts/diagnostics/live_vs_dataset.py "$CK" "$R/live_obs_45.npz" \
  > "$R/sonuc_gap45.log" 2>&1
log "UCURUM n=45 K=16 ile OLCULDU"
