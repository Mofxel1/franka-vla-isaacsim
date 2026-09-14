#!/bin/bash
# %56'lik model (train_combo) icin README videosu.
# DIKKAT: 3 kamerali model -> eval'e --side_cam SART; sabit kamerayla egitildi
# -> --fix_cam SART. Ikisi de yoksa model eksik/yanlis girdiyle calisir.
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
CK=~/franka_runs/train_combo/checkpoints/004000/pretrained_model
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
  > "$R/server_video56.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_video56.log" 2>/dev/null && break; sleep 5; done
conda activate isaaclab
timeout 1800 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 32 --env_spacing 25.0 --headless --fix_cam --side_cam \
  --record_waves 5 --record_dir ~/isaac_captures/combo \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_video56.log" 2>&1
kill $SRV 2>/dev/null; sleep 5
log "VIDEO56: $(grep -a 'SONUC' $R/eval_video56.log | tail -1)"
