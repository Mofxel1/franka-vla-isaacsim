#!/bin/bash
# Izlemek icin: 8 ortami yan yana gosteren dosemeli video, 3 dalga = 24 bolum.
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
CK=~/franka_runs/train_fixcam/checkpoints/004000/pretrained_model
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
  > "$R/server_video.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_video.log" 2>/dev/null && break; sleep 5; done
conda activate isaaclab
timeout 1800 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 32 --env_spacing 25.0 --headless --fix_cam \
  --record_waves 4 --record_dir ~/isaac_captures/fixcam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_video.log" 2>&1
kill $SRV 2>/dev/null; sleep 5
log "VIDEO HAZIR: $(grep -a 'SONUC' $R/eval_video.log | tail -1)"
