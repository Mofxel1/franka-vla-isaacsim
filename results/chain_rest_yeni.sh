#!/bin/bash
# ISINMA DUZELTMESI TESTI: render-only isinma kolu evde birakiyor mu?
# Beklenen ee pozu (egitim, kare 2): (+0.375, +0.057, +0.328)
# Eski isinmada olculen        : (+0.284, +0.506, +0.107)  -- 45cm sapma
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
CKPT=~/franka_runs/train_rest/checkpoints/004000/pretrained_model
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CKPT" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_restyeni.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_restyeni.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless \
  --dump_obs "$R/live_obs_rest_yeni.npz" --dump_step 2 \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_restyeni.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "REST YENI ISINMA OLCULDU"
