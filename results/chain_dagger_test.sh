#!/bin/bash
# DAgger altyapisi DOGRULAMA kosusu (kucuk). Tam toplama oncesi:
#  - toplu cikarim calisiyor mu (8 ortam bagimsiz aksiyon aliyor mu)
#  - etiket UZMANIN komutu mu (politikanin degil)
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
CKPT=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CKPT" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8766 \
  > "$R/server_daggertest.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_daggertest.log" 2>/dev/null && break; sleep 5; done
conda activate isaaclab
timeout 900 python -u scripts/collect_demos.py --num_envs 8 --num_episodes 6 \
  --env_spacing 25.0 --seed 501 --headless --dagger_port 8766 \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out $D/dagger_test.hdf5 > "$R/collect_daggertest.log" 2>&1
kill $SRV 2>/dev/null; sleep 5
log "DAGGER DOGRULAMA KOSUSU BITTI"
