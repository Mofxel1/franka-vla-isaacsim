#!/bin/bash
# GUVENILIR BASARI ORANI: n=20 cok kucuktu (2/20 ile 1/20 ayrismaz, GA %0-25).
# Iki sabit-kamera modeli icin 60'ar bolum -> GA yaklasik yariya iner.
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
run () {  # $1=ckpt $2=etiket
  conda activate lerobot
  nohup python -u scripts/policy_server.py --ckpt "$1" --device cuda \
    --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
    > "$R/server_$2.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_$2.log" 2>/dev/null && break; sleep 10; done
  conda activate isaaclab
  timeout 4200 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
    --num_episodes 12 --env_spacing 25.0 --headless --fix_cam \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    > "$R/eval_$2.log" 2>&1
  kill $SRV 2>/dev/null; sleep 8
  log "UZUNLUK TESTI: $2 -> $(grep -a 'SONUC' $R/eval_$2.log | tail -1)"
}
run ~/franka_runs/train_fixcam/checkpoints/004000/pretrained_model uzunluk

log "UZUNLUK TESTI TAMAMLANDI"
