#!/bin/bash
# 2026-09-13: eval'de kup env.step()'ten SONRA okunuyordu -> bolum bitince
# Isaac Lab ortami step() icinde sifirliyor ve final_z hep taze kupun
# yuksekligi (0.055) cikiyordu. success = peak>0.10 AND final>0.10 oldugu
# icin BASARI YAPISAL OLARAK IMKANSIZDI. Projedeki tum 0/20 bunun eseri.
# Duzeltildi. Simdi en iyi modeller YENIDEN olculuyor.
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
run () {  # $1=ckpt $2=etiket $3=ekstra eval bayraklari
  conda activate lerobot
  nohup python -u scripts/policy_server.py --ckpt "$1" --device cuda \
    --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
    > "$R/server_$2.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_$2.log" 2>/dev/null && break; sleep 10; done
  conda activate isaaclab
  timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
    --num_episodes 20 --env_spacing 25.0 --headless $3 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    > "$R/eval_$2.log" 2>&1
  kill $SRV 2>/dev/null; sleep 8
  log "DOGRU OLCUM: $2 -> $(grep -a 'SONUC' $R/eval_$2.log | tail -1)"
}
run ~/franka_runs/train_fixdag/checkpoints/004000/pretrained_model d_fixdag "--fix_cam"
run ~/franka_runs/train_fixcam/checkpoints/004000/pretrained_model d_fixcam "--fix_cam"
run ~/franka_runs/train_geo3/checkpoints/006000/pretrained_model  d_geo3   ""
log "DOGRU OLCUM TAMAMLANDI"
