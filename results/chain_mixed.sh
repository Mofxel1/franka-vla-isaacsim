#!/bin/bash
# Karma veri seti hazir olunca: egit -> olc -> kapali dongu
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_mixed
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_mixed
BASE=~/franka_runs/train_aux_long2/checkpoints/008000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

for i in $(seq 1 140); do
  python -c "
import sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_mixed', root='$ROOT')
sys.exit(0 if d.num_episodes>900 else 1)" 2>/dev/null && break
  sleep 30
done
log "karma veri seti hazir"

rm -rf "$O"
lerobot-train \
  --dataset.repo_id=franka_lift_mixed --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" \
  --batch_size=32 --steps=6000 --save_freq=3000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_mixed.log" 2>&1
log "karma egitim bitti"

{
echo "===== KARMA VERI SETI (150 kare + ilk 60'in 2 kopyasi, yardimci gorev) ====="
for c in 003000 006000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"
    python scripts/diagnostics/localize_test.py "$d" 40 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_mixed.log" 2>&1
log "karma model olculdu"

BEST="$O/checkpoints/006000/pretrained_model"
[ -d "$BEST" ] || BEST="$O/checkpoints/003000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" \
  --device cuda --relative_actions --n_action_steps 25 --port 8765 \
  > "$R/server_mixed.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_mixed.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py \
  --num_episodes 10 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_mixed.gif" > "$R/eval_mixed.log" 2>&1
kill $SRV 2>/dev/null
log "KARMA SONUC HAZIR (olcum + kapali dongu)"
