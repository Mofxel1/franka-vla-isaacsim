#!/bin/bash
# Nesne-merkezli veri hazir olunca: egit -> olc -> kapali dongu
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_objc
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_objc
BASE=~/franka_runs/train_mixed2/checkpoints/012000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

for i in $(seq 1 140); do
  python -c "
import sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_objc', root='$ROOT')
sys.exit(0 if d.num_episodes>900 else 1)" 2>/dev/null && break
  sleep 30
done
log "nesne-merkezli veri seti hazir"

rm -rf "$O"
lerobot-train \
  --dataset.repo_id=franka_lift_objc --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" \
  --batch_size=32 --steps=6000 --save_freq=3000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_objc.log" 2>&1
log "nesne-merkezli egitim bitti"

{
echo "===== NESNE-MERKEZLI AKSIYON (x,y kupe gore) ====="
for c in 003000 006000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"
    K=8 FRAME=2 OBJ_CENTRIC=1 python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_objc.log" 2>&1
log "nesne-merkezli model olculdu"

BEST="$O/checkpoints/006000/pretrained_model"
[ -d "$BEST" ] || BEST="$O/checkpoints/003000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" \
  --device cuda --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_objc.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_objc.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 3000 python -u scripts/eval_policy_isaacsim.py \
  --num_episodes 15 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_objc.gif" > "$R/eval_objc.log" 2>&1
kill $SRV 2>/dev/null
log "NESNE-MERKEZLI SONUC HAZIR"
