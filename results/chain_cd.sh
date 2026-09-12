#!/bin/bash
# Deney 1: B'yi uzat (aux_repeat=1, 4000 -> 8000 adim)
# Deney 2: yardimci gorev agirligini artir (aux_repeat=4, %50 pay), B ile ayni baslangic
R=~/Projects/franka_vla_data/results
O=~/franka_runs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
measure(){ python scripts/diagnostics/localize_test.py "$1" 40 2>/dev/null \
  | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }

# ---------- Deney 1: B'yi uzat ----------
log "Deney1 basladi: B uzatma (aux_repeat=1, +4000 adim)"
rm -rf "$O/train_aux_long"
lerobot-train \
  --dataset.repo_id=franka_lift_aux \
  --dataset.root=/home/orhan/Projects/franka_vla_data/lerobot/franka_lift_aux \
  --policy.path=/data/franka_vla/train_aux/checkpoints/004000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O/train_aux_long" \
  --batch_size=32 --steps=4000 --save_freq=2000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_aux_long.log" 2>&1
log "Deney1 egitimi bitti"
{
echo "===== DENEY 1: yardimci gorev UZATILDI (aux_repeat=1) ====="
for c in 002000 004000; do
  d="$O/train_aux_long/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- kumulatif $((4000+10#$c)) adim ---"; measure "$d"; }
done
} >> "$R/sonuc_deney1.log" 2>&1
log "Deney1 olculdu"

# ---------- aux4 veri seti hazir mi ----------
for i in $(seq 1 80); do
  python -c "
import os,sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_aux4', root=os.path.expanduser('~/Projects/franka_vla_data/lerobot/franka_lift_aux4'))
sys.exit(0 if d.num_episodes>300 else 1)" 2>/dev/null && break
  sleep 30
done
log "aux4 veri seti hazir"

# ---------- Deney 2: agirlik artirildi ----------
log "Deney2 basladi: yardimci gorev agirligi %20 -> %50"
rm -rf "$O/train_aux4"
lerobot-train \
  --dataset.repo_id=franka_lift_aux4 \
  --dataset.root=/home/orhan/Projects/franka_vla_data/lerobot/franka_lift_aux4 \
  --policy.path=/data/franka_vla/train_early/checkpoints/002000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O/train_aux4" \
  --batch_size=32 --steps=4000 --save_freq=2000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_aux4.log" 2>&1
log "Deney2 egitimi bitti"
{
echo "===== DENEY 2: yardimci gorev AGIRLIGI %50 (aux_repeat=4) ====="
echo "(B ile ayni baslangic: train_early@2000, ayni adim sayisi)"
for c in 002000 004000; do
  d="$O/train_aux4/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"; measure "$d"; }
done
} >> "$R/sonuc_deney2.log" 2>&1
log "Deney2 olculdu -- HEPSI BITTI"
