#!/bin/bash
# A egitimi bitince: A'yi olc -> B'yi egit -> B'yi olc
A_PID="$1"
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data

log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

# --- 1) A bitsin ---
while kill -0 "$A_PID" 2>/dev/null; do sleep 30; done
log "A egitimi bitti"

# --- 2) A'yi olc ---
{
echo "=========== A: kirpilmis veri, toplam 6000 adim ==========="
for c in 002000 004000; do
  d="/data/franka_vla/train_early2/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && python scripts/diagnostics/localize_test.py "$d" 40 2>/dev/null \
    | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
done
} >> "$R/sonuc_A.log" 2>&1
log "A olculdu"

# --- 3) B veri seti hazir mi ---
for i in $(seq 1 60); do
  python -c "
import os,sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_aux', root=os.path.expanduser('~/Projects/franka_vla_data/lerobot/franka_lift_aux'))
sys.exit(0 if d.num_episodes>300 else 1)" 2>/dev/null && break
  sleep 30
done
log "B veri seti hazir"

# --- 4) B'yi egit (A ile AYNI baslangic noktasi, AYNI adim sayisi) ---
rm -rf /data/franka_vla/train_aux
lerobot-train \
  --dataset.repo_id=franka_lift_aux \
  --dataset.root=/home/orhan/Projects/franka_vla_data/lerobot/franka_lift_aux \
  --policy.path=/data/franka_vla/train_early/checkpoints/002000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/data/franka_vla/train_aux \
  --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 \
  > "$R/train_aux.log" 2>&1
log "B egitimi bitti"

# --- 5) B'yi olc ---
{
echo "=========== B: yardimci gorev (kup konumu aksiyonda), 4000 adim ==========="
d="/data/franka_vla/train_aux/checkpoints/004000/pretrained_model"
[ -d "$d" ] && python scripts/diagnostics/localize_test.py "$d" 40 2>/dev/null \
  | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_B.log" 2>&1
log "B olculdu -- HEPSI BITTI"
