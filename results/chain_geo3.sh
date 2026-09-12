#!/bin/bash
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_geo3
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_geo2
BASE=~/franka_runs/train_geo2/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
lerobot-train --dataset.repo_id=franka_lift_geo2 --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=6000 --save_freq=6000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_geo3.log" 2>&1
log "geo3 egitimi bitti (kumulatif 12000)"
{
echo "===== GEO2 UZATMA (kumulatif 9000 / 12000) ====="
for c in 006000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- kumulatif $((6000+10#$c)) ---"
    HDF5=/home/orhan/franka_runs/data/franka_lift_geo.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 \
      python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_geo3.log" 2>&1
log "GEO3 OLCULDU"
