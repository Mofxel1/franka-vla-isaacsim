#!/bin/bash
# SIFIRDAN egitim: onceden egitilmis VLM + yeni geometrideki veri.
# Eski geometriye uyarlanmis train_objc2 bulasması YOK.
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_temiz
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_geo2
HDF5=/home/orhan/franka_runs/data/franka_lift_geo.hdf5
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_geo2 --dataset.root="$ROOT" \
  --policy.type=smolvla --policy.device=cuda --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  --policy.num_vlm_layers=16 --policy.expert_width_multiplier=0.75 \
  --policy.self_attn_every_n_layers=2 --policy.attention_mode=cross_attn \
  --policy.load_vlm_weights=true \
  --output_dir="$O" --batch_size=32 --steps=8000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_temiz.log" 2>&1
log "temiz egitim bitti"
{
echo "===== SIFIRDAN EGITIM (onceden egitilmis VLM, yeni geometri) ====="
for c in 004000 008000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"
    HDF5=$HDF5 K=8 FRAME=2 OBJ_CENTRIC=1 \
      python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_temiz.log" 2>&1
log "TEMIZ EGITIM OLCULDU"
