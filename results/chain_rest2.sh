#!/bin/bash
# chain_rest.sh cevirmeyi BITIRDI ama ozet satirindaki eski tek-dosya varsayimi
# yuzunden sifir olmayan kodla cikti (veri saglam, dogrulandi). Kaldigi yerden.
R=~/Projects/franka_vla_data/results
D=/home/orhan/franka_runs/data
O=~/franka_runs/train_rest
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_rest
BASE=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_rest --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_rest.log" 2>&1
log "rest egitimi bitti"
{
echo "===== REST KARELERI CIKARILDI (skip_first=12) ====="
echo "veri: 546 bolum 49140 kare | aksiyon std[:3] [0.0117 0.0293 0.1053]"
for rs in 0 12; do
  echo "--- REST_SKIP=$rs ---"
  HDF5=$D/franka_lift_geo.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=$rs \
    python scripts/diagnostics/localize_test.py "$O/checkpoints/004000/pretrained_model" 45 2>/dev/null \
    | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
done
} >> "$R/sonuc_rest.log" 2>&1
log "REST OLCULDU"
