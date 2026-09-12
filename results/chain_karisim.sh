#!/bin/bash
# TEMIZ + GURULTULU karisim: keskinlik (temiz) + sapmadan toparlanma (DART)
R=~/Projects/franka_vla_data/results
D=/home/orhan/franka_runs/data
O=~/franka_runs/train_karisim
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_karisim
BASE=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

# Birlestirme YOK -- cevirici iki dosyayi dogrudan okuyor (10 GB kopya olmuyor)
python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/franka_lift_geo.hdf5 $D/franka_lift_dart.hdf5 \
  --repo_id franka_lift_karisim --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 2 \
  --only_success --max_abs 1.5 --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_karisim.log" 2>&1
log "cevrildi"

# --- otomatik istatistik dogrulamasi (uc kez bu asamada bozukluk kacti) ---
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_karisim', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('aksiyon std[:3]', np.round(st['std'][:3],4), ' (beklenen ~[0.08,0.06,0.11])')
print('yardimci std   ', np.round(st['std'][8:10],4), ' (beklenen ~[0.06,0.14])')
ok = st['std'][2] < 0.3 and st['std'][8] < 0.3
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ -- egitim baslatilmayacak')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_karisim.log" 2>&1 || { log "ISTATISTIK DOGRULAMASI BASARISIZ -- durduruldu"; exit 1; }
log "istatistik dogrulamasi gecti"

rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_karisim --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_karisim.log" 2>&1
log "karisim egitimi bitti"
{
echo "===== KARISIM (temiz + gurultulu) ====="
HDF5=$D/franka_lift_geo.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 \
  python scripts/diagnostics/localize_test.py "$O/checkpoints/004000/pretrained_model" 45 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_karisim.log" 2>&1
log "KARISIM OLCULDU"
