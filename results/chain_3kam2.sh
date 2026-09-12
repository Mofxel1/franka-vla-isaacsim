#!/bin/bash
# 3 kamera: cevir + egit + olc. Toplama chain_3kam.sh'de bitti.
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
O=~/franka_runs/train_3kam
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_3kam
# DIKKAT: geo3'un kendi config'inde input_features DOLU; lerobot make_policy
# (factory.py:305) sadece BOS ise veri setinden dolduruyor. Dogrudan geo3'ten
# devam etseydik yan kamera SESSIZCE yok sayilacakti. Bu kopya config'e
# observation.images.side eklenmis hali (agirliklar sembolik bagli).
BASE=~/franka_runs/base_geo3_3kam
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/s3_s401.hdf5 $D/s3_s402.hdf5 $D/s3_s403.hdf5 \
  --repo_id franka_lift_3kam --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --only_success --max_abs 1.5 --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_3kam.log" 2>&1
log "3kam cevrildi"

# --- kapi: veri saglam MI ve YAN KAMERA veri setinde VAR MI ---
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_3kam', root='$ROOT'); st=d.meta.stats['action']
keys=[k for k in d.meta.features if k.startswith('observation.images')]
print('bolum',d.num_episodes,'kare',d.num_frames)
print('kameralar:', keys)
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4))
ok = ('observation.images.side' in keys) and st['std'][2]<0.3 and st['std'][8]<0.3 and d.num_episodes>300
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_3kam.log" 2>&1 || { log "3KAM KAPISI BASARISIZ -- durduruldu"; exit 1; }
log "3kam kapisi gecti"

rm -rf "$O"
# 3 goruntu -> gorsel token 1.5x -> 6 GB kartta batch 32 sigmayabilir.
# Once 32 denenir, VRAM patlarsa 16'ya dusulur (adim iki katina cikarilir).
BS=32; STEPS=4000
lerobot-train --dataset.repo_id=franka_lift_3kam --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=$BS --steps=$STEPS --save_freq=$STEPS --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_3kam.log" 2>&1
if ! [ -d "$O/checkpoints/$(printf %06d $STEPS)/pretrained_model" ]; then
  log "3kam batch=32 BASARISIZ (muhtemelen VRAM) -> batch=16 / 8000 adim"
  rm -rf "$O"; BS=16; STEPS=8000
  lerobot-train --dataset.repo_id=franka_lift_3kam --dataset.root="$ROOT" \
    --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
    --output_dir="$O" --batch_size=$BS --steps=$STEPS --save_freq=$STEPS --log_freq=250 \
    --num_workers=2 --seed=1000 > "$R/train_3kam_bs16.log" 2>&1
fi
log "3kam egitimi bitti (batch=$BS adim=$STEPS)"
{
echo "===== UC KAMERA (on + yan + bilek) ====="
HDF5=$D/s3_s401.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py \
  "$O/checkpoints/$(printf %06d $STEPS)/pretrained_model" 45 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_3kam.log" 2>&1
log "3KAM OLCULDU"
