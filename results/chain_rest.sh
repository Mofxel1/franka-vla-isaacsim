#!/bin/bash
# REST KARELERI CIKARILMIS veri: uzman durum makinesi bolum basinda 0.2 s
# (= 12 kare) REST yapar ve des_ee_pose = ee_pose yazar ("neredeysen kal").
# O karelerde etiket kupe degil kolun rastgele pozuna bakar -> ogrenilemez.
# skip_first 2 -> 12. repeat_early kopyalari da S0'dan sonra basliyor.
R=~/Projects/franka_vla_data/results
D=/home/orhan/franka_runs/data
O=~/franka_runs/train_rest
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_rest
BASE=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

python -u scripts/convert_to_lerobot.py --hdf5 $D/franka_lift_geo.hdf5 \
  --repo_id franka_lift_rest --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --only_success --max_abs 1.5 --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_rest.log" 2>&1 || { log "REST CEVIRME PATLADI"; exit 1; }
log "rest cevrildi (skip_first=12)"

# --- istatistik kapisi ---
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_rest', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('aksiyon std[:3]', np.round(st['std'][:3],4), ' (beklenen ~[0.08,0.06,0.11])')
print('yardimci std   ', np.round(st['std'][8:10],4), ' (beklenen ~[0.06,0.14])')
ok = st['std'][2] < 0.3 and st['std'][8] < 0.3 and d.num_episodes > 300
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_rest.log" 2>&1 || { log "REST ISTATISTIK KAPISI BASARISIZ -- durduruldu"; exit 1; }
log "rest istatistik kapisi gecti"

rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_rest --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_rest.log" 2>&1
log "rest egitimi bitti"
{
echo "===== REST KARELERI CIKARILDI (skip_first=12) ====="
for rs in 0 12; do
  echo "--- REST_SKIP=$rs ---"
  HDF5=$D/franka_lift_geo.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=$rs \
    python scripts/diagnostics/localize_test.py "$O/checkpoints/004000/pretrained_model" 45 2>/dev/null \
    | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
done
} >> "$R/sonuc_rest.log" 2>&1
log "REST OLCULDU"
