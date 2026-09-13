#!/bin/bash
# SABIT KAMERA veri setini cevir + istatistik kapisi.
# Sonda sonucu (2026-09-13): x korelasyonu randomize 0.35 -> sabit 0.72-0.81.
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_fixcam
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/fixcam_s701.hdf5 $D/fixcam_s702.hdf5 \
  --repo_id franka_lift_fixcam --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --only_success --max_abs 1.5 --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_fixcam.log" 2>&1
log "fixcam cevrildi"
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_fixcam', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('kameralar:', [k for k in d.meta.features if k.startswith('observation.images')])
print('aksiyon std[:3]', np.round(st['std'][:3],4), ' (REST\'siz beklenen ~[0.012,0.030,0.105])')
print('yardimci std   ', np.round(st['std'][8:10],4))
ok = st['std'][2]<0.3 and st['std'][8]<0.3 and d.num_episodes>300
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_fixcam.log" 2>&1 || { log "FIXCAM KAPISI BASARISIZ"; exit 1; }
log "fixcam kapisi gecti -- EGITIME HAZIR"
