#!/bin/bash
R=~/Projects/franka_vla_data/results
D=/home/orhan/franka_runs/data
O=~/franka_runs/train_dart2
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_dart2
BASE=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

python -u scripts/convert_to_lerobot.py --hdf5 $D/franka_lift_dart.hdf5 \
  --repo_id franka_lift_dart2 --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 2 \
  --only_success --max_abs 1.5 --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_dart2.log" 2>&1
log "dart2 cevrildi"

# istatistik dogrulamasi -- gecmiste iki kez bu asamada bozukluk kacti
python -c "
import numpy as np
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_dart2', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4))
print('aksiyon maks[:3]', np.round(st['max'][:3],3))
" >> "$R/sonuc_dart2.log" 2>&1

rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_dart2 --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=6000 --save_freq=6000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_dart2.log" 2>&1
log "dart2 egitimi bitti"
{
echo "===== DART2 (gurultulu + sinir filtresi) ====="
HDF5=$D/franka_lift_dart.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 \
  python scripts/diagnostics/localize_test.py "$O/checkpoints/006000/pretrained_model" 45 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_dart2.log" 2>&1
log "dart2 olculdu"
nohup python -u scripts/policy_server.py --ckpt "$O/checkpoints/006000/pretrained_model" \
  --device cuda --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_dart2.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_dart2.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_dart2.gif" > "$R/eval_dart2.log" 2>&1
kill $SRV 2>/dev/null
log "DART2 SONUC HAZIR"
