#!/bin/bash
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_geo2
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_geo2
BASE=~/franka_runs/train_objc2/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
python -u scripts/convert_to_lerobot.py --hdf5 /home/orhan/franka_runs/data/franka_lift_geo.hdf5 \
  --repo_id franka_lift_geo2 --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 2 --only_success \
  --overwrite --root /home/orhan/franka_runs/lerobot > "$R/convert_geo2.log" 2>&1
log "geo2 cevrildi (only_success)"
python -c "
import numpy as np
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_geo2', root='$ROOT')
st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('yardimci ort',np.round(st['mean'][8:10],4),'std',np.round(st['std'][8:10],4))
" >> "$R/sonuc_geo2.log" 2>&1
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_geo2 --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=6000 --save_freq=3000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_geo2.log" 2>&1
log "geo2 egitimi bitti"
{
echo "===== GUNCEL GEOMETRI + only_success ====="
for c in 003000 006000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"
    HDF5=/home/orhan/franka_runs/data/franka_lift_geo.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 \
      python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_geo2.log" 2>&1
log "geo2 olculdu"
BEST="$O/checkpoints/006000/pretrained_model"; [ -d "$BEST" ] || BEST="$O/checkpoints/003000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 > "$R/server_geo2.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_geo2.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 3000 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_geo2.gif" > "$R/eval_geo2.log" 2>&1
kill $SRV 2>/dev/null
log "GEO2 SONUC HAZIR"
