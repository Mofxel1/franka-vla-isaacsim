#!/bin/bash
# Veri seti hazir olunca: tam bolum + yardimci gorev ile egit -> olc -> kapali dongu
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_full_aux
BASE=~/franka_runs/train_aux_long2/checkpoints/008000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

# --- veri seti hazir mi (tam bolum: ~76000 kare) ---
for i in $(seq 1 120); do
  python -c "
import os,sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_full_aux', root=os.path.expanduser('~/Projects/franka_vla_data/lerobot/franka_lift_full_aux'))
sys.exit(0 if d.num_frames>70000 else 1)" 2>/dev/null && break
  sleep 30
done
log "tam bolum + yardimci gorev veri seti hazir"

# --- egit ---
rm -rf "$O"
lerobot-train \
  --dataset.repo_id=franka_lift_full_aux \
  --dataset.root=/home/orhan/Projects/franka_vla_data/lerobot/franka_lift_full_aux \
  --policy.path="$BASE" \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" \
  --batch_size=32 --steps=6000 --save_freq=3000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_full_aux.log" 2>&1
log "tam bolum egitimi bitti"

# --- olc ---
{
echo "===== TAM BOLUM + YARDIMCI GOREV (algiyi ogrenmis modelden devam) ====="
for c in 003000 006000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"
    python scripts/diagnostics/localize_test.py "$d" 40 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_full_aux.log" 2>&1
log "tam bolum modeli olculdu"

# --- kapali dongu (ARTIK ANLAMLI: model tum gorevi gordu) ---
BEST="$O/checkpoints/006000/pretrained_model"
[ -d "$BEST" ] || BEST="$O/checkpoints/003000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" \
  --device cuda --relative_actions --n_action_steps 25 --port 8765 \
  > "$R/server_full.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_full.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py \
  --num_episodes 10 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_full.gif" > "$R/eval_full.log" 2>&1
kill $SRV 2>/dev/null
log "kapali dongu bitti -- TAM BOLUM SONUCU HAZIR"
