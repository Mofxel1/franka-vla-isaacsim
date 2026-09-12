#!/bin/bash
# Guncel kamera geometrisiyle yeniden topla -> cevir -> egit -> olc -> kapali dongu
R=~/Projects/franka_vla_data/results
D=/home/orhan/franka_runs/data
O=~/franka_runs/train_geo
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_geo
BASE=~/franka_runs/train_objc2/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

# --- topla: 100'erlik iki parti (RAM siniri) ---
conda activate isaaclab
for S in 101 102; do
  python -u scripts/collect_demos.py --num_envs 8 --num_episodes 100 \
    --env_spacing 25.0 --seed $S --headless \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/geo_s$S.hdf5 > "$R/collect_geo_s$S.log" 2>&1
  log "parti $S toplandi"
done
conda activate lerobot
python -u scripts/merge_hdf5.py --out $D/franka_lift_geo.hdf5 \
  $D/geo_s101.hdf5 $D/geo_s102.hdf5 > "$R/merge_geo.log" 2>&1
rm -f $D/geo_s101.hdf5 $D/geo_s102.hdf5
log "birlestirildi"

# --- cevir: objc recetesi + skip_first ---
python -u scripts/convert_to_lerobot.py --hdf5 $D/franka_lift_geo.hdf5 \
  --repo_id franka_lift_geo --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 2 \
  --overwrite --root /home/orhan/franka_runs/lerobot > "$R/convert_geo.log" 2>&1
log "cevrildi"

# --- egit ---
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_geo --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=6000 --save_freq=3000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_geo.log" 2>&1
log "egitim bitti"

# --- olc ---
{
echo "===== GUNCEL KAMERA GEOMETRISIYLE TOPLANAN VERI ====="
for c in 003000 006000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- $((10#$c)) adim ---"
    K=8 FRAME=2 OBJ_CENTRIC=1 python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_geo.log" 2>&1
log "olculdu"

# --- kapali dongu ---
BEST="$O/checkpoints/006000/pretrained_model"; [ -d "$BEST" ] || BEST="$O/checkpoints/003000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 > "$R/server_geo.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_geo.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 3000 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_geo.gif" > "$R/eval_geo.log" 2>&1
kill $SRV 2>/dev/null
log "GEO SONUC HAZIR"
