#!/bin/bash
# SABIT KAMERA: egit + olc + kapali dongu.
# DIKKAT: veri --fix_cam ile toplandi; eval de --fix_cam ile kosmali.
# Aksi halde boru hattinin iki ucu yine uyusmaz (bu projede 7 kez oldu).
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
O=~/franka_runs/train_fixcam
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_fixcam
BASE=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot

# cevirme bitene kadar bekle
for i in $(seq 1 200); do
  grep -q "fixcam kapisi gecti" "$R/chain.log" && break
  grep -q "FIXCAM KAPISI BASARISIZ" "$R/chain.log" && { log "fixcam egitimi IPTAL (kapi)"; exit 1; }
  sleep 20
done

rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_fixcam --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_fixcam.log" 2>&1
log "fixcam egitimi bitti"
CK="$O/checkpoints/004000/pretrained_model"
[ -d "$CK" ] || { log "FIXCAM CHECKPOINT YOK"; exit 1; }

{
echo "===== SABIT KAMERA MODELI ====="
echo "--- cevrimdisi, kendi dagilimindan TAZE veri (fixcam_s702) ---"
HDF5=$D/fixcam_s702.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py "$CK" 45 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_fixcam.log" 2>&1
log "fixcam cevrimdisi olculdu"

nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
  > "$R/server_fixcam.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_fixcam.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless --fix_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_fixcam.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "FIXCAM KAPALI DONGU HAZIR"
