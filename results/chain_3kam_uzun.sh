#!/bin/bash
# 3kam'i 4000 adim DAHA egit (3 kamerada kumulatif 8000).
# Gerekce: model 2 goruntulu bir checkpoint'ten basladi ve ucuncu kamerayi
# kullanmayi ogrenmek icin sadece 4000 adim aldi; kayip 0.032'de hala dusuyordu.
# Canli x egimi zaten 1.009 / kor 0.817 (iyi) ama -32 mm SABIT KAYMA var.
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_3kam_uzun
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_3kam
BASE=~/franka_runs/train_3kam/checkpoints/004000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_3kam --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_3kam_uzun.log" 2>&1
log "3kam uzatma egitimi bitti (3 kamerada kumulatif 8000)"
CK="$O/checkpoints/004000/pretrained_model"
[ -d "$CK" ] || { log "3KAM UZATMA CHECKPOINT YOK -- durduruldu"; exit 1; }
{
echo "===== 3KAM UZATMA (3 kamerada kumulatif 8000) ====="
echo "--- cevrimdisi, TAZE veri (s3_val404) ---"
HDF5=/data/franka_vla/data/s3_val404.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py "$CK" 30 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_3kam_uzun.log" 2>&1
log "3kam uzatma cevrimdisi olculdu"
# kapali dongu -- YENI (render-only) isinma ile
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_3kamuzun.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_3kamuzun.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless --side_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_3kamuzun.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "3KAM UZATMA KAPALI DONGU HAZIR"
