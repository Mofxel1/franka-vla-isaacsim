#!/bin/bash
# 4000. adim checkpoint'i dusunce egitimi durdur ve olc
PID="$1"
R=~/Projects/franka_vla_data/results
CK=~/franka_runs/train_aux_long2/checkpoints/004000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

while kill -0 "$PID" 2>/dev/null; do
  [ -f "$CK/model.safetensors" ] && { sleep 20; kill -INT "$PID" 2>/dev/null; sleep 10; kill "$PID" 2>/dev/null; break; }
  sleep 30
done
log "B uzatma 4000'de durduruldu (kumulatif 12000)"
sleep 15
{
echo "===== B UZATMA: kumulatif 12000 adim ====="
python scripts/diagnostics/localize_test.py "$CK" 40 2>/dev/null \
  | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_12000.log" 2>&1
log "kumulatif 12000 olculdu"
