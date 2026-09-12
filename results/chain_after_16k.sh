#!/bin/bash
# Egitim bitince: 12000 ve 16000'i olc -> en iyisiyle kapali dongu testi
PID="$1"
R=~/Projects/franka_vla_data/results
CKDIR=~/franka_runs/train_aux_long2/checkpoints
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

while kill -0 "$PID" 2>/dev/null; do sleep 30; done
log "B uzatma bitti (kumulatif 16000)"
sleep 20

conda activate lerobot
{
echo "===== B UZATMA: kumulatif 12000 ve 16000 ====="
for c in 004000 008000; do
  d="$CKDIR/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- kumulatif $((8000+10#$c)) adim ---"
    python scripts/diagnostics/localize_test.py "$d" 40 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_16000.log" 2>&1
log "kumulatif 12000+16000 olculdu"

# --- kapali dongu: en iyi checkpoint ---
BEST="$CKDIR/008000/pretrained_model"
[ -d "$BEST" ] || BEST="$CKDIR/004000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" \
  --device cuda --relative_actions --n_action_steps 25 --port 8765 \
  > "$R/server_16k.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_16k.log" 2>/dev/null && break; sleep 10; done
log "politika sunucusu hazir, kapali dongu basliyor"

conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py \
  --num_episodes 10 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif "$R/eval_16k.gif" > "$R/eval_16k.log" 2>&1
kill $SRV 2>/dev/null
log "kapali dongu bitti -- HAZIR"
