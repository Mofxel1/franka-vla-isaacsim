#!/bin/bash
# Egitim bitince: lokalizasyon (TEMIZ kare) -> kapali dongu A/B (tek ornek vs ortalama)
PID="$1"
R=~/Projects/franka_vla_data/results
O=~/franka_runs/train_mixed2
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

while kill -0 "$PID" 2>/dev/null; do sleep 30; done
log "karma uzatma bitti (kumulatif 18000)"
sleep 20
conda activate lerobot

BEST="$O/checkpoints/012000/pretrained_model"
[ -d "$BEST" ] || BEST="$O/checkpoints/008000/pretrained_model"

{
echo "===== KARMA UZATMA -- TEMIZ KARE (FRAME=2) ile olcum ====="
for c in 004000 008000 012000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- kumulatif $((6000+10#$c)) adim ---"
    K=8 FRAME=2 python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_mixed2.log" 2>&1
log "lokalizasyon olculdu (temiz kare)"

# ---------- KAPALI DONGU A/B ----------
run_eval () {   # $1 = n_samples, $2 = etiket
  nohup python -u scripts/policy_server.py --ckpt "$BEST" \
    --device cuda --relative_actions --n_action_steps 25 \
    --n_samples "$1" --port 8765 > "$R/server_$2.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_$2.log" 2>/dev/null && break; sleep 10; done
  conda activate isaaclab
  timeout 3000 python -u scripts/eval_policy_isaacsim.py \
    --num_episodes 15 --env_spacing 25.0 --headless \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out_gif "$R/eval_$2.gif" > "$R/eval_$2.log" 2>&1
  kill $SRV 2>/dev/null; sleep 8
  conda activate lerobot
  log "kapali dongu bitti: $2 (n_samples=$1)"
}

run_eval 1 "tek"
run_eval 8 "ort8"

{
echo "===== KAPALI DONGU A/B: tek ornek vs 8 ornek ortalamasi ====="
for t in tek ort8; do
  echo "--- $t ---"
  grep -E "SONUC|^\[IZ\]   " "$R/eval_$t.log" 2>/dev/null
  echo "kaldirma yuksekligi (tepe_z) dagilimi:"
  grep -oE "tepe_z=[0-9.]+" "$R/eval_$t.log" 2>/dev/null | cut -d= -f2 | sort -rn | head -5 | tr '\n' ' '
  echo
done
} >> "$R/sonuc_ab_samples.log" 2>&1
log "A/B SONUCU HAZIR"
