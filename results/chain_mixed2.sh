#!/bin/bash
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
{
echo "===== KARMA UZATMA: kumulatif 10000 / 14000 / 18000 ====="
for c in 004000 008000 012000; do
  d="$O/checkpoints/$c/pretrained_model"
  [ -d "$d" ] && { echo "--- kumulatif $((6000+10#$c)) adim ---"
    python scripts/diagnostics/localize_test.py "$d" 45 2>/dev/null \
      | grep -E "checkpoint|n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "; }
done
} >> "$R/sonuc_mixed2.log" 2>&1
log "karma uzatma olculdu"

BEST="$O/checkpoints/012000/pretrained_model"
[ -d "$BEST" ] || BEST="$O/checkpoints/008000/pretrained_model"
nohup python -u scripts/policy_server.py --ckpt "$BEST" \
  --device cuda --relative_actions --n_action_steps 25 --port 8765 \
  > "$R/server_m2.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_m2.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 3600 python -u scripts/eval_policy_isaacsim.py \
  --num_episodes 20 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --dump_obs "$R/live_obs_m2.npz" --out_gif "$R/eval_m2.gif" \
  > "$R/eval_m2.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
conda activate lerobot
K=16 python scripts/diagnostics/live_vs_dataset.py "$BEST" "$R/live_obs_m2.npz" \
  >> "$R/sonuc_mixed2.log" 2>&1
log "KARMA UZATMA SONUCU HAZIR (olcum + kapali dongu + ucurum)"
