#!/bin/bash
# SABIT KAMERA + DAgger.
# Gerekce (2026-09-13 olcumleri):
#   sabit kamera  -> cevrimdisi 41.4 -> 31.6 mm (hassasiyet KAZANILDI)
#                    ama canli a20 42.5 -> 114.7 mm (saglamlik KAYBEDILDI)
#   x egimi 0.932 (az tahmin, guvenli) -> 1.145 (fazla tahmin, kupu deviriyor)
# Randomizasyon bir duzenleyici gorevi goruyormus. Ikisine birden ihtiyac var:
#   sabit kamera (dusuk taban) + DAgger (duz egri).
# DUN EKSIK KALAN: DAgger karelerinde kup firlatilinca yardimci etiket sisiyordu
#   (yardimci std 0.0556 -> 0.0764). Artik --cube_box ile kup calisma alanindan
#   ciktigi KAREDEN itibaren bolum kesiliyor (bolum atilmiyor; erken kareler degerli).
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
O=~/franka_runs/train_fixdag
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_fixdag
DRIVER=~/franka_runs/train_fixcam/checkpoints/004000/pretrained_model
BASE=~/franka_runs/train_fixcam/checkpoints/004000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

# ---------- 1) DAgger toplama, SABIT kamera, fixcam modeli suruyor ----------
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$DRIVER" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8766 \
  > "$R/server_fixdag.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_fixdag.log" 2>/dev/null && break; sleep 5; done
conda activate isaaclab
for S in 801 802; do
  timeout 3000 python -u scripts/collect_demos.py --num_envs 8 --num_episodes 100 \
    --env_spacing 25.0 --seed $S --headless --fix_cam --dagger_port 8766 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/fixdag_s$S.hdf5 > "$R/collect_fixdag_s$S.log" 2>&1
  log "fixdag parti $S toplandi: $(stat -c%s $D/fixdag_s$S.hdf5 2>/dev/null) bayt"
done
kill $SRV 2>/dev/null; sleep 5
log "FIXDAG TOPLAMA TAMAM"

# ---------- 2) Cevirme: uzman (fixcam) + DAgger, kup kutusu filtresiyle ----------
conda activate lerobot
python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/fixcam_s701.hdf5 $D/fixcam_s702.hdf5 $D/fixdag_s801.hdf5 $D/fixdag_s802.hdf5 \
  --repo_id franka_lift_fixdag --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --max_abs 1.5 --cube_box 0.15 0.85 -0.50 0.50 \
  --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_fixdag.log" 2>&1
log "fixdag cevrildi"
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_fixdag', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4), ' (fixcam referansi [0.0571 0.1422])')
ok = st['std'][2]<0.3 and st['std'][8]<0.3 and d.num_episodes>500
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_fixdag.log" 2>&1 || { log "FIXDAG KAPISI BASARISIZ"; exit 1; }
log "fixdag kapisi gecti"

# ---------- 3) Egitim ----------
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_fixdag --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_fixdag.log" 2>&1
log "fixdag egitimi bitti"
CK="$O/checkpoints/004000/pretrained_model"
[ -d "$CK" ] || { log "FIXDAG CHECKPOINT YOK"; exit 1; }

# ---------- 4) Olcum ----------
{
echo "===== SABIT KAMERA + DAgger ====="
HDF5=$D/fixcam_s702.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py "$CK" 45 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_fixdag.log" 2>&1
log "fixdag cevrimdisi olculdu"
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
  > "$R/server_fixdagson.log" 2>&1 &
SRV2=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_fixdagson.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless --fix_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_fixdag.log" 2>&1
kill $SRV2 2>/dev/null; sleep 8
log "FIXDAG KAPALI DONGU HAZIR"
