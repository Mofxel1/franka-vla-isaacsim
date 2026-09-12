#!/bin/bash
# DAgger TAM KOSU: topla -> cevir -> egit -> olc
#
# Teshis (2026-09-12): hedefleme hatasi adim 20'de 43mm, adim 60'ta 77mm.
# Veri setinde bu bozulma YOK (kare taramasi f40'ta 6.2mm). Yani sorun gec
# kareler degil, kolun DAGITIM DISINA cikmasi = kovaryat kayma.
# DART (uzmanin kendi yorungesine gurultu) denendi, tabani iki katina cikardi.
# DAgger politikanin GERCEKTEN gezdigi durumlardan uzman duzeltmesi toplar.
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
O=~/franka_runs/train_dagger
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_dagger
BASE=~/franka_runs/train_geo3/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

# ---------- 1) TOPLAMA: politika surer, uzman etiketler ----------
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$BASE" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8766 \
  > "$R/server_dagger.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_dagger.log" 2>/dev/null && break; sleep 5; done
conda activate isaaclab
for S in 601 602; do
  timeout 3000 python -u scripts/collect_demos.py --num_envs 8 --num_episodes 100 \
    --env_spacing 25.0 --seed $S --headless --dagger_port 8766 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/dagger_s$S.hdf5 > "$R/collect_dagger_s$S.log" 2>&1
  log "DAgger parti $S toplandi: $(stat -c%s $D/dagger_s$S.hdf5 2>/dev/null) bayt"
done
kill $SRV 2>/dev/null; sleep 5
log "DAGGER TOPLAMA TAMAM"

# ---------- 2) CEVIRME: orijinal uzman verisi + DAgger verisi ----------
# Standart DAgger toplulastirir: D_1 U D_2 U ... Tek basina DAgger verisiyle
# egitmek keskinligi kaybettirebilir (DART dersi).
conda activate lerobot
python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/franka_lift_geo.hdf5 $D/dagger_s601.hdf5 $D/dagger_s602.hdf5 \
  --repo_id franka_lift_dagger --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --max_abs 1.5 --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_dagger.log" 2>&1
log "dagger cevrildi"
# NOT: --only_success YOK. DAgger'da bolumlerin cogu basarisiz (politika
# beceriksiz) ama ETIKET uzmanin ve degerli olan tam da o durumlar.

python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_dagger', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4))
ok = st['std'][2]<0.3 and st['std'][8]<0.3 and d.num_episodes>500
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_dagger.log" 2>&1 || { log "DAGGER KAPISI BASARISIZ -- durduruldu"; exit 1; }
log "dagger kapisi gecti"

# ---------- 3) EGITIM ----------
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_dagger --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_dagger2.log" 2>&1
log "dagger egitimi bitti"
CK="$O/checkpoints/004000/pretrained_model"
[ -d "$CK" ] || { log "DAGGER CHECKPOINT YOK -- durduruldu"; exit 1; }

# ---------- 4) OLCUM ----------
{
echo "===== DAGGER (uzman verisi + politika dagilimi) ====="
echo "--- cevrimdisi, TAZE veri (s3_val404) ---"
HDF5=$D/s3_val404.hdf5 K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py "$CK" 30 2>/dev/null \
  | grep -E "n=|medyan|ortalama|  x:|  y:|YARDIMCI|    "
} >> "$R/sonuc_dagger.log" 2>&1
log "dagger cevrimdisi olculdu"

nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_daggerson.log" 2>&1 &
SRV2=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_daggerson.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 2400 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 20 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_dagger2.log" 2>&1
kill $SRV2 2>/dev/null; sleep 8
log "DAGGER KAPALI DONGU HAZIR"
