#!/bin/bash
# KAZANANLARI BIRLESTIR. Uc bagimsiz kazanc, hicbiri birlikte denenmedi:
#   sabit kamera  %38   (--fix_cam)
#   DART gurultu  %31   (--action_noise 0.020)
#   ucuncu kamera %18   (--side_cam)
# Ayrica ilk dalga duzeltmesi: masa carpisma govdesi ilk reset'te hazir degil,
# kup yere dusuyor -> cevirmede ilk 8 bolum atlaniyor, evalde ilk 8 sayilmiyor.
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
O=~/franka_runs/train_combo
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_combo
BASE=~/franka_runs/train_fixcam/checkpoints/004000/pretrained_model
BASE3=~/franka_runs/base_fixcam_3kam
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

# ---------- 1) TOPLAMA: sabit kamera + yan kamera + DART gurultusu ----------
conda activate isaaclab
for S in 901 902; do
  python -u scripts/collect_demos.py --num_envs 8 --num_episodes 100 \
    --env_spacing 25.0 --seed $S --headless --fix_cam --side_cam \
    --action_noise 0.020 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/combo_s$S.hdf5 > "$R/collect_combo_s$S.log" 2>&1
  log "combo parti $S: $(stat -c%s $D/combo_s$S.hdf5 2>/dev/null) bayt, basarili $(grep -ac BASARILI $R/collect_combo_s$S.log)/104"
done
log "COMBO TOPLAMA TAMAM"

# ---------- 2) 3 kameraya hazir baslangic checkpoint'i ----------
# LeRobot make_policy (factory.py:305) veri setindeki kameralari SADECE
# checkpoint config'inin input_features'i BOSSA dolduruyor. Mevcut bir
# checkpoint'ten devam ederken yeni kamera SESSIZCE duser.
rm -rf "$BASE3"; mkdir -p "$BASE3"
for f in "$BASE"/*; do b=$(basename "$f"); [ "$b" = config.json ] && continue; ln -s "$f" "$BASE3/$b"; done
conda activate lerobot
python -c "
import json
c=json.load(open('$BASE/config.json'))
c['input_features']['observation.images.side'] = dict(c['input_features']['observation.images.front'])
json.dump(c, open('$BASE3/config.json','w'), indent=2)
print('girdi ozellikleri:', list(json.load(open('$BASE3/config.json'))['input_features']))
" >> "$R/sonuc_combo.log" 2>&1

# ---------- 3) CEVIRME ----------
python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/combo_s901.hdf5 $D/combo_s902.hdf5 \
  --repo_id franka_lift_combo --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --skip_episodes 8 --only_success --max_abs 1.5 \
  --cube_box 0.15 0.85 -0.50 0.50 \
  --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_combo.log" 2>&1
log "combo cevrildi"
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_combo', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('kameralar:', [k for k in d.meta.features if k.startswith('observation.images')])
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4))
ok = ('observation.images.side' in d.meta.features and st['std'][2]<0.3
      and st['std'][8]<0.3 and d.num_episodes>300)
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_combo.log" 2>&1 || { log "COMBO KAPISI BASARISIZ"; exit 1; }
log "combo kapisi gecti"

# ---------- 4) EGITIM ----------
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_combo --dataset.root="$ROOT" \
  --policy.path="$BASE3" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_combo.log" 2>&1
log "combo egitimi bitti"
CK="$O/checkpoints/004000/pretrained_model"
[ -d "$CK" ] || { log "COMBO CHECKPOINT YOK"; exit 1; }

# ---------- 5) OLCUM: dogrudan basari orani ----------
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
  > "$R/server_combo.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_combo.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 4200 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 200 --env_spacing 25.0 --headless --fix_cam --side_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_combo.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "COMBO SONUC: $(grep -a 'SONUC' $R/eval_combo.log | tail -1)"
