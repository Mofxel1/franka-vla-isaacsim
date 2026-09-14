#!/bin/bash
# DAgger, FAZSIZ UZMANLA. combo (%56) uzerine.
#
# Ilk DAgger denemesi %1 vermisti. Sebep teshis edildi: uzman fazli bir durum
# makinesi; politika kavrama pozuna varamayinca APPROACH'ta takiliyor ve
# "tutucuyu kapat" etiketi HIC uretilmiyor (104 bolumun 96'sinda). Birlesik
# veride sinyal seyreliyor ve model "tutucuyu acik tut" ogreniyor.
#
# Fazsiz uzman fazi HER ADIMDA geometriden hesapliyor; tek basina surdugunde
# 22/24 (%92) basarili -- fazli makinenin %88'iyle ayni seviyede (dogrulandi).
#
# KRITIK DOGRULAMA: toplama bitince "kapali kare orani"na bak. Uzman verisinde
# %41, eski DAgger'da %3.4 idi. Fazsiz uzmanla belirgin yukari cikmali; cikmazsa
# duzeltme ISE YARAMAMIS demektir ve egitime gecmeye gerek yok.
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
O=~/franka_runs/train_combodag
ROOT=/home/orhan/franka_runs/lerobot/franka_lift_combodag
DRIVER=~/franka_runs/train_combo/checkpoints/004000/pretrained_model
BASE=~/franka_runs/train_combo/checkpoints/004000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

# ---------- 1) TOPLAMA: combo suruyor, FAZSIZ uzman etiketliyor ----------
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$DRIVER" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8766 \
  > "$R/server_combodag.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_combodag.log" 2>/dev/null && break; sleep 5; done
conda activate isaaclab
# RAM: 3 kamera + politika sunucusu ile 100'luk parti OOM ile oluyor.
# OLCULDU (2026-09-14): iki parti de bolum 99'da, anon-rss 14.9 GB ile SIGKILL.
# Bolum basina ~0.15 GB -> 50'lik parti ~7.5 GB + Isaac 3 + sunucu 2.5 = 13 GB.
for S in 1001 1002 1003 1004; do
  timeout 3000 python -u scripts/collect_demos.py --num_envs 8 --num_episodes 50 \
    --env_spacing 25.0 --seed $S --headless --fix_cam --side_cam \
    --stateless_expert --dagger_port 8766 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/combodag_s$S.hdf5 > "$R/collect_combodag_s$S.log" 2>&1
  log "combodag parti $S: basarili $(grep -ac BASARILI $R/collect_combodag_s$S.log)/$(grep -ac 'TOPLA] bolum' $R/collect_combodag_s$S.log) | yazildi: $(ls -la $D/combodag_s$S.hdf5 2>/dev/null | awk '{print $5}')"
done
kill $SRV 2>/dev/null; sleep 5
log "COMBODAG TOPLAMA TAMAM"

# ---------- 2) KRITIK KAPI: "kapat" etiketi uretildi mi ----------
conda activate lerobot
python -c "
import numpy as np, h5py, sys
tot=[]; hic=0; n=0
for s in (1001,1002,1003,1004):
    h=h5py.File(f'$D/combodag_s{s}.hdf5','r')
    for e in sorted(h['data'].keys()):
        g=h['data'][e]['action'][12:162,7]
        tot.append((g<0).mean()); n+=1
        if (g<0).sum()==0: hic+=1
    h.close()
oran=100*np.mean(tot)
print(f'kapali kare orani %{oran:.1f}  (UZMAN %41.1 / eski DAgger %3.4)')
print(f'tutucuyu HIC kapatmayan bolum {hic}/{n}  (eski DAgger 96/104)')
ok = oran > 15
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ -- fazsiz uzman da etiket uretmiyor')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_combodag.log" 2>&1 || { log "COMBODAG ETIKET KAPISI BASARISIZ -- durduruldu"; exit 1; }
log "combodag etiket kapisi gecti"

# ---------- 3) CEVIRME: combo (uzman) + combodag (DAgger) ----------
python -u scripts/convert_to_lerobot.py \
  --hdf5 $D/combo_s901.hdf5 $D/combo_s902.hdf5 $D/combodag_s1001.hdf5 $D/combodag_s1002.hdf5 $D/combodag_s1003.hdf5 $D/combodag_s1004.hdf5 \
  --repo_id franka_lift_combodag --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --skip_episodes 8 --max_abs 1.5 --cube_box 0.15 0.85 -0.50 0.50 \
  --overwrite --root /home/orhan/franka_runs/lerobot \
  > "$R/convert_combodag.log" 2>&1
log "combodag cevrildi"
# NOT: --only_success YOK. DAgger'da bolumlerin cogu basarisiz ama ETIKET
# uzmanin ve degerli olan tam da o durumlar.
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_combodag', root='$ROOT'); st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('kameralar:', [k for k in d.meta.features if k.startswith('observation.images')])
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4), ' (combo referansi [0.0543 0.1371])')
ok = ('observation.images.side' in d.meta.features and st['std'][2]<0.3
      and st['std'][8]<0.3 and d.num_episodes>600)
print('DOGRULAMA:', 'GECTI' if ok else 'BASARISIZ')
sys.exit(0 if ok else 1)
" >> "$R/sonuc_combodag.log" 2>&1 || { log "COMBODAG KAPISI BASARISIZ"; exit 1; }
log "combodag kapisi gecti"

# ---------- 4) EGITIM ----------
rm -rf "$O"
lerobot-train --dataset.repo_id=franka_lift_combodag --dataset.root="$ROOT" \
  --policy.path="$BASE" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$O" --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000 > "$R/train_combodag.log" 2>&1
log "combodag egitimi bitti"
CK="$O/checkpoints/004000/pretrained_model"
[ -d "$CK" ] || { log "COMBODAG CHECKPOINT YOK"; exit 1; }

# ---------- 5) OLCUM ----------
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
  > "$R/server_combodagson.log" 2>&1 &
SRV2=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_combodagson.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 4200 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
  --num_episodes 200 --env_spacing 25.0 --headless --fix_cam --side_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  > "$R/eval_combodag.log" 2>&1
kill $SRV2 2>/dev/null; sleep 8
log "COMBODAG SONUC: $(grep -a 'SONUC' $R/eval_combodag.log | tail -1)  (combo referansi %56)"
