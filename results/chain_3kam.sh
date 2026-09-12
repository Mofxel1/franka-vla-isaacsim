#!/bin/bash
# UC KAMERA: on + YAN + bilek. Gerekce (2026-09-08 olcumu):
#   canli adim 0'da x kor 0.722 / y kor 0.849, kayma x -39mm / y -8.6mm
#   -> zayif eksen x, yani ON kameranin DERINLIK ekseni.
# Yan kamera (eye 0.50,-1.45,0.70 -> tgt 0.50,0,0.15) y boyunca bakar;
# x o goruste YANAL olur -- kameranin en iyi olctugu sey. Ucgenleme.
# Birlestirme YOK: cevirici coklu dosya okuyor (disk 14 GB, birlestirme sigmaz).
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

conda activate isaaclab
# 3 kamera -> kare basina ~1.5x RAM. 100'luk parti 2 kamerada ~7.5 GB idi,
# 3 kamerada ~11 GB olur; 17 GB bosta var ama guvenli taraf: 70'lik partiler.
for S in 401 402 403; do
  python -u scripts/collect_demos.py --num_envs 8 --num_episodes 70 \
    --env_spacing 25.0 --seed $S --headless --side_cam \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/s3_s$S.hdf5 > "$R/collect_3kam_s$S.log" 2>&1
  log "3kam parti $S toplandi: $(stat -c%s $D/s3_s$S.hdf5 2>/dev/null) bayt"
done
log "3 KAMERA TOPLAMA TAMAM"
