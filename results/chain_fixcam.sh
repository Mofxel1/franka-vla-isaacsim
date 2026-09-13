#!/bin/bash
# SABIT KAMERA TESTI. Hipotez (2026-09-13):
#   Kamera her bolumde r+/-0.25m, aci+/-0.35rad, z-0.20/+0.25 oynuyor.
#   "Kup su pikselde -> kup dunyada surada" eslemesi HER BOLUMDE degisiyor.
#   208 bolumle model hem kamerayi cikarmayi hem kupu bulmayi ogrenemiyor.
# Test: SADECE kamerayi sabitle (isik/masa randomize kalir -- tek degisken),
#       ayni donuk-ozellik sondasini kosur, DR'li veriyle karsilastir.
#   DR'li referans (164 bolum): DONUK SigLIP 61.2 mm | x kor 0.358 | y kor 0.872
R=~/Projects/franka_vla_data/results
D=/data/franka_vla/data
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate isaaclab
for S in 701 702; do
  python -u scripts/collect_demos.py --num_envs 8 --num_episodes 100 \
    --env_spacing 25.0 --seed $S --headless --fix_cam \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/fixcam_s$S.hdf5 > "$R/collect_fixcam_s$S.log" 2>&1
  log "sabit-kamera parti $S toplandi: $(stat -c%s $D/fixcam_s$S.hdf5 2>/dev/null) bayt"
done
log "SABIT KAMERA TOPLAMA TAMAM"
conda activate lerobot
{
echo "===== SABIT KAMERA (isik/masa hala randomize) ====="
HDF5=$D/fixcam_s701.hdf5 N_EP=250 PER_EP=12 EPOCHS=40 \
  python -u scripts/diagnostics/frozen_feat_probe.py 2>/dev/null \
  | grep -E "cozunurluk|kup erken|bolum x|TABAN|DONUK SigLIP|sifirdan CNN|^ +x: egim|^ +y: egim|->"
} >> "$R/sonuc_fixcam.log" 2>&1
log "SABIT KAMERA SONDASI OLCULDU"
