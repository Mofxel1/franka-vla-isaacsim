#!/bin/bash
# ELENEN MODELLERI DOGRU METRIKLE YENIDEN OLC.
#
# Bu projedeki TUM model karsilastirmalari bozuk eval ile yapildi:
#   - final_z sifirlanmis kupten okunuyordu -> success imkansizdi
#   - 0. ortamin aksiyonu 8 ortama yayiniliyordu -> gercegin ~1/20'si olculuyordu
# Kanit: train_fixcam eski olcumde 1/60 (%2), yeni olcumde 76/200 (%38).
# Yani "0/20, ise yaramadi" diye elenen dallarin bir kismi YANLIS elenmis olabilir.
#
# DIKKAT: her model KENDI egitim rejimiyle olculur.
#   randomize kamerayla egitilenlerde --fix_cam YOK
#   uc kameraliya --side_cam SART (yoksa model eksik girdiyle calisir)
R=~/Projects/franka_vla_data/results
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
run () {  # $1=ckpt $2=etiket $3=eval bayraklari
  conda activate lerobot
  nohup python -u scripts/policy_server.py --ckpt "$1" --device cuda \
    --object_centric --n_action_steps 25 --n_samples 1 --port 8765 \
    > "$R/server_$2.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_$2.log" 2>/dev/null && break; sleep 5; done
  conda activate isaaclab
  timeout 4200 python -u scripts/eval_policy_isaacsim.py --num_envs 8 \
    --num_episodes 200 --env_spacing 25.0 --headless $3 \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    > "$R/eval_$2.log" 2>&1
  kill $SRV 2>/dev/null; sleep 8
  log "YENIDEN: $2 -> $(grep -a 'SONUC' $R/eval_$2.log | tail -1)"
}
run ~/franka_runs/train_geo3/checkpoints/006000/pretrained_model  y_geo3  ""
run ~/franka_runs/train_rest/checkpoints/004000/pretrained_model  y_rest  ""
run ~/franka_runs/train_3kam/checkpoints/004000/pretrained_model  y_3kam  "--side_cam"
run ~/franka_runs/train_dart2/checkpoints/006000/pretrained_model y_dart2 ""
log "YENIDEN OLCUM TAMAMLANDI"
