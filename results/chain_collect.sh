#!/bin/bash
# 100'erlik iki parti topla, birlestir, dogrula (RAM siniri: ~7.5GB/100 bolum)
R=~/Projects/franka_vla_data/results
D=/home/orhan/franka_runs/data
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }

conda activate isaaclab
for S in 77 78; do
  python -u scripts/collect_demos.py \
    --num_envs 8 --num_episodes 100 --env_spacing 25.0 --seed $S --headless \
    --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
    --out $D/clean_s$S.hdf5 > "$R/collect_s$S.log" 2>&1
  log "parti $S toplandi: $(ls -la $D/clean_s$S.hdf5 2>/dev/null | awk '{print $5}') bayt"
done

conda activate lerobot
python -u scripts/merge_hdf5.py --out $D/franka_lift_clean.hdf5 \
  $D/clean_s77.hdf5 $D/clean_s78.hdf5 > "$R/merge_clean.log" 2>&1
log "birlestirildi"
rm -f $D/clean_s77.hdf5 $D/clean_s78.hdf5

python -c "
import h5py, numpy as np, os
n=h5py.File('$D/franka_lift_clean.hdf5')
o=h5py.File(os.path.expanduser('~/Projects/franka_vla_data/data/franka_lift_merged.hdf5'))
for nm,h in [('YENI',n),('ESKI',o)]:
    eps=sorted(h['data'].keys()); d=[]
    for e in range(0,min(200,len(eps)),20):
        w=h['data'][eps[e]]['observation.images.wrist'][:4].astype(np.float32)
        d.append(np.abs(w[1]-w[0]).mean())
    print(f'{nm}: {len(eps)} bolum | kare0->1 bilek farki ort {np.mean(d):5.1f}')
n.close(); o.close()" >> "$R/dogrulama_clean.log" 2>&1
log "TOPLAMA TAMAM ve dogrulandi"
