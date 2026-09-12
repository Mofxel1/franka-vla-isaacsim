#!/bin/bash
R=~/Projects/franka_vla_data/results
CK=~/franka_runs/train_objc2/checkpoints/006000/pretrained_model
source ~/miniconda3/etc/profile.d/conda.sh; export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
log(){ echo "[$(date +%H:%M)] $*" >> "$R/chain.log"; }
conda activate lerobot
nohup python -u scripts/policy_server.py --ckpt "$CK" --device cuda \
  --object_centric --n_action_steps 25 --n_samples 8 --port 8765 \
  > "$R/server_dump2.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "dinleniyor" "$R/server_dump2.log" 2>/dev/null && break; sleep 10; done
conda activate isaaclab
timeout 3000 python -u scripts/eval_policy_isaacsim.py \
  --num_envs 8 --dump_step 2 --num_episodes 45 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --dump_obs "$R/live_obs_step2.npz" --out_gif "$R/eval_dump2.gif" \
  > "$R/eval_dump2.log" 2>&1
kill $SRV 2>/dev/null; sleep 8
log "adim-2 canli gozlemler kaydedildi"
conda activate lerobot
python3 -c "
import h5py, os, numpy as np
d=np.load('$R/live_obs_step2.npz')
h=h5py.File(os.path.expanduser('~/Projects/franka_vla_data/data/franka_lift_merged.hdf5'))
eps=sorted(h['data'].keys())
ds=np.array([h['data'][eps[i]]['observation.images.front'][2] for i in range(0,240,6)][:40],np.float32)
h.close()
def st(a):
    u=np.mean([len(np.unique(x.astype(np.uint8).reshape(-1,3),axis=0)) for x in a[:8]])
    g=a.mean(-1); return a.mean(), u, float(np.abs(np.diff(g,axis=1)).mean()+np.abs(np.diff(g,axis=0)).mean())
for nm,a in [('veri seti kare2',ds),('canli adim2',d['front'][:40].astype(np.float32))]:
    b,u,e=st(a); print(f'{nm:>18} | parlaklik {b:6.1f} | renk {u:6.0f} | kenar {e:6.2f}')
" >> "$R/sonuc_step2.log" 2>&1
K=16 FRAME=2 OBJ_CENTRIC=1 python scripts/diagnostics/live_vs_dataset.py \
  "$CK" "$R/live_obs_step2.npz" >> "$R/sonuc_step2.log" 2>&1
log "ADIM-2 UCURUM OLCULDU"
