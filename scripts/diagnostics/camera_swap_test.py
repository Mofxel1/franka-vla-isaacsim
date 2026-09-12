"""HANGI KAMERA BOZUYOR?

Canli goruntude modelin kup tahmini saglam (65mm) ama hareket plani cokuyor
(197mm). Aksiyonlar GORELI oldugu icin dogru plan hem kupun hem KOLUN yerini
bilmeyi gerektirir; durum vektoru sifirli oldugundan kol pozu BILEK kamerasindan
cikarilmak zorunda.

Dort kombinasyon denenir:
  veri-on + veri-bilek   (referans, iyi olmali)
  canli-on + canli-bilek (kapali dongu, kotu)
  canli-on + veri-bilek  -> duzelirse sucu BILEK kamerasi
  veri-on + canli-bilek  -> bozulursa sucu BILEK kamerasi

Kolun ilk karede evde oldugu ve her iki kaynakta ayni oldugu varsayimiyla,
bilek goruntusunu takas etmek mesru: tasidigi bilgi (kol pozu) ayni.
"""
import os, sys, numpy as np, torch, h5py
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

CKPT, NPZ = sys.argv[1], sys.argv[2]
DEV, K = "cuda", int(os.environ.get("K", "8"))
TASK = "pick up the cube and lift it"
cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); pol.eval(); pol.to(DEV)
pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})
ZS = np.zeros(16, dtype=np.float32)

def plan(f, w):
    b = {"observation.images.front": torch.from_numpy(f).float().permute(2,0,1).unsqueeze(0)/255.,
         "observation.images.wrist": torch.from_numpy(w).float().permute(2,0,1).unsqueeze(0)/255.,
         "observation.state": torch.from_numpy(ZS).unsqueeze(0), "task": [TASK]}
    with torch.no_grad():
        return np.mean([post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy() for _ in range(K)], axis=0)

def descent(ch, ee):
    a = ch[:, :3] + ee
    return a[int(np.argmin(a[:, 2])), :2]

d = np.load(NPZ); n = len(d["front"])
h = h5py.File(os.path.expanduser("~/Projects/franka_vla_data/data/franka_lift_merged.hdf5"))
eps = sorted(h["data"].keys())
ds_front, ds_wrist, ds_ee, ds_cube = [], [], [], []
for i in range(0, len(eps), max(1, len(eps)//n)):
    if len(ds_front) >= n: break
    ep = h["data"][eps[i]]
    ds_front.append(ep["observation.images.front"][0]); ds_wrist.append(ep["observation.images.wrist"][0])
    ds_ee.append(ep["observation.state.ee_pos"][0]); ds_cube.append(ep["observation.state.object_pos"][0][:2])
h.close()

# bilek goruntusu istatistikleri
lw = d["wrist"].astype(np.float32); dw = np.array(ds_wrist, dtype=np.float32)
lf = d["front"].astype(np.float32); df = np.array(ds_front, dtype=np.float32)
print("goruntu istatistikleri (parlaklik ort / std / benzersiz renk):")
for nm, a in [("veri-on ", df), ("canli-on ", lf), ("veri-bilek ", dw), ("canli-bilek", lw)]:
    u = len(np.unique(a[0].astype(np.uint8).reshape(-1, 3), axis=0))
    print(f"  {nm}: {a.mean():6.1f} / {a.std():5.1f} / {u}")

combos = [
    ("veri-on  + veri-bilek  (referans)", lambda i: (ds_front[i], ds_wrist[i]), ds_ee, ds_cube),
    ("canli-on + canli-bilek (kapali dongu)", lambda i: (d["front"][i], d["wrist"][i]),
        [d["state"][i][9:12] for i in range(n)], [d["cube"][i][:2] for i in range(n)]),
    ("canli-on + VERI-bilek", lambda i: (d["front"][i], ds_wrist[i]),
        [d["state"][i][9:12] for i in range(n)], [d["cube"][i][:2] for i in range(n)]),
    ("VERI-on  + canli-bilek", lambda i: (ds_front[i], d["wrist"][i]), ds_ee, ds_cube),
]
print(f"\n{'kombinasyon':>38} | {'plan medyan':>11} | {'x kor':>6} | {'okuma medyan':>12}")
print("-" * 82)
for name, get, ees, cubes in combos:
    err, pred, true, aux = [], [], [], []
    for i in range(n):
        f, w = get(i); ch = plan(f, w)
        xy = descent(ch, ees[i]); c = np.array(cubes[i])
        err.append(np.linalg.norm(xy - c)); pred.append(xy); true.append(c)
        if ch.shape[1] >= 10: aux.append(np.mean(ch[:, 8:10], axis=0))
    err, pred, true = np.array(err), np.array(pred), np.array(true)
    xk = np.corrcoef(true[:,0], pred[:,0])[0,1]
    am = np.median(np.linalg.norm(np.array(aux)-true, axis=1))*1000 if aux else float('nan')
    print(f"{name:>38} | {np.median(err)*1000:8.1f}mm | {xk:+6.3f} | {am:9.1f}mm")
