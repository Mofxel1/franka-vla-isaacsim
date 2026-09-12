"""CEVRIMDISI / CANLI UCURUMU

Ayni model, ayni olcum, iki farkli goruntu kaynagi:
  - VERI SETI goruntusu (HDF5'ten ham kare)      -> egitimde gorulen dagitim
  - CANLI eval goruntusu (Isaac Sim'den npz)     -> kapali donguda gelen dagitim

Olcum localize_test.py ile ayni: bolum basi gozlemden 50 adimlik plan alinir,
planin z'si en dusuk adimi = inis noktasi, XY'si kupun GERCEK XY'siyle karsilastirilir.

Ikisi BENZERSE  -> sorun goruntude degil, algi seviyesinde; egitime devam mantikli.
CANLI KOTUYSE   -> egitim/dagitim uyumsuzlugu var; algi ne kadar iyilesirse
                   iyilessin kapali dongu calismaz, once bu kapatilmali.

Kullanim:
  python live_vs_dataset.py <checkpoint> <canli_obs.npz>
npz, eval_policy_isaacsim.py --dump_obs ile uretilir.
"""
import os, sys, numpy as np, torch, h5py
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

CKPT, NPZ = sys.argv[1], sys.argv[2]
DEV, K = os.environ.get("PROBE_DEV", "cuda"), int(os.environ.get("K", "8"))
# Veri setinde 0. kare BAYAT (onceki bolumun son karesi). Varsayilan 2.
FR = int(os.environ.get("FRAME", "2"))
# Nesne-merkezli model: x,y kupe gore -> modelin kendi tahminiyle mutlaklastir
OBJC = os.environ.get("OBJ_CENTRIC", "0") == "1"
TASK = "pick up the cube and lift it"

cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); pol.eval(); pol.to(DEV)
pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})
ZS = np.zeros(16, dtype=np.float32)


def plan(front_u8, wrist_u8):
    b = {"observation.images.front":
            torch.from_numpy(front_u8).float().permute(2, 0, 1).unsqueeze(0) / 255.,
         "observation.images.wrist":
            torch.from_numpy(wrist_u8).float().permute(2, 0, 1).unsqueeze(0) / 255.,
         "observation.state": torch.from_numpy(ZS).unsqueeze(0),
         "task": [TASK]}
    with torch.no_grad():
        return np.mean([post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy()
                        for _ in range(K)], axis=0)


def descent_xy(ch, ee):
    if OBJC:
        absxyz = ch[:, :3].copy()
        absxyz[:, 0] += ch[:, 8]
        absxyz[:, 1] += ch[:, 9]
    else:
        absxyz = ch[:, :3] + ee
    return absxyz[int(np.argmin(absxyz[:, 2])), :2]


def report(name, err, pred, true, aux=None):
    err, pred, true = np.array(err), np.array(pred), np.array(true)
    print(f"\n--- {name} (n={len(err)}) ---")
    print(f"  hareket plani : medyan {np.median(err)*1000:6.1f} mm | "
          f"ortalama {err.mean()*1000:6.1f} mm")
    for j, ax in enumerate("xy"):
        print(f"    {ax}: egim {np.polyfit(true[:,j], pred[:,j], 1)[0]:+.3f}  "
              f"kor {np.corrcoef(true[:,j], pred[:,j])[0,1]:+.3f}")
    if aux is not None and len(aux):
        aux = np.array(aux)
        ae = np.linalg.norm(aux - true, axis=1)
        print(f"  dogrudan okuma: medyan {np.median(ae)*1000:6.1f} mm | "
              f"ortalama {ae.mean()*1000:6.1f} mm")
        for j, ax in enumerate("xy"):
            print(f"    {ax}: egim {np.polyfit(true[:,j], aux[:,j], 1)[0]:+.3f}  "
                  f"kor {np.corrcoef(true[:,j], aux[:,j])[0,1]:+.3f}")


# ---------- CANLI ----------
d = np.load(NPZ)
n_live = len(d["front"])
e, p, t, a = [], [], [], []
for i in range(n_live):
    ch = plan(d["front"][i], d["wrist"][i])
    xy = descent_xy(ch, d["state"][i][9:12])
    cube = d["cube"][i][:2]
    e.append(np.linalg.norm(xy - cube)); p.append(xy); t.append(cube)
    if ch.shape[1] >= 10:
        a.append(np.mean(ch[:, 8:10], axis=0))

# ---------- VERI SETI (ayni sayida bolum) ----------
# Kamera geometrisi 2026-09-04'te degisti; model hangi veriyle egitildiyse
# olcum de ONUNLA yapilmali. HDF5 ortam degiskeniyle secilir.
_H5 = os.environ.get("HDF5", "~/Projects/franka_vla_data/data/franka_lift_merged.hdf5")
h = h5py.File(os.path.expanduser(_H5))
eps = sorted(h["data"].keys())
e2, p2, t2, a2 = [], [], [], []
for i in range(0, len(eps), max(1, len(eps) // n_live)):
    if len(e2) >= n_live: break
    ep = h["data"][eps[i]]
    ch = plan(ep["observation.images.front"][FR], ep["observation.images.wrist"][FR])
    xy = descent_xy(ch, ep["observation.state.ee_pos"][FR])
    cube = ep["observation.state.object_pos"][FR][:2]
    e2.append(np.linalg.norm(xy - cube)); p2.append(xy); t2.append(cube)
    if ch.shape[1] >= 10:
        a2.append(np.mean(ch[:, 8:10], axis=0))
h.close()

print(f"\ncheckpoint: {CKPT}")
print(f"veri seti kare: {FR}" + ("  | NESNE-MERKEZLI" if OBJC else ""))
report("VERI SETI goruntusu (egitim dagitimi)", e2, p2, t2, a2)
report("CANLI eval goruntusu (kapali dongu dagitimi)", e, p, t, a)
print(f"\n  referans: sabit-orta 136mm | CNN 12mm | kavrama esigi 20mm")
print("\n  Ikisi benzerse: sorun goruntude degil, egitime devam mantikli.")
print("  Canli belirgin kotuyse: dagitim uyumsuzlugu ONCE kapatilmali.")
