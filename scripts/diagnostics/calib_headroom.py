"""
KALIBRASYON TAVANI: hatanin ne kadari DUZELTILEBILIR bir sistematik bozulma?

Canli olcumde model su anda: x egim 0.932, y egim 0.919, x kayma -11 mm.
Egimin 1'den kucuk olmasi "ortalamaya cekilme" demek -- MSE ile egitilen her
regresor belirsizlik altinda bunu yapar. Ama bu SISTEMATIK: cikarim sirasinda
geri olceklenebilir.

Bu betik hatayi uc kademede ayirir:

  1. HAM                      modelin oldugu gibi hatasi
  2. KAYMA duzeltilmis        sabit ofset cikarilir
  3. KAYMA + EGIM duzeltilmis regresyon dogrusu tersine cevrilir

  3. kademe 20 mm'ye yaklasiyorsa -> modelin TASIDIGI bilgi neredeyse yeterli;
     kalan is kalibrasyon, yeni veri/mimari degil
  3. kademe hala uzaksa         -> bilgi gercekten eksik

DIKKAT: duzeltme katsayilari AYNI veriden ogrenilirse bu bir ust sinirdir
(iyimser). Bu yuzden K-katli capraz dogrulama ile de raporlanir: katsayilar
verinin bir kismindan ogrenilip DIGER kisminda uygulanir.

Kullanim: python calib_headroom.py <npz> <ckpt>
"""
import os, sys
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

NPZ = sys.argv[1]
CKPT = os.path.expanduser(sys.argv[2])
DEV = os.environ.get("PROBE_DEV", "cuda")
K = int(os.environ.get("K", "8"))
TASK = "pick up the cube and lift it"

d = np.load(NPZ)
cube = d["cube"][:, :2]
cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); pol.eval(); pol.to(DEV)
pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})
want = [k for k in cfg.input_features if k.startswith("observation.images")]
short = [k.rsplit(".", 1)[1] for k in want]
print(f"{NPZ}\n  n={len(cube)} | kameralar={short}\n")

zs = np.zeros(16, dtype=np.float32)
pred = []
for i in range(len(cube)):
    b = {"observation.state": torch.from_numpy(zs).unsqueeze(0), "task": [TASK]}
    for key, s in zip(want, short):
        b[key] = torch.from_numpy(d[s][i]).float().permute(2, 0, 1).unsqueeze(0) / 255.
    with torch.no_grad():
        ch = np.mean([post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy()
                      for _ in range(K)], axis=0)
    pred.append(np.mean(ch[:, 8:10], axis=0))
pred = np.array(pred)


def med(p):
    return np.median(np.linalg.norm(p - cube, axis=1)) * 1000


print(f"  {'kademe':<34}{'medyan':>9}")
print("  " + "-" * 43)
print(f"  {'1. HAM':<34}{med(pred):>8.1f} mm")

p2 = pred - np.median(pred - cube, axis=0)
print(f"  {'2. KAYMA duzeltilmis':<34}{med(p2):>8.1f} mm")

p3 = pred.copy()
for j in range(2):
    a, b_ = np.polyfit(cube[:, j], pred[:, j], 1)
    p3[:, j] = (pred[:, j] - b_) / a
print(f"  {'3. KAYMA + EGIM duzeltilmis':<34}{med(p3):>8.1f} mm   (ust sinir, iyimser)")

# --- K-katli: katsayilar BASKA verilerden ogrenilir ---
rng = np.random.default_rng(0)
idx = rng.permutation(len(cube))
folds = np.array_split(idx, min(5, len(cube)))
p4 = pred.copy()
for f in folds:
    tr = np.setdiff1d(idx, f)
    for j in range(2):
        a, b_ = np.polyfit(cube[tr, j], pred[tr, j], 1)
        p4[f, j] = (pred[f, j] - b_) / a
print(f"  {'4. ayni ama CAPRAZ DOGRULANMIS':<34}{med(p4):>8.1f} mm   <-- gercekci")

print(f"\n  kavrama esigi 20 mm")
for j, ax in enumerate("xy"):
    a, b_ = np.polyfit(cube[:, j], pred[:, j], 1)
    print(f"    {ax}: egim {a:+.3f}  kesme {b_:+.4f}  "
          f"kayma {np.median(pred[:,j]-cube[:,j])*1000:+.1f} mm")
