"""
Kucultme (shrinkage) olcumu: model tahmini = EGIM x uzman + sabit.

Egim 1.0 = model mesafeyi dogru tahmin ediyor.
Egim 0.7 = model %30 kisa tahmin ediyor -> kola kupe varmadan duruyor.

DIKKAT - SmolVLA cikarimi STOKASTIK: flow matching rastgele gurultuden
basladigi icin ayni girdiye her cagride farkli cevap verir (std ~0.008,
aksiyonlar ~0.05 mertebesinde, yani %15 bagil gurultu). Bu yuzden:
  - her sorgu K kez tekrarlanip ORTALAMASI alinir
  - bolum basina birden fazla kareden ornek alinir
  - egim icin bootstrap ile %90 guven araligi hesaplanir
Tek ornekli olcum yaniltici: 2026-08-24'te ayni checkpoint icin 0.950 ve 0.492
okunup "iyilesme"/"gerileme" saniidi; ikisi de ayni guven araligindaydi.

Kullanim:
  python measure_slope.py <checkpoint_yolu> [repo_id] [cihaz] [K]
"""
import os, sys, numpy as np, torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = os.path.expanduser(sys.argv[1])
REPO = sys.argv[2] if len(sys.argv) > 2 else "franka_lift_rel2"
DEV  = sys.argv[3] if len(sys.argv) > 3 else "cuda"
K    = int(sys.argv[4]) if len(sys.argv) > 4 else 8
N_EP = int(os.environ.get("N_EP", "40"))
FRAMES = [20, 30, 45, 60]        # yaklasma evresinin farkli anlari

ROOT = os.path.expanduser(f"~/Projects/franka_vla_data/lerobot/{REPO}")
ds = LeRobotDataset(REPO, root=ROOT)
pol = SmolVLAPolicy.from_pretrained(CKPT); pol.eval(); pol.to(DEV); pol.config.device = DEV
pre, post = make_pre_post_processors(
    policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})

step = max(1, ds.num_episodes // N_EP)
starts = [int(ds.meta.episodes["dataset_from_index"][i]) for i in range(0, ds.num_episodes, step)]

R, P = [], []
for st in starts:
    for off in FRAMES:
        s = ds[st + off]
        b = {kk: s[kk].unsqueeze(0) for kk in
             ["observation.images.front", "observation.images.wrist", "observation.state"]}
        b["task"] = [s["task"]]
        acc = []
        for _ in range(K):
            with torch.no_grad():
                pol.reset()
                acc.append(post(pol.select_action(pre(b)))[0].cpu().numpy()[:3])
        P.append(np.mean(acc, axis=0))
        R.append(s["action"][:3].numpy())
R, P = np.array(R), np.array(P)


def boot_ci(r, p, n=800):
    rng = np.random.default_rng(0)
    out = [np.polyfit(r[i], p[i], 1)[0]
           for i in (rng.integers(0, len(r), len(r)) for _ in range(n))]
    return np.percentile(out, [5, 95])


print(f"checkpoint : {os.path.basename(os.path.dirname(CKPT))}   veri: {REPO}")
print(f"orneklem   : {len(R)} nokta ({len(starts)} bolum x {len(FRAMES)} kare), "
      f"her nokta {K} cikarimin ortalamasi")
print()
print(f"{'eksen':>6} | {'EGIM':>7} | {'%90 guven araligi':>19} | {'kor.':>6} | {'ort.hata':>9}")
print("-" * 62)
for k, ax in enumerate("xyz"):
    a, _ = np.polyfit(R[:, k], P[:, k], 1)
    lo, hi = boot_ci(R[:, k], P[:, k])
    c = np.corrcoef(R[:, k], P[:, k])[0, 1]
    e = np.abs(R[:, k] - P[:, k]).mean()
    verdict = "OK" if lo > 0.85 else ("kuculme" if hi < 0.85 else "belirsiz")
    print(f"{ax:>6} | {a:7.3f} | [{lo:6.3f}, {hi:6.3f}]  | {c:6.3f} | {e:9.4f}  {verdict}")
print()
print("OK       = guven araliginin ALTI 0.85 ustunde -> olcek dogru")
print("kuculme  = guven araliginin USTU 0.85 altinda -> model kisa tahmin ediyor")
print("belirsiz = aralik 0.85'i iceriyor -> orneklem yetersiz, karar verme")
