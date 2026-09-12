"""
KESIN TEST: kol EVDEYKEN model kupun yerini GORUNTUDEN okuyabiliyor mu?

Bolumun 10. karesini veriyoruz (kol henuz hareket etmemis, ee_pos kupun yerini
ELE VERMIYOR: R^2=0.089). Modelin urettigi 50 adimlik planin ilerideki
adimlarina bakiyoruz. Plan kupe dogru gidiyorsa model GORUYOR demektir.
"""
import os, sys, numpy as np, torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
import h5py

CKPT = os.path.expanduser(sys.argv[1])
REPO = sys.argv[2]
DEV  = os.environ.get("PROBE_DEV", "cuda")
ROOT = os.path.expanduser(f"~/Projects/franka_vla_data/lerobot/{REPO}")
HDF5 = os.path.expanduser(sys.argv[3])   # kup konumu icin ham veri

ds = LeRobotDataset(REPO, root=ROOT)
# DIKKAT: from_pretrained modeli CHECKPOINT CONFIG'indeki cihaza yukler (genelde
# cuda). Sonradan .to("cpu") demek gec kalir -- GPU doluysa yukleme aninda OOM olur.
# Cihazi ONCEDEN config uzerinden ayarlamak sart.
from lerobot.configs.policies import PreTrainedConfig
import json as _json

# LoRA checkpointleri SADECE adaptor agirliklarini tutar (adapter_model.safetensors,
# ~5 MB). Tam model yok -> once adapter_config.json'daki TABAN modeli yukleyip
# uzerine adaptoru uygulamak gerekiyor.
_adapter_cfg = os.path.join(CKPT, "adapter_config.json")
if os.path.exists(_adapter_cfg):
    _base = _json.load(open(_adapter_cfg))["base_model_name_or_path"]
    print(f"  [PEFT] taban model: {_base}")
    _cfg = PreTrainedConfig.from_pretrained(_base); _cfg.device = DEV
    pol = SmolVLAPolicy.from_pretrained(_base, config=_cfg)
    from peft import PeftModel
    pol = PeftModel.from_pretrained(pol, CKPT)
    pol = pol.merge_and_unload()      # adaptoru agirliklara kaynastir
    print("  [PEFT] adaptor uygulandi ve kaynastirildi")
else:
    _cfg = PreTrainedConfig.from_pretrained(CKPT); _cfg.device = DEV
    pol = SmolVLAPolicy.from_pretrained(CKPT, config=_cfg)
pol.eval(); pol.to(DEV)
pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})

h = h5py.File(HDF5, "r"); heps = sorted(h["data"].keys())
OFF = 10          # kol evde, sizinti R^2=0.089
# Cikarim stokastik (flow matching): her sorgu K kez tekrarlanip ortalanir.
# Orneklem kucukse egim tahmininin guven araligi +-0.15'e kadar cikiyor ve
# checkpointler arasi farklar ayirt edilemiyor. Ortam degiskeniyle buyutulebilir.
K    = int(os.environ.get("K", "6"))
N_EP = int(os.environ.get("N_EP", "30"))

rows = []
n_ep = min(len(heps), ds.num_episodes)
for i in range(0, n_ep, max(1, n_ep // N_EP)):
    st = int(ds.meta.episodes["dataset_from_index"][i])
    s = ds[st + OFF]
    b = {kk: s[kk].unsqueeze(0) for kk in
         ["observation.images.front","observation.images.wrist","observation.state"]}
    b["task"] = [s["task"]]
    chunks = []
    for _ in range(K):
        with torch.no_grad():
            chunks.append(post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy())
    ch = np.mean(chunks, axis=0)          # (50, 8)
    # plan boyunca kumulatif y hareketi (goreli aksiyon: her adim bir delta)
    plan_y = ch[:, 1]
    ee_y  = float(s["observation.state"][10])
    cube_y = float(h["data"][heps[i]]["observation.state.object_pos"][OFF][1])
    rows.append((cube_y, ee_y, plan_y[5], plan_y[15], plan_y[30], plan_y[49]))
h.close()

r = np.array(rows)
cube_y, ee_y = r[:,0], r[:,1]
print(f"checkpoint: {os.path.basename(os.path.dirname(CKPT))}   n={len(r)} bolum, kare {OFF}")
print(f"kup_y araligi: {cube_y.min():+.3f} .. {cube_y.max():+.3f}   ee_y std: {ee_y.std():.4f}")
print()
def _boot_ci(x, y, n=800):
    rng = np.random.default_rng(0)
    out = [np.polyfit(x[i], y[i], 1)[0]
           for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return np.percentile(out, [5, 95])

print(f"{'plan adimi':>11} | {'korelasyon':>11} | {'egim':>7} | {'%90 guven araligi':>19}")
print("-" * 60)
for j, name in [(2,"5"), (3,"15"), (4,"30"), (5,"49")]:
    p = r[:,j]
    if p.std() <= 1e-9:
        print(f"{name:>11} | {'sabit cikti':>11} | {'-':>7} | {'-':>19}")
        continue
    c = np.corrcoef(cube_y, p)[0,1]
    sl = np.polyfit(cube_y, p, 1)[0]
    lo, hi = _boot_ci(cube_y, p)
    print(f"{name:>11} | {c:11.3f} | {sl:7.3f} | [{lo:6.3f}, {hi:6.3f}]")
print()
print("Korelasyon yuksek -> model kolu evdeyken bile kupu GORUYOR")
print("Korelasyon ~0     -> model GORMUYOR, ortalama plani uretiyor")
