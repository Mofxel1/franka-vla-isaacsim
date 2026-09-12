"""
KUMULATIF PLAN TESTI: modelin 50 adimlik planinin TOPLAMI kupe variyor mu?

Goreli aksiyonda her adim bir "su kadar ilerle" komutu. Simdiye kadar hep TEK
adimin egimine baktik; asil onemli olan 50 adimin TOPLAM yer degistirmesi.
Model dogru yone baslayip yari yolda duruyorsa, tek-adim egimi iyi gorunur
ama kol kupe hic varamaz -- kapali dongude gordugumuz tam olarak bu.

Olcum: kare 10'da (kol evde) plan uretilir, deltalar toplanir ve uzmanin
ayni penceredeki toplam yer degistirmesiyle karsilastirilir.
"""
import os, sys, json, numpy as np, torch, h5py
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

CKPT = os.path.expanduser(sys.argv[1])
REPO = sys.argv[2]
HDF5 = os.path.expanduser(sys.argv[3])
DEV  = os.environ.get("PROBE_DEV", "cuda")
K    = int(os.environ.get("K", "8"))
N_EP = int(os.environ.get("N_EP", "60"))
OFF  = int(os.environ.get("FRAME", "10"))

ds = LeRobotDataset(REPO, root=os.path.expanduser(f"~/Projects/franka_vla_data/lerobot/{REPO}"))
ad = os.path.join(CKPT, "adapter_config.json")
if os.path.exists(ad):
    base = json.load(open(ad))["base_model_name_or_path"]
    cfg = PreTrainedConfig.from_pretrained(base); cfg.device = DEV
    pol = SmolVLAPolicy.from_pretrained(base, config=cfg)
    from peft import PeftModel
    pol = PeftModel.from_pretrained(pol, CKPT).merge_and_unload()
else:
    cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
    pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg)
pol.eval(); pol.to(DEV); pol.config.device = DEV
# Dil komutunu token'a ceviren ve girdileri normalize eden on-islemci SART;
# atlanirsa predict_action_chunk 'observation.language.tokens' bulamiyor.
from lerobot.policies.factory import make_pre_post_processors
_ppath = base if os.path.exists(ad) else CKPT
pre, post = make_pre_post_processors(
    policy_cfg=pol.config, pretrained_path=_ppath,
    preprocessor_overrides={"device_processor": {"device": DEV}})

h = h5py.File(HDF5, "r"); heps = sorted(h["data"].keys())
n = min(len(heps), ds.num_episodes)
rows = []
for i in range(0, n, max(1, n // N_EP)):
    st = int(ds.meta.episodes["dataset_from_index"][i])
    s = ds[st + OFF]
    b = {kk: s[kk].unsqueeze(0) for kk in
         ["observation.images.front", "observation.images.wrist", "observation.state"]}
    b["task"] = [s["task"]]
    ch = []
    for _ in range(K):
        with torch.no_grad():
            ch.append(pol.predict_action_chunk(pre(b))[0].cpu().numpy())
    ch = np.mean(ch, axis=0)                      # (50, 8)
    model_disp = ch[:, :3].sum(axis=0)            # planin TOPLAM yer degistirmesi

    ep = h["data"][heps[i]]
    N = int(ep.attrs["num_samples"])
    hi = min(OFF + ch.shape[0], N)
    exp_disp = ep["action"][OFF:hi, :3].sum(axis=0)   # uzmanin ayni penceredeki toplami
    ee = ep["observation.state.ee_pos"][OFF]
    cube = ep["observation.state.object_pos"][OFF]
    rows.append(np.concatenate([model_disp, exp_disp, cube[:2] - ee[:2]]))
h.close()

r = np.array(rows)
print(f"checkpoint: {os.path.basename(os.path.dirname(CKPT))}   n={len(r)} bolum, kare {OFF}, {K} tekrar")
print()
print(f"{'eksen':>6} | {'model toplam':>13} | {'uzman toplam':>13} | {'oran':>6} | {'kor.':>6}")
print("-" * 60)
for k, ax in enumerate("xyz"):
    m, e = r[:, k], r[:, 3 + k]
    ratio = np.abs(m).mean() / max(np.abs(e).mean(), 1e-9)
    c = np.corrcoef(e, m)[0, 1] if m.std() > 1e-9 else float("nan")
    print(f"{ax:>6} | {np.abs(m).mean():12.4f}m | {np.abs(e).mean():12.4f}m | {ratio:6.2f} | {c:6.3f}")
print()
print("oran ~1.0 -> plan uzman kadar yol katediyor")
print("oran <<1  -> model YARI YOLDA DURUYOR (kapali dongude kupe varamaz)")
