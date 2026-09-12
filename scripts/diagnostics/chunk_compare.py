"""
PLAN KARSILASTIRMA: modelin 50 adimlik plani, uzmanin ayni karelerdeki
aksiyonlariyla adim adim karsilastirilir. IKISI DE goreli, IKISI DE LeRobot
veri setinden -- boylece elma-armut karsilastirmasi olmaz.

Goreli aksiyon "su kadar ilerle" DEGIL, "bulundugun yerden D uzaktaki noktaya
git" demek; her adimda yeniden hesaplanir. Bu yuzden deltalari TOPLAMAK
anlamsizdir (yapildi, 35 metre gibi sacma sonuc verdi).

Anlamli olan: her plan adiminda modelin verdigi komut, uzmanin o kareye
karsilik gelen komutuyla ayni buyuklukte ve yonde mi.
"""
import os, sys, json, numpy as np, torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

CKPT = os.path.expanduser(sys.argv[1]); REPO = sys.argv[2]
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
    pol = PeftModel.from_pretrained(pol, CKPT).merge_and_unload(); _pp = base
else:
    cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
    pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); _pp = CKPT
pol.eval(); pol.to(DEV); pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=_pp,
    preprocessor_overrides={"device_processor": {"device": DEV}})

starts = [int(ds.meta.episodes["dataset_from_index"][i])
          for i in range(0, ds.num_episodes, max(1, ds.num_episodes // N_EP))]
ends = [int(ds.meta.episodes["dataset_to_index"][i])
        for i in range(0, ds.num_episodes, max(1, ds.num_episodes // N_EP))]

M, E = [], []
for st, en in zip(starts, ends):
    s = ds[st + OFF]
    b = {kk: s[kk].unsqueeze(0) for kk in
         ["observation.images.front", "observation.images.wrist", "observation.state"]}
    b["task"] = [s["task"]]
    ch = []
    for _ in range(K):
        with torch.no_grad():
            ch.append(post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy())
    ch = np.mean(ch, axis=0)                      # (H, 8)
    H = ch.shape[0]
    exp = np.stack([ds[min(st + OFF + j, en - 1)]["action"].numpy() for j in range(H)])
    M.append(ch[:, :3]); E.append(exp[:, :3])
M, E = np.array(M), np.array(E)                   # (n, H, 3)

print(f"checkpoint: {os.path.basename(os.path.dirname(CKPT))}  n={len(M)} bolum, "
      f"kare {OFF}, plan uzunlugu {M.shape[1]}, {K} tekrar")
print()
print(f"{'plan adimi':>11} | {'model |D|':>10} | {'uzman |D|':>10} | {'oran':>6} | {'MAE':>8} | {'kor(y)':>7}")
print("-" * 68)
for j in [0, 5, 10, 20, 35, 49]:
    if j >= M.shape[1]: continue
    m, e = M[:, j, :], E[:, j, :]
    mm, em = np.abs(m).mean(), np.abs(e).mean()
    c = np.corrcoef(e[:, 1], m[:, 1])[0, 1] if m[:, 1].std() > 1e-9 else float("nan")
    print(f"{j:>11} | {mm:9.4f}m | {em:9.4f}m | {mm/max(em,1e-9):6.2f} | "
          f"{np.abs(m-e).mean():7.4f}m | {c:7.3f}")
print()
print("oran ~1 ve kor yuksek -> model uzmani dogru taklit ediyor")
print("oran <<1 -> model daha KUCUK komutlar veriyor (yavas/eksik hareket)")
