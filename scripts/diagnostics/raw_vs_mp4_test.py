"""
KESIN TEST: MP4 sikistirmasi modeli bozuyor mu?

Model MP4-kodlanmis goruntulerle egitildi ama kapali donguda Isaac Sim'den
HAM piksel aliyor. Ayni kareyi iki kaynaktan da verip tahminleri karsilastiriyoruz:
  - LeRobot veri seti  -> MP4 kodlanmis/cozulmus (egitimde gorulen bicim)
  - ham HDF5           -> sikistirilmamis (kapali donguda gelen bicime yakin)
Ikisi ayni kareye ait, tek fark sikistirma.

Tahminler ciddi farkliysa: egitim/dagitim uyumsuzlugu var, cozum ya veri setini
sikistirmasiz uretmek ya da kapali donguda ayni kodlamayi uygulamak.
"""
import os, sys, json, numpy as np, torch, h5py
from lerobot.datasets import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

CKPT = os.path.expanduser(sys.argv[1]); REPO = sys.argv[2]
HDF5 = os.path.expanduser(sys.argv[3])
DEV  = os.environ.get("PROBE_DEV", "cuda")
K    = int(os.environ.get("K", "8"))
N_EP = int(os.environ.get("N_EP", "40"))
OFF  = int(os.environ.get("FRAME", "10"))

ds = LeRobotDataset(REPO, root=os.path.expanduser(f"~/Projects/franka_vla_data/lerobot/{REPO}"))
cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg)
pol.eval(); pol.to(DEV); pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})

h = h5py.File(HDF5, "r"); heps = sorted(h["data"].keys())
n = min(len(heps), ds.num_episodes)
diffs, mp4_p, raw_p, exps, pix = [], [], [], [], []
for i in range(0, n, max(1, n // N_EP)):
    st = int(ds.meta.episodes["dataset_from_index"][i])
    s = ds[st + OFF]
    ep = h["data"][heps[i]]

    def run(front, wrist):
        b = {"observation.images.front": front.unsqueeze(0),
             "observation.images.wrist": wrist.unsqueeze(0),
             "observation.state": s["observation.state"].unsqueeze(0),
             "task": [s["task"]]}
        out = []
        for _ in range(K):
            with torch.no_grad():
                out.append(post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy())
        return np.mean(out, axis=0)

    # MP4'ten (egitimde gorulen)
    a_mp4 = run(s["observation.images.front"], s["observation.images.wrist"])
    # ham HDF5'ten (kapali donguda gelen bicime yakin)
    fr = torch.from_numpy(ep["observation.images.front"][OFF]).float().permute(2,0,1)/255.0
    wr = torch.from_numpy(ep["observation.images.wrist"][OFF]).float().permute(2,0,1)/255.0
    a_raw = run(fr, wr)

    pix.append(np.abs(s["observation.images.front"].numpy() - fr.numpy()).mean())
    mp4_p.append(a_mp4[5, :3]); raw_p.append(a_raw[5, :3])
    diffs.append(np.abs(a_mp4[:, :3] - a_raw[:, :3]).mean())
    exps.append(ds[st + OFF + 5]["action"].numpy()[:3])
h.close()

mp4_p, raw_p, exps = np.array(mp4_p), np.array(raw_p), np.array(exps)
print(f"checkpoint: {os.path.basename(os.path.dirname(CKPT))}  n={len(mp4_p)} bolum, kare {OFF}")
print(f"\ngoruntu piksel farki (MP4 vs ham): {np.mean(pix)*255:.2f}/255 ortalama")
print(f"tahmin farki (plan boyunca ort.)   : {np.mean(diffs)*1000:.2f} mm")
print()
print(f"{'kaynak':>8} | {'uzmanla MAE':>12} | {'kor(y)':>7} | {'egim(y)':>8}")
print("-" * 46)
for name, P in [("MP4", mp4_p), ("ham", raw_p)]:
    mae = np.abs(P - exps).mean()
    c = np.corrcoef(exps[:,1], P[:,1])[0,1]
    sl = np.polyfit(exps[:,1], P[:,1], 1)[0]
    print(f"{name:>8} | {mae*1000:11.2f}mm | {c:7.3f} | {sl:8.3f}")
print()
print("Iki satir BENZER ise sikistirma sorun DEGIL.")
print("'ham' belirgin kotuyse -> egitim/dagitim uyumsuzlugu bulundu.")
