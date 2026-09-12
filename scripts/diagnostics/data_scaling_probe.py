"""
OGRENME EGRISI: daha cok VERI toplamak lokalizasyonu duzeltir mi?

Elimizde ~208 bolum var, yani ~208 FARKLI kup konumu. SmolVLA'nin resmi
recetesi 1.28M ornek oneriyor. Ama veri toplamaya saatler harcamadan once
sunu bilmek gerek: egri hala dik mi, yoksa doymus mu?

Yontem: TEST seti SABIT tutulur, sadece EGITIM seti buyutulur. Her boyutta
donuk SigLIP ozellikleri uzerine ayni kucuk kafa egitilir (~10 s).

  egri hala dusuyorsa  -> veri toplamak ISE YARAR, ne kadar gerektigi tahmin
                          edilebilir
  egri duzlestiyse     -> darbogaz veri DEGIL; toplamak bosa gider

NOT: bu sondanin MUTLAK sayilari SmolVLA'nin sayilari degil (kafa cok kucuk,
bolum basina 12 kare). Onemli olan EGRININ SEKLI.

Kullanim: PER_EP=12 EPOCHS=40 python data_scaling_probe.py [hdf5] [ckpt]
"""
import os, sys, time
import numpy as np, h5py, torch, torch.nn as nn

HDF5 = os.path.expanduser(os.environ.get("HDF5", sys.argv[1] if len(sys.argv) > 1
    else "/data/franka_vla/data/franka_lift_geo.hdf5"))
CKPT = os.path.expanduser(os.environ.get("CKPT", sys.argv[2] if len(sys.argv) > 2
    else "~/franka_runs/train_geo3/checkpoints/006000/pretrained_model"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PER_EP = int(os.environ.get("PER_EP", "12"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
N_TEST = int(os.environ.get("N_TEST", "50"))     # SABIT test bolumu sayisi
REPEAT = int(os.environ.get("REPEAT", "3"))      # her boyut icin tekrar (gurultu)
FR_LO, FR_HI = 12, 60
CAM = os.environ.get("CAM", "front")

# ------------------------------------------------------------------ veri
imgs, cubes, n_skip = [], [], 0
with h5py.File(HDF5, "r") as f:
    for e in sorted(f["data"].keys()):
        ep = f["data"][e]
        o = ep["observation.state.object_pos"][:]
        d = np.linalg.norm(o[:, :2] - o[0, :2], axis=1)
        # kup erken kimildarsa etiket bozuk (kol sifirlamada carpiyor, ~%22)
        if (d > 0.002).any() and int(np.argmax(d > 0.002)) <= FR_HI:
            n_skip += 1; continue
        n = ep[f"observation.images.{CAM}"].shape[0]
        idx = np.linspace(FR_LO, min(FR_HI, n - 1), PER_EP).astype(int)
        imgs.append(ep[f"observation.images.{CAM}"][idx])
        cubes.append(o[idx][:, :2])
n_ep = len(imgs)
print(f"cihaz {DEV} | {os.path.basename(HDF5)} | kullanilabilir {n_ep} bolum "
      f"(kup erken kimildadigi icin {n_skip} atlandi)")

rng = np.random.default_rng(0)
order = rng.permutation(n_ep)
te_eps, pool = order[:N_TEST], order[N_TEST:]
print(f"  SABIT test: {len(te_eps)} bolum | egitim havuzu: {len(pool)} bolum\n")

def gather(sel):
    X = np.concatenate([imgs[i] for i in sel]).astype(np.float32) / 255.
    Y = np.concatenate([cubes[i] for i in sel]).astype(np.float32)
    return X, Y

Xte, Yte = gather(te_eps)

# ----------------------------------------------------- donuk SigLIP ozellikleri
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
from lerobot.configs.policies import PreTrainedConfig
cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); pol.eval(); pol.to(DEV)
vlm = pol.model.vlm_with_expert

def feats(X, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            t = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).to(DEV)
            t = resize_with_pad(t, cfg.resize_imgs_with_padding[1],
                                cfg.resize_imgs_with_padding[0], pad_value=0)
            out.append(vlm.embed_image(t * 2. - 1.).float().cpu().numpy())
    return np.concatenate(out)

t0 = time.time()
Fte = feats(Xte)
# egitim havuzunun ozelliklerini BIR KEZ cikar, sonra alt kume al
Xpool, Ypool = gather(pool)
Fpool = feats(Xpool)
n_tok, dim = Fte.shape[1], Fte.shape[2]
print(f"  ozellik cikarimi bitti: {n_tok} token x {dim} boyut ({time.time()-t0:.0f}s)\n")
del pol, vlm; torch.cuda.empty_cache()


class TokenHead(nn.Module):
    def __init__(self, n_tok, dim, red=32):
        super().__init__()
        self.proj = nn.Linear(dim, red)
        self.mlp = nn.Sequential(nn.Flatten(), nn.LayerNorm(n_tok * red),
                                 nn.Linear(n_tok * red, 256), nn.ReLU(),
                                 nn.Linear(256, 2))
    def forward(self, x):
        return self.mlp(torch.relu(self.proj(x)))


def fit_eval(Ftr, Ytr, seed):
    torch.manual_seed(seed)
    net = TokenHead(n_tok, dim).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ft, yt = torch.as_tensor(Ftr), torch.as_tensor(Ytr)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(ft))
        for i in range(0, len(ft), 64):
            b = perm[i:i + 64]
            loss = nn.functional.mse_loss(net(ft[b].to(DEV)), yt[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        pr = np.concatenate([net(torch.as_tensor(Fte[i:i+128]).to(DEV)).cpu().numpy()
                             for i in range(0, len(Fte), 128)])
    err = np.linalg.norm(pr - Yte, axis=1)
    sl = np.polyfit(Yte[:, 0], pr[:, 0], 1)[0]
    co = np.corrcoef(Yte[:, 0], pr[:, 0])[0, 1]
    return np.median(err) * 1000, sl, co


base = np.median(np.linalg.norm(Yte - Ypool.mean(0), axis=1)) * 1000
print(f"  TABAN (hep ortalamayi tahmin et): {base:.1f} mm\n")
print(f"  {'bolum':>6} {'ornek':>7} | {'medyan mm':>10} {'x egim':>8} {'x kor':>7}")
print("  " + "-" * 46)

sizes = [s for s in (20, 40, 60, 80, 100, len(pool)) if s <= len(pool)]
res = []
for ns in sizes:
    meds, sls, cos = [], [], []
    for r in range(REPEAT):
        sub = np.random.default_rng(100 + r).permutation(len(pool))[:ns]
        # havuzdaki bolum i -> Fpool'da [i*PER_EP, (i+1)*PER_EP) satirlari
        rows = np.concatenate([np.arange(i * PER_EP, (i + 1) * PER_EP) for i in sub])
        m, sl, co = fit_eval(Fpool[rows], Ypool[rows], seed=r)
        meds.append(m); sls.append(sl); cos.append(co)
    res.append((ns, np.mean(meds)))
    print(f"  {ns:>6} {ns*PER_EP:>7} | {np.mean(meds):>10.1f} {np.mean(sls):>8.3f} "
          f"{np.mean(cos):>7.3f}   (+-{np.std(meds):.1f})")

print("\n" + "=" * 52)
if len(res) >= 3:
    (n1, m1), (n2, m2) = res[-3], res[-1]
    dec = (m1 - m2) / m1 * 100
    print(f"  son iki katlama ({n1} -> {n2} bolum): {m1:.1f} -> {m2:.1f} mm  ({dec:+.1f}%)")
    if dec > 8:
        print("  -> EGRI HALA DIK. Daha cok bolum toplamak ISE YARAR.")
    elif dec > 3:
        print("  -> Egri yavasliyor. Toplamak yardim eder ama azalan getiriyle.")
    else:
        print("  -> EGRI DUZ. Darbogaz veri MIKTARI degil; toplamak bosa gider.")
