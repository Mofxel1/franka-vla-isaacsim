"""
DONUK SIGLIP SONDASI: SmolVLA'nin gordugu ozelliklerde kupun yeri VAR MI?

SmolVLA varsayilan olarak `freeze_vision_encoder=True` VE `train_expert_only=True`
ile egitiliyor -> 450M parametrenin sadece 100M'i (aksiyon uzmani) ogreniyor.
Goru kodlayici bizim goruntulerimize hic uyarlanmiyor.

Bu sonda iki yolu AYNI veriyle, AYNI bolum-bazli ayrimla karsilastirir:

  A) DONUK SigLIP ozellikleri  -> kucuk kafa      (SmolVLA'nin gordugu sey)
  B) Sifirdan kucuk CNN                            (bilgi goruntude var mi)

  A ~ B  -> dondurma darbogaz DEGIL; goru kodlayiciyi cozmek bosa gider
  A >> B -> darbogaz KESIN olarak dondurma; cozmeye deger

--- NEDEN BOLUM BAZLI AYRIM ---
Onceki sonda (vision_probe.py) kare bazli ayiriyordu. Bir bolum icinde kupun
konumu SABIT; ayni bolumun kareleri hem egitime hem dogrulamaya dagilinca model
"bu bolum boyle gorunuyor -> kup surada" ezberleyip dogrulamada iyi puan alir.
Bu sonda bolumleri ayirir: bir bolumun TUM kareleri ya egitimde ya testtedir.

Kullanim:
  N_EP=200 PER_EP=12 EPOCHS=30 python frozen_feat_probe.py [hdf5] [ckpt]
"""
import os, sys, time
import numpy as np, h5py, torch, torch.nn as nn

HDF5 = os.path.expanduser(os.environ.get(
    "HDF5", sys.argv[1] if len(sys.argv) > 1
    else "/data/franka_vla/data/franka_lift_geo.hdf5"))
CKPT = os.path.expanduser(os.environ.get(
    "CKPT", sys.argv[2] if len(sys.argv) > 2
    else "~/franka_runs/train_geo3/checkpoints/006000/pretrained_model"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_EP = int(os.environ.get("N_EP", "200"))
PER_EP = int(os.environ.get("PER_EP", "12"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
# REST fazi 12 kare surer (des_ee_pose = ee_pose). O karelerde kol rastgele
# savrulmus olabiliyor -> 12'den basla. 60'a kadar kol kupu kapatmiyor.
FR_LO, FR_HI = 12, 60
CAM = os.environ.get("CAM", "front")

print(f"cihaz {DEV} | veri {os.path.basename(HDF5)} | kamera {CAM}")

# ---------------------------------------------------------------- veri
imgs_by_ep, cube_by_ep = [], []
n_skip = 0
with h5py.File(HDF5, "r") as f:
    eps = sorted(f["data"].keys())
    step = max(1, len(eps) // N_EP)
    for e in eps[::step]:
        ep = f["data"][e]
        n = ep[f"observation.images.{CAM}"].shape[0]
        # Bolumlerin ~%22'sinde sifirlama aninda savrulan kol kupe daha ilk
        # karelerde carpiyor. O bolumlerde "kupun yeri" etiketi kare boyunca
        # degisiyor ve sonda bozuk etiket ogreniyor. Kup FR_HI'dan once
        # kimildiyorsa bolumu tamamen atla.
        o = ep["observation.state.object_pos"][:]
        d = np.linalg.norm(o[:, :2] - o[0, :2], axis=1)
        mv = int(np.argmax(d > 0.002)) if (d > 0.002).any() else 10 ** 9
        if mv <= FR_HI:
            n_skip += 1
            continue
        idx = np.linspace(FR_LO, min(FR_HI, n - 1), PER_EP).astype(int)
        imgs_by_ep.append(ep[f"observation.images.{CAM}"][idx])
        cube_by_ep.append(ep["observation.state.object_pos"][idx][:, :2])
n_ep = len(imgs_by_ep)
print(f"  kup erken kimildadigi icin atlanan bolum: {n_skip}")

# BOLUM bazli ayrim: bir bolumun tum kareleri ayni tarafta
rng = np.random.default_rng(0)
order = rng.permutation(n_ep)
n_tr = int(n_ep * 0.75)
tr_eps, te_eps = order[:n_tr], order[n_tr:]
print(f"  {n_ep} bolum x {PER_EP} kare | egitim {len(tr_eps)} bolum / "
      f"test {len(te_eps)} bolum  (BOLUM BAZLI ayrim)")

def gather(sel):
    X = np.concatenate([imgs_by_ep[i] for i in sel]).astype(np.float32) / 255.0
    Y = np.concatenate([cube_by_ep[i] for i in sel]).astype(np.float32)
    return X, Y

Xtr, Ytr = gather(tr_eps)
Xte, Yte = gather(te_eps)
print(f"  egitim {len(Xtr)} ornek / test {len(Xte)} ornek")

# taban cizgisi: her zaman egitim ortalamasini tahmin et
base = np.linalg.norm(Yte - Ytr.mean(0), axis=1)
print(f"\n  TABAN (hep ortalamayi tahmin et): medyan {np.median(base)*1000:.1f} mm")


def report(name, pred, true, t0):
    err = np.linalg.norm(pred - true, axis=1)
    line = (f"  {name:28s} medyan {np.median(err)*1000:6.1f} mm | "
            f"ort {err.mean()*1000:6.1f} mm | {time.time()-t0:.0f}s")
    for j, ax in enumerate("xy"):
        sl = np.polyfit(true[:, j], pred[:, j], 1)[0]
        co = np.corrcoef(true[:, j], pred[:, j])[0, 1]
        line += f"\n  {'':28s}   {ax}: egim {sl:+.3f} kor {co:+.3f}"
    print(line, flush=True)
    return np.median(err) * 1000


def train_head(net, Ftr, Ytr_, Fte, epochs, lr=1e-3, bs=64):
    net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ft = torch.as_tensor(Ftr); yt = torch.as_tensor(Ytr_)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(ft))
        for i in range(0, len(ft), bs):
            b = perm[i:i + bs]
            xb, yb = ft[b].to(DEV), yt[b].to(DEV)
            loss = nn.functional.mse_loss(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    out = []
    with torch.no_grad():
        fe = torch.as_tensor(Fte)
        for i in range(0, len(fe), 128):
            out.append(net(fe[i:i + 128].to(DEV)).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------- A) DONUK SIGLIP ozellikleri
print("\n=== A) DONUK SigLIP ozellikleri + kucuk kafa ===", flush=True)
t0 = time.time()
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig

cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); pol.eval(); pol.to(DEV)
vlm = pol.model.vlm_with_expert
print(f"  goru kodlayici: donuk={cfg.freeze_vision_encoder} "
      f"train_expert_only={cfg.train_expert_only}")


def siglip_feats(X, bs=16):
    """SmolVLA'nin aksiyon uzmanina verdigi goruntu token'lari."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            t = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).to(DEV)
            # SmolVLA'nin kendi on islemesi: 512'ye pad'li resize, sonra [-1,1]
            from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
            t = resize_with_pad(t, cfg.resize_imgs_with_padding[1],
                                cfg.resize_imgs_with_padding[0], pad_value=0)
            t = t * 2.0 - 1.0
            h = vlm.embed_image(t)              # (B, token, dim)
            out.append(h.float().cpu().numpy())
    return np.concatenate(out)


Ftr = siglip_feats(Xtr); Fte = siglip_feats(Xte)
n_tok, dim = Ftr.shape[1], Ftr.shape[2]
print(f"  ozellik sekli: {n_tok} token x {dim} boyut  ({time.time()-t0:.0f}s)")


class TokenHead(nn.Module):
    """Token basina indirge, sonra duzlestir -- UZAMSAL bilgi korunur.
    (Ortalama havuzlama konum bilgisini yok ederdi, lokalizasyonda tam da
    ihtiyacimiz olan sey o.)"""
    def __init__(self, n_tok, dim, red=32):
        super().__init__()
        self.proj = nn.Linear(dim, red)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(n_tok * red),
            nn.Linear(n_tok * red, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, x):
        return self.mlp(torch.relu(self.proj(x)))


pred_a = train_head(TokenHead(n_tok, dim), Ftr, Ytr, Fte, EPOCHS)
med_a = report("DONUK SigLIP + kafa", pred_a, Yte, t0)
del pol, vlm, Ftr, Fte
torch.cuda.empty_cache()

# ------------------------------------------------- B) sifirdan CNN
print("\n=== B) Sifirdan kucuk CNN (ayni bolum-bazli ayrim) ===", flush=True)
t0 = time.time()


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 16, 32, 64, 64]
        L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i + 1], 3, 2, 1), nn.BatchNorm2d(c[i + 1]), nn.ReLU()]
        self.f = nn.Sequential(*L, nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                               nn.Linear(64 * 16, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.f(x)


cnn = SmallCNN()
print(f"  parametre: {sum(p.numel() for p in cnn.parameters()):,}")
pred_b = train_head(cnn, Xtr.transpose(0, 3, 1, 2).copy(), Ytr,
                    Xte.transpose(0, 3, 1, 2).copy(), EPOCHS, lr=2e-3, bs=32)
med_b = report("sifirdan CNN", pred_b, Yte, t0)

# ------------------------------------------------- karar
print("\n" + "=" * 62)
print(f"  taban {np.median(base)*1000:.1f} mm | "
      f"DONUK SigLIP {med_a:.1f} mm | sifirdan CNN {med_b:.1f} mm")
if med_a > med_b * 1.5:
    print("  -> DONUK OZELLIKLER DARBOGAZ. Goru kodlayiciyi cozmek DEGER.")
elif med_a < med_b * 1.1:
    print("  -> Donuk ozellikler yeterli. Darbogaz baska yerde; cozmek BOSA GIDER.")
else:
    print("  -> Belirsiz; fark kucuk.")
