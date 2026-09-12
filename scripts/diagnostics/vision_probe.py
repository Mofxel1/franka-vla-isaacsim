"""
GORU SONDASI: kupun konumu bizim goruntulerimizden CIKARILABILIR mi?

Kucuk bir CNN egitiyoruz -- tek gorevi on kamera goruntusunden kupun (x,y)
koordinatini tahmin etmek. VLA yok, dil yok, aksiyon yok, durum vektoru yok.

  Sonda BASARILI -> bilgi goruntude VAR; sorun SmolVLA'nin onu ogrenememesi
  Sonda BASARISIZ -> goruntuler yetersiz (cozunurluk/kamera mesafesi);
                     hicbir model olmayan bilgiyi cikaramaz, once veri duzeltilmeli

Karsilastirma icin "ortalama tahmin" temel cizgisi de raporlanir.
"""
import os, sys, time
import numpy as np, h5py, torch, torch.nn as nn

HDF5 = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                          else "~/Projects/franka_vla_data/data/franka_lift_merged.hdf5")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_EP = int(os.environ.get("N_EP", "200"))      # kac bolumden ornek
PER_EP = int(os.environ.get("PER_EP", "12"))   # bolum basina kac kare
EPOCHS = int(os.environ.get("EPOCHS", "12"))

print(f"cihaz: {DEV}")
print("veri yukleniyor...", flush=True)
X, Y = [], []
with h5py.File(HDF5, "r") as f:
    eps = sorted(f["data"].keys())
    step = max(1, len(eps) // N_EP)
    for e in eps[::step]:
        ep = f["data"][e]
        n = int(ep.attrs["num_samples"])
        # kupe henuz dokunulmamis kareler: kolun onu kapatmadigi erken evre
        idx = np.linspace(0, min(60, n - 1), PER_EP).astype(int)
        imgs = ep["observation.images.front"][idx]          # (PER_EP,224,224,3)
        cube = ep["observation.state.object_pos"][idx][:, :2]  # (PER_EP,2) x,y
        X.append(imgs); Y.append(cube)
X = np.concatenate(X).astype(np.float32) / 255.0
Y = np.concatenate(Y).astype(np.float32)
print(f"  {len(X)} ornek, goruntu {X.shape[1:]}, hedef {Y.shape[1:]}")

# egitim/dogrulama ayrimi -- BOLUM bazli degil kare bazli, ama bolumler zaten
# farkli kup konumlarina sahip oldugu icin yeterli
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
X, Y = X[perm], Y[perm]
n_val = len(X) // 5
Xtr, Ytr, Xva, Yva = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]
print(f"  egitim {len(Xtr)}, dogrulama {len(Xva)}")

Xtr = torch.from_numpy(Xtr).permute(0, 3, 1, 2)
Xva = torch.from_numpy(Xva).permute(0, 3, 1, 2).to(DEV)
Ytr = torch.from_numpy(Ytr); Yva_t = torch.from_numpy(Yva).to(DEV)

def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())

net = nn.Sequential(
    blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 128), blk(128, 128),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2)).to(DEV)
print(f"  sonda parametre sayisi: {sum(p.numel() for p in net.parameters()):,}")

opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
BS = 64
t0 = time.time()
for ep_i in range(EPOCHS):
    net.train()
    p = torch.randperm(len(Xtr))
    tot = 0.0
    for i in range(0, len(Xtr), BS):
        b = p[i:i+BS]
        xb, yb = Xtr[b].to(DEV), Ytr[b].to(DEV)
        loss = nn.functional.mse_loss(net(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    sched.step()
    net.eval()
    with torch.no_grad():
        pv = net(Xva)
        vmae = (pv - Yva_t).abs().mean(0).cpu().numpy()
    print(f"  epoch {ep_i+1:2d}/{EPOCHS}  egitim_mse={tot/len(Xtr):.5f}  "
          f"dogrulama_MAE x={vmae[0]*1000:.1f}mm y={vmae[1]*1000:.1f}mm", flush=True)

with torch.no_grad():
    pv = net(Xva).cpu().numpy()
base = np.abs(Yva - Yva.mean(0)).mean(0)      # "hep ortalamayi soyle" temel cizgisi
mae = np.abs(pv - Yva).mean(0)
print()
print("=" * 58)
print(f"{'':14} {'x':>10} {'y':>10}")
print(f"{'temel cizgi':14} {base[0]*1000:9.1f}mm {base[1]*1000:9.1f}mm   (hep ortalama)")
print(f"{'SONDA':14} {mae[0]*1000:9.1f}mm {mae[1]*1000:9.1f}mm")
for k, ax in enumerate("xy"):
    c = np.corrcoef(Yva[:, k], pv[:, k])[0, 1]
    print(f"  {ax}: korelasyon={c:.3f}   iyilesme={base[k]/max(mae[k],1e-9):.1f}x")
print("=" * 58)
print(f"sure: {time.time()-t0:.0f}s")
print()
print("SONDA temel cizgiden BELIRGIN iyiyse -> bilgi goruntude VAR")
print("SONDA ~ temel cizgi ise -> goruntuden kup konumu CIKARILAMIYOR")
