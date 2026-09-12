"""
CANLI GORUNTULERI CEVRIMDISI BESLE: ucurum GORUNTULERDE mi, SERVIS YOLUNDA mi?

Sorun: 3 kameralı modelin canli x korelasyonu 0.285, ayni modelin taze veri
uzerinde cevrimdisi korelasyonu 0.794. Iki aciklama var:

  a) canli GORUNTULER egitim goruntulerinden farkli   -> veri/render sorunu
  b) goruntuler ayni, ama SERVIS YOLU bozuyor          -> boru hatti sorunu
     (goruntu sirasi, dtype, olcekleme, eksik kamera, chunk zamanlamasi...)

Bu test ikisini ayirir: eval sirasinda dokulen GERCEK canli goruntuleri alip
ayni modele CEVRIMDISI veririz.

  sonuc ~0.28 cikarsa -> (a) goruntuler; sorun render/veri tarafinda
  sonuc ~0.79 cikarsa -> (b) goruntuler iyi; sorun canli servis yolunda

Kullanim:
  python offline_on_live.py <npz> <ckpt> [ckpt2 ...]
"""
import os, sys
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

NPZ = sys.argv[1]
CKPTS = sys.argv[2:]
DEV = os.environ.get("PROBE_DEV", "cuda")
K = int(os.environ.get("K", "8"))
TASK = "pick up the cube and lift it"

d = np.load(NPZ)
cube = d["cube"][:, :2]
print(f"veri: {NPZ}")
print(f"  {len(cube)} canli gozlem | anahtarlar {sorted(d.files)}\n")


def run(ckpt):
    cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.device = DEV
    pol = SmolVLAPolicy.from_pretrained(ckpt, config=cfg); pol.eval(); pol.to(DEV)
    pol.config.device = DEV
    pre, post = make_pre_post_processors(
        policy_cfg=pol.config, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": DEV}})
    want = [k for k in cfg.input_features if k.startswith("observation.images")]
    short = [k.rsplit(".", 1)[1] for k in want]
    missing = [s for s in short if s not in d.files]
    print(f"{os.path.basename(os.path.dirname(ckpt.rstrip('/')))}"
          f"  kameralar={short}" + (f"  EKSIK={missing}" if missing else ""))
    if missing:
        print("  -> dokumde bu kamera yok, atlaniyor\n"); return None

    zs = np.zeros(16, dtype=np.float32)
    preds = []
    for i in range(len(cube)):
        b = {"observation.state": torch.from_numpy(zs).unsqueeze(0),
             "task": [TASK]}
        for key, s in zip(want, short):
            img = d[s][i]
            b[key] = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.
        with torch.no_grad():
            ch = np.mean([post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy()
                          for _ in range(K)], axis=0)
        preds.append(np.mean(ch[:, 8:10], axis=0))   # modelin kup tahmini
    preds = np.array(preds)
    err = np.linalg.norm(preds - cube, axis=1)
    print(f"  medyan {np.median(err)*1000:6.1f} mm | ort {err.mean()*1000:6.1f} mm")
    for j, ax in enumerate("xy"):
        sl = np.polyfit(cube[:, j], preds[:, j], 1)[0]
        co = np.corrcoef(cube[:, j], preds[:, j])[0, 1]
        print(f"    {ax}: egim {sl:+.3f}  kor {co:+.3f}  "
              f"kayma {np.mean(preds[:,j]-cube[:,j])*1000:+.1f} mm")
    print()
    del pol; torch.cuda.empty_cache()
    return np.median(err) * 1000


for c in CKPTS:
    run(os.path.expanduser(c))

print("=" * 62)
print("Karsilastirma noktalari (2026-09-08 olcumleri):")
print("  3kam  CANLI dongu, adim 0 : medyan 79.6 mm | x kor 0.285")
print("  3kam  cevrimdisi taze veri: medyan 33.6 mm | x kor 0.794")
print("  geo3  CANLI dongu, adim 0 : medyan 59.9 mm | x kor 0.722")
print("  geo3  cevrimdisi taze veri: medyan 41.4 mm | x kor 0.846")
