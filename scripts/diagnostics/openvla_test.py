"""
OpenVLA-7B SIFIRDAN (ince ayarsiz) test.

Soru: OpenVLA, hic egitilmeden bizim Franka sahnemizde kupu goruyor mu?
Olcum: farkli kup konumlarina sahip karelerden tahmin alip, tahmin edilen
aksiyonun y bileseninin kupun gercek y'siyle korelasyonuna bakiyoruz.
(Ayni yontem SmolVLA icin vision_test.py'de kullanildi.)

DIKKAT: OpenVLA transformers 4.40.1 ile yazildi, bizde 5.5.4 var -> ozel
modelleme kodu (trust_remote_code) uyumsuz olabilir. Bu script once YUKLEMEYI
dener, basarisiz olursa hatayi net raporlar.
"""
import os, sys, time
import numpy as np, torch, h5py
from PIL import Image

MODEL = "openvla/openvla-7b"
HDF5 = os.path.expanduser("~/Projects/franka_vla_data/data/franka_lift_merged.hdf5")
N_EP = int(os.environ.get("N_EP", "24"))
FRAME = int(os.environ.get("FRAME", "10"))     # kol evde, sizinti yok
PROMPT_TASK = "pick up the cube and lift it"

print("=" * 60)
print("ADIM 1: model yukleniyor (15 GB, 4-bit)")
print("=" * 60, flush=True)
t0 = time.time()
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print(f"  processor OK ({time.time()-t0:.0f}s)", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    # DIKKAT: 4-bit modeller yuklendikleri cihazda KALIR, .to() ile tasinamaz.
    # device_map="cuda:0" (string) iceride .to() tetikleyip
    # "`.to` is not supported for `4-bit`" hatasi veriyor.
    # Dogrusu sozluk bicimi: {"": 0} -> tum modul 0. GPU'ya.
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
        quantization_config=bnb, low_cpu_mem_usage=True, device_map={"": 0})
    vla.eval()
    print(f"  model OK ({time.time()-t0:.0f}s)", flush=True)
except Exception as e:
    import traceback
    print(f"\n  YUKLEME BASARISIZ: {type(e).__name__}")
    print(f"  {str(e)[:400]}")
    print("\n  --- TAM IZ ---")
    traceback.print_exc()
    print("\n  -> transformers surum uyumsuzlugu muhtemel. Cozum: ayri bir conda")
    print("     ortaminda transformers==4.40.1 ile denemek.")
    sys.exit(1)

print(f"  GPU bellek: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# hangi unnorm_key'ler var
keys = list(getattr(vla, "norm_stats", {}).keys())
franka = [k for k in keys if any(t in k for t in ("austin", "viola", "toto", "furniture"))]
UNNORM = os.environ.get("UNNORM_KEY") or (franka[0] if franka else (keys[0] if keys else None))
print(f"  kullanilabilir unnorm_key sayisi: {len(keys)}, secilen: {UNNORM}", flush=True)

print()
print("=" * 60)
print(f"ADIM 2: {N_EP} bolumden kare {FRAME} ile tahmin")
print("=" * 60, flush=True)

f = h5py.File(HDF5, "r")
eps = sorted(f["data"].keys())
step = max(1, len(eps) // N_EP)
rows = []
t1 = time.time()
for i, e in enumerate(eps[::step][:N_EP]):
    ep = f["data"][e]
    img = Image.fromarray(ep["observation.images.front"][FRAME])
    cube = ep["observation.state.object_pos"][FRAME]
    prompt = f"In: What action should the robot take to {PROMPT_TASK}?\nOut:"
    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        act = vla.predict_action(**inputs, unnorm_key=UNNORM, do_sample=False)
    rows.append((float(cube[0]), float(cube[1]), float(act[0]), float(act[1])))
    if i % 5 == 0:
        print(f"  {i+1}/{N_EP}  kup=({cube[0]:+.3f},{cube[1]:+.3f})  "
              f"aksiyon=({act[0]:+.4f},{act[1]:+.4f})  [{time.time()-t1:.0f}s]", flush=True)
f.close()

r = np.array(rows)
print()
print("=" * 60)
print("SONUC")
print("=" * 60)
for k, ax in [(0, "x"), (1, "y")]:
    cube_v, act_v = r[:, k], r[:, k+2]
    if act_v.std() < 1e-9:
        print(f"  {ax}: model SABIT aksiyon uretiyor (std=0) -> sahneyi okumuyor")
        continue
    c = np.corrcoef(cube_v, act_v)[0, 1]
    print(f"  {ax}: korelasyon={c:+.3f}   aksiyon std={act_v.std():.4f}")
print()
print(f"toplam sure: {time.time()-t0:.0f}s")
print()
print("Korelasyon |0.5| ustunde -> OpenVLA sifirdan kupu GORUYOR, yol acik")
print("Korelasyon ~0            -> sifirdan calismiyor, ince ayar sart")
