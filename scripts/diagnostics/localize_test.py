"""Lokalizasyon testi: model kupun YERINI bulabiliyor mu?

Bolum basi gozlemden (kol evde, durum sifir) 50 adimlik plan alinir; planin
z'si en dusuk adimi = inis noktasi. Onun XY'si kupun GERCEK XY'si ile
karsilastirilir. Kavrama icin <20 mm gerekir (kup 4 cm).

Referans degerler:
  "her zaman ortayi tahmin et" temel cizgisi  136 mm
  gorsel sonda CNN (246K parametre)            12 mm
Kullanim: python localize_test.py <checkpoint_klasoru> [bolum_sayisi]
"""
import os, sys, numpy as np, torch, h5py
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

CKPT = sys.argv[1]
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 40
# Veri setinde 0. KARE BAYAT (onceki bolumun son karesi) -- collect_demos.py'de
# sifirlama sonrasi isinma adimi yoktu, 305 bolumun 304'u etkilendi.
# Varsayilan 2: kol hala evde ama render tazelenmis.
FR   = int(os.environ.get("FRAME", "2"))
# NESNE-MERKEZLI model: x,y kupe gore uretilmis -> modelin KENDI kup tahminiyle
# mutlaklastir (cikarimda da boyle yapiliyor). z zaten mutlak.
OBJC = os.environ.get("OBJ_CENTRIC", "0") == "1"
DEV, K = os.environ.get("PROBE_DEV", "cuda"), int(os.environ.get("K", "8"))
# Uzman durum makinesi bolum basinda 0.2 s REST yapar: des_ee_pose = ee_pose
# ("neredeysen orada kal"). 50 Hz'de tam 12 kare. O karelerde etiket kupe DEGIL
# kolun o anki (rastgele) pozuna bakar -- ogrenilemez gurultu. Model bunu taklit
# ettigi icin planin ilk adimlarinda z masa altina sapiyor ve argmin(z) oraya
# kilitleniyor. Okumada bu pencere atlanmali.
REST = int(os.environ.get("REST_SKIP", "12"))
TASK = "pick up the cube and lift it"

cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = DEV
pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg); pol.eval(); pol.to(DEV)
pol.config.device = DEV
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}})

# Kamera geometrisi 2026-09-04'te degisti; model hangi veriyle egitildiyse
# olcum de ONUNLA yapilmali. HDF5 ortam degiskeniyle secilir.
_H5 = os.environ.get("HDF5", "~/Projects/franka_vla_data/data/franka_lift_merged.hdf5")
h = h5py.File(os.path.expanduser(_H5))
eps = sorted(h["data"].keys())
zs = np.zeros(16, dtype=np.float32)
errs, pred, true, aux = [], [], [], []
for i in range(0, len(eps), max(1, len(eps) // N)):
    if len(errs) >= N: break
    ep = h["data"][eps[i]]
    def _img(name):
        return torch.from_numpy(ep[name][FR]).float().permute(2, 0, 1).unsqueeze(0) / 255.
    b = {"observation.images.front": _img("observation.images.front"),
         "observation.images.wrist": _img("observation.images.wrist"),
         "observation.state": torch.from_numpy(zs).unsqueeze(0),
         "task": [TASK]}
    # UC KAMERA: yan kamera hem HDF5'te hem modelin girdilerinde varsa gonder.
    # Modelin beklediginden AZ goruntu gondermek sessizce bozar; FAZLA gondermek
    # de oyle -- bu yuzden iki tarafi da kontrol ediyoruz.
    if "observation.images.side" in ep and "observation.images.side" in pol.config.input_features:
        b["observation.images.side"] = _img("observation.images.side")
    with torch.no_grad():
        ch = np.mean([post(pol.predict_action_chunk(pre(b)))[0].cpu().numpy() for _ in range(K)], axis=0)
    if OBJC:
        absxyz = ch[:, :3].copy()
        absxyz[:, 0] += ch[:, 8]          # modelin kendi kup tahmini
        absxyz[:, 1] += ch[:, 9]
    else:
        absxyz = ch[:, :3] + ep["observation.state.ee_pos"][FR]
    j = REST + int(np.argmin(absxyz[REST:, 2]))
    xy = absxyz[j, :2]
    cube = ep["observation.state.object_pos"][FR][:2]
    errs.append(np.linalg.norm(xy - cube)); pred.append(xy); true.append(cube)
    if ch.shape[1] >= 10:
        # YARDIMCI GOREV: modelin kupun yeri hakkindaki DOGRUDAN tahmini
        aux.append(np.mean(ch[:, 8:10], axis=0))
h.close()

errs, pred, true = np.array(errs), np.array(pred), np.array(true)
print(f"\ncheckpoint: {CKPT}")
print(f"n={len(errs)} bolum, kare {FR}, REST_SKIP={REST}" + (" | NESNE-MERKEZLI" if OBJC else ""))
print(f"  medyan hata : {np.median(errs)*1000:6.1f} mm")
print(f"  ortalama    : {errs.mean()*1000:6.1f} mm   (min {errs.min()*1000:.0f} / max {errs.max()*1000:.0f})")
for j, ax in enumerate("xy"):
    sl = np.polyfit(true[:, j], pred[:, j], 1)
    c = np.corrcoef(true[:, j], pred[:, j])[0, 1]
    print(f"  {ax}: egim {sl[0]:+.3f}  kesme {sl[1]:+.3f}  kor {c:+.3f}  "
          f"sapma {np.mean(pred[:,j]-true[:,j])*1000:+.0f}mm")
if aux:
    aux = np.array(aux)
    ae = np.linalg.norm(aux - true, axis=1)
    print(f"\n  YARDIMCI GOREV -- modelin kup icin DOGRUDAN tahmini:")
    print(f"    medyan {np.median(ae)*1000:6.1f} mm | ortalama {ae.mean()*1000:6.1f} mm")
    for j, axn in enumerate("xy"):
        print(f"    {axn}: egim {np.polyfit(true[:,j], aux[:,j], 1)[0]:+.3f}  "
              f"kor {np.corrcoef(true[:,j], aux[:,j])[0,1]:+.3f}")
print(f"\n  referans: sabit-orta 136mm | CNN 12mm | kavrama esigi 20mm")
