"""
Birden fazla gosterim HDF5'ini tek dosyada birlestirir.

Bolum isimleri yeniden numaralandirilir (episode_000000, 000001, ...).
Kunye bilgileri ilk dosyadan alinir; 'seed' alani birlestirilen tohumlarin
listesine cevrilir.

Kullanim:
  python merge_hdf5.py --out birlesik.hdf5 dosya1.hdf5 dosya2.hdf5
"""
import argparse, os
import re
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
parser.add_argument("inputs", nargs="+")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

out = os.path.expanduser(args.out)
inputs = [os.path.expanduser(p) for p in args.inputs]

for p in inputs:
    if not os.path.exists(p):
        raise SystemExit(f"HATA: bulunamadi -> {p}")
if os.path.exists(out) and not args.overwrite:
    raise SystemExit(f"HATA: {out} zaten var. --overwrite kullan.")

print(f"[BIRLESTIR] {len(inputs)} dosya -> {out}")
seeds, total, idx = [], 0, 0

with h5py.File(out, "w") as dst:
    dg = dst.create_group("data")
    for fi, path in enumerate(inputs):
        with h5py.File(path, "r") as src:
            if fi == 0:
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
            else:
                # tutarlilik kontrolu: farkli fps/cozunurluk birlestirilmemeli
                for k in ("fps", "image_resolution", "action_space", "env"):
                    if k in src.attrs and k in dst.attrs and src.attrs[k] != dst.attrs[k]:
                        raise SystemExit(
                            f"HATA: '{k}' uyusmuyor ({dst.attrs[k]} vs {src.attrs[k]}) -> {path}")
            if "seed" in src.attrs:
                # Zaten birlestirilmis bir dosyanin 'seed' ozniteligi LISTE olur
                # ([101, 102]); int() onu cevirmeye calisip patliyordu
                # (2026-09-07: karisim zinciri bu yuzden basladigi anda coktu).
                # 'seed' uc bicimde gelebiliyor:
                #   int            -> tek toplamadan
                #   dizi [101 102] -> birlestirilmis dosyadan
                #   str '[101, 102]' -> h5py bazi surumlerde boyle yaziyor
                _sd = src.attrs["seed"]
                if isinstance(_sd, (bytes, np.bytes_)):
                    _sd = _sd.decode()
                if isinstance(_sd, str):
                    _sd = [int(x) for x in re.findall(r"-?\d+", _sd)]
                seeds.extend(np.atleast_1d(_sd).astype(int).tolist())

            eps = sorted(src["data"].keys())
            n_fr = 0
            for e in eps:
                src.copy(src["data"][e], dg, name=f"episode_{idx:06d}")
                n_fr += int(src["data"][e].attrs["num_samples"])
                idx += 1
            total += n_fr
            print(f"[BIRLESTIR]   {os.path.basename(path)}: {len(eps)} bolum, {n_fr:,} kare", flush=True)

    dst.attrs["num_episodes"] = idx
    dst.attrs["seed"] = str(seeds)          # birden fazla tohum -> metin
    dst.attrs["merged_from"] = [os.path.basename(p) for p in inputs]

mb = os.path.getsize(out) / 1e6
print("=" * 56)
print(f"[BIRLESTIR] TAMAM: {idx} bolum | {total:,} kare | {mb:,.0f} MB")
print("=" * 56)
