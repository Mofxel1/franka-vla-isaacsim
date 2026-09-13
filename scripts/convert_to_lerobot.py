"""
Bizim HDF5 -> LeRobot v3.0 veri seti donusumu.

Ne degisiyor:
  - 152 ayri bolum grubu      -> tek duz tablo (parquet) + bolum sinirlari ust veride
  - ham piksel dizileri       -> MP4 video (kamera basina)
  - 9 ayri observation.state.* -> tek 16 boyutlu "observation.state" vektoru
  - dosya oznitelgi "task"    -> tasks.jsonl + task_index

observation.state icerigi (16 boyut):
  [0:9]   eklem acilari  (7 kol + 2 parmak)
  [9:12]  el konumu      (x, y, z)
  [12:16] el yonelimi    (quaternion)

DIKKAT: kupun konumu BILEREK disarida birakildi -- gercek robotta olculemez.
        (HDF5'te duruyor, fikir degisirse oradan alinir.)

Kullanim:
  conda activate lerobot
  python convert_to_lerobot.py --hdf5 ~/Projects/franka_vla_data/data/franka_lift_150ep.hdf5
"""

import argparse, os, shutil, time
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5", type=str, nargs="+",
                    default=[os.path.expanduser("~/Projects/franka_vla_data/data/franka_lift_150ep.hdf5")],
                    help="Bir veya BIRDEN FAZLA HDF5. Coklu verilirse bolumler "
                         "sirayla eklenir -- ayrica birlestirmeye (ve 10+ GB'lik "
                         "kopyaya) gerek kalmaz. 2026-09-08: merge_hdf5 ile "
                         "birlestirme diski doldurmustu.")
parser.add_argument("--repo_id", type=str, default="franka_lift_sim",
                    help="veri seti adi (HF Hub'a yuklerken kullanici/ad seklinde olur)")
parser.add_argument("--root", type=str,
                    default=os.path.expanduser("~/Projects/franka_vla_data/lerobot"),
                    help="cikti klasoru")
parser.add_argument("--limit", type=int, default=None, help="sadece ilk N bolum (test icin)")
parser.add_argument("--zero_state", action="store_true",
                    help="observation.state alanini BIRAK ama icini SIFIRLA. "
                         "SmolVLA state'i zorunlu tutuyor (yoksa KeyError), o yuzden "
                         "kaldiramayiz; ama sifirlayinca hicbir bilgi tasimaz ve model "
                         "goruntuye mecbur kalir. Sizinti icin bkz. proje notlari.")
parser.add_argument("--no_state", action="store_true",
                    help="observation.state'i veri setine HIC KOYMA (vision-only). "
                         "Durum vektoru kolun konumunu tasiyor, o da kupun yerini ele "
                         "veriyor (kare 20'de R^2=0.86); model bu yuzden GORMEYI hic "
                         "ogrenmiyor. Kaldirinca goruntuden baska kaynak kalmiyor.")
parser.add_argument("--max_frames", type=int, default=None,
                    help="her bolumden sadece ILK N kareyi al. Sizinti tablosu: kupun "
                         "yeri kol konumundan kare 15'te R^2=0.63, kare 20'de 0.86, kare "
                         "60'ta 0.99 ile tahmin edilebiliyor. 250 karelik bolumun sadece "
                         "ilk ~15 karesi GORU gerektiriyor; kalan %94 kaybi domine edip "
                         "modelin gormeyi ogrenmesini gereksiz kiliyor. Kirpinca kayip "
                         "goru-kritik karelerden gelir.")
parser.add_argument("--object_centric", action="store_true",
                    help="NESNE-MERKEZLI aksiyon: konumun x,y'sini KUPE gore yaz "
                         "(hedef - kup), z'yi MUTLAK birak. Kupun z'si sabit "
                         "(std 2.2mm) oldugu icin kup-z tahminine gerek kalmaz.\n"
                         "Neden: olculdu ki model kupu 14.6mm ile biliyor "
                         "(x egim 0.919, kor 0.980) ama hareket planinda x'e hic "
                         "tepki vermiyor (egim 0.149, kor 0.042). Aksiyon KOLA gore "
                         "tanimlandigi icin kup bilgisi devrede olmak zorunda degil. "
                         "Kupe gore tanimlanirsa cikarimda mutlak hedef ancak modelin "
                         "KENDI kup tahminiyle uretilebilir -> algi yapisal olarak "
                         "donguye girer. --aux_cube ile birlikte kullanilmali.")
parser.add_argument("--aux_cube", action="store_true",
                    help="YARDIMCI GOREV: aksiyon vektorunun sonuna kupun (x,y) "
                         "konumunu ekle -> 8 boyut yerine 10. Model her plan "
                         "adiminda kupun nerede oldugunu da soylemek zorunda kalir; "
                         "kayip bunu cezalandirdigi icin GORMEYI ogrenmek zorunlu "
                         "hale gelir. Aksiyon kaybinin asla uretmedigi gradyani "
                         "uretir. SmolVLA aksiyonu max_action_dim=32'ye doldurdugu "
                         "icin katman boyutlari degismez, mevcut checkpoint yuklenir. "
                         "Cikarimda son iki boyut atilir (ya da tani icin okunur).")
parser.add_argument("--aux_repeat", type=int, default=1,
                    help="--aux_cube ile: kupun (x,y)'sini aksiyon vektorunde N kez "
                         "tekrarla. Kayip tum boyutlari esit agirliklar; tekrar sayisi "
                         "yardimci gorevin kayiptaki PAYINI belirler. N=1 -> 2/10 = %20, "
                         "N=4 -> 8/16 = %50. Algi/eylem kopuklugunun sebebi sinyalin "
                         "zayifligiysa bunu artirmak dogrudan test eder.")
parser.add_argument("--only_success", action="store_true",
                    help="Sadece BASARILI bolumleri cevir (HDF5 'success' ozniteligi). "
                         "Orijinal veri seti boyle toplanmis (305/305 basarili). "
                         "Basarisiz bolumlerde uzman kupu firlatabiliyor; o karelerde "
                         "yardimci etiket (kup x,y) metrelerce uzakta oluyor ve "
                         "normalizasyonu bozuyor. Olculdu: filtresiz sette yardimci "
                         "std 5.09 m (olmasi gereken 0.055), model de olcegi 90 kat "
                         "sismis bir esleme ogrendi (x egimi 2.27 yerine ~1.0).")
parser.add_argument("--cube_box", type=float, nargs=4, default=None,
                    metavar=("X0", "X1", "Y0", "Y1"),
                    help="Kup bu XY kutusundan CIKTIGI kareden itibaren bolumu KES "
                         "(bolumu atma -- erken kareler degerli). DAgger verisinde "
                         "politika kupe carpip firlatiyor; sonraki kareler 'masadaki "
                         "kupu bul' gorevini ogretmiyor ve normalizasyonu sisiriyor "
                         "(yardimci std 0.0556 -> 0.0764). OLCULDU (2026-09-13): "
                         "basarili UZMAN bolumlerinde kupun XY gezintisi medyan "
                         "175 mm, yani 'kimildadi mi' esigi YANLIS olurdu -- uzman "
                         "kupu tasiyor. Dogru olcut calisma alani siniri. "
                         "Onerilen: 0.15 0.85 -0.50 0.50 (dogma araligi + 25cm pay).")
parser.add_argument("--max_abs", type=float, default=1.5,
                    help="Bir bolumde |aksiyon[:3]| veya |kup| bu sinirin (metre) "
                         "ustune cikarsa bolumu ATLA. --only_success YETMIYOR: "
                         "basari olcutu 'tepe_z>0.10 ve son_z>0.10'; kup FIRLATILIP "
                         "havada kalirsa son_z 75 m olur ve 'basarili' sayilir. "
                         "Gurultulu toplamada (--action_noise) bu vaka sikliyor ve "
                         "normalizasyonu cokertiyor (olculdu: aksiyon z std 0.112 -> "
                         "2.466, yardimci std 0.056 -> 3.427). 0 = kapali.")
parser.add_argument("--skip_first", type=int, default=0,
                    help="Her bolumun ILK N karesini atla. Toplamada sifirlama "
                         "sonrasi render bir kare gecikiyor: 0. karenin goruntusu "
                         "ONCEKI bolumun son karesi (tutucunun havada kupu tuttugu "
                         "an). Olculdu: kare0->1 piksel farki 55-128, 1->2 ~3, "
                         "sonrasi ~0.5; kol ise sabit. 305 bolumun 304'u etkilenmis. "
                         "Onerilen: 2. (Simulasyon dongusune ek adim eklemek "
                         "DENENDI ve daha kotu cikti -- bkz. collect_demos.py notu.)")
parser.add_argument("--repeat_early", type=int, default=0,
                    help="Her bolumun ILK --early_len karesini N kez FAZLADAN, ayri "
                         "bolum olarak yaz. Amac: gorevin tamamini iceren veri setinde "
                         "goru-kritik karelerin kayiptaki payini artirmak. Tam bolumde "
                         "bu pay %6 ve model gormeyi UNUTUYOR (docs/SONUCLAR.md, tam "
                         "bolum denemesi: 68mm -> 125mm, korelasyon negatife dondu). "
                         "150 kare + 2 kopya ile pay %17 olur.")
parser.add_argument("--early_len", type=int, default=60,
                    help="--repeat_early kopyalarinin uzunlugu (kare)")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--relative_actions", action="store_true",
                    help="aksiyonun konum kismini GORELI (hedef - mevcut EE) yaz. "
                         "Mutlak aksiyon durumla ~0.9 korele oldugu icin model goruntu "
                         "yerine propriosepsiyonu ezberliyor (nedensel karisiklik).")
args = parser.parse_args()

from lerobot.datasets import LeRobotDataset

STATE_DIM = 16
ACTION_DIM = 8


def build_state(ep, t):
    """9 eklem + 3 konum + 4 yonelim = 16 boyut."""
    return np.concatenate([
        ep["observation.state.joint_pos"][t],   # 9
        ep["observation.state.ee_pos"][t],      # 3
        ep["observation.state.ee_quat"][t],     # 4
    ]).astype(np.float32)


def main():
    srcs = [h5py.File(p, "r") for p in args.hdf5]
    src = srcs[0]
    fps = float(src.attrs["fps"])
    task = str(src.attrs["task"])
    res = int(src.attrs["image_resolution"])
    # (dosya, bolum_adi) ciftleri -- coklu kaynak
    pairs = [(f, e) for f in srcs for e in sorted(f["data"].keys())]
    if args.limit:
        pairs = pairs[:args.limit]
    eps = [e for _, e in pairs]
    print(f"[CEVIR] {len(args.hdf5)} kaynak dosya, toplam {len(pairs)} bolum")

    out_root = os.path.join(args.root, args.repo_id)
    if os.path.exists(out_root):
        if args.overwrite:
            print(f"[CEVIR] mevcut klasor siliniyor: {out_root}")
            shutil.rmtree(out_root)
        else:
            raise SystemExit(f"HATA: {out_root} zaten var. --overwrite kullan.")

    print(f"[CEVIR] kaynak : {args.hdf5}")
    print(f"[CEVIR] hedef  : {out_root}")
    print(f"[CEVIR] {len(eps)} bolum | {fps:.0f} Hz | {res}px | gorev: \"{task}\"")

    # LeRobot sema tanimi
    features = {
        "observation.images.front": {
            "dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {
            "dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]},
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM + 2 * args.aux_repeat if args.aux_cube else ACTION_DIM,),
            "names": None},
    }

    # Yan kamera HDF5'te varsa otomatik dahil et (uc kameralı toplama)
    _probe = pairs[0][0]["data"][pairs[0][1]]
    HAS_SIDE = "observation.images.side" in _probe
    if HAS_SIDE:
        features["observation.images.side"] = {
            "dtype": "video", "shape": (res, res, 3), "names": ["height", "width", "channels"]}
        print("[CEVIR] YAN KAMERA bulundu -> uc kameralı veri seti")

    if not args.no_state:
        features["observation.state"] = {
            "dtype": "float32", "shape": (STATE_DIM,), "names": None}

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(fps),
        root=out_root,
        features=features,
        use_videos=True,
    )
    # Aksiyon biciminin ne oldugu inference tarafinda BILINMELI.
    _sd = 'YOK' if args.no_state else ('SIFIRLANMIS (vision-only)' if args.zero_state else 'VAR')
    print(f"[CEVIR] durum vektoru: {_sd}")
    _af = ('NESNE-MERKEZLI (x,y kupe gore; z mutlak)' if args.object_centric
           else ('GORELI (delta)' if args.relative_actions else 'MUTLAK'))
    print(f"[CEVIR] aksiyon bicimi: {_af}")
    if args.aux_cube:
        _nd = ACTION_DIM + 2 * args.aux_repeat
        print(f"[CEVIR] YARDIMCI GOREV acik: aksiyon {ACTION_DIM} -> {_nd} boyut "
              f"(kup x,y {args.aux_repeat} kez tekrarli, kayiptaki pay ~%{200*args.aux_repeat//_nd})")

    t0 = time.time()
    total_frames = 0
    n_skipped_abs = 0
    n_skipped_box = n_truncated = 0
    for k, (_f, name) in enumerate(pairs):
        ep = _f["data"][name]
        if args.only_success and not bool(ep.attrs.get("success", False)):
            continue
        n = int(ep.attrs["num_samples"])
        S0 = args.skip_first
        if args.max_abs > 0:
            _a = np.abs(ep["action"][:, :3]).max()
            _c = np.abs(ep["observation.state.object_pos"][:, :3]).max()
            if _a > args.max_abs or _c > args.max_abs:
                n_skipped_abs += 1
                continue
        if args.cube_box:
            _o = ep["observation.state.object_pos"][:, :2]
            x0, x1, y0, y1 = args.cube_box
            _in = (_o[:, 0] >= x0) & (_o[:, 0] <= x1) & (_o[:, 1] >= y0) & (_o[:, 1] <= y1)
            if not _in[0]:
                n_skipped_box += 1
                continue                      # daha ilk kareden disarida: bolumu at
            _bad = np.flatnonzero(~_in)
            if len(_bad):
                _cut = int(_bad[0])
                if _cut - S0 < 20:
                    n_skipped_box += 1
                    continue                  # kesince geriye anlamli veri kalmiyor
                n_truncated += 1
                n = min(n, _cut)
        if args.max_frames:
            n = min(n, args.max_frames + S0)
        front = ep["observation.images.front"][:]
        wrist = ep["observation.images.wrist"][:]
        side = ep["observation.images.side"][:] if HAS_SIDE else None
        actions = ep["action"][:]

        ee_pos_all = ep["observation.state.ee_pos"][:]
        obj_pos_all = (ep["observation.state.object_pos"][:]
                       if (args.aux_cube or args.object_centric) else None)

        def write_episode(limit):
            for t in range(S0, limit):
                _add_frame(t)
            ds.save_episode()
            return max(0, limit - S0)

        def _add_frame(t):
            act = actions[t].astype(np.float32).copy()
            if args.object_centric:
                # x,y KUPE gore; z MUTLAK kalir (masa yuksekligi sabit).
                act[0] -= obj_pos_all[t][0]
                act[1] -= obj_pos_all[t][1]
            elif args.relative_actions:
                # Konum kismini farka cevir: hedef - mevcut EE konumu.
                # Quaternion ve gripper AYNEN kalir (uzman yonelimi zaten sabit
                # tutuyor, bilgi tasimiyor).
                act[:3] = act[:3] - ee_pos_all[t]
            if args.aux_cube:
                # Kupun MUTLAK (x,y)'si. Plan boyunca sabit -- kup yaklasma
                # fazinda hareket etmiyor. Model bunu ancak GORUNTUDEN cikarabilir
                # (durum vektoru sifirlanmis).
                _c = obj_pos_all[t][:2].astype(np.float32)
                act = np.concatenate([act] + [_c] * args.aux_repeat)
            # NOT: lerobot 0.6.1'de "task" ayri parametre degil, frame sozlugunun
            # icinde bir anahtar olmali.
            frame = {
                "observation.images.front": front[t],
                "observation.images.wrist": wrist[t],
                "action": act,
                "task": task,
            }
            if HAS_SIDE:
                frame["observation.images.side"] = side[t]
            if not args.no_state:
                st = build_state(ep, t)
                if args.zero_state:
                    st = np.zeros_like(st)
                frame["observation.state"] = st
            ds.add_frame(frame)

        total_frames += write_episode(n)
        # Goru-kritik pencereyi fazladan kez yaz (ayri bolumler olarak)
        for _ in range(args.repeat_early):
            total_frames += write_episode(min(args.early_len + S0, n))

        if (k + 1) % 10 == 0 or k == len(eps) - 1:
            el = time.time() - t0
            print(f"[CEVIR] {k+1}/{len(eps)} bolum | {total_frames:,} kare | "
                  f"{el:.0f}s | tahmini kalan {el/(k+1)*(len(eps)-k-1):.0f}s", flush=True)

    # ZORUNLU: parquet yazicilarini kapat, yoksa dosyalar bozuk kalir
    ds.finalize()
    src.close()

    # boyut karsilastirmasi
    def dirsize(p):
        return sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(p) for f in fs)
    # NOT: --hdf5 artik nargs="+" -> LISTE. Tek dosya varsayan bu ozet satiri
    # 2026-09-08'de cevirme bittikten SONRA patlamisti (veri saglamdi).
    src_mb_full = sum(os.path.getsize(_p) for _p in args.hdf5) / 1e6
    n_all = 0
    for _p in args.hdf5:
        with h5py.File(_p, "r") as _f:
            n_all += len(_f["data"].keys())
    src_mb = src_mb_full * len(eps) / n_all      # sadece cevrilen bolumlerin payi
    out_mb = dirsize(out_root) / 1e6

    print("\n" + "=" * 60)
    print(f"[CEVIR] TAMAM -> {out_root}")
    print(f"[CEVIR] {len(eps)} bolum | {total_frames:,} kare")
    print(f"[CEVIR] boyut: {src_mb:,.0f} MB  ->  {out_mb:,.0f} MB  "
          f"({src_mb/max(out_mb,1):.1f}x kucuk)")
    if args.max_abs > 0:
        print(f"[CEVIR] sinir disi ({args.max_abs} m) atlanan bolum: {n_skipped_abs}")
    if args.cube_box:
        print(f"[CEVIR] kup kutusu {args.cube_box}: {n_truncated} bolum KESILDI, "
              f"{n_skipped_box} bolum atlandi")
    print(f"[CEVIR] sure: {time.time()-t0:.0f}s")
    print("=" * 60)


main()
