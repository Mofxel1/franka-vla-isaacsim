# Franka VLA — Isaac Sim'de görü-dil-aksiyon ile küp kaldırma

Isaac Sim'de Franka Panda koluyla sentetik veri topla → LeRobot biçimine çevir →
SmolVLA'yı eğit → kapalı döngüde simülatörde test et.

---

## Önce buraya bak

| dosya | ne var |
|---|---|
| **[docs/DURUM.md](docs/DURUM.md)** | **Nerede kaldık, sıradaki adım.** Buradan başla. |
| [docs/YOL_HARITASI.md](docs/YOL_HARITASI.md) | Nereye gidiyoruz — fazlar, Nova 5, robot köpek |
| [docs/SONUCLAR.md](docs/SONUCLAR.md) | Hangi model, hangi test, ne çıktı — tek tablo |
| [docs/KOMUTLAR.md](docs/KOMUTLAR.md) | Kopyala-yapıştır komutlar ve tuzaklar |

Yeni bir ölçüm yapınca sonucu `SONUCLAR.md`'ye ekle. Bir hipotezi elediğinde
oradaki Test 8 tablosuna yaz — aynı şeyi ikinci kez denememek için.

---

## Klasör düzeni

```
franka_vla_data/
├── README.md                   bu dosya
├── docs/                       durum, sonuçlar, komutlar
├── scripts/                    boru hattı (düz tutuldu, importlar bozulmasın)
│   ├── collect_demos.py            1. Isaac Sim'de uzman verisi topla
│   ├── pick_lift_sm.py                uzman durum makinesi
│   ├── domain_randomizer.py           ışık/kamera/masa randomizasyonu
│   ├── merge_hdf5.py               1b. HDF5 dosyalarını birleştir
│   ├── convert_to_lerobot.py       2. LeRobot biçimine çevir
│   ├── policy_server.py            3. modeli soket üzerinden servis et
│   ├── eval_policy_isaacsim.py     4. kapalı döngü değerlendirme
│   ├── bridge_protocol.py             ortamlar arası ham bayt protokolü
│   ├── isaac_shutdown.py              close() asılmasına karşı watchdog
│   ├── vram_probe.py / watch_demo.py / diagnose_policy.py
│   ├── diagnostics/            ölçüm betikleri (bağımsız çalışır)
│   │   ├── localize_test.py        ← ASIL METRİK
│   │   ├── vision_probe.py            bilgi görüntüde var mı (CNN)
│   │   ├── vision_test.py             eğim testi
│   │   ├── chunk_compare.py           plan büyüklüğü
│   │   ├── measure_slope.py / raw_vs_mp4_test.py / openvla_test.py
│   └── archive/                artık geçersiz betikler
├── results/                    ölçüm çıktıları
├── data      -> /data/franka_vla/data       (ham HDF5)
├── lerobot   -> /data/franka_vla/lerobot    (LeRobot veri setleri)
└── runs      -> /data/franka_vla            (tüm eğitim çıktıları: train_*)
```

Üçü de symlink; asıl veri `/data` bölümünde (ev dizini yetmiyor).
Checkpoint yolu: `runs/train_zs/checkpoints/010000/pretrained_model`

---

## Veri setleri (2026-09-12)

Ham veri (`data/` → `/data/franka_vla/data`):

| ad | ne | boyut |
|---|---|---|
| `franka_lift_geo.hdf5` | 2 kameralı ham kaynak, 208 bölüm, güncel geometri | 5.0 GB |
| `s3_val404.hdf5` | **3 kameralı doğrulama seti — hiçbir model görmedi** | 1.2 GB |

`s3_val404.hdf5` dürüst ölçümün tek kaynağı: modeller kendi eğitim verilerinde
22-33 mm, bu sette 33-41 mm veriyor. **Silme.**

LeRobot veri setleri (`lerobot/`):

| ad | ne | boyut |
|---|---|---|
| `franka_lift_3kam` | 3 kamera, nesne-merkezli, REST'siz (`--skip_first 12`) | 235 MB |
| `franka_lift_rest` | 2 kamera, REST'siz | 155 MB |
| `franka_lift_geo2` | 2 kamera, güncel geometri | 153 MB |
| `franka_lift_dart2` | DART gürültülü (dal kapandı) | 155 MB |

Modeller (`runs/`): `train_geo3` (kapalı döngüde en iyi), `train_3kam`
(çevrimdışında en iyi), `train_rest`, `train_dart2`.

2026-09-12'de silindi: `franka_lift_merged.hdf5` (eski geometri),
`franka_lift_dart.hdf5`, `s3_s40{1,2,3}.hdf5` (çevrildi), `train_objc2`.

---

## Üç Python ortamı

| ortam | ne için | not |
|---|---|---|
| `conda activate lerobot` | eğitim, çıkarım, tanılama | |
| `conda activate isaaclab` | Isaac Sim simülasyonu | Kit kendi numpy 1.26'sını yükler |
| `source /data/venv_openvla/bin/activate` | OpenVLA testi | transformers 4.40.1 + accelerate 0.29.3 |

Ortamlar arası veri **pickle ile taşınmaz** (numpy sürüm çakışması) —
`bridge_protocol.py` ham bayt kullanır.

---

## Git

Bu klasör bir git deposu. `.gitignore` şunları **dışarıda tutar**:

- `data/`, `lerobot/`, `runs/` symlink'leri ve tüm `*.hdf5` / `*.safetensors` /
  `*.npz` — veri ve model ağırlıkları git'e girmez
- `*.gif`, `*.mp4` — yeniden üretilebilir
- `results/{train,convert,collect,server,merge}_*.log` — ilerleme çubuğu spam'i

**Repoda duran ve önemli olan:**

- `scripts/` — boru hattının tamamı
- `docs/` — projenin hafızası (DURUM, SONUCLAR, KOMUTLAR, YOL_HARITASI)
- `results/*.sh` — her deneyin tam reçetesi (33 zincir betiği)
- `results/sonuc_*.log` — ölçüm çıktıları
- `results/*.png` — kamera geometrisi bulgusunun görsel kanıtı
- `results/chain.log` — tüm deneylerin zaman çizelgesi
