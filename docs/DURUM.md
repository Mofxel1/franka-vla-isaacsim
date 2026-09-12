# Durum — nerede kaldık

Son güncelleme: 2026-09-12

---

## Tek cümlelik özet

**2026-09-08 sonu.** Gün boyunca üç dal denendi, üçü de kapandı ama ölçüm
altyapısı çok daha sağlam. Kapalı döngü hâlâ **0/20**.

Taze veriyle (tohum 404) dürüst sıralama:

| model | çevrimdışı medyan | canlı adım 0 | canlı x kor |
|---|---|---|---|
| geo3 (2 kam) | 41.4 mm | **59.9 mm** | **0.722** |
| rest (2 kam) | 41.4 mm | 60.3 mm | — |
| 3kam (3 kam) | **33.6 mm** | 79.6 mm | 0.285 |

Çevrimdışı en iyi 3 kamera, **canlıda en iyi geo3**. Bir sonraki iş: yan
kameranın canlı görüntüsü ile eğitim görüntüsünü karşılaştırmak (ön kamerada
aynı uyuşmazlık 2026-09-04'te bulunmuştu, yan kamera için hiç bakılmadı).

## Kanıtın özeti

```
lokalizasyon medyan hatası (kavrama için <20 mm gerekli)
  sabit-orta tahmini               136 mm
  train_zs@10000                   131 mm   ← taban çizgisiyle aynı
  train_geo3@6000  (veri seti)      22.4 mm  ← EN İYİ, REST_SKIP=12 ile
  train_karisim@4000               38.0 mm  ← karışım: işe yaramadı
  train_dart2@6000                 49.4 mm  ← gürültülü: en kötü
  246K parametreli CNN              12 mm   ← bilgi görüntüde VAR
```

NOT: 2026-09-08 öncesindeki tüm "medyan hata" sayıları (650-680 mm bandı)
ölçüm hatasıydı — bkz. SONUCLAR.md, REST penceresi.

Ayrıntılar ve tüm ölçümler: [SONUCLAR.md](SONUCLAR.md)

## Neden bu kadar geç anlaşıldı

Bütün açık döngü metrikleri mükemmel görünüyordu (kayıp 0.029, pozisyon MAE 5 mm,
quaternion MAE 0.0005, tutucu %100). Hepsi modelin **uzmanın aksiyon dizisiyle**
uyumunu ölçüyordu. Kol hareket hâlindeyken "aynen devam et" demek bu metrikleri
doyuruyor — küpün nerede olduğunu bilmeye gerek yok.

**Ders: politikayı uzmanla değil, görevin hedefiyle karşılaştır.**
Doğru metrik `scripts/diagnostics/localize_test.py`.

## Kök neden

Sızıntı tablosu (Test 6): 250 karelik bölümde görü sadece ilk ~15 karede gerekli.
Kalan %94 eğitim kaybını domine ediyor, model görmeyi öğrenmeden kaybın neredeyse
tamamını düşürebiliyor. Bu yüzden 5000→10000 adımda biraz düzelip **10000→15000
arasında hiç kıpırdamadı**.

---

## En iyi model şu an

Hepsi **tohum 404** (hiçbirinin görmediği taze veri) ile ölçüldü:

| model | kamera | çevrimdışı medyan | canlı adım 0 | canlı x kor |
|---|---|---|---|---|
| `train_geo3/checkpoints/006000` | 2 | 41.4 mm | **59.9 mm** | **0.722** |
| `train_rest/checkpoints/004000` | 2 | 41.4 mm | 60.3 mm | — |
| `train_3kam/checkpoints/004000` | 3 | **33.6 mm** | 79.6 mm | 0.285 |

- **Kapalı döngü için en iyi: `train_geo3`.** Hepsi 0/20.
- **Çevrimdışı için en iyi: `train_3kam`** ama canlıda çöküyor.
- Kavrama eşiği **20 mm** (küp 4 cm). En iyimiz taze veride 33.6 mm — yaklaşık
  **2 kat** uzaktayız.

UYARI: modeli kendi eğitim verisinde ölçme. Aynı modeller kendi verilerinde
22-33 mm veriyor; taze veride 33-41 mm. Fark ezber.

---

## 2026-09-04'te bulunan BEŞ hata

Hepsi aynı aileden: **boru hattının iki ucunda farklı varsayım.**

| # | hata | etki |
|---|---|---|
| 1 | `localize_test.py` HDF5 yolunu koda gömüyordu | yeni modeli eski veriyle test ettim |
| 2 | **kamera geometrisi değişmiş** | eğitim verisi ≠ eval sahnesi — uçurumun asıl sebebi |
| 3 | `--only_success` eksikti | 6 fırlatılmış küp (en uzak 174.9 m) normalizasyonu çökertti |
| 4 | toplama 8 ortam / eval 1 ortam | x eğimi 0.202 → 0.494 |
| 5 | ölçüm adımı: veri kare 2 / eval adım 0 | görüntü istatistiği farkı buradan |

**En önemlisi 2.** `franka_lift_merged.hdf5` haftalar önce farklı bir kamera
konumuyla toplanmış; parametreler sonradan değişmiş ve hiçbir yerde kayıtlı
değildi. Üç bağımsız yoldan doğrulandı:

```
                      masa alanı   ufuk satırı
ESKİ veri seti          22.9%        78.8
YENİ toplama            32.9%        66.3      ← eval'le aynı
CANLI eval              31.9%        63.1
```
+ yeni veriyle ince ayar kaybı 0.505'ten başladı (önceki koşu 0.033'te bitmişti).

Bu, "canlıda derinlik (x) çöküyor ama yanal (y) sağlam" bulgusunu açıklıyor:
y kaba piksel konumundan okunur, x perspektif geometrisinden — kamera taşınınca
sadece x bozulur.

---

## Sıradaki adım (2026-09-08 akşamı yazıldı)

1. **Yan kamera geometri kontrolü** *(ucuz, ~30 dk, en yüksek öncelik)*
   Toplama HDF5'indeki `observation.images.side` ile canlı eval'deki yan
   görüntünün istatistiklerini karşılaştır (masa alanı, ufuk satırı, parlaklık)
   — ön kamerada 2026-09-04'te tam bu yöntemle uçurumun sebebi bulunmuştu.
   Canlı x korelasyonunun 0.794'ten 0.285'e çökmesi buna işaret ediyor.
2. **3kam'ı daha uzun eğit** *(~75 dk)* — kayıp 0.032'de hâlâ düşüyordu.
   Ancak 1. adım bir uyuşmazlık bulursa bu boşa gider; ÖNCE 1.
3. Aşağıdaki eski liste (DART varyantları, gerçek DAgger) hâlâ geçerli.

### Eski liste

**2026-09-07 durumu:** kovaryat kayma teşhisi **doğrulandı** ama ilk çözüm denemesi
net kazanç vermedi. DART bozulma eğrisini düzleştirdi (+80% → +21%) ama tabanı
iki katına çıkardı (61 → 142 mm). Kapalı döngü 0/20.

> **2026-09-07 akşam — karışım denemesi ÇALIŞMADAN çöktü.** `merge_hdf5.py`,
> zaten birleştirilmiş bir dosyanın `seed` özniteliğini (`[101, 102]` — liste)
> `int()`'e çevirmeye çalışıp patladı. Boş veri seti üretildi; **çevirme sonrası
> istatistik kapısı doğru davranıp eğitimi başlatmadı** (bu kapı bugün eklendi ve
> ilk işinde işe yaradı). `merge_hdf5.py` düzeltildi (`np.atleast_1d(...)`),
> yarın doğrudan çalıştırılabilir. Ders: uzun zinciri başlattıktan sonra ilk
> birkaç dakikada kontrol et — bu sefer 3 saat boşa gitti.

**Varyantların durumu:**

1. ~~Temiz + gürültülü karışım~~ — **DENENDİ, ÇALIŞMADI** (2026-09-08).
   Sıralama temiz (22) > karışım (38) > gürültülü (49) mm. Dal kapandı.
2. **Daha düşük gürültü (0.010).** Denenmedi. Karışım da işe yaramadığı için
   önceliği düşük.
3. **Daha uzun eğitim.** `train_3kam` için hâlâ geçerli (kayıp 0.032'de
   düşüyordu). Yan kamera geometri kontrolünden SONRA.
4. **Gerçek DAgger.** Hâlâ denenmedi, altyapı yok. En pahalı ama kovaryat
   kaymaya en doğrudan saldırı.

### 2026-09-08 — REST penceresi bulundu

Uzman durum makinesi her bölümün başında 0.2 s (= **12 kare**) REST yapıyor ve
`des_ee_pose = ee_pose` yazıyor. O karelerde etiket kupe değil kolun **o anki
rastgele pozuna** bakıyor; model durumu sıfırlanmış olduğu için öğrenilemez.

- **Ölçüm etkisi:** `argmin(z)` o pencereye kilitleniyordu → dört modelin de
  hatası ~650 mm görünüyordu. `REST_SKIP=12` ile gerçek değerler 22-49 mm.
- **Eğitim etkisi:** `--skip_first 2` 12 karenin sadece 2'sini atıyordu,
  `--repeat_early` kalanı 3 katına çıkarıyordu (kareler ~%11).
- **SONUÇ (test edildi, ÇÜRÜTÜLDÜ):** REST karelerini eğitimden çıkarmak
  (`--skip_first 12`, `train_rest`) canlı hataya hiçbir şey yapmadı —
  adım 0'da 59.9 (geo3) vs 60.3 mm (rest). REST gerçek bir hataydı (ölçümü ve
  x normalizasyonunu bozuyordu) ama **canlı uçurumun sebebi değil.**
- **Yan bulgu:** nesne-merkezli aksiyonun x varyansının neredeyse tamamı REST
  karelerinden geliyormuş (std 0.0827 → 0.0117, aralık [−0.842,+0.184] →
  [−0.173,+0.184]). Model x'i 7 kat fazla büyük bir std ile normalize ediyordu.

---

## Disk durumu — DAR

- **2026-09-12 temizliği: `/` 2.9 → 11 GB, `/data` 4.7 → 20 GB.**
  Silindi: `s3_s40{1,2,3}.hdf5` (7.5 GB, `franka_lift_3kam`'e çevrilmişti),
  `franka_lift_merged.hdf5` (7.5 GB, eski geometri), `franka_lift_dart.hdf5`
  (5.3 GB, DART dalı kapandı), `train_objc2` (3 GB, eski geometri).
- **Korunanlar ve sebepleri:** `franka_lift_geo.hdf5` (2 kameralı tek ham veri,
  yeniden çevirme gerekirse şart), `s3_val404.hdf5` (**hiçbir modelin görmediği
  doğrulama seti — dürüst ölçümün tek kaynağı, SAKIN SİLME**), dört lerobot
  veri seti (ham veri olmadan da eğitim yapılabilir), `train_{geo3,rest,3kam,dart2}`.
- Bir eğitim koşusu (2 checkpoint) 3 GB. `--save_freq`'i toplam adım
  sayısına eşitle ki tek checkpoint yazsın.
- 2026-09-04'te silindi: `train_geo`, `train_objc`, `train_mixed2`,
  `train_aux_long2`, `lerobot/franka_lift_{geo,objc,mixed}`
- `/data` 1.7 GB, pratikte dolu.
- **Veri toplama 100'erlik partiler hâlinde yapılmalı** — 200 bölüm tek seferde
  ~15 GB RAM istiyor (sistemde 23 GB), OOM ile ölüyor.

---

## Ortam / donanım kısıtları

- Ubuntu 22.04, RTX 3060 Laptop **6 GB**, sürücü 535, CUDA 12.2
- Isaac Sim **4.5.0** + Isaac Lab **v2.1.0** — sabit, sürücü 535 yüzünden 5.x'e geçilemez
- Sürücü sürüm kontrolü reddediyor → `--kit_args="--/rtx/verifyDriverVersion/enabled=false"`
  (mutlaka `=` biçimi; boşluklu biçim argparse'ta patlar)
- `simulation_app.close()` asılıyor → `isaac_shutdown.safe_close()`
- Üç ayrı Python ortamı: `lerobot` (eğitim/çıkarım), `isaaclab` (simülasyon),
  `/data/venv_openvla` (OpenVLA, transformers 4.40.1 + accelerate 0.29.3)
- Kit kendi numpy 1.26'sını yüklüyor → ortamlar arası veri `bridge_protocol.py`
  ile **ham bayt** olarak taşınır, pickle çalışmaz
- `/data` bölümünde ~6.6 GB boş — checkpoint biriktirmeye dikkat
- Her zaman `python -u`, yoksa çıktı tamponlanır ve iş donmuş görünür
- `pgrep -f <desen>` / `pkill -f` **kendi kabuğunu da eşleştirir** — PID'leri bir
  komutta listele, ayrı komutta literal PID ile öldür
