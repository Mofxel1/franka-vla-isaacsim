# Sonuçlar — hangi model, hangi test, ne çıktı

Tek kaynak. Yeni bir ölçüm yapınca buraya satır ekle.

---

## Modeller

| ad | veri seti | aksiyon | durum vektörü | adım | checkpoint |
|---|---|---|---|---|---|
| `train_rel2` | `franka_lift_rel2` | göreli | **var** | 12000 | `/data/franka_vla/train_rel2/checkpoints/012000` |
| `train_zs` | `franka_lift_zs` | göreli | **sıfır** | 5000 / 10000 / 15000 | `/data/franka_vla/train_zs/checkpoints/0{05,10,15}000` |
| `train_lora` | `franka_lift_zs` + görü LoRA | göreli | sıfır | 3000 / 6000 | `/data/franka_vla/train_lora/checkpoints/00{3,6}000` |
| `train_noisy` | `franka_lift_noisy` (DART gürültü) | göreli | sıfır | 4000 / 8000 | `/data/franka_vla/train_noisy/checkpoints/00{4,8}000` |
| `train_early` | `franka_lift_early` (ilk 60 kare) | göreli | sıfır | 2000 | `/data/franka_vla/train_early/checkpoints/002000` |
| `train_early2` (A) | `franka_lift_early`, `train_early@2000`'den devam | göreli | sıfır | 2000 / 4000 | `/data/franka_vla/train_early2/checkpoints/00{2,4}000` |
| `train_aux` (B) | `franka_lift_aux` (**aksiyon 10 boyut**, son ikisi küp x,y) | göreli | sıfır | 4000 | `/data/franka_vla/train_aux/checkpoints/004000` |
| **`train_aux_long` (D1)** | `franka_lift_aux`, `train_aux@4000`'den devam | göreli | sıfır | 2000 / 4000 (küm. 6000 / **8000**) | `~/franka_runs/train_aux_long/checkpoints/004000` ← **EN İYİ** |
| `train_aux4` (D2) | `franka_lift_aux4` (**16 boyut**, küp x,y 4 kez) | göreli | sıfır | 2000 / 4000 | `~/franka_runs/train_aux4/checkpoints/004000` |

> `train_zs/checkpoints/last` → **015000**'i gösterir, ama en iyisi **010000**.
> Test ederken checkpoint'i açıkça yaz.
>
> `train_early`, `train_zs@10000`'den **ince ayar**; 3816. adımda elle durduruldu,
> sadece 002000 kaydedildi.

---

## Test 1 — Lokalizasyon (asıl metrik)

`scripts/diagnostics/localize_test.py`. Bölüm başı gözlemden (kol evde, durum sıfır)
50 adımlık plan alınır; planın z'si en düşük adımı = iniş noktası. Onun XY'si küpün
**gerçek** XY'si ile karşılaştırılır. Küp 4 cm, kavrama için <20 mm gerekir.

| model | n | medyan | ortalama | x eğim | y eğim | x kor | y kor |
|---|---|---|---|---|---|---|---|
| **sabit-orta tahmini** (taban) | 305 | **136 mm** | 138 | 0.00 | 0.00 | — | — |
| `train_zs@10000`, veri seti görüntüsü | 40 | **131 mm** | 131 | 0.084 | 0.073 | 0.270 | 0.296 |
| `train_early@2000` (A, kümülatif 2000) | 40 | 115 mm | 132 | 0.105 | 0.179 | 0.150 | 0.304 |
| `train_early2@2000` (A, kümülatif 4000) | 40 | 131 mm | 144 | 0.096 | 0.117 | 0.120 | 0.216 |
| `train_early2@4000` (A, kümülatif 6000) | 40 | **138 mm** | 136 | 0.009 | 0.196 | 0.016 | 0.337 |
| `train_aux@4000` (B, hareket planı) | 40 | **112 mm** | 130 | 0.041 | 0.281 | 0.038 | 0.449 |
| `train_aux@4000` (B, **doğrudan okuma**) | 40 | **94 mm** | 114 | 0.246 | 0.275 | 0.548 | 0.497 |
| `train_aux_long@2000` (D1, küm. 6000) | 40 | 93 mm | 114 | 0.104 | 0.401 | 0.106 | 0.554 |
| `train_aux_long@2000` (D1, **doğrudan okuma**) | 40 | 92 mm | 101 | 0.312 | 0.376 | 0.630 | 0.618 |
| `train_aux_long@4000` (D1, küm. **8000**) | 40 | **84 mm** | 100 | 0.321 | 0.439 | 0.614 | 0.617 |
| `train_aux_long@4000` (D1, **doğrudan okuma**) | 40 | **82 mm** | 101 | 0.352 | 0.383 | 0.691 | 0.585 |
| `train_aux4@4000` (D2, ağırlık %50) | 40 | 110 mm | 123 | 0.056 | 0.300 | 0.067 | 0.512 |
| `train_aux4@4000` (D2, **doğrudan okuma**) | 40 | 103 mm | 113 | 0.248 | 0.304 | 0.568 | 0.517 |
| `train_aux_long2@4000` (küm. 12000) | 40 | 71 mm | 94 | 0.486 | 0.481 | 0.751 | 0.670 |
| `train_aux_long2@8000` (küm. **16000**) | 40 | **68 mm** | 81 | 0.503 | 0.559 | 0.722 | 0.785 |
| `train_aux_long2@8000` (**doğrudan okuma**) | 40 | **62 mm** | 75 | 0.577 | 0.556 | **0.834** | 0.795 |
| `train_full_aux@6000` (TAM bölüm) | 40 | 125 mm | 128 | 0.137 | 0.083 | 0.348 | 0.294 |
| `train_full_aux@6000` (**doğrudan okuma**) | 40 | 160 mm | 189 | −0.165 | −0.028 | **−0.181** | −0.029 |
| `train_mixed@3000` (karma) | 40 | 110 mm | 120 | 0.137 | 0.187 | 0.342 | 0.491 |
| `train_mixed@6000` (karma) | 40 | **90 mm** | 101 | 0.283 | 0.386 | 0.549 | 0.714 |
| `train_mixed@6000` (**doğrudan okuma**) | 40 | 95 mm | 108 | 0.357 | 0.396 | 0.565 | 0.596 |
| `train_zs@10000`, **canlı** eval görüntüsü | 12 | 193 mm | 194 | — | — | — | — |
| görsel sonda CNN (246K parametre) | — | **12 mm** | — | ~1.0 | ~1.0 | 0.976 | 0.977 |

**Okuma:** eğimler 1.0 olmalıyken 0.01-0.28. Model küpün yerine göre çok zayıf tepki
veriyor, neredeyse sabit bir nokta üretiyor. `train_zs` ve A pratikte sabit-orta
tahmininin aynısı; **B (yardımcı görev) ilk anlamlı hareketi yaptı** ama yeterli değil.
246K'lık CNN aynı görüntülerden 12 mm çıkarıyor — **bilgi görüntüde var,
SmolVLA zor çıkarıyor.**

> Uyarı: bu tabloya 110 mm diye giren bir değer vardı, 12 bölümlük ölçümdendi.
> 40 bölümle doğrusu 131 mm. Az örnekli ölçüme güvenme — n=40 bile bu iş için az,
> A'nın 115/131/138 salınımı gerçek bir eğilim değil titremeydi.

### A denemesi (2026-08-31) — ELENDİ

Veriyi görü-kritik pencereye kırpmak (`franka_lift_early`, her bölümden ilk 60
kare) kaybı doğru yere yoğunlaştırdı ama lokalizasyonu **düzeltmedi**: 6000 adımda
138 mm, yani sabit-orta tabanına geri döndü, x eğimi 0.009'a çöktü.
Kayıp 0.227 → 0.034. **Kaybın bileşimini değiştirmek yetmiyor.**

### B denemesi (2026-08-31) — YÖN DOĞRU, BÜYÜKLÜK YETERSİZ

Yardımcı görev: küpün (x,y) konumu aksiyon vektörünün sonuna eklendi
(`franka_lift_aux`, 8 → 10 boyut). SmolVLA aksiyonu `max_action_dim=32`'ye
doldurduğu için katman boyutları değişmedi, mevcut checkpoint sorunsuz yüklendi —
kütüphaneye hiç dokunmadan yardımcı kayıp elde edildi.

A ile **adil karşılaştırma**: ikisi de `train_early@2000`'den başladı, ikisi de
4000 adım. Tek fark veri seti.

Sonuç: korelasyonlar iki katına çıktı (x 0.27 → 0.55). Bugüne kadarki en büyük
hareket. Ama 94 mm hâlâ 20 mm eşiğinin çok uzağında, eğimler 0.25-0.28 (olması
gereken 1.0).

**Algı/eylem kopukluğu — GEÇİCİYMİŞ.** 4000 adımda modelin *bildiği* (doğrudan
okuma 94 mm, x eğim 0.246) ile *hareketine yansıttığı* (112 mm, x eğim 0.041)
ayrışıyordu; bunu yapısal bir risk sandım. 8000 adımda kapandı: 84 mm vs 82 mm,
eğimler 0.321 vs 0.352. **Yapısal değil, eğitimin erken evresiymiş.**

### Deney 1 (2026-08-31) — SÜRE KALDIRAÇ, DOYMUŞ DEĞİL

B'yi aynı veriyle 8000 adıma uzatmak. Projedeki ilk sürekli iyileşme:

| kümülatif | hareket planı | x eğim | y eğim | x kor | y kor | doğrudan okuma |
|---|---|---|---|---|---|---|
| 4000 | 112 mm | 0.041 | 0.281 | 0.038 | 0.449 | 94 mm |
| 6000 | 93 mm | 0.104 | 0.401 | 0.106 | 0.554 | 92 mm |
| 8000 | **84 mm** | 0.321 | 0.439 | 0.614 | 0.617 | **82 mm** |

Her sütun tek yönde, **doyma belirtisi yok**. x korelasyonu 0.04 → 0.61.

### Deney 2 (2026-08-31) — AĞIRLIK KALDIRAÇ DEĞİL

Yardımcı görevin kayıptaki payını %20'den %50'ye çıkarmak (küp x,y aksiyonda 4 kez
tekrar, `franka_lift_aux4`, 16 boyut). B ile aynı başlangıç, aynı adım sayısı.

| | medyan | x eğim | y eğim | doğrudan okuma |
|---|---|---|---|---|
| B (ağırlık %20) | 112 mm | 0.041 | 0.281 | 94 mm |
| D2 (ağırlık %50) | 110 mm | 0.056 | 0.300 | 103 mm |

Fark yok. **Belirleyici olan ağırlık değil, süre.**

### 2026-09-01 — TAM BÖLÜM ALGIYI SİLİYOR

`train_aux_long2@16000` (68 mm, doğrudan okuma korelasyonu 0.83) modelinden devam
edip **tam 250 karelik bölümlerle** 6000 adım eğitildi (`franka_lift_full_aux`).

Sonuç: 68 mm → **125 mm**, doğrudan okuma 62 mm → 160 mm, korelasyon **+0.83 → −0.18**.
Model küpü bulmayı öğrenmişken unuttu ve kısayola geri döndü.

**Bu, sızıntı teşhisinin en net kanıtı:** 250 karenin %94'ünde doğru cevap
"hareketine devam et" olduğu için, tam veri seti görsel öğrenmeyi aktif olarak siler.

Ayrıca: sadece ilk 60 kareyle eğitilen model kapalı döngüde **kavrayamaz** — uzman
tutucuyu ~89. adımda kapatıyor, model o kareyi hiç görmedi. O modelle yapılan
kapalı döngü testi geçersizdir.

### 2026-09-01 — KARMA VERİ SETİ: doğru reçete

`franka_lift_mixed`: her bölüm **150 kareye** kesildi (kavrama ~89, kaldırma
eşiği ~112-135'te tamamlanıyor — ölçüldü) + her bölümün **ilk 60 karesi 2 kez
fazladan** ayrı bölüm olarak yazıldı. Görü-kritik karelerin payı %6 → **%17**.

| | lokalizasyon | doğrudan okuma | x kor |
|---|---|---|---|
| tam bölüm | 125 mm | 160 mm | −0.18 |
| karma @3000 | 110 mm | 138 mm | 0.37 |
| karma @6000 | **90 mm** | 95 mm | 0.55 |

Algı korundu **ve** görev öğrenildi. 110 → 90 mm, **doymuş değil**.

Kapalı döngü hâlâ 0/10 ama ilk kez anlamlı davranış:
`bölüm 8 tepe_z = 0.100 m` (eşik tam 0.10), `bölüm 10 tepe_z = 0.296 m`
(gerçek kaldırma, sonra düşürme). Önceki en iyi tek bir 0.196 m idi.

**Kalan sorun — çevrimdışı/canlı uçurumu:** çevrimdışı 90 mm iken kapalı döngüde
20. adımda 509 mm. Bu uçurum eski modelde de vardı (veri seti 110 mm vs canlı
193 mm). Algı ne kadar iyileşirse iyileşsin bu kapanmazsa kapalı döngü çalışmaz.
**Sıradaki ölçüm bu olmalı.**

### 2026-09-02 — ÇEVRİMDIŞI/CANLI UÇURUMU ÖLÇÜLDÜ

`scripts/diagnostics/live_vs_dataset.py` (yeni). Aynı model (`train_mixed@6000`),
aynı ölçüm, iki görüntü kaynağı. Canlı gözlemler
`eval_policy_isaacsim.py --dump_obs` ile toplandı. **n=45, K=16.**

| kaynak | hareket planı | doğrudan okuma |
|---|---|---|
| veri seti görüntüsü | 97.5 mm | 93.5 mm |
| **canlı eval görüntüsü** | **193.3 mm** | **60.9 mm** |

**Model canlı görüntüde küpü İYİ görüyor** (61 mm, veri setindekinden bile iyi;
iki ayrı koşuda 65 / 61 mm, kararlı). Ama hareket planı 193 mm.

Eksen kırılımı — canlı görüntüde:

| eksen | okuma eğimi | okuma kor | plan eğimi | plan kor |
|---|---|---|---|---|
| y (yanal) | **0.978** | **0.917** | 0.959 | 0.871 |
| x (derinlik) | 0.331 | 0.431 | 0.194 | 0.177 |

Kamera `[1.5, 0.7, 0.85]` → `[0.30, 0, 0.20]` baktığı için **y yanal, x derinlik**.
Model yanal konumu neredeyse kusursuz okuyor, derinliği okuyamıyor. Tek kameradan
derinlik kestirme problemi. (CNN sondası x'i 5.5 mm ile çıkarıyordu — yani
imkânsız değil, SmolVLA için zor.)

İki ayrı eksik: **(1) derinlik zayıf, (2) okuma → hareket aktarımı kopuk.**
İkincisi daha önce de görülmüş ve uzun eğitimle kapanmıştı (4000→8000).

> **ÖLÇÜM UYARISI.** Bu ölçüm ilk kez n=16, K=8 ile yapıldı ve referans satırı
> iki koşuda 69 mm / x kor +0.83 ile 108 mm / x kor −0.16 verdi — yani sonucu
> tersine çevirecek kadar oynadı. SmolVLA akış eşleştirme tabanlı, her çağrı
> farklı plan örnekler. **n=16 kesinlikle yetersiz; n≥45 ve K≥16 kullan.**
> Bu projede gürültülü ölçüme dördüncü kanışımdı.

### 2026-09-02 — VERİ HATASI: 0. KARE BAYAT (305 bölümün 304'ü)

`collect_demos.py`'de bölüm sıfırlaması sonrası **ısınma adımı yoktu**. Aynı hatayı
`eval_policy_isaacsim.py`'de bulup düzeltmiştik (2026-08-28), toplama tarafına
bakılmamıştı.

Kanıt — bilek kamerasında ardışık kareler arası ortalama piksel farkı:

```
bölüm   0:   1.7   1.0   0.6 ...    ← ilk bölüm, öncesi yok, TEMİZ
bölüm  60:  60.1   3.7   1.7 ...    ← 0→1 farkı 60
bölüm 120:  55.3   3.2   1.5 ...
bölüm 180: 127.8   2.9   1.8 ...
bölüm 240:  67.4   3.2   1.5 ...
```

Kolun `ee_pos`'u 0-3. karelerde sabit (kol kıpırdamıyor), ama görüntü 0→1 arasında
uçuyor. Yani **0. karenin görüntüsü önceki bölümün son karesi** — tutucunun havada
küpü tuttuğu an. Görsel doğrulama: `results/kamera_karsilastirma.png` 1. satırda
küp dört bölümde de aynı yerde ve kocaman (tutucunun içinde), oysa küpün konumu
bölümler arası 199×499 mm değişiyor.

**Etkisi — ana metrik bozuktu.** `localize_test.py` ölçümü tam da 0. karede
yapıyordu. Yani projedeki bütün veri-seti lokalizasyon sayıları, modele "küp
nerede" diye sorarken ona **önceki bölümün son karesini** göstererek elde edildi.
Canlı ölçümlerin veri setinden iyi çıkması (61 mm vs 93 mm) de bununla açıklanıyor:
canlı taraf düzeltilmişti, veri seti tarafı değildi.

Eğitim üzerindeki etkisi daha sınırlı (tam sette 250 karede 1), ama karma sette
ilk pencere 3 kez tekrarlandığı için 0. kare fazladan ağırlık alıyordu.

**Düzeltmeler:**
- `collect_demos.py`: sıfırlama + randomizasyon sonrası `env.step(actions)` ısınma
- `localize_test.py`: `FRAME` ortam değişkeni, **varsayılan 2** (kol hâlâ evde,
  render tazelenmiş). Eski sayılarla karşılaştırırken bunu hesaba kat.

Mevcut HDF5 (`franka_lift_merged.hdf5`) hâlâ bozuk 0. kareler içeriyor; yeniden
toplamaya kadar ölçümlerde kare ≥1 kullan.

#### Temiz kareyle ölçüm — ALGI ASLINDA ÇÖZÜLMÜŞ

`train_mixed@6000`, aynı model, sadece kare 0 yerine kare 2 (n=20, K=4, CPU):

| | medyan | x eğim | x kor | y eğim | y kor |
|---|---|---|---|---|---|
| doğrudan okuma, **kare 2** | **24.4 mm** | **1.000** | **0.970** | **1.002** | **0.981** |
| doğrudan okuma, kare 0 (bozuk) | 93.5 mm | 0.404 | 0.563 | 0.464 | 0.645 |

Eğimler tam 1.0, korelasyonlar 0.97-0.98. **Model küpü neredeyse mükemmel
buluyor.** Kavrama eşiği 20 mm, CNN sondası 12 mm — model 24 mm'de.

**Projedeki bütün lokalizasyon sayıları (136 / 131 / 125 / 90 / 68 mm) ölçüm
artefaktıydı.** Modele bir önceki bölümün son karesi gösterilip "küp nerede"
diye soruluyordu.

Bunun bir sonucu daha: "derinlik (x) okunamıyor" teşhisi de artefaktmış. Bozuk
karede x eğimi 0.33 çıkıyordu, temiz karede **1.000**. Üçüncü (yan) kamera
gerekçesi böylece ortadan kalktı — kod yazıldı ve duruyor
(`--side_cam`, `collect_demos.py` / `eval_policy_isaacsim.py` / `convert_to_lerobot.py`),
ama ölçülen bir soruna dayanmıyor.

**Kalan gerçek sorun:** model küpü 24 mm ile biliyor ama kapalı döngüde hedefi
193 mm şaşırıyor. Yani algı değil, **algıdan eyleme aktarım** ve kapalı döngü
kararlılığı.

### 2026-09-02 — TEMİZ KARE TEYİDİ ve ALGI/EYLEM KOPUKLUĞU

Karma eğitim kümülatif 18000'e uzatıldı (`train_mixed2`). Temiz kare (FRAME=2),
n=45, K=8:

| kümülatif | plan medyan | plan x eğim | plan x kor | **okuma medyan** | okuma x eğim | okuma x kor |
|---|---|---|---|---|---|---|
| 14000 | 18.0 mm | 0.137 | 0.038 | **15.3 mm** | 0.908 | 0.981 |
| 18000 | 18.9 mm | 0.149 | 0.042 | **14.6 mm** | 0.919 | 0.980 |

**Algı kesinleşti: 14.6 mm**, x eğim 0.919 / kor 0.980, y eğim 0.991 / kor 0.992.
Kavrama eşiği 20 mm, CNN sondası 12 mm.

**Ama hareket planı bunu KULLANMIYOR.** Aynı modelin içinde küpün x'i 0.919
eğimle biliniyor, hareket planında x'e tepki 0.149 / kor 0.042. İki checkpoint'te
de aynı, n=45 — gürültü değil, yapısal.

Sebep: aksiyon **kola göre** tanımlı (`hedef − ee_pos`). Model küpü hiç
kullanmadan da geçerli bir delta üretebiliyor; aux kaybı onu görmeye zorluyor ama
o bilgi aksiyon başlığına akmıyor.

### 2026-09-02 — A/B: ÖRNEK ORTALAMASI ÇÖZMEDİ

Hipotez: plan medyanı 18.9 mm ama ortalaması 92 mm, maks 753 mm — yani arada bir
felaket örnek geliyor. SmolVLA akış eşleştirme tabanlı; çevrimdışı ölçümlerde
K=8 ortalama alınıyordu, sunucu ise **tek örnek** çekiyordu.
`policy_server.py --n_samples` eklendi. Aynı model, 15 bölüm:

| | başarı | kavrama anı hatası | en yüksek tepe_z |
|---|---|---|---|
| tek örnek | 0/15 | 3608 mm | 0.887 m (savrulma) |
| 8 örnek ortalama | 0/15 | 69 mm | 0.336 m |

Ortalama **uçarılığı kırptı** ama başarıyı getirmedi. 20. adımda kol hâlâ küpten
700-800 mm uzakta. **Sorun örnekleme gürültüsü değil.**

> Not: örnek ortalaması zaten kalıcı çözüm olarak görülmüyor — akış eşleştirme
> çok modlu dağılım modellemek için var, ortalama modlar arasında geçersiz
> aksiyon üretebilir. Çok nesneli göreve geçince kesin çöker. Burada sadece
> **teşhis aracı** olarak kullanıldı ve arızanın kaynağını eledi.

### 2026-09-02 — NESNE-MERKEZLİ AKSİYON *(çalışıyor)*

Ölçülen kopukluğu yapısal olarak kapatma denemesi.

```
önce:  aksiyon = hedef − KOLUN konumu   → küp bilgisi devrede olmak zorunda değil
şimdi: aksiyon = hedef − KÜPÜN konumu   (x,y; z mutlak kalır, küp z'si sabit)
       çıkarımda: mutlak hedef = aksiyon + modelin KENDİ küp tahmini (boyut 8,9)
```

Model küp tahminini kullanmadan mutlak hedef üretemiyor → algı zorunlu hale
geliyor. `franka_lift_objc` (915 bölüm / 82.350 kare). Doğrulama:
aksiyon ort[x,y] = [−0.014, +0.004] (küp çerçevesinde sıfıra yakın, doğru),
yardımcı ort = [0.499, −0.009] (küpün gerçek dağılımı).

Ek fayda: zor iş (küpü bulmak) yardımcı boyutlara, kolay iş (küp çerçevesinde
neredeyse sabit yörünge) aksiyon boyutlarına ayrılıyor.

Bayraklar: `convert_to_lerobot.py --object_centric`,
`policy_server.py --object_centric`, `localize_test.py` için `OBJ_CENTRIC=1`.

#### SONUÇ — kapalı döngüde hedefleme 10 KAT düzeldi

`train_objc@6000` (n=45, temiz kare): doğrudan okuma **13.7 mm**
(x eğim 0.930 kor 0.981, y eğim 0.992 kor 0.991).

Kapalı döngü, 15 bölüm — **modelin hedefi ↔ küp** mesafesi:

| adım | kola göre (önceki) | **nesne-merkezli** |
|---|---|---|
| 20 | 740 mm | **75 mm** |
| 40 | 682 mm | **109 mm** |
| 60 | 741 mm | **116 mm** |

Projedeki en büyük kapalı döngü iyileşmesi. Ham iz (bölüm 1, küp `(0.563,0.202)`):
model 10. adımdan itibaren `(0.532, 0.215)` hedefliyor — **34 mm** — ve kol
oraya doğru inerek gidiyor. İlk kez canlı ortamda doğru hedef + doğru hareket.

**Ama başarı hâlâ 0/15 ve 15 bölümün HEPSİNDE tutucu hiç kapanmıyor.**
Bazı bölümler de hâlâ felaket (birinde küp 7.4 m'ye uçtu; hedef ortalaması
1011 mm, medyanı 75 mm — yani dağılım çok çarpık).

**Darboğaz yer değiştirdi:** "nereye gideceğini bilmiyor" → "ne zaman
kavrayacağını bilmiyor" + kararlılık.

#### Kalan arıza: iniş küpün YANINA düşüyor, model vazgeçiyor

Uzun iz (bölüm 1, küp `(0.421, 0.136)`, 160 adım):

```
adım 11-45   hedef (0.49, 0.15, 0.18)   kol varıyor
adım 55-65   hedef z 0.07'ye iniyor      kol iniyor: 0.181 -> 0.096   OK
adım 75      hedef z 0.16'ya ÇIKIYOR     kol geri yükseliyor          ARIZA
adım 85-145  hedef z 0.11-0.16 salınıyor, kol 0.13-0.15'te asılı
adım 155     tutucu -0.80 ama kol z=0.142, küpten 63 mm uzakta
```

Hedef x'i **tutarlı ~60-70 mm fazla** (0.49 hedefliyor, küp 0.421). Kol küpün
yanına iniyor, bilek kamerasında küp göremiyor, kapatmıyor, geri çekilip tekrar
deniyor — sonsuz salınım.

**Tutucu mantığı sağlam** (ayrıca ölçüldü): veri seti karelerinde kare 95'te
model −0.82 (uzman −0.83), kare 110'da −0.99 (uzman −1.00); 8 örnek ortalaması
bunu bozmuyor (örnekler hemfikir, neg oranı %92-100). Yani model doğru anda
kapatmayı biliyor, sadece o ana hiç gelmiyor.

**Zincir:** canlı görüntüde x tahmini sapıyor → hedef 60-70 mm yanlış → iniş
küpün yanına → kavrama yok → salınım.

Bu, sabah ölçülen çevrimdışı/canlı uçurumuyla örtüşüyor (canlıda x eğimi 0.33,
y eğimi 0.98). **Kapatılamayan tek boşluk bu.**

> **Metrik uyarısı.** `localize_test`'in "hareket planı" bileşeni nesne-merkezli
> modda güvenilir değil: iniş adımını `argmin z` ile seçiyor ve **uzmanın kendi
> verisinde** bile 1.0 yerine **1.315** eğim veriyor. Model ile uzman farklı plan
> adımlarını seçince kıyas bozuluyor. Bu modda sadece **doğrudan okuma** sayısına
> ve kapalı döngüye bak.

### 2026-09-03 — CANLI/VERİ SETİ UÇURUMU: gerçek ve SADECE x'te

İki tarafı da **temiz kareyle** (dataset FRAME=2) karşılaştırma, `train_objc@6000`,
n=45, K=16. Dünkü karşılaştırma canlıyı bozuk 0. kareyle kıyaslıyordu, geçersizdi.

| kaynak | doğrudan okuma | x eğim | x kor | y eğim | y kor |
|---|---|---|---|---|---|
| veri seti (kare 2) | **15.1 mm** | 0.926 | 0.981 | 0.992 | 0.991 |
| canlı, num_envs=1 | 64.1 mm | 0.202 | 0.323 | 0.971 | 0.949 |
| canlı, **num_envs=8** | **52.8 mm** | **0.494** | 0.562 | 0.947 | 0.922 |

**y kusursuz transfer oluyor, x çöküyor.** Ön kamerada y yanal (piksel
konumundan doğrudan), x ise derinlik (ince ipuçlarından) — kırılgan olan o.

#### Sebep 1 bulundu: ORTAM SAYISI asimetrisi

`collect_demos.py` varsayılanı **num_envs=8**, `eval_policy_isaacsim.py` ise
**num_envs=1**'i koda gömmüştü. TiledCamera N ortamı tek dokuya render ediyor ve
`randomize_env_lights` her ortama ayrı lamba koyuyor → görüntü dağılımı N'e bağlı.

Eval'i 8'e çıkarmak, **modelde hiçbir şey değişmeden** x eğimini 0.202 → 0.494
yaptı. `eval_policy_isaacsim.py --num_envs` eklendi.

Bu, projede yakalanan **üçüncü** aynı-tip hata:
```
1. eğitimde randomize sahne  /  evalde sabit sahne
2. toplamada ısınma yok      /  evalde ısınma var
3. toplamada 8 ortam         /  evalde 1 ortam
```
Hepsi "boru hattının iki ucunda farklı dağılım". Kalan fark (0.494 → 0.926)
henüz açıklanamadı.

#### ELENDİ: render yakınsaması hipotezi

Canlı görüntülerde benzersiz renk 7031 vs veri setinde 3157 → "RTX zamansal
biriktirmesi oturmamış" sanıldı. Isınma 1 → 8 adıma çıkarıldı: **kötüleşti**
(okuma 52.8 → 61.5 mm, x eğim 0.494 → −0.035, y de 0.947 → 0.691).

Sebep muhtemelen düzeltmenin kendisi: ısınma komutu `hold_action()` yapılmıştı,
o da sıfırlamadan hemen sonra **bayat** `ee_frame` verisini okuyup kolu önceki
bölümün pozuna çağırıyor olabilir. Isınma `env.step(zeros)` hâline döndürüldü.

#### ELENDİ: toplamaya ısınma adımı eklemek — VERİYİ BOZDU

`collect_demos.py`'ye bölüm sonrası `env.step(actions)` eklendi. `actions`
**önceki bölümün son komutu** — yeni sıfırlanmış ortam için alakasız bir hedef.
Kol her bölüm başında savruldu: ee_pos ilk karelerde z=−0.192'ye (masa altı)
kadar gitti, kare0→1 farkı 76 → **84-111** oldu. 208 bölümlük yeni veri eskisinden
beter çıktı, **silindi**. Geri alındı.

**Doğru çözüm:** simülasyona dokunma, çevirmede at →
`convert_to_lerobot.py --skip_first 2`. Eski veri zaten yalnızca 0. karede bozuk
(0→1 farkı 55-128, 1→2 ≈3, sonrası ≈0.5).

#### Temiz kareyle eğitim (`train_objc2`) — çevrimdışı DEĞİŞMEDİ

`franka_lift_objc2` = objc reçetesi + `--skip_first 2`. 6000 adım.

| | doğrudan okuma | x eğim | y eğim |
|---|---|---|---|
| `train_objc@6000` (bozuk kareli) | 13.7 mm | 0.930 | 0.992 |
| `train_objc2@6000` (temiz) | 15.3 mm | 0.939 | 0.988 |

Fark yok — bozuk kareler eğitimi ölçülebilir biçimde bozmuyormuş. Kapalı döngü
**0/20** (num_envs=8), tutucu yalnızca 2 bölümde kapandı (hata 77 ve 141 mm).

#### YENİ BULGU: hedefleme bölüm ilerledikçe BOZULUYOR

`train_objc2` kapalı döngü, bölüm 1 tam izi, küp `(0.518, −0.226)`:

```
adım 11-24   hedef (0.490, -0.223)   küpten  28 mm   ← başlangıç çok iyi
adım 45      kol hedefe varmış
adım 50+     hedef z salınıyor: 0.177 → 0.097 → 0.081 → 0.168 → 0.086 → 0.070
adım 150     hedef (0.531, -0.148)   küpten  80 mm   ← uzaklaşmış
adım 153     tutucu kapandı, hata 77 mm
```

Model bölüm **başında** küpü doğru buluyor, kol yaklaştıkça kestirim bozuluyor ve
inişe commit edemiyor. Muhtemel mekanizma: kol küpün üstüne gelince ön kamerada
küpü kendi gövdesiyle **kapatıyor**, bilek kamerasında küp çok yakın ve farklı.

**Bu, bütün lokalizasyon ölçümlerimizin bölüm BAŞINDA (kare 2) yapılmış olmasının
bizi yanıltmış olabileceği anlamına geliyor.** Ölçüm sırada: aynı test kare
2/20/40/60/80/100'de.

#### Yan bulgular

- **Toplama hızlı:** 8 ortamla 100 bölüm ≈ 3 dakika (2.5 saat sanılıyordu).
- **200 bölüm tek seferde RAM'e sığmıyor.** `collect_demos.py` bütün bölümleri
  bellekte biriktirip sonda yazıyor: 200 × 249 × 2 kamera × 224²×3 ≈ **15 GB**,
  sistemde 23 GB. 189. bölümde OOM ile öldürüldü, dosya hiç yazılmadı.
  **100'erlik partiler + `merge_hdf5.py` kullan.**

### 2026-09-04 — KAMERA GEOMETRİSİ DEĞİŞMİŞ (uçurumun asıl sebebi)

Canlı/veri seti uçurumunun kaynağı görünüm değil **geometri**. Güncel kodla
toplanan örnek, eski veri setine değil **eval'e** benziyor:

| kaynak | masa alanı | ufuk satırı | dikey merkez |
|---|---|---|---|
| ESKİ veri seti (eğitim) | 22.9% | 78.8 | 186.1 |
| YENİ toplama (güncel kod) | 32.9% | 66.3 | 174.2 |
| CANLI eval | 31.9% | 63.1 | 175.7 |

Yani `franka_lift_merged.hdf5` **farklı bir kamera konumuyla** toplanmış; o
zamandan beri parametreler değişmiş ve hiçbir yerde kayıtlı değil.

Bu, "canlıda x çöküyor ama y sağlam" bulgusunu açıklıyor: y yanal konum, kamera
kaysa da kadrajdaki yeri kabaca korunur; x derinlik, doğrudan perspektif
geometrisinden çıkarılır ve kamera taşınınca eşleme bozulur.

**Bağımsız doğrulama:** yeni veriyle ince ayara başlayınca kayıp **0.505**'ten
başladı (önceki koşu 0.033'te bitmişti). Geometri aynı olsaydı bu sıçrama olmazdı.

#### ELENDİ: görünüm hipotezleri

- Benzersiz renk farkı (7031 vs 3157) → `--dump_step` ile iki taraf aynı adımdan
  alınınca **kapandı** (3353 vs 3157), ama lokalizasyon uçurumu **kapanmadı**
  (canlı 57 mm / veri seti 15 mm). Yani fark özet istatistiklerde değil.
- Ortam sayısı (toplama 8 / eval 1): gerçek ama kısmi etken, x eğimi
  0.202 → 0.494. `eval_policy_isaacsim.py --num_envs` eklendi.

#### ÖLÇÜM HATASI: `localize_test.py` HDF5 yolunu koda gömüyordu

Yeni geometriyle eğitilen model, **eski** geometrideki görüntülerle test edildi
(101 mm). Düzeltildi: `HDF5` ortam değişkeni (`localize_test.py`,
`live_vs_dataset.py`). **Model hangi veriyle eğitildiyse ölçüm de onunla.**

#### VERİ HATASI: `--only_success` eksikti — normalizasyonu çökertti

Orijinal veri **305/305 başarılı**, küp hiç 1 m dışına çıkmamış. Yeni toplamada
`--only_success` kullanmadım: 208 bölümün 26'sı başarısız, **6'sında uzman küpü
fırlattı** (en uzak **174.9 m**). O karelerdeki yardımcı etiket devasa:

| set | yardımcı ortalama | yardımcı std |
|---|---|---|
| objc2 (referans) | [0.499, −0.009] | [0.055, 0.140] |
| geo (filtresiz) | [0.820, −0.038] | **[5.094, 1.667]** |
| geo2 (`--only_success`) | [0.503, −0.015] | [0.056, 0.136] |

Model ölçeği 90 kat şişmiş bir eşleme öğrendi: **x eğimi 2.27** (olması gereken
~1.0), okuma 119 mm. Filtreden sonra düzeldi.
`convert_to_lerobot.py --only_success` eklendi.

#### Geometrisi eşleşen ilk model (`train_geo2`)

`train_objc2`'den (eski geometri) 6000 adım ince ayar, `franka_lift_geo2`:

| | doğrudan okuma | x eğim | y eğim | hareket planı |
|---|---|---|---|---|
| geo (filtresiz) | 119 mm | 2.266 | 0.752 | 920 mm |
| geo2 @3000 | 38.3 mm | 0.730 | 0.799 | 654 mm |
| geo2 @6000 | **35.1 mm** | 0.782 | 0.825 | 667 mm |
| objc2 (eski geometri, kendi verisinde) | 15.3 mm | 0.939 | 0.988 | 18 mm |

Yardımcı başlık toparlandı ve **doymadı** (38 → 35 mm). Ama **aksiyon başlığı
toparlanmadı** (667 mm, sistematik −530 mm x sapması). Muhtemel sebep: eski
geometriye uyarlanmış bir modeli 6000 adımda yeni geometriye çevirmek yetmiyor;
basit regresyon olan yardımcı başlık çabuk uyuyor, 50 adımlık plan üreten aksiyon
başlığı uymuyor.

Kapalı döngü 0/20, ama medyan hedef hatası 40. adımda **77 mm**, 60. adımda
**72 mm** — `objc2`'nin 120-133 mm'sinden iyi.

#### UZATMA TERS ETKİ YAPTI (`train_geo3`, kümülatif 12000)

`train_geo2@6000`'den 6000 adım daha, aynı veri (`franka_lift_geo2`).

| | doğrudan okuma | kapalı döngü hedef hatası (medyan) |
|---|---|---|
| | | adım 20 / 40 / 60 |
| geo2 @6000 | 35.1 mm | — / **77 mm** / **72 mm** |
| geo3 @12000 | **28.3 mm** | 106 / 162 / 158 mm |

**Çevrimdışı okuma iyileşti, kapalı döngü kötüleşti.** Başarı 0/20 (ikisinde de).
Hareket planı metriği de değişmedi (667 mm, sistematik −534 mm x sapması) —
6000 adım daha eğitmek aksiyon başlığını düzeltmedi.

Normalizasyon şüphesi **elendi**: checkpoint'in kaydettiği istatistikler veri
setiyle birebir aynı (ort `[-0.0128, 0.0036, 0.2054]`, std `[0.0827, 0.0591, 0.1123]`).

**Bu, "daha uzun eğit" seçeneğini bu noktada eliyor** ve iki metriğin ters yönde
hareket etmesi başlı başına açık bir soru olarak kalıyor.

### 2026-09-05 — ALGI/EYLEM AYRIMI: modelin kendi tahmini canlıda ölçüldü

İlk kez **kapalı döngü sırasında** modelin kendi küp tahmini loglandı
(`policy_server` yanıta `cube_pred` ekliyor, `eval_policy_isaacsim` adım adım
karşılaştırıyor). `train_geo3@12000`, 20 bölüm:

| adım | 0 | 20 | 40 | 60 | 80 | 120 | 160 |
|---|---|---|---|---|---|---|---|
| tahmin hatası (medyan) | **59.9** | 61.4 | 68.0 | 72.3 | 94.7 | 97.5 | **107.6 mm** |

Aynı model veri seti görüntülerinde **28.3 mm**. İki ayrı etki birden:

**1. Baştan kalan dağılım farkı.** Adım 0'da 60 mm — geometri düzeltmesine rağmen
çevrimdışı (28 mm) ile canlı arasında hâlâ 2× fark var. Kavrama eşiği 20 mm
olduğu için bu tek başına yeterli sebep.

**2. Bölüm boyunca bozulma: 60 → 108 mm.** Model kendi ürettiği hareketlerle
eğitimde görmediği durumlara giriyor, orada algısı bozuluyor, bozuldukça daha
yanlış hareket ediyor. **Kovaryat kayma**, ders kitabı örneği.

**Bu, "çevrimdışı okuma iyileşirken kapalı döngü kötüleşiyor" çelişkisini
açıklıyor:** çevrimdışı ölçüm hep UZMANIN gittiği yollardaki görüntülerle
yapılıyor; model kendi gittiği yollarda yaşıyor ve orada kör. Okumayı uzman
yolunda iyileştirmek, model yolundaki körlüğü düzeltmiyor.

Kapalı döngü yine 0/20 (hedef hatası adım 60'ta medyan 73 mm, ortalama 3178 mm —
bazı bölümler tamamen dağılıyor).

### 2026-09-07 — DART (gürültülü yörünge, temiz etiket): MEKANİZMA ÇALIŞTI, NET SONUÇ KÖTÜ

Kovaryat kaymaya doğrudan saldırı. `collect_demos.py --action_noise 0.020`
(uzman komutuna 2 cm gürültü uygulanır, **temiz** komut etiket olarak kaydedilir).
Gürültü kalibrasyonu: 0.010 → 35/40 başarılı, 0.020 → 36/40. 200 bölüm toplandı.
`train_geo3`'ten 6000 adım — aynı geometri, aynı reçete, tek değişken veri.

**Kapalı döngüde modelin kendi küp tahmini (medyan):**

| adım | 20 | 40 | 60 | 100 | 160 | bozulma |
|---|---|---|---|---|---|---|
| geo3 (temiz) | 61 | 68 | 72 | 94 | **108 mm** | +80% |
| dart2 (gürültülü) | 142 | 151 | 113 | 226 | **173 mm** | **+21%** |

**Bozulma eğrisi düzleşti (+80% → +21%)** — teşhis doğruydu, DART o mekanizmaya
etki etti. **Ama taban iki katına çıktı** (61 → 142 mm). Net sonuç daha kötü,
kapalı döngü 0/20.

Çevrimdışı okuma neredeyse değişmedi (28.3 → 32.8 mm) — beklenen, çünkü o ölçüm
hâlâ uzmanın yollarındaki görüntülerle yapılıyor, DART'ın hedefi orası değil.
y eğimi düzeldi (0.815 → 1.004).

Yorum: gürültülü veri modeli daha geniş ama daha bulanık bir dağılımda eğitiyor —
sapmalara dayanıklılık kazanıyor, keskinliğini kaybediyor. Denenmemiş varyantlar:
daha düşük gürültü (0.010), daha uzun eğitim, **temiz + gürültülü karışım**.

#### VERİ HATASI: `--only_success` fırlatılan küpü "başarılı" sayıyor

İlk DART denemesi çöktü (okuma 82.6 mm, hareket planı 1052 mm). Sebep:
başarı ölçütü `tepe_z > 0.10 ve son_z > 0.10`; küp **fırlatılıp havada kalırsa**
son_z 75 m olur ve ölçüt sağlanır. Gürültüyle fırlatma sıklaşınca bu vaka çoğaldı.

| | aksiyon std[:3] | yardımcı std | aksiyon z maks |
|---|---|---|---|
| geo2 (temiz) | [0.083, 0.059, 0.112] | [0.056, 0.136] | 0.50 m |
| dart (filtresiz) | [0.086, 0.063, **2.466**] | [**3.427**, 2.006] | **75.22 m** |
| dart2 (`--max_abs 1.5`) | [0.086, 0.063, 0.113] | [0.056, 0.144] | 0.50 m |

**550 bölümden SADECE 1 tanesi** bu hasarı verdi. `convert_to_lerobot.py --max_abs`
eklendi (aksiyon ve küp konumuna 1.5 m akıl sınırı, varsayılan açık).

Bu bozukluk projede **üçüncü kez** oldu ve üçünde de ancak eğitim bittikten sonra
fark edildi (~85 dk kayıp). Artık zincirlere **çevirme sonrası otomatik istatistik
doğrulaması** konuyor — `aksiyon std` ve `yardımcı std` eğitim başlamadan loglanır.

---

### 2026-09-08 — KARIŞIM DENEYİ + ÖLÇÜM HATASI: REST PENCERESİ

Temiz (`franka_lift_geo`) + gürültülü (`franka_lift_dart`) birleşik veri seti,
1095 bölüm / 98550 kare, `train_geo3@6000`'den 4000 adım. İstatistik kapısı geçti
(aksiyon std [0.084, 0.061, 0.112], yardımcı std [0.056, 0.140] — geo2/dart2 ile
neredeyse aynı, normalizasyon sağlam).

**İlk okuma 674 mm.** Ama dört modelin sonucu yan yana konunca metriğin kendisi
şüpheli hâle geldi:

| model | veri | medyan hata | x sapması |
|---|---|---|---|
| geo2 | temiz | 667 mm | −555 mm |
| geo3 | temiz (uzatma) | 667 mm | −534 mm |
| dart2 | gürültülü | 679 mm | −569 mm |
| karisim | karışım | 674 mm | −577 mm |

Modele göre değişmeyen bir sayı modeli ölçmüyor. Gerçek etiketlerle aynı okuma
**0.0 mm** verdi (metrik sağlam), ama modelin çıktısı bölüm bölüm açılınca:

```
bolum            argz       z   rel_x   hata_mm
episode_000000     47  +0.119  -0.000      6.1   ← dogru
episode_000004     12  +0.124  -0.001     10.2   ← dogru
episode_000008      7  +0.121  -0.556    590.4   ← plan basta sapiyor
episode_000012      0  +0.051  -0.719    787.7   ← z masa ALTINDA
```

45 bölümün 39'unda planın **ilk birkaç adımı** z'yi masa altına (0.05) ve x'i
robotun arkasına (−0.6) atıyor; `argmin(z)` gerçek kavrama noktası yerine o
sıçramayı yakalıyor. Model küpü aslında 6-10 mm hatayla buluyor.

#### Kök neden: uzman durum makinesinin REST fazı

`pick_lift_sm.py`:

```python
if state == PickSmState.REST:
    des_ee_pose[tid] = ee_pose[tid]      # "neredeysen orada kal"
PickSmWaitTime.REST = 0.2 s               # 50 Hz -> tam 12 kare
```

Isaac Lab'in stok davranışı, hata değil. Ölçüldü: **her veri setinde, her bölümde
tam 12 kare** `action == ee_pos` (geo 208/208, dart 208/208, merged 305/305).
Bölümlerin **%22'sinde** (46/208) kol bu pencerede masa altına savruluyor.
DÜZELTME (2026-09-08): bu `--randomize_arm` DEĞİL — geo toplaması o bayrağı
kullanmadı. Kol kare 0'da evde (z=0.385), kare 1'de −0.064'e düşüyor: sıfırlama
anındaki **bayat `ee_pose` okuması**. REST fazı "neredeysen kal" derken o bayat
pozu komut olarak kopyalıyor ve kolu oraya çekiyor. Bayat-kare ailesinin bir üyesi.

Taklit öğrenmesi için bu zehir: o karelerde etiket kolun **o anki rastgele
pozuna** bakıyor, küpe değil. Model durumu sıfırlanmış olduğu için o pozu
bilemez → **görüntüyle ilişkisiz, öğrenilemez gürültü**. `--skip_first 2` bunun
sadece 2 karesini atıyordu; `--repeat_early 2` kalan 10 kareyi **3 katına**
çıkarıyordu (eğitim karelerinin ~%11'i).

#### Düzeltilmiş okuma — algı sanılandan çok daha iyi

`localize_test.py`'ye `REST_SKIP` eklendi (varsayılan 12): `argmin(z)` ilk 12
adımı atlar.

| model | REST_SKIP=0 | **REST_SKIP=12** | yardımcı görev |
|---|---|---|---|
| **geo3** (temiz) | 660 mm | **22.4 mm** | 26.5 mm |
| karisim (karışım) | 657 mm | 38.0 mm | 37.9 mm |
| dart2 (gürültülü) | 672 mm | 49.4 mm | 49.6 mm |

Aksiyon okuması ile yardımcı görev artık birbirini tutuyor (22 vs 26) — model
tutarlı. **geo3 = 22.4 mm**, kavrama eşiği 20 mm'nin hemen üstünde.

**Karışım deneyinin cevabı: işe yaramadı.** Sıralama temiz (22) > karışım (38) >
gürültülü (49). Gürültü keskinliği bozuyor ve temiz veriyle seyreltmek onu
kurtarmıyor. Ama artık bu geçerli bir ölçüm.

**Açık hipotez:** kapalı döngü eval'i adım 0'da, yani tam REST penceresinde
başlıyor. Model orada "mevcut pozda kal" etiketini taklit etmeyi öğrenmiş ama
pozu bilmiyor → ortalama bir saçmalık üretiyor. Bu, haftalardır açıklanamayan
**canlı 60 mm / veri seti 28 mm** farkı için güçlü bir aday.

---

### 2026-09-08 (akşam) — REST'i eğitimden çıkarmak: MEKANİZMA DOĞRU, CANLI UÇURUM ÇÜRÜTÜLDÜ

`--skip_first 12` ile yeniden çevirme (546 bölüm / 49140 kare), `train_geo3@6000`'den
4000 adım (`train_rest`).

**Beklenmedik yan bulgu — x ekseninde 7 katlık normalizasyon şişmesi:**

| | x | y | z | x aralığı |
|---|---|---|---|---|
| geo2 (REST var) | 0.0827 | 0.0591 | 0.1123 | [−0.842, +0.184] |
| rest (REST yok) | **0.0117** | 0.0293 | 0.1053 | [−0.173, +0.184] |

Nesne-merkezli koordinatta uzman her zaman küpü hedefler, yani `rel_x ≈ 0` olmalı.
Şimdiye kadarki x varyansının **neredeyse tamamı REST karelerinden geliyormuş**.
Model x'i 7 kat fazla büyük bir std ile normalize ediyordu; gerçek sinyal normalize
aralığın 1/7'sine sıkışıyordu. Ortalama da −0.0128'den +0.001'e (merkeze) oturdu.

**Çevrimdışı sonuç (kare 2, REST_SKIP=12):**

| model | REST_SKIP=0 | REST_SKIP=12 | yardımcı |
|---|---|---|---|
| geo3 (REST var) | 660 mm | **~24 mm** (22.4 / 25.8) | 25-27 mm |
| rest (REST yok) | **149 mm** | ~35 mm (36.0 / 33.6) | 37-39 mm |

Sol sütun mekanizmayı **doğruluyor**: REST kareleri eğitimden çıkınca modelin planı
artık bölüm başında masa altına dalmıyor (660 → 149 mm). Ama doğruluk kötüleşti.

**Bu karşılaştırma adil değil:** normalizasyon köklü değişti (x std 7 kat) ama model
eski normalizasyonla eğitilmiş geo3'ten devam ettirildi; çıkış katmanı yeni ölçeğe
uyum sağlarken 4000 adımda kesildi (kayıp hâlâ düşüyordu, 0.018).

#### ÖLÇÜM TUZAĞI: FRAME=12'de medyan iyi, eğim ÇÖKÜK

| | medyan | ortalama | x eğim |
|---|---|---|---|
| geo3 @ kare 2 | 25.8 mm | 40.3 | **+0.798** |
| geo3 @ kare 12 | 19.3 mm | **82.4** | **+0.035** ← kör |
| rest @ kare 12 | 22.8 mm | 82.3 | +0.026 ← kör |

Kare 12'de medyan düşüyor ama x eğimi sıfıra çöküyor — model x'i okumayı bırakıp
sabit tahmin ediyor (küpün x aralığı dar olduğu için medyan iyi görünüyor). Bu,
projenin başındaki "sabit-orta tahmini" tuzağının aynısı. İki model kare 12'de
birebir aynı sayıları veriyor → orada model değil, kolun kamerayı kapatması
ölçülüyor. **Dürüst ölçüm noktası kare 2.**

#### KAPALI DÖNGÜ — hipotez ÇÜRÜTÜLDÜ

```
adım      0     20     40     60     80    100    120    140    160
geo3    59.9   61.4   68.0   72.3   94.7   93.7   97.5  104.1  107.6
rest    60.3  100.0   91.3   91.4  113.5  101.1  101.3  108.8  106.5
```

Adım 0'da **59.9 vs 60.3 mm**. REST karelerini eğitimden çıkarmak canlı hataya
hiçbir şey yapmadı. Her ikisi de 0/20.

**REST penceresi canlı uçurumun sebebi DEĞİL.** (Ölçüm hatası ve normalizasyon
şişmesi olarak gerçekti ve düzeltildi, ama canlı uçurumu açıklamıyor.)

#### YENİ VE GÜÇLÜ İPUCU: canlı hata modele bağlı değil

| model | çevrimdışı | canlı adım 0 |
|---|---|---|
| geo3 | ~24 mm | 59.9 mm |
| rest | ~35 mm | 60.3 mm |

Çevrimdışı doğrulukları **11 mm farklı** olan iki model canlıda **aynı** sayıyı
veriyor. Darboğaz modelin algısı olsaydı daha iyi çevrimdışı model canlıda da
daha iyi olurdu. Canlı hata ~60 mm'ye **çivilenmiş** → kaynak modelde değil,
eval boru hattında sabit bir bileşen.

Bunu ayırmak için `eval_policy_isaacsim.py`'ye **işaretli** hata + canlı eğim
raporu eklendi (`[ALGI] ... kayma x/y`, `[ALGI-EGIM]`). Önceki sürüm sadece norm
kaydediyordu, yani kaymayı saçılmadan ayırmak imkânsızdı.

---

### 2026-09-08 (akşam) — ÜÇ KAMERA (Orhan'ın önerisi) — ÇEVRİMDIŞI KÜÇÜK KAZANÇ, CANLI DAHA KÖTÜ

**Gerekçe — ölçümle eşleşiyor.** Canlı adım 0'da eksen ayrımı ilk kez yapıldı:

```
x: eğim +0.715  kor +0.722   kayma −39.0 mm
y: eğim +0.673  kor +0.849   kayma  −8.6 mm
```

Zayıf eksen **x** — ön kameradan bakınca *derinlik* olan eksen; kayma da neredeyse
tamamen orada. Yandan bakan kamera (eye 0.50,−1.45,0.70 → tgt 0.50,0,0.15, yani
optik ekseni y boyunca) x'i derinlik olmaktan çıkarıp **yanal piksel konumu**
yapar. Öneri, ölçülen arızanın üstüne oturuyor.

**Kontrollü A/B — tek değişken üçüncü kamera:**

| koşu | kamera | veri | başlangıç | adım | çevrimdışı |
|---|---|---|---|---|---|
| `train_rest` | 2 (ön+bilek) | REST'siz | geo3@6000 | 4000 | 35 mm |
| `train_3kam` | **3** (+yan) | REST'siz | geo3@6000 | 4000 | ? |

Veri: 216 bölüm → 564 bölüm / 50760 kare, aksiyon std [0.0147, 0.0368, 0.1113].
VRAM 5624/6144 MiB (batch 32 sığdı), 1.06 s/adım.

#### TUZAK: LeRobot üçüncü kamerayı SESSİZCE düşürüyordu

`lerobot/policies/factory.py:305`:

```python
if not cfg.input_features:
    cfg.input_features = {... veri setinden ...}
```

Checkpoint'in config'inde `input_features` **doluysa** veri setindeki kameralar
dikkate alınmıyor. `--policy.path=train_geo3` ile doğrudan devam etseydik yan
kamera sessizce düşecek, model iki görüntüyle eğitilecek ve "3 kamera işe
yaramadı" diye **yanlış bir sonuç** çıkaracaktık.

Çözüm: `~/franka_runs/base_geo3_3kam` — config'e `observation.images.side`
eklenmiş, ağırlıklar sembolik bağlı (32 KB). Güvenli çünkü SmolVLA görüntü
başına parametre tutmuyor (her görüntü aynı SigLIP'ten geçip token olarak
ekleniyor) ve `VISUAL: IDENTITY` (görüntü için normalizasyon istatistiği yok).

`localize_test.py`'ye de aynı çift kontrol eklendi (yan kamera hem HDF5'te hem
`pol.config.input_features` içinde varsa gönderilir). `eval_policy_isaacsim.py`
çağrılırken **`--side_cam` şart**.

**Değerlendirme kuralı:** sadece medyana bakma. Bugün iki kez medyanın iyi
görünüp eğimin çökük olduğu duruma düşüldü (FRAME=12 tuzağı). Yan kameranın
işe yaradığının kanıtı **x eğiminin 0.74'ten belirgin yükselmesi**.

---

### 2026-09-08 (akşam) — ÜÇ KAMERA SONUÇLARI

#### Çevrimdışı — ÖNEMLİ: ilk okuma EĞİTİM VERİSİNDEYDİ

3kam'ın kendi eğitim verisinde (s3_s401) x eğimi +0.903 çıktı ve "önerinin
işe yaradığı" rapor edildi. **Yanlıştı.** Üç modelin de görmediği taze veriyle
(tohum 404, 32 bölüm) ölçünce:

| model | medyan | ortalama | x eğim | x kor | y eğim |
|---|---|---|---|---|---|
| geo3 (2 kam) | 41.4 mm | 43.2 | 0.705 | **0.846** | 0.847 |
| rest (2 kam) | 41.4 mm | 48.6 | 0.674 | 0.799 | 0.794 |
| 3kam (3 kam) | **33.6 mm** | 48.9 | 0.747 | 0.794 | 0.809 |

- **Gerçek:** medyan 41.4 → 33.6 mm (−%19), iki ölçüm setinde de tutarlı.
- **Gerçek değil:** x eğimindeki 0.90 sıçraması (kendi verisinde 0.903, taze
  veride 0.747). Aradaki fark ezber.
- x **korelasyonu iyileşmedi** (0.794 vs geo3'ün 0.846); ortalama hata da
  düşmedi (48.9 vs 43.2) → büyük sapmalar duruyor.

**DERS:** bir modeli kendi eğitim verisinde ölçme. Bugün bu tuzağa düşüldü ve
ancak taze veri toplanınca (5 dk) yakalandı.

#### Kapalı döngü — 3 KAMERA DAHA KÖTÜ

```
                adım 0 medyan   x kayma   x eğim   x kor    y eğim   y kor
geo3  (2 kam)      59.9 mm      −39.0     +0.715   +0.722   +0.673   +0.849
3kam  (3 kam)      79.6 mm      −59.3     +0.327   +0.285   +0.934   +0.944
```

Her ikisi de 0/20. Canlıda 3 kamera modeli **belirgin biçimde daha kötü**:
medyan 59.9 → 79.6 mm, x kayması −39 → −59 mm, x korelasyonu 0.722 → 0.285.

**Ama y CANLIDA ÇOK İYİLEŞTİ:** eğim 0.673 → 0.934, korelasyon 0.849 → 0.944.

Yani yan kamera canlıda y'yi düzeltip x'i çökertiyor — beklenenin tam tersi
(yan kameranın amacı x'ti). Çevrimdışı x korelasyonu 0.794 iken canlıda 0.285:
3 kamera modelinin **canlı/çevrimdışı uçurumu 2 kameralıdan çok daha büyük**.

**En olası açıklama (SINANMADI):** yan kameranın canlı görüntüsü eğitim
görüntüsünden farklı. Ön kamerada aynı sorun 2026-09-04'te bulunmuştu (masa
alanı %22.9 vs %32.9). Yan kamera için aynı karşılaştırma HİÇ YAPILMADI.
Sıradaki adım bu: toplama HDF5'indeki yan görüntü ile canlı eval'deki yan
görüntünün istatistiklerini karşılaştır.

**İkinci olasılık:** model az eğitilmiş (kayıp 0.032'de hâlâ düşüyordu;
2 görüntüyle eğitilmiş bir başlangıçtan 3. kamerayı öğrenmesi için 4000 adım
kısa olabilir).

---

### 2026-09-12 — CANLI/ÇEVRİMDIŞI UÇURUMU ÇÖZÜLDÜ: eval'in ısınma adımı

**Sebep:** `eval_policy_isaacsim.py` içindeki `warmup()` fonksiyonu
`env.step(torch.zeros(...))` yapıyordu. IK-Abs aksiyon uzayında sıfır aksiyon
*"hedef EE pozu (0,0,0), quaternion (0,0,0,0)"* demek — geçersiz bir hedef.
IK çözücü saçma eklem hedefleri üretiyor ve kol tek adımda fırlıyor. Politika
sonra eğitimde **hiç görülmemiş** bir kol konfigürasyonundan başlıyor. Kol
kameranın gördüğü en büyük nesne olduğu için gözlemin tamamı dağıtım dışı.

**Düzeltme:** ısınma artık fizik adımı atmıyor — `sim.render()` +
`scene.update(0.0)`. Isınmanın amacı zaten nişanlama sonrası kamerayı
tazelemekti. Eski davranış `--legacy_warmup` ile korundu (A/B için).

| | adım 0 medyan | x eğim | x kor | y eğim | y kor | x kayma | kayma çıkınca |
|---|---|---|---|---|---|---|---|
| ESKİ ısınma | 59.9 mm | 0.715 | 0.722 | 0.673 | 0.849 | −39.0 mm | 54.6 mm |
| **YENİ ısınma** | **36.4 mm** | **0.932** | **0.840** | **0.919** | **0.976** | **−11.1 mm** | **31.2 mm** |

**Uçurum kapandı, hatta tersine döndü:** canlı 36.4 mm, aynı modelin çevrimdışı
taze veri sonucu 41.4 mm.

**Mekanizma doğrulandı** (`--dump_step 0`, hiçbir komut verilmeden):

```
EE pozu, adim/kare 0
  EGITIM      (+0.458, -0.001, +0.386)
  CANLI YENI  (+0.463, +0.000, +0.385)
  FARK        5.3 mm      eklem sapmasi: toplam 0.124 rad, en buyuk 0.022
```

#### DERS: yanlış sonuçtan yanlış ders çıkarıldı, dal 9 gün kapalı kaldı

2026-09-03'te ısınma `hold_action()` ile düzeltilmeye çalışıldı ve **daha kötü**
çıktı (61.5 vs 52.8 mm). Koda *"Sifir komutta kal"* notu düşüldü ve dal kapandı.
Yön doğruymuş: o denemede kolu tutmak için okunan `ee_frame` sıfırlamadan hemen
sonra **bayattı**, kol önceki bölümün pozuna çağrılıyordu. Yani hem eski hem yeni
davranış bozuktu; ölçüm "sıfır komut daha iyi" dedi ve yanlış sonuç kalıcı bir
nota dönüştü.

Render-only ısınma her iki tuzaktan da kaçıyor: hiçbir komut verilmiyor.

#### Düzeltilmiş ısınmayla TÜM modeller yeniden ölçüldü

Isınma düzeltmesinden önceki her kapalı döngü sayısı fırlatılmış kolla alındı ve
**geçersiz**. Yeniden ölçüm (hepsi render-only ısınma, n=20, adım 0):

| model | kamera | adım 0 medyan | x eğim | x kor | y eğim | y kor | x kayma | kayma çıkınca |
|---|---|---|---|---|---|---|---|---|
| `train_geo3` | 2 | **36.4 mm** | 0.932 | 0.840 | 0.919 | 0.976 | −11.1 mm | **31.2 mm** |
| `train_rest` | 2 (REST'siz) | 48.1 mm | **1.023** | **0.860** | 0.748 | 0.952 | **+1.6 mm** | 40.8 mm |
| `train_3kam` | 3 | 58.3 mm | 1.009 | 0.817 | 0.848 | 0.912 | −32.2 mm | 37.9 mm |

Üçü de **0/20**.

Okunacaklar:

- **Tek bir model hükmetmiyor.** geo3 medyanda, `rest` x kalibrasyonunda
  (kayma +1.6 mm ≈ sıfır) ve x eğim/korelasyonunda en iyi.
- **3 kamera hâlâ daha kötü ama sebebi körlük DEĞİL:** x eğimi 1.009, kor 0.817
  — geo3 kadar iyi. Fark neredeyse tamamen −32 mm'lik **sabit kayma**. Kaymayı
  çıkarınca 37.9 vs 31.2, uçurum kapanıyor.
- **REST dalı geçersiz gerekçeyle kapanmıştı.** 2026-09-08'de rest vs geo3
  60.3 vs 59.9 mm ölçülüp "REST'i çıkarmak canlıya etki etmiyor" denmişti;
  o iki ölçüm de bozuk ısınmayla alınmıştı. Düzeltilmiş ölçümde ikisi
  belirgin biçimde farklı (48.1 vs 36.4) ve `rest` x kaymasını sıfırlıyor.

**ELENEN hipotez — yardımcı hedefin geç karelerden kayması.** −32 mm'lik sabit
kayma için: küp kavranıp kaldırılınca kolla birlikte robota doğru gidiyor, geç
kareler yardımcı etiketi aşağı çekebilir diye düşünüldü. Ölçüldü: eğitim
penceresindeki (12-162) küp x ortalaması, başlangıç x'inden sadece **+0.5 mm**
farklı. Kaymayı açıklamıyor.

#### 3kam UZATMA — AŞIRI ÖĞRENME, dal kapandı

`train_3kam`'a 4000 adım daha (3 kamerada kümülatif 8000). Gerekçe: kayıp
0.032'de hâlâ düşüyordu, model üçüncü kamerayı kullanmayı tam öğrenmemiş olabilir.

| | eğitim kaybı | çevrimdışı (taze) | x eğim | x kor | canlı adım 0 |
|---|---|---|---|---|---|
| 3kam @4000 | 0.032 | **33.6 mm** | **0.747** | **0.794** | **58.3 mm** |
| 3kam @8000 | **0.022** | 37.4 mm | 0.684 | 0.783 | 60.4 mm |

**Kayıp düştü, görev metriği yükseldi** — aşırı öğrenmenin klasik imzası. Model
az eğitilmiş değilmiş. Hem çevrimdışı hem canlı kötüleşti, 0/20.

Bu, projenin temel dersinin bir örneği daha: **eğitim kaybı görev metriği
değil.** Aynı tuzağa bugün üç kez çarpıldı — açık döngü metrikleri, modeli kendi
verisinde ölçmek, ve şimdi eğitim kaybı.

#### Günün sonu tablosu (hepsi düzeltilmiş ısınmayla)

| model | kamera | çevrimdışı (taze) | canlı adım 0 | x kayma |
|---|---|---|---|---|
| **`train_geo3`** | 2 | 41.4 mm | **36.4 mm** | −11.1 mm |
| `train_rest` | 2 (REST'siz) | 41.4 mm | 48.1 mm | **+1.6 mm** |
| `train_3kam` @4000 | 3 | **33.6 mm** | 58.3 mm | −32.2 mm |
| `train_3kam` @8000 | 3 | 37.4 mm | 60.4 mm | −27.0 mm |

Hepsi 0/20. **En iyi canlı model `train_geo3`** — en basit olanı.

#### Bu düzeltmenin ÇÖZMEDİĞİ

Kapalı döngü hâlâ **0/20**. İki sorun duruyor:

1. Algı canlıda 36.4 mm, kavrama eşiği 20 mm — hâlâ ~1.8 kat uzak
2. Bölüm ilerledikçe bozulma (adım 160'ta medyan 150 mm) — ama bu sayı
   devrilen küplerle kirli, algı ölçüsü olarak kullanılamaz

Ayrıca canlı görüntüler hâlâ eğitim görüntülerinden ~2 kat daha değişken
(parlaklık std 30.7 vs 14.6). Algı artık çalıştığı için engel değil, ama
açıklanmadı.

---

### 2026-09-12 — İKİ DAL DAHA KAPANDI + BİR İDDİA ÇÜRÜTÜLDÜ

**Yan kamera geometri kontrolü — UYUŞMAZLIK YOK.**

```
            parlaklik   masa alani   ufuk satiri
YAN  fark      -1.4      -4.9 puan    -0.3 satir
ON   fark      +2.2      -8.2 puan    +0.3 satir   <- eslestigini bildigimiz kontrol
```

Yan kameranın canlı/eğitim farkı, eşleştiğini bildiğimiz ön kameranınkinden daha
küçük. (2026-09-04'teki gerçek uyuşmazlıkta ufuk 12 satır kaymıştı.)

**Görü kodlayıcıyı çözmek — DEĞMEZ.** `freeze_vision_encoder=True` VE
`train_expert_only=True` → 450M'nin sadece 100M'i eğitiliyor. Ama donuk
özellikler sıfırdan CNN'den **daha iyi**:

| | medyan | x kor | y kor |
|---|---|---|---|
| taban (hep ortalama) | 105.3 mm | — | — |
| DONUK SigLIP + kafa | **61.2 mm** | 0.358 | 0.872 |
| sıfırdan CNN (192K) | 82.0 mm | 0.295 | 0.618 |

`scripts/diagnostics/frozen_feat_probe.py`. Ayrıca `train_expert_only` sonra
çalışıp **VLM'in tamamını** (görü dahil) dondurduğu için tek başına
`freeze_vision_encoder=false` geçmek sessiz bir boş deney olurdu.

**"246K CNN 12 mm" İDDİASI ÇÜRÜDÜ.** `vision_probe.py` eğitim/doğrulama ayrımını
**kare bazlı** yapıyordu. Bir bölüm içinde küpün konumu sabit ve her bölümden 12
kare alınıyor; karelerin bir kısmı eğitime bir kısmı doğrulamaya düşünce model
"bu sahne böyle görünüyor → küp şurada" ezberliyor. **Bölüm bazlı** ayrımla aynı
mimari 82 mm veriyor, 12 mm değil.

Bu, "bilgi görüntüde VAR" iddiasının dayanağıydı. İddia yanlış olmayabilir ama
**kanıtlanmış değil** — bölüm bazlı ayrımda eğitimde sadece 123 farklı küp konumu
kalıyor ve her iki sonda da veri açlığı çekiyor. Doğru ifade: *elimizde
ulaşılabilir taban için geçerli bir gösterim yok.*

**Servis yolu suçsuz.** Canlı dökümdeki gerçek görüntüler modele çevrimdışı
verildi (`scripts/diagnostics/offline_on_live.py`):

| | canlı döngü | çevrimdışı taze veri | çevrimdışı CANLI görüntülerle |
|---|---|---|---|
| 3kam | 79.6 mm | 33.6 mm | 70.6 mm |
| geo3 | 59.9 mm | 41.4 mm | 56.9 mm |

Çevrimdışı-canlı-görüntülerle ≈ canlı döngü → sorun görüntülerde, soket/sıra/
zamanlama değil. (Sonradan anlaşıldı: görüntüleri bozan şey ısınmanın fırlattığı
koldu.)

---

### 2026-09-12 (akşam) — ÜÇ KALDIRAÇ DAHA ELENDİ

Isınma düzeltmesinden sonra tek engel kaldı: **algı 36 mm, gereken 20 mm.**
Hangi kaldıracın bunu kapatacağını bulmak için üç ucuz test.

#### 1. Kalibrasyon — KALDIRAÇ DEĞİL (3 mm)

Canlı eğimler 1'in altında (x 0.932, y 0.919) = "ortalamaya çekilme". MSE ile
eğitilen her regresör bunu yapar ve **sistematik** olduğu için çıkarımda geri
ölçeklenebilir. Ne kadar kazanç var? (`scripts/diagnostics/calib_headroom.py`)

| kademe | medyan |
|---|---|
| 1. HAM | 43.3 mm |
| 2. kayma düzeltilmiş | 51.2 mm |
| 3. kayma + eğim düzeltilmiş | 37.3 mm *(iyimser üst sınır)* |
| 4. aynısı ama **çapraz doğrulanmış** | **40.3 mm** |

En iyi ihtimalle 3 mm. Kalan hata **gerçek saçılma**, düzeltilebilir bir
bozulma değil. Model bilgiyi taşıyıp yanlış ölçeklemiyor; bilgi belirsiz.

#### 2. Çözünürlük — KALDIRAÇ DEĞİL (ters yönde)

Görüntüler 224px toplanıp SigLIP'e 512'ye büyütülerek veriliyor, yani gerçek
detay eklenmiyor. 512px'te toplamak işe yarar mı? Ucuz testi: **düşürüp** bak.

| çözünürlük | medyan |
|---|---|
| 224 px (ham) | 67.7 mm |
| 160 px | 65.1 mm |
| **112 px** | **59.6 mm** |
| 80 px | 74.0 mm |

Çözünürlüğü **düşürmek iyileştiriyor**. 224px bağlayıcı değil; 512px toplama
dalı açılmadan kapandı. *(Çekince: sonda SmolVLA'dan zayıf, 60-68 vs 36 mm.)*

#### 3. Daha çok veri — BELİRSİZ

Öğrenme eğrisi, **test seti sabit** tutulup eğitim seti büyütülerek
(`scripts/diagnostics/data_scaling_probe.py`):

| bölüm | medyan | x eğim | x kor |
|---|---|---|---|
| 20 | 105.1 mm | 0.054 | 0.104 |
| 40 | 90.3 mm | 0.144 | 0.287 |
| 60 | 81.2 mm | 0.184 | 0.369 |
| 80 | 77.6 mm | 0.203 | 0.383 |
| 100 | 74.9 mm | 0.266 | 0.452 |
| 114 | 75.5 mm | 0.291 | 0.497 |

**Medyan doyuyor ama eğim/korelasyon hiç düzleşmiyor** (60→114 bölümde eğim
%58 artmış). Betiğin otomatik hükmü ("eğri düz") sadece medyana bakıyor ve
yanıltıcı: medyan sondanın kendi kapasitesiyle sınırlı, bilgi çıkarımı değil.

Yine de kesin değil: sonda x eğiminde 0.29'da, SmolVLA canlıda 0.93'te —
farklı rejimler. Kesin cevap için SmolVLA'yı yarım veriyle eğitip karşılaştırmak
gerek (~2 x 100 dk).

---

### 2026-09-12 (gece) — DAgger: EĞRİYİ DÜZLEŞTİRDİ, TABANI YÜKSELTTİ

Teşhis: hedefleme hatası adım 20'de 43 mm, adım 60'ta 77 mm. Veri setinde bu
bozulma **yok** (kare taraması f40'ta 6.2 mm) → sorun geç kareler değil, kolun
dağıtım dışına çıkması. Ders kitabı cevabı DAgger.

**Altyapı** (yeni): `policy_server.py`'ye **toplu çıkarım** (8 ortam tek forward
geçişiyle bağımsız aksiyon → toplama 2 saat yerine 16 dk),
`collect_demos.py --dagger_port` (politika sürer, uzman etiketler).

Doğrulandı (küçük koşu): ortamlar bağımsız (8/8 farklı küp), etiket uzmanın
(en düşük-z adımda aksiyon_xy == küp_xy, **0.0 mm** — uzmanın imzası), politika
sürüyor (|uzman hedefi − kol| medyan 56-83 mm).

**Veri:** 208 DAgger bölümü + 208 orijinal = 1206 bölüm / 108.540 kare.

**Sonuç:**

| | çevrimdışı (taze) | canlı adım 0 | hedefleme a20 | a40 | a60 |
|---|---|---|---|---|---|
| geo3 | **41.4 mm** | **36.4 mm** | 43 mm | 73 | 77 (+%79) |
| dagger | 47.3 mm | 75.9 mm | 72 mm | 80 | 65 (**düz**) |

0/20, tutucu **20/20** bölümde hiç kapanmadı.

**DART ile BİREBİR aynı takas:** eğri düzleşti, taban yükseldi. İki farklı
kovaryat-kayma çaresi, aynı sonuç.

**Muhtemel sebep (SINANMADI):** DAgger bölümlerinde politika küpe çarpıp
fırlatıyor; o andan sonra "küpün konumu" etiketi başlangıçtaki küpü bulmayı
öğrenmek için çöp. Veri setinin yardımcı std'si bunu ele veriyor:
**0.0556 → 0.0764**. Bu kareler filtrelenmedi. Düzeltme: küp ilk kımıldadıktan
sonraki kareleri at (`frozen_feat_probe.py`'deki filtrenin aynısı).

#### ELENDİ: örnek ortalaması 0/20'nin sebebi değil

DAgger toplamasında (n_samples=1) politika **%9.6 başarılı** (20/208), aynı
model eval'de (n_samples=8) **%0**. Ortalama şüphelendi, A/B yapıldı:

| | n_samples=8 | n_samples=1 |
|---|---|---|
| adım 0 medyan | 36.4 mm | 41.3 mm |
| adım 20 | 42.5 mm | 39.3 mm |
| tutucu kapanmadı | 19/20 | 18/20 |
| SONUÇ | 0/20 | 0/20 |
| **tepe_z en yüksek** | **30.5 m** | **1.95 m** |

İkisi de 0/20 → ortalama sebep **değil**. Ama son satır önemli: ortalama
alınca küp 30 metreye fırlıyor, tek örnekle 2 metreye. Akış eşleştirmenin
çok modlu planlarını ortalamak davranışı şiddetlendiriyor.

*(Not: bu A/B 2026-09-02'de de yapılmış ve "ikisi de 0/15" çıkmıştı — ama o
ölçüm bozuk ısınmayla alınmıştı, yani geçersizdi. Şimdi geçerli.)*

#### AÇIK SORU — yarının en değerli ipucu

**DAgger toplamasında %9.6, eval'de %0.** Aynı simülatör, aynı checkpoint,
aynı aksiyon biçimi. n_samples elendi. Kalan farklar:

- eval **0. ortamın** aksiyonunu 8 ortama yayınlıyor; DAgger'da her ortam
  kendi aksiyonunu alıyor *(env 0 için fark etmemeli ama doğrulanmadı)*
- DR uygulama sıklığı: eval env-0 bölüm sonunda, toplama herhangi bir ortam
  bitince
- `policy.reset()` zamanlaması: `handle_batch` her yeniden planlamada,
  tek-ortam yolu sadece bölüm başında

Bu çelişki çözülürse kapalı döngü sonucu tamamen değişebilir.

---

### 2026-09-13 — KAMERA RANDOMIZASYONU: kendi koydugumuz engel

**Soru:** şirketler VLA'yı çalıştırabiliyor, biz neden alamıyoruz?

Kamera randomizasyon aralıklarına bakınca:

```python
r  = 1.655 m ± 0.25       # kamera mesafesi
th = base ± 0.35 rad      # ±20 derece
z  = 0.85 + (−0.20, +0.25)
hedef noktasi ± 0.06 m
```

**Kamera her bölümde farklı bir yerde.** Sabit kamerada "küp şu pikselde → küp
dünyada şurada" tek bir sabit fonksiyondur ve birkaç yüz örnekle öğrenilir.
Bizde her bölümde değişiyor: model önce görüntüden kamera pozunu çıkarmak,
sonra projeksiyonu tersine çevirmek zorunda. 208 bölümle bu çok zor.

#### Ölçüm — SADECE kamerayı sabitle (ışık/masa randomize kalır)

`collect_demos.py --fix_cam` ve `eval_policy_isaacsim.py --fix_cam` eklendi
(tek değişken ilkesi; `--no_dr` üçünü birden kapatıp sonucu yorumlanamaz kılardı).

Aynı donuk-özellik sondası, aynı bölüm-bazlı ayrım:

| veri | bölüm | SigLIP medyan | **SigLIP x kor** | CNN medyan | **CNN x kor** |
|---|---|---|---|---|---|
| randomize kamera | 164 | 61.2 mm | 0.358 | 82.0 mm | 0.295 |
| randomize kamera | 60 | 85.5 mm | 0.352 | 102.1 mm | 0.317 |
| **sabit kamera** | 85 | **51.6 mm** | **0.718** | **45.7 mm** | **0.807** |

- **x korelasyonu iki katına çıkıyor.** İki randomize koşu farklı bölüm
  sayılarıyla neredeyse aynı değeri verdi (0.358 / 0.352) → gürültü değil,
  verinin kararlı özelliği.
- Sabit kamera **daha AZ veriyle** (85 vs 164) daha iyi sonuç verdi.
- y'de de iyileşme var ama ılımlı (0.80-0.87 → 0.92-0.96). Mekanizmayla uyumlu:
  kamerayı oynatmak en çok **derinliği** bozar — ve derinlik zaten haftalardır
  bizim zayıf eksenimizdi.

#### Nereden geldi — kendi düzeltmemizin yan etkisi

2026-09-04'te "eğitim randomize, eval sabit" uyuşmazlığı bulunmuştu. Düzeltme
olarak **eval'e randomizasyon eklendi**. Ters yön — eğitimden çıkarmak — hiç
değerlendirilmedi.

Randomizasyon yanlış değil; Faz 3'te Nova 5'e geçerken sim-to-real için
gerekecek. Ama **Faz 1'i sakatlıyor.** Doğru sıra: önce sabit kamerayla görevi
çalıştır, sonra randomizasyonu ekle ve bedelini ölç.

Endüstriyle farkımızın özeti: ağır domain randomization milyonlarca örnekle
kullanılır. LeRobot topluluğunda 50-100 bölümle çalışan ince ayarlar var — ama
**sabit kamera, sabit sahne** ile. Biz az verinin üstüne çok değişkenlik koyduk.

---

### 2026-09-13 — İLK BAŞARI. `success` YAPISAL OLARAK İMKÂNSIZMIŞ

**Bütün proje boyunca gördüğümüz `0/20` bir ölçüm hatasıydı.**

`eval_policy_isaacsim.py` döngü sırası:

```
275  env.step(act)               <- Isaac Lab bolum bitince env 0'i BURADA sifirlar
278  obj = scene["object"].data
292  z = obj.root_pos_w[0,2]     <- sifirlamadan SONRA okunuyor
294  final_z = z                 <- YENI bolumun taze kupu: hep 0.055
296  if term[0]: success = peak_z > 0.10 AND final_z > 0.10
```

Bölüm biter bitmez `env.step()` ortamı sıfırlayıp küpü masaya geri koyuyor;
`final_z` o taze küpün yüksekliğini (0.055) okuyor. Eşik 0.10. **Koşul hiçbir
zaman sağlanamıyordu.** Kanıt: her eval çıktısında `son_z` birebir 0.055 —
tepe_z 1.459 m olan bölümlerde bile.

Dosyanın kendi açıklaması (satır 11) bu otomatik sıfırlamayı **biliyor** ve
kamera için elle hallediyor (`aim_front_cam` + ısınma); küp ölçümü için hesaba
katmamış. Toplama tarafı doğruydu — tampona `env.step()`'ten **önce** yazıyor,
ve orada aynı model %22 başarılı çıkıyordu. İki uç arasındaki bu çelişki
2026-09-12 gecesinde fark edilmiş ama sebebi bir gün sonra bulundu.

**Düzeltme:** küp okuması `env.step()` ÖNCESİNE alındı (satır 254). Artık
`final_z` bölümün son gözlemlenebilir küp yüksekliği.

#### DÜZELTİLMİŞ SONUÇLAR — projenin ilk kavramaları

| model | kamera | başarı | tutucu kapanmadı | ALGI a0 |
|---|---|---|---|---|
| **`train_fixcam`** | sabit | **2/20 (%10)** | 15/20 | 41.0 mm |
| `train_fixdag` | sabit + DAgger | 1/20 (%5) | 18/20 | 40.0 mm |
| `train_geo3` | randomize | 0/20 (%0) | 18/20 | 48.5 mm |

- **İlk çalışan kavramalar.** Örnek: `tepe_z=0.419m son_z=0.419m BASARILI`.
- `geo3`'ün 0/20'si **gerçekmiş** — sadece ölçüm hatası değil. Randomize kamera
  modeli gerçekten beceremiyor.
- Ölçüm hatası **sabit kameralı modellerin gerçek başarılarını** gizliyormuş.
- n=20 çok küçük: 2/20 ile 1/20 istatistiksel olarak ayrışmaz. Güvenilir sayı
  için daha çok bölüm gerek.

#### Düzeltmeden sonra: gerçek başarı oranı ~%4-5, %10 değil

n=20 çok küçüktü. 60'ar bölümle tekrar:

| model | n=20 | n=60 | birleşik |
|---|---|---|---|
| `train_fixcam` | 2/20 (%10) | 1/60 (%2) | **3/80 (%3.8)** |
| `train_fixdag` | 1/20 (%5) | 3/60 (%5) | **4/80 (%5.0)** |

2/20'lik %10 küçük örneklem şansıymış. İki model ayrışmıyor (3/80 vs 4/80).

#### ÇÖZÜLMEMİŞ: toplama %24, eval %4

Aynı model (`train_fixcam`), aynı simülatör:

```
                 tepe_z>0.10    son_z>0.10
toplama (208)       45%           24%      <- kaldiriyor ve TUTUYOR
eval (60)           30%            2%      <- kaldiriyor ama DUSURUYOR
```

Tutucu kapanma oranı ikisinde de benzer (~%25) — model her ikisinde de
kavramaya teşebbüs ediyor, ama toplamada tutuyor.

**Eşitlenip elenen farklar:** ortam config'i (aynı `parse_env_cfg`, aynı TASK,
aynı `env_spacing`, aynı kameralar), bölüm uzunluğu (ikisi de **250 kare**,
ölçüldü), başarı ölçütü (ikisi de `peak>0.10 AND final>0.10`, aynı sabit),
çıkarım yolu (`select_action` da 25'lik kuyruk tutuyor, `predict_action_chunk`
+ sunucu kuyruğuna denk), kamera rejimi (ikisi de `--fix_cam`).

**ELENDİ: aksiyon yayını.** Eval 0. ortamın aksiyonunu 8 ortama yayınlıyor ve
sadece 0. ortamı ölçüyordu; toplama her ortama kendi aksiyonunu veriyor. Eval
paralel ölçüme çevrildi (aşağıda) → 1/16 (%6). Hâlâ %24 değil.

#### PARALEL ÖLÇÜM (yeni varsayılan)

`eval_policy_isaacsim.py` artık 8 ortamın **hepsini** bağımsız ölçüyor: her
ortam kendi gözleminden kendi aksiyonunu alır (toplu çıkarım), her ortamın
bölümü ayrı kaydedilir. Eski davranış `--legacy_broadcast` ile korundu.

- Yapısal asimetri kalktı (toplama tarafı zaten böyle çalışıyordu).
- **Duvar saati ~8 kat düştü:** 16 bölüm 2 dakika (eskiden 20 bölüm 6 dakika).
  200 bölümlük güvenilir bir ölçüm artık ~25 dakika.
- Doğrulama: ilk dalgada 8 bölümün de `son_z=0.021` çıkması, toplama
  verisindeki ilk dalganın **birebir aynı imzası**. Eskiden eval'de bu değer
  her bölümde 0.055'ti (sıfırlanmış küp).

---

#### Bu ailede SEKİZİNCİ hata

Boru hattının iki ucunda farklı varsayım. Ve en pahalısı: haftalarca
"model çalışmıyor" diye hipotez eledik, oysa ölçüt hiçbir zaman
sağlanamıyordu. Önceki yedisi ölçümü *bozuyordu*; bu, başarıyı **imkânsız**
kılıyordu.

**Ders:** bir metrik hiç değişmiyorsa (her bölümde `son_z` = 0.055), metriğin
kendisinden şüphelen. Sabit bir sayı bir şeyi ölçmüyor demektir — bu projede
2026-09-08'de dört modelin de ~650 mm vermesiyle aynı imza, aynı hata ailesi,
beş gün arayla iki kez.

---

### 2026-09-13 (akşam) — SIRALAMA BAŞTAN AŞAĞI DEĞİŞTİ

Düzeltilmiş paralel eval ile **200'er bölüm**, her model kendi eğitim rejiminde:

| model | rejim | başarı | çevrimdışı algı |
|---|---|---|---|
| **`train_fixcam`** | sabit kamera | **76/200 (%38)** | 31.6 mm |
| `train_dart2` | randomize + DART | 62/200 (%31) | 49.4 mm |
| `train_3kam` | randomize, 3 kamera | 35/200 (%18) | 33.6 mm |
| `train_geo3` | randomize, 2 kamera | 24/200 (%12) | 41.4 mm |
| `train_rest` | randomize, REST'siz | 22/200 (%11) | 41.4 mm |
| `train_fixdag` | sabit + DAgger | 2/200 (%1) | **26.3 mm** |

#### YANLIŞ ELENEN İKİ DAL

- **DART ikinci en iyi model (%31).** "Eğriyi düzleştirdi ama tabanı iki katına
  çıkardı, net sonuç kötü, 0/20" diye elenmişti.
- **Üç kamera gerçekten işe yarıyor (%18 vs geo3'ün %12'si).** "İşe yaramadı"
  diye elenmişti.

Her iki eleme de bozuk eval ile yapılmıştı.

#### EN ÖNEMLİ BULGU: `localize_test.py` GÖREV BAŞARISINI ÖNGÖRMÜYOR

```
cevrimdisi lokalizasyon  <->  kapali dongu basarisi
   korelasyon: +0.310      (iyi bir vekil -1'e yakin olurdu)
```

En çarpıcı çift:

| | çevrimdışı | başarı |
|---|---|---|
| `train_fixdag` | **26.3 mm (EN İYİ)** | **%1 (EN KÖTÜ)** |
| `train_dart2` | **49.4 mm (EN KÖTÜ)** | **%31 (2.)** |

`localize_test.py` bu projede "ASIL METRİK" diye adlandırılmış ve haftalarca
bütün kararlar ona dayandırılmıştı. **Öngörü gücü yok.**

Bu, projenin ilk dersinin aynısı, bir kat yukarıda: açık döngü metrikleri
(uzmanla MAE) yanıltıyordu, yerine bir vekil kondu, o vekil de yanıltıyor.
Ölçüm noktası bölüm başındaki tek bir plan; görev ise 250 adımlık kapalı bir
döngü. Aradaki her şey — sapmadan toparlanma, tutucu zamanlaması, kolun
küpü devirmemesi — bu vekilde hiç görünmüyor. DART'ın kazandırdığı tam da bu
görünmeyen kısımdı.

#### BUNDAN SONRA: doğrudan başarı oranı ölç

Paralel eval sayesinde 200 bölüm **~8 dakika**. Vekil metriğe gerek yok.
`localize_test.py` tanı aracı olarak kalır (modelin ne gördüğünü okumak için),
ama **karar metriği kapalı döngü başarı oranıdır**.

#### Denenmemiş: kazananları birleştir

Üç bağımsız kazanç var ve hiçbiri birlikte denenmedi:
**sabit kamera (%38) + DART (%31) + üç kamera (%18)**.

---

## Test 2 — Eğim testi (eski metrik, artık ikincil)

`scripts/diagnostics/vision_test.py`. Kare 10'da 50 adımlık plan; plan adımı 5'in
y bileşeni ile küpün y'si arasındaki regresyon eğimi. n=153, bootstrap GA.

| model | eğim | %95 GA |
|---|---|---|
| OpenVLA-7B sıfır-atış | kor 0.179 | reddedildi |
| `train_rel2@12000` (durum AÇIK) | 0.430 | — |
| `train_zs@5000` | 0.512 | [0.439, 0.583] |
| `train_zs@10000` | **0.667** | [0.602, 0.730] |
| `train_zs@15000` | 0.633 | [0.560, 0.706] |
| `train_lora@6000` | 0.677 | [0.604, 0.753] |

**Okuma:** 10000→15000 iyileşme yok, **doymuş**. LoRA ile görü kodlayıcısını açmak
da fayda etmedi. Bu tablo "daha çok eğitim" ve "görü kodlayıcısı" seçeneklerini eler.

---

## Test 3 — Aksiyonun tüm kanalları

`train_zs@10000`, veri seti kareleri. Aksiyon 8 boyutlu: `pozisyon(3)+quat(4)+tutucu(1)`.

| kare | pozisyon MAE | quaternion MAE | tutucu işaret doğruluğu |
|---|---|---|---|
| 10 | 14.4 mm | 0.0049 | %100.0 |
| 30 | 5.3 mm | 0.0005 | %99.9 |
| 55 | 6.6 mm | 0.0004 | %97.8 |

(uzman quaternion std 0.43; uzman tutucu binary {−1,+1}, kapanma adımı medyan 89,
aralık 81-95, bölüm 250 adım)

**Okuma — ÖNEMLİ:** Bu sayılar mükemmel görünüyor ama **yanıltıcı**. Modelin
*uzmanın aksiyon dizisiyle* uyumunu ölçüyorlar. Kol zaten hareket hâlindeyken
"aynen devam et" demek bu metrikleri doyuruyor; küpün nerede olduğunu bilmek
gerekmiyor. **Politikayı uzmanla değil, görevin hedefiyle karşılaştır** (Test 1).

---

## Test 4 — Plan büyüklüğü karşılaştırması

`scripts/diagnostics/chunk_compare.py`, `train_zs@10000`, kare 10, n=61.

| plan adımı | model \|Δ\| | uzman \|Δ\| | oran | MAE | kor(y) |
|---|---|---|---|---|---|
| 5 | 0.1122 m | 0.1241 m | 0.90 | 30 mm | 0.971 |
| 10 | 0.0999 | 0.0934 | 1.07 | 21 mm | 0.978 |
| 20 | 0.0581 | 0.0533 | 1.09 | 10 mm | 0.982 |
| 35 | 0.0204 | 0.0172 | 1.18 | 4 mm | 0.986 |
| 49 | 0.0263 | 0.0314 | 0.84 | 10 mm | 0.972 |

**Okuma:** "model eksik hareket ediyor" hipotezini eler. Ama Test 3 ile aynı tuzağa
düşüyor — uzmanla uyumu ölçüyor, görevi değil.

---

## Test 5 — Kapalı döngü (Isaac Sim)

`scripts/eval_policy_isaacsim.py`, 10 bölüm. Tutucunun kapandığı andaki XY hatası.

| koşu | model | plan adımı | başarı | medyan hedef hatası | en iyi |
|---|---|---|---|---|---|
| DR yok + ısınma yok (ESKİ, geçersiz) | `zs@10000` | 25 | 0/10 | kaldırma yok | — |
| DR yok + ısınma yok (ESKİ, geçersiz) | 305 bölüm temiz | 25 | 0/10 | kaldırma yok | — |
| DR yok + ısınma yok (ESKİ, geçersiz) | `noisy@8000` | 25 | 0/10 | kaldırma yok | — |
| **DR düzeltmesi** | `zs@10000` | 25 | 0/10 | 400 mm | 56 mm |
| **DR + ısınma düzeltmesi** | `zs@10000` | 25 | 0/10 | **384 mm** | **28 mm** |
| DR + ısınma | `zs@10000` | **1** | 0/10 | 625 mm | 159 mm |
| DR + ısınma | `noisy@8000` | 25 | 0/10 | 380 mm | 37 mm |

Erken adım ölçümü (`zs@10000`, kol henüz sapmadan):

| adım | modelin hedefi ↔ küp | kolun yeri ↔ küp |
|---|---|---|
| 20 | 539 mm | 603 mm |
| 40 | 560 mm | 731 mm |
| 60 | 552 mm | 674 mm |

**Okuma:**
- `n_action_steps=1` **çok daha kötü**. SmolVLA akış eşleştirme; her çağrı farklı
  bir plan örnekliyor, ardışık komutlar tutmuyor. **Chunking şart, 25 kullan.**
- Hata 20. adımda zaten 539 mm → hata birikmesi/kovaryat kayma değil, **baştan var**.
- `ee_hata ≈ hedef_hata` her satırda → IK ve kontrol sağlam, yanlış olan modelin hedefi.

Ham iz örneği (küp `(0.440, −0.017)`):
```
adım  0-9 : hedef (0.44, -0.015, 0.39)   küpün TAM XY'si — ama bu kısım "bekle",
adım 10   : hedef (0.45, -0.215, 0.19)   görü gerektirmiyor
adım 11-24: hedef (0.43, -0.24,  0.18)   inmeye başlayınca y'de 220 mm sapma
```
x hep doğru, y'de hem eğim düşük (~0.73) hem sabit ~−0.22 m kaydırma.

---

## Test 6 — Sızıntı tablosu (projenin temel bulgusu)

Küpün y'sinin, kolun konumundan ne kadar tahmin edilebildiği (R²):

| kare | 0 | 10 | 15 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|
| R² | 0.006 | 0.089 | 0.634 | 0.859 | 0.930 | 0.995 |

**Okuma:** 250 karelik bir bölümde görü **sadece ilk ~15 karede** gerekli. Kalan
%94'ünde doğru aksiyon "hareketine devam et"ten çıkıyor. Eğitim kaybı bu %94
tarafından yönetiliyor → model görmeyi öğrenmeden kaybın neredeyse tamamını
düşürebiliyor. `franka_lift_early` veri seti bunu düzeltmek için üretildi.

---

## Test 7 — Görsel sonda (bilgi görüntüde var mı?)

`scripts/diagnostics/vision_probe.py`. Sıfırdan eğitilen 246K parametreli CNN,
9760 görüntü, 3 dakika.

| eksen | hata | korelasyon | ortalama-tahmin tabanına göre |
|---|---|---|---|
| x | 5.5 mm | 0.976 | 8.8× iyi |
| y | 11.8 mm | 0.977 | 10.4× iyi |

**Okuma:** Görsel bilgi 224 pikselde fazlasıyla var. Çözünürlük, kamera mesafesi,
veri seti büyüklüğü **darboğaz değil**. Aynı randomize kamera pozlarıyla eğitildiği
için "kamera pozu modele verilmiyor, görev belirsiz" itirazını da eler.

---

## Test 8 — Elenmiş teknik şüpheler

| şüphe | ölçüm | sonuç |
|---|---|---|
| MP4 sıkıştırması modeli bozuyor | ham 34.5 mm vs MP4 30.3 mm | fark küçük, **elendi** |
| sıfır-state eğitildi, gerçek state gönderiliyor | sıfır 32.2 mm vs gerçek 30.5 mm | model o kanalı zaten yok sayıyor, **zararsız** |
| göreli→mutlak dönüşümü yanlış | aynı kare ee_pos ile maks hata **0.0000 mm** (önceki kare ile 124 mm) | konvansiyon **doğru** |
| IK / kontrol sorunu | `ee_hata ≈ hedef_hata` | kol komutu takip ediyor, **elendi** |
| hata birikmesi (kovaryat kayma) | 20. adımda zaten 539 mm; DART gürültülü model de aynı | **elendi** |
| daha çok eğitim | eğim 10k→15k: 0.667→0.633 | doymuş, **elendi** |
| görü kodlayıcısını aç (LoRA) | 0.667 → 0.677 | **elendi** |

---

## Bulunan gerçek hatalar (düzeltildi)

1. **Eval'de domain randomization yoktu.** Eğitim verisinin her karesi randomize
   sahneden geliyordu (gök ışığı 800-5500, masa üstü lambası 1.5e4-2.2e5, kamera
   ±0.35 rad / ±0.25 m / −0.20+0.25 m); eval varsayılan sahneyi kullanıyordu.
   → `eval_policy_isaacsim.py`'ye `DomainRandomizer` eklendi (`--no_dr`, `--dr_seed`).

2. **Bölüm başı bayat gözlem.** Isınma adımı sadece döngü başındaydı; 2. bölümden
   itibaren her bölümün ilk gözlemi bir öncekinin son karesiydi ve `n_action_steps`
   kadar adım o bayat kareden üretiliyordu. → her bölüm sonrası ısınma adımı eklendi.

Bu iki hata, **daha önceki tüm kapalı döngü sonuçlarını geçersiz kılıyordu**
(DART gürültülü modelin 0/10'u dahil — yeniden ölçüldü, yine 0/10).

3. `masa mesh baglama: MESH BULUNAMADI` — masa rengi randomizasyonu hiç çalışmamış.
   Aynı kod toplamada da çalışmadığı için eğitim/eval tutarlı, **sorun değil**.

4. **ÖLÇÜM HATASI: REST penceresi** (2026-09-08). `localize_test.py` planın
   ilk 12 adımını da `argmin(z)` aramasına dahil ediyordu; o adımlar uzmanın
   REST fazını (des_ee_pose = ee_pose) taklit ettiği için z masa altına
   sapıyordu. Dört modelin de gerçek hatası ~650 mm yerine 22-49 mm çıktı.
   → `REST_SKIP` env değişkeni eklendi (varsayılan 12).

5. **VERİ HATASI: REST kareleri eğitimde** (2026-09-08). Aynı 12 kare
   `--skip_first 2` ile eğitim setinde kalıyor, `--repeat_early` ile 3 katına
   çıkıyordu. → `--skip_first 12` ile yeniden çevrildi (`train_rest`).
