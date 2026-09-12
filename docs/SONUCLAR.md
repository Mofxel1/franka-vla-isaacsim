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
