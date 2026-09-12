# Komutlar

Son güncelleme: 2026-09-12. Her blok kopyala-yapıştır çalışır.

> **2026-09-12 uyarısı:** bu dosya bir kez tamamen eskimişti — içindeki veri
> setlerinin ve checkpoint'lerin hiçbiri artık yoktu (`franka_lift_merged`,
> `franka_lift_zs`, `train_zs`, `--relative_actions`). Yapıştırınca hata
> veriyordu. Bir şeyi sildiğinde ya da adını değiştirdiğinde **buraya da bak.**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data
```

Güncel varlıklar:

```
ham veri   data/franka_lift_geo.hdf5     2 kamera, 208 bölüm
           data/s3_val404.hdf5           3 kamera, 32 bölüm — DOĞRULAMA, silme
veri seti  lerobot/franka_lift_{3kam,rest,geo2,dart2}
model      runs/train_{geo3,3kam,rest,dart2}
en iyi     runs/train_geo3/checkpoints/006000/pretrained_model  (kapalı döngü)
           runs/train_3kam/checkpoints/004000/pretrained_model  (çevrimdışı)
```

---

## 1. Veri toplama (ortam: `isaaclab`)

```bash
conda activate isaaclab
python -u scripts/collect_demos.py \
  --num_envs 8 --num_episodes 100 --env_spacing 25.0 --seed 101 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out /data/franka_vla/data/franka_lift_s101.hdf5
```

Önemli bayraklar:
- `--env_spacing 25.0` — **8.0 varsayılanı yetersiz**, ortamlar birbirini görüyor
- `--side_cam` — üçüncü (yan) kamerayı da kaydet. Yan kamera y ekseni boyunca
  bakar, böylece ön kameranın *derinlik* ekseni olan x orada *yanal* olur
- `--action_noise 0.020` — DART: temiz uzman etiketini kaydeder, bozulmuş
  aksiyonu uygular *(dal kapandı, tabanı iki katına çıkardı)*
- `--no_dr` — domain randomization kapat (**normalde AÇIK bırak**)

Tuzaklar:
- **100'erlik partiler hâlinde topla.** 200 bölüm tek seferde ~15 GB RAM ister
  (sistemde 23 GB), OOM ile ölür. 3 kamerada 70'erlik yap.
- `--num_envs` toplamada kaçsa eval'de de o olmalı — asimetri x eğimini
  0.202'ye düşürmüştü.
- Birleştirmeye **gerek yok**: çevirici çoklu dosya okuyor (aşağıda). Birleştirme
  diskte gereksiz bir kopya yaratıyor ve bir kez diski doldurdu.

---

## 2. LeRobot'a çevirme (ortam: `lerobot`)

```bash
conda activate lerobot
python -u scripts/convert_to_lerobot.py \
  --hdf5 /data/franka_vla/data/franka_lift_s101.hdf5 /data/franka_vla/data/franka_lift_s102.hdf5 \
  --repo_id franka_lift_yeni \
  --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --only_success --max_abs 1.5 \
  --overwrite --root /home/orhan/franka_runs/lerobot
```

Bayrakların **neden** böyle olduğu:

| bayrak | sebep |
|---|---|
| `--object_centric` | x,y küpe göre yazılır. Mutlak aksiyon durumla ~0.9 korele; model görüntü yerine propriosepsiyonu ezberliyordu |
| `--zero_state` | durum alanı kalır ama sıfırlanır. `--no_state` **çalışmaz** (SmolVLA `KeyError: 'observation.state'`) |
| `--aux_cube` | aksiyon 8 → 10 boyut; model küpün x,y'sini doğrudan tahmin etmeyi de öğrenir. Bu aynı zamanda **modelin ne gördüğünü okumamızı** sağlar |
| `--skip_first 12` | uzman durum makinesi bölüm başında 0.2 s REST yapıp `des_ee_pose = ee_pose` yazıyor — öğrenilemez etiket. **2 yetmez, 12 olacak** |
| `--repeat_early 2 --early_len 60` | görü sadece ilk ~15 karede gerekli; 250 karelik bölümde kalan %94 kaybı domine ediyor |
| `--only_success --max_abs 1.5` | fırlatılan küp "başarılı" sayılabiliyor (son_z 75 m). 550 bölümden **1 tanesi** normalizasyonu çökertmişti |

Süre: 2 kamera ~22 dk / 208 bölüm, 3 kamera ~31 dk / 216 bölüm.

**Çevirmeden sonra mutlaka istatistik kapısı çalıştır** (zincirlerde var):

```bash
python -c "
import numpy as np, sys
from lerobot.datasets import LeRobotDataset
d=LeRobotDataset('franka_lift_yeni', root='/home/orhan/franka_runs/lerobot/franka_lift_yeni')
st=d.meta.stats['action']
print('bolum',d.num_episodes,'kare',d.num_frames)
print('kameralar:', [k for k in d.meta.features if k.startswith('observation.images')])
print('aksiyon std[:3]', np.round(st['std'][:3],4))
print('yardimci std   ', np.round(st['std'][8:10],4))
sys.exit(0 if st['std'][2]<0.3 and st['std'][8]<0.3 else 1)
"
```

Beklenen (REST'siz, nesne-merkezli): aksiyon std `[~0.012, ~0.030, ~0.105]`,
yardımcı std `[~0.056, ~0.140]`. x std'si 0.08 civarıysa REST kareleri
içeride kalmış demektir.

---

## 3. Eğitim (ortam: `lerobot`)

```bash
conda activate lerobot
lerobot-train \
  --dataset.repo_id=franka_lift_yeni \
  --dataset.root=/home/orhan/franka_runs/lerobot/franka_lift_yeni \
  --policy.path=/home/orhan/franka_runs/train_geo3/checkpoints/006000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/home/orhan/franka_runs/train_yeni \
  --batch_size=32 --steps=4000 --save_freq=4000 --log_freq=250 \
  --num_workers=2 --seed=1000
```

Tuzaklar:
- `--policy.push_to_hub=false` **şart**, yoksa `repo_id argument missing`
- `output_dir` **önceden var olmamalı** → `FileExistsError`. `mkdir` yapma.
- `--policy.path=lerobot/smolvla_base` **çalışmaz**: temel model
  `camera1/camera2/camera3` bekliyor, bizim setimiz `front/wrist` kullanıyor
- `--policy.type=smolvla` (sıfırdan) → 6 GB'a **sığmıyor**, OOM
- `--save_freq`'i `--steps`'e eşitle → tek checkpoint yazılır (her biri ~1.2 GB)
- Hız: 2 kamera ~1.17 adım/s, 3 kamera ~1.06 adım/s → 4000 adım ≈ 70 dk
- 3 kamerada VRAM 5624/6144 MiB — batch 32 **ancak** sığıyor

### KAMERA SAYISI DEĞİŞTİRİYORSAN — sessiz tuzak

LeRobot `make_policy` (`factory.py:305`) veri setindeki kameraları **sadece**
checkpoint config'inin `input_features`'ı BOŞSA dolduruyor. Mevcut bir
checkpoint'ten devam ederken yeni kamera **sessizce düşer.**

```bash
SRC=/home/orhan/franka_runs/train_geo3/checkpoints/006000/pretrained_model
DST=/home/orhan/franka_runs/base_geo3_3kam
rm -rf $DST; mkdir -p $DST
for f in "$SRC"/*; do b=$(basename "$f"); [ "$b" = config.json ] && continue; ln -s "$f" "$DST/$b"; done
python -c "
import json
c=json.load(open('$SRC/config.json'))
c['input_features']['observation.images.side'] = dict(c['input_features']['observation.images.front'])
json.dump(c, open('$DST/config.json','w'), indent=2)
print(list(json.load(open('$DST/config.json'))['input_features']))
"
```

Güvenli: SmolVLA görüntü başına parametre tutmaz (her görüntü aynı SigLIP'ten
geçip token olur) ve `VISUAL: IDENTITY` — yeni ağırlık ya da istatistik gerekmez.

Durdurma (PID'i literal yaz, `pkill -f` kendi kabuğunu öldürür):
```bash
pgrep -af lerobot-train      # PID'i oku
kill -INT <PID>
```

---

## 4. Lokalizasyon ölçümü — ASIL METRİK (ortam: `lerobot`)

Simülatör gerekmez, ~3 dakika.

```bash
conda activate lerobot
HDF5=/data/franka_vla/data/s3_val404.hdf5 \
K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py \
  /home/orhan/franka_runs/train_geo3/checkpoints/006000/pretrained_model 45
```

Ortam değişkenleri:

| değişken | ne |
|---|---|
| `HDF5` | hangi veriyle ölçülecek. **Modelin eğitim verisini KULLANMA** — ezber ölçersin (0.903 vs 0.747) |
| `REST_SKIP=12` | planın ilk 12 adımını atla; REST fazı taklidi `argmin(z)`'yi kaçırıyor |
| `FRAME=2` | dürüst ölçüm noktası. **FRAME=12 tuzak**: medyan iyileşir ama x eğimi 0.03'e çöker (kol kamerayı kapatıyor) |
| `K` | örnek sayısı; akış eşleştirme her çağrıda farklı plan üretir |
| `OBJ_CENTRIC=1` | nesne-merkezli modelde şart |

**Sadece medyana bakma.** Medyan iyi görünüp eğim çökmüş olabilir. Kanıt için
`x: egim` ve `kor` değerlerine bak.

Referanslar: sabit-orta 136 mm · kavrama eşiği **20 mm** · CNN 12 mm

Diğer tanılamalar:
```bash
# bilgi goruntude var mi (CNN sifirdan egitilir, ~3 dk)
N_EP=100 PER_EP=100 EPOCHS=30 python scripts/diagnostics/vision_probe.py
```

---

## 5. Kapalı döngü (iki ortam, iki terminal)

**Terminal 1 — politika sunucusu (`lerobot`):**
```bash
conda activate lerobot
python -u scripts/policy_server.py \
  --ckpt /home/orhan/franka_runs/train_geo3/checkpoints/006000/pretrained_model \
  --device cuda --object_centric --n_action_steps 25 --n_samples 8 --port 8765
```

- `--n_action_steps 25` — **1 kullanma**, çok daha kötü (625 mm vs 384 mm).
  Akış eşleştirme her çağrıda farklı plan örnekler; chunking şart.
- `--object_centric` — veri seti öyle çevrildiyse **şart**
- Açılışta `aksiyon bicimi` satırını **oku** ve beklediğinle eşleştiğini gör

**Terminal 2 — simülasyon (`isaaclab`):**
```bash
conda activate isaaclab
python -u scripts/eval_policy_isaacsim.py \
  --num_envs 8 --num_episodes 20 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif /tmp/eval.gif
```

- **Model 3 kameralıysa `--side_cam` EKLE.** Yoksa yan kamera sahnede oluşmaz,
  sunucuya gitmez, model eksik girdiyle çalışır ve sessizce bozulur.
- `--dr_seed 1234` — toplamadaki tohumdan farklı olmalı
- `--dump_obs dosya.npz` — her bölümün ilk gözlemini kaydet

Çıktıda okunacaklar:
```
[ALGI]      modelin kup tahmininin adim adim hatasi + ISARETLI kayma
[ALGI-EGIM] canli egim/korelasyon -- cevrimdisi egim testinin canli esi
[IZ] [HAM]  hedefleme ve ham iz
```

**Sadece adım 0 temiz algı ölçümüdür.** Sonraki adımlarda kol küpü devirirse
küpün gerçek konumu değişir, korelasyon doğal olarak ölür — bu algı bozulması
değil. Ortalama hatanın binlerce mm'ye fırlaması bunun işareti.

Süre: 20 bölüm ≈ 10 dakika.

---

## Genel kurallar

- Her zaman `python -u` — yoksa çıktı tamponlanır, iş donmuş görünür
- **Uzun zinciri başlattıktan sonra ilk 90 saniyede kontrol et.** Bir kez 3 saat,
  bir kez bütün akşam bu yüzden gitti.
- `pgrep -f` / `pkill -f` kendi kabuğunu eşleştirir: PID'leri bir komutta listele,
  **ayrı** komutta literal PID ile öldür
- Checkpoint yüklerken cihazı config üzerinden ayarla, yoksa checkpoint'in kendi
  cihazına yükler (OOM veya dtype hatası):
  ```python
  from lerobot.configs.policies import PreTrainedConfig
  cfg = PreTrainedConfig.from_pretrained(CKPT); cfg.device = "cuda"
  pol = SmolVLAPolicy.from_pretrained(CKPT, config=cfg)
  pol.to("cuda"); pol.config.device = "cuda"
  ```
- Pil modunda GPU saatleri yarıya düşer, eğitim 2.1x yavaşlar — uzun işten önce
  şarjı kontrol et
