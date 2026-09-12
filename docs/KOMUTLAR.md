# Komutlar

Her blok kopyala-yapıştır çalışır. Ortamı aktive etmeyi unutma.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
cd ~/Projects/franka_vla_data/scripts
```

---

## 1. Veri toplama (ortam: `isaaclab`)

```bash
conda activate isaaclab
python -u collect_demos.py \
  --num_episodes 150 --env_spacing 25.0 --seed 42 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out ~/Projects/franka_vla_data/data/franka_lift_s42.hdf5
```

Önemli bayraklar:
- `--env_spacing 25.0` — **8.0 varsayılanı yetersiz**, ortamlar birbirini görüyor
- `--action_noise 0.01` — DART tarzı: temiz uzman etiketini kaydeder, bozulmuş
  aksiyonu uygular
- `--only_success` — sadece başarılı bölümleri yaz
- `--no_dr` — domain randomization kapat (**normalde AÇIK bırak**)

Birleştirme:
```bash
python -u merge_hdf5.py --out data/franka_lift_merged.hdf5 data/franka_lift_s4*.hdf5
```

---

## 2. LeRobot'a çevirme (ortam: `lerobot`)

```bash
conda activate lerobot
python -u convert_to_lerobot.py \
  --hdf5 ~/Projects/franka_vla_data/data/franka_lift_merged.hdf5 \
  --repo_id franka_lift_zs \
  --relative_actions --zero_state --overwrite
```

- `--relative_actions` — **şart**. Mutlak aksiyon durumla ~0.9 korele, model
  görüntü yerine propriosepsiyonu ezberliyor
- `--zero_state` — durum alanını bırakır ama sıfırlar. `--no_state` **çalışmaz**
  (SmolVLA `KeyError: 'observation.state'` atar)
- `--max_frames 60` — her bölümden ilk N kare (görü-kritik pencere)

~10-13 dakika sürer, 305 bölüm için ~135× sıkıştırma.

---

## 3. Eğitim (ortam: `lerobot`)

```bash
conda activate lerobot
cd ~/Projects/franka_vla_data
lerobot-train \
  --dataset.repo_id=franka_lift_early \
  --dataset.root=/home/orhan/Projects/franka_vla_data/lerobot/franka_lift_early \
  --policy.path=/data/franka_vla/train_zs/checkpoints/010000/pretrained_model \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/data/franka_vla/train_early \
  --batch_size=32 --steps=6000 --save_freq=2000 --log_freq=250 \
  --num_workers=2 --seed=1000
```

Tuzaklar:
- `--policy.push_to_hub=false` **şart**, yoksa `repo_id argument missing` hatası
- `output_dir` **önceden var olmamalı** → `FileExistsError`. `mkdir` yapma.
- `--policy.path=lerobot/smolvla_base` **artık çalışmıyor**: temel model
  `camera1/camera2/camera3` bekliyor, bizim setimiz `front/wrist` kullanıyor.
  Kendi checkpoint'imizden devam et.
- `--policy.type=smolvla` (sıfırdan) → 6 GB'a **sığmıyor**, OOM
- Hız: ~1.26 adım/s → 6000 adım ≈ 80 dakika
- Checkpoint ~1.5 GB. `/data`'da yer az, `save_freq`'i düşük tutma.

Durdurma (PID'i literal yaz, `pkill -f` kendi kabuğunu öldürür):
```bash
pgrep -af lerobot-train      # PID'i oku
kill -INT <PID>
```

---

## 4. Kapalı döngü değerlendirme (iki ortam, iki terminal)

**Terminal 1 — politika sunucusu (`lerobot`):**
```bash
conda activate lerobot
python -u policy_server.py \
  --ckpt /data/franka_vla/train_zs/checkpoints/010000/pretrained_model \
  --device cuda --relative_actions --n_action_steps 25 --port 8765
```

- `--n_action_steps 25` — **1 kullanma**, çok daha kötü (625 mm vs 384 mm).
  SmolVLA akış eşleştirme; her çağrı farklı plan örnekliyor, chunking şart.
- `--relative_actions` — veri seti göreli çevrildiyse **şart**

**Terminal 2 — simülasyon (`isaaclab`):**
```bash
conda activate isaaclab
python -u eval_policy_isaacsim.py \
  --num_episodes 10 --env_spacing 25.0 --headless \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out_gif /tmp/eval.gif
```

- `--dr_seed 1234` (varsayılan) — toplamadaki tohumdan farklı olmalı, ezber değil
  genelleme ölçülsün
- `--no_dr` — randomizasyonu kapat (**eğitim dağılımının dışına çıkarır**, sadece
  karşılaştırma için)
- `--dump_obs dosya.npz` — her bölümün ilk gözlemini kaydet (çevrimdışı analiz için)

Süre: 10 bölüm ≈ 15 dakika.

---

## 5. Tanılama testleri (ortam: `lerobot`)

**Lokalizasyon — asıl metrik.** Simülatör gerekmez, ~3 dakika:
```bash
conda activate lerobot
cd ~/Projects/franka_vla_data/scripts/diagnostics
python localize_test.py /data/franka_vla/train_early/checkpoints/002000/pretrained_model 40
```
Referanslar: sabit-orta 136 mm · kavrama eşiği 20 mm · CNN 12 mm

**Görsel sonda** — bilgi görüntüde var mı (CNN sıfırdan eğitilir, ~3 dakika):
```bash
N_EP=100 PER_EP=100 EPOCHS=30 python vision_probe.py
```

**Eğim testi** (eski metrik):
```bash
N_EP=153 K=8 python vision_test.py /data/franka_vla/train_zs/checkpoints/010000/pretrained_model franka_lift_zs
```

**Plan büyüklüğü** — model uzmanla aynı ölçekte mi komut veriyor:
```bash
python chunk_compare.py /data/franka_vla/train_zs/checkpoints/010000/pretrained_model franka_lift_zs
```

---

## Genel kurallar

- Her zaman `python -u` — yoksa çıktı tamponlanır, iş donmuş görünür
- Uzun işleri arka planda çalıştır ve log yolunu peşinen not al
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
