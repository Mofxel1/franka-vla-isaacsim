# Yol haritası

Orhan'ın hedefi: VLA'yı **öğrenmek** ve bununla bir proje geliştirmek.
Sıralama: Franka (sim) → Dobot Nova 5 (gerçek).

(Robot köpek fikri bu haritadan çıkarıldı — lokomosyon/navigasyon ayrı bir
problem sınıfı, ayrı proje olarak ele alınacak.)

Bu belge "nereye gidiyoruz"u tutar. Günlük durum [DURUM.md](DURUM.md)'de,
ölçümler [SONUCLAR.md](SONUCLAR.md)'de.

---

## Faz 1 — Tek küp, simülasyon *(şu an burada)*

**Amaç:** boru hattını kurmak ve VLA'nın nasıl çalıştığını/bozulduğunu öğrenmek.

Durum: algı çözüldü (15.3 mm, eğim 0.91/0.98). Kalan iş, bilineni kararlı
biçimde harekete çevirmek. Kapalı döngü henüz 0/N.

**Bitti sayılma ölçütü:** kapalı döngüde tutarlı kaldırma (≥5/10).

Bu fazda öğrenilenler kalıcı sermaye:
- Politikayı **uzmanla değil görevin hedefiyle** karşılaştırmak
- Kısayol öğrenmeyi (causal confusion) teşhis etmek
- Gürültülü ölçüme güvenmemek (n≥45, K≥16)
- Karma veri seti: karar anlarının kayıptaki payını korumak

---

## Faz 2 — Çok nesne + dil

**Neden:** şu an 305 bölümün hepsinde aynı talimat var
(*"pick up the cube and lift it"*). Model dili tamamen yok sayabilir, hiçbir şey
kaybetmez. **VLA'nın V ve A'sı çalışıyor, L çalışmıyor.**

**Değişiklik:** masaya ikinci nesne (farklı renk) + talimat çeşitliliği
(*"pick up the red cube"* / *"the blue one"*).

Bu tek hamle üç şeyi birden zorlar:
1. Dil bağlantısı gerçek olur — talimat davranışı seçmek zorunda
2. Yardımcı görev genelleşir: "küpün konumu" değil,
   **"talimatta geçen nesnenin konumu"**
3. Aksiyon dağılımı çok modlu olur → örnek ortalaması gibi kestirmeler
   çöker, bizi doğru çözüme mecbur bırakır

**Uyarı:** Faz 1 bitmeden geçilmemeli. İki sorun aynı anda açıkken hangisinin
neye sebep olduğu ayrılamaz.

---

## Faz 3 — Dobot Nova 5, gerçek robot

**Taşınan:** boru hattı mimarisi (toplama → LeRobot → eğitim → kapalı döngü),
iki ortam arası köprü, ölçüm disiplini, veri tasarımı dersleri.

**Aksiyon uzayı seçimi buraya doğrudan taşınıyor.** Model mutlak EE pozu +
tutucu üretiyor (`ee_pose_abs(pos3+quat4)+gripper1`); IsaacLab'de bunu IK-Abs
kontrolcüsü çalıştırıyor. Nova 5'te aynı rolü basit bir IK çözücü / MoveIt
üstlenir. Yani politikanın çıktısı değişmeden gerçek robota bağlanır.

**Dikkat — planlayıcı değil servo katmanı.** MoveIt'in tam planlama hattı
(çarpışmasız yörünge üret, çalıştır) gecikmeli ve tek seferlik; VLA ise 50 Hz'de
kapalı döngü komut üretiyor. Doğru eşleşme MoveIt Servo ya da doğrudan kontrol
hızında IK. Tam planlayıcı kullanılırsa kapalı döngü bozulur.

**Veri toplama riski — ve senin fikrinin bunu çözmesi.** Bu fazın asıl sürprizi
model değil veri: simülasyonda küpün yerini BİLEN bir uzman durum makinesi veri
üretiyordu, gerçekte o ayrıcalık yok. Ama IK + basit bir algılama (AprilTag ya
da kalibre kamera + renk/nesne dedektörü) ile **aynı durum makinesi gerçek
robotta da uzman olarak çalışabilir**. Böylece 300 bölümü teleoperasyonla tek
tek toplamak gerekmez. Teleoperasyon yedek plan olarak kalsın.

> **Dedektör UZMAN içindir, VLA için değil.** Ayrım:
> ```
> uzman (veri üretici)  küpün yerini bilmek zorunda -> simde simülatör,
>                       gerçekte AprilTag / kalibre kamera + dedektör
> VLA   (öğrenen)       her zaman sadece iki kamera + talimat görür
> ```
> Dedektör iki yerde kullanılır: durum makinesini sürmek ve yardımcı görev
> etiketini (küp x,y) kaydetmek. Politikanın **girdisine hiç girmez** —
> model o konumu tahmin etmeyi öğreniyor.
>
> **TUZAK:** AprilTag'i politikanın kullandığı kameraların gördüğü yere koyma.
> Küpün üstünde işaret varsa model küpü değil İŞARETİ aramayı öğrenir ve
> işaretsiz ortamda çöker — bu projedeki sızıntı hikâyesinin aynısı.
> Çözüm: dedektör kamerası politikanın girdisi olmasın (ayrı kalibre tepe
> kamerası), ya da işaret küpün görünmeyen yüzünde olsun.

**Kalan taşınmayanlar:** kamera yerleşimi ve kalibrasyon baştan,
`pick_lift_sm.py` Franka'ya özel (yeniden yazılacak).

**Hazır olan:** Nova 5 URDF kaynakları `~/dobot_ws`'te
(effort=0, gripper yok, `package://` yolları düzeltilmeli).

---

## Kalıcı ilkeler

1. **Görevin hedefini ölç, uzmana benzerliği değil.** Açık döngü metrikleri
   (uzmanla MAE, korelasyon) hareket sürekliliğiyle doyar ve yanıltır.
2. **Az örnekli ölçüme karar verdirme.** Bu projede dört kez yanılttı.
3. **Kestirmeler kalıcı çözüm değil.** Örnek ortalaması, sabit kamera, tek
   talimat — hepsi basit görevde çalışır, karmaşıklaşınca çöker.
4. **Tek seferde tek değişken.** İki şey aynı anda değişirse sonuç yorumlanamaz.
5. **Boru hattının her ucunda aynı dağılım.** Eğitimde randomize, eval'de sabit
   sahne; toplamada ısınmasız, eval'de ısınmalı — bu tür asimetriler haftalar
   yedi.
