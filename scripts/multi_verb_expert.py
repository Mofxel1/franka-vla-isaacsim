"""FAZ 2 -- COK FIILLI FAZSIZ UZMAN.

NEDEN COK FIIL: tek fiille ("pick up the X") talimat yalnizca 1 bit tasir --
hangi kup. Model dili tamamen yok sayip rastgele kup secse bile %50 alir.
Dilin gercekten is yapmasi icin talimatin NESNEYI degil DAVRANISI da secmesi
gerekir.

FIILLER (hareket yapisi kasten farkli secildi):
  LIFT  "pick up the X cube"              kavra -> kaldir            (BIRAKMA YOK)
  STACK "put the X cube on the Y cube"    kavra -> tasi -> BIRAK     (hedef NESNE)
  PLACE "put the X cube on the L side"    kavra -> tasi -> BIRAK     (hedef BOLGE)
  PUSH  "push the X cube to the L"        KAVRAMA YOK, temasla it

STACK ile PLACE bilerek esli: ikisi de kavra-tasi-birak, ama biri nesne biri
bolge hedefliyor. Model bu iki yuvayi ayirmak zorunda kalir.
PUSH kumenin en degerlisi: tutucunun hic kapanmadigi tek fiil.

FAZSIZ TASARIM (bkz. stateless_expert.py): faz icsel tutulmaz, HER ADIMDA
geometriden hesaplanir. Sebebi olculmustu -- fazli durum makinesi DAgger'da
APPROACH'ta takilip "tutucuyu kapat" etiketini HIC uretmiyordu (%41 -> %3.4).
Fazsiz uzman ayrica bedava toparlanma verir: kup elden kacarsa mesafe kosulu
bozulur ve uzman kendiliginden "geri don, yeniden yaklas" der.
"""
import torch

OPEN, CLOSE = 1.0, -1.0

LIFT, STACK, PLACE, PUSH = 0, 1, 2, 3
FIIL_ADI = {LIFT: "lift", STACK: "stack", PLACE: "place", PUSH: "push"}

# Bolgeler. PLACE ve PUSH AYNI hedefi paylasir: ikisi de kupu "sola" goturur,
# ama biri kavrayip tasiyip birakarak, digeri hic kavramadan iterek. Boylece
# model bitis durumundan fiili CIKARAMAZ -- hareketi secmek icin DILI
# kullanmak zorunda kalir. Bu, fiil temellendirmesinin en sert testi.
BOLGE_Y = 0.28        # masa kenari ~0.474 (olculdu); pay birakiliyor
BOLGELER = {"left": (0.50, BOLGE_Y), "right": (0.50, -BOLGE_Y)}
YONLER = {"left": (0.0, 1.0), "right": (0.0, -1.0)}

CUBE = 0.042           # kup kenari
MASA_Z = 0.021         # kup masada dururken merkez yuksekligi


class MultiVerbExpert:
    """Fazsiz, cok fiilli uzman. `compute` her adimda sifirdan karar verir."""

    def __init__(self, num_envs, device,
                 above=0.10,         # yaklasirken kupun kac cm ustu
                 close_tol=0.015,    # bu mesafede tutucu KAPANIR
                 align_tol=0.02,     # xy'de bu kadar hizaliysa INILIR
                 grasp_tol=0.03,     # kapali + bu mesafe = KAVRAMIS
                 finger_closed=0.06, # parmak toplami bunun altinda = kapali
                 yerlestir_tol=0.025,# birakma icin xy hizalama toleransi
                 itme_mesafe=0.15):  # kup bu kadar itilmeli
        self.n = num_envs
        self.dev = device
        self.above = above
        self.close_tol = close_tol
        self.align_tol = align_tol
        self.grasp_tol = grasp_tol
        self.finger_closed = finger_closed
        self.yerlestir_tol = yerlestir_tol
        self.itme_mesafe = itme_mesafe
        self._fingers = torch.full((num_envs,), 0.08, device=device)

    def set_fingers(self, finger_total):
        self._fingers = finger_total.to(self.dev).flatten()[:self.n]

    def reset_idx(self, env_ids=None):
        """Fazsiz uzmanda sifirlanacak icsel durum YOK -- sorunun kaynagi
        zaten icsel durumdu."""
        return

    # ------------------------------------------------------------------
    def compute(self, ee_pos, hedef_pos, diger_pos, fiil, yer_hedef, quat):
        """Hepsi (n,...) tensor.

        ee_pos     (n,3)  uc islevci konumu
        hedef_pos  (n,3)  talimatta gecen kupun konumu
        diger_pos  (n,3)  otekinin konumu (STACK icin hedef)
        fiil       (n,)   LIFT/STACK/PLACE/PUSH
        yer_hedef  (n,2)  PLACE icin bolge merkezi, PUSH icin birim yon
        quat       (n,4)  komut edilecek yonelim (sabit, asagi bakan)

        Donen: (n,8) = pos(3) + quat(4) + tutucu(1)
        """
        n = ee_pos.shape[0]
        closed = self._fingers < self.finger_closed
        d_cube = torch.linalg.norm(ee_pos - hedef_pos, dim=-1)
        d_xy = torch.linalg.norm(ee_pos[:, :2] - hedef_pos[:, :2], dim=-1)

        kavradi = closed & (d_cube < self.grasp_tol)
        kavrama_pozunda = d_cube < self.close_tol
        ustunde = (d_xy < self.align_tol) & (ee_pos[:, 2] > hedef_pos[:, 2])

        # --- varsayilan: kupun ustune yaklas, sonra in, sonra kapat ---
        ust = hedef_pos.clone(); ust[:, 2] = ust[:, 2] + self.above
        pos = ust.clone()
        pos = torch.where(ustunde.unsqueeze(-1), hedef_pos, pos)
        pos = torch.where(kavrama_pozunda.unsqueeze(-1), hedef_pos, pos)
        grip = torch.full((n, 1), OPEN, device=self.dev, dtype=ee_pos.dtype)
        grip[kavrama_pozunda | kavradi] = CLOSE

        # --- KAVRADIKTAN SONRA: fiile gore ayrisir ---
        # Birakma hedefi (STACK/PLACE icin). Once USTUNE git, sonra IN, sonra AC.
        birak_xy = torch.zeros((n, 2), device=self.dev, dtype=ee_pos.dtype)
        birak_z = torch.zeros(n, device=self.dev, dtype=ee_pos.dtype)

        m_stack = fiil == STACK
        m_place = fiil == PLACE
        m_lift = fiil == LIFT

        # STACK: otekinin tam ustu, bir kup boyu yukarida
        birak_xy = torch.where(m_stack.unsqueeze(-1), diger_pos[:, :2], birak_xy)
        birak_z = torch.where(m_stack, diger_pos[:, 2] + CUBE, birak_z)
        # PLACE: bolge merkezi, masa yuksekliginde
        birak_xy = torch.where(m_place.unsqueeze(-1), yer_hedef, birak_xy)
        birak_z = torch.where(m_place, torch.full_like(birak_z, MASA_Z), birak_z)

        birak_hedef = torch.cat([birak_xy, birak_z.unsqueeze(-1)], dim=-1)
        birak_ust = birak_hedef.clone(); birak_ust[:, 2] = birak_hedef[:, 2] + self.above

        # birakma noktasinin USTUNDE miyiz?
        d_birak_xy = torch.linalg.norm(ee_pos[:, :2] - birak_xy, dim=-1)
        ust_hizali = d_birak_xy < self.yerlestir_tol
        # yeterince alcaldik mi? (kup birakilacak yuksekligin hemen uzerinde)
        alcaldi = ee_pos[:, 2] <= birak_hedef[:, 2] + 0.02

        tasiyor = kavradi & (m_stack | m_place)
        pos = torch.where((tasiyor & ~ust_hizali).unsqueeze(-1), birak_ust, pos)
        pos = torch.where((tasiyor & ust_hizali).unsqueeze(-1), birak_hedef, pos)
        # hizalandi VE alcaldi -> BIRAK
        birakma_ani = tasiyor & ust_hizali & alcaldi
        grip[birakma_ani] = OPEN

        # birakildiktan sonra: kup yerinde ve tutucu acik -> yukari cekil
        yerlesti = (~closed) & (torch.linalg.norm(hedef_pos[:, :2] - birak_xy, dim=-1)
                                < self.yerlestir_tol + 0.02) & (m_stack | m_place)
        cekil = birak_ust.clone()
        pos = torch.where(yerlesti.unsqueeze(-1), cekil, pos)
        grip[yerlesti] = OPEN

        # LIFT: kavradiysa yukari kaldir (birakma yok)
        kaldir = hedef_pos.clone(); kaldir[:, 2] = MASA_Z + 0.25
        pos = torch.where((kavradi & m_lift).unsqueeze(-1), kaldir, pos)

        # --- PUSH: kavrama yok, tutucu KAPALI (parmaklar itici gibi) ---
        m_push = fiil == PUSH
        if bool(m_push.any()):
            yon = yer_hedef                                    # (n,2) birim yon
            geri = hedef_pos[:, :2] - yon * (CUBE * 0.5 + 0.055)
            # HAVUC-SOPA HATASI (2026-09-15'te olculdu): itme hedefi ONCE
            # `kup_konumu + yon * mesafe` idi, yani HER ADIMDA kupun O ANKI
            # yerinden hesaplaniyordu. Kup ittikce hedef onunde kaciyor ve kol
            # kupu masadan dusurene kadar koveliyordu -- hedef 150 mm iken
            # olculen ilerleme 314..1078 mm. Fazsiz tasarimin dogal tuzagi:
            # hedef HAREKETLI bir seye gore tanimlanirsa yakinsamaz.
            # Duzeltme: hedef SABIT bir y (bolge cizgisi).
            hedef_y = yon[:, 1] * BOLGE_Y
            itis_sonu = torch.stack(
                [hedef_pos[:, 0], hedef_y + yon[:, 1] * 0.03], dim=-1)
            z_itme = torch.full_like(hedef_pos[:, 2], MASA_Z)

            baslangic = torch.cat([geri, z_itme.unsqueeze(-1)], dim=-1)
            bas_ust = baslangic.clone(); bas_ust[:, 2] = MASA_Z + self.above
            son = torch.cat([itis_sonu, z_itme.unsqueeze(-1)], dim=-1)

            # kupun ARKASINDA miyiz? (itme yonunde kupun gerisinde)
            rel = ee_pos[:, :2] - hedef_pos[:, :2]
            arkada = (rel * yon).sum(-1) < -(CUBE * 0.5)
            d_bas_xy = torch.linalg.norm(ee_pos[:, :2] - geri, dim=-1)
            bas_hizali = d_bas_xy < 0.03
            # ALCAK ESIGI KRITIK. Ilk surumde MASA_Z+0.05 (=0.071) idi ve
            # OLCULDU: kol daha 7 cm yuksekteyken "alcaldim" sayilip itme
            # komutuna geciyor, yatay hareket kupun UZERINDEN gecip gripper
            # kupun tepesine oturuyordu (kup hareketi 0.0 mm, ee-kup xy 5 mm,
            # ee z min 0.048 -- kupun tepesi 0.042). Esik kupun ust yuzeyinin
            # ALTINDA olmali, yoksa itme degil tirmanma olur.
            alcak = ee_pos[:, 2] < MASA_Z + 0.012          # 0.033 < kup tepesi 0.042

            # 1) hizali degilsek: baslangic noktasinin USTUNE git (kupu
            #    yolda devirmemek icin once yukaridan)
            p_push = bas_ust.clone()
            # 2) baslangicin ustundeysek: IN (kupun ARKASINDA, bos alanda)
            p_push = torch.where((bas_hizali & ~alcak).unsqueeze(-1), baslangic, p_push)
            # 3) hem arkada hem GERCEKTEN alcaktaysak: IT
            p_push = torch.where((arkada & alcak).unsqueeze(-1), son, p_push)
            # 4) BOLGEYE VARDI -> yukari cekil. Bu olmadan uzman itmeye devam
            #    eder ve kupu masadan dusurur.
            vardi = (hedef_pos[:, 1] - hedef_y) * yon[:, 1] >= 0.0
            # CEKILME DIK YUKARI OLMALI. Ilk surumde hedef KUPUN konumuydu;
            # kol kupun ARKASINDA oldugu icin "kupun ustune cik" komutu
            # yukselirken ILERI gitmek demekti ve kupu itmeye devam ediyordu
            # (kup masadan dusuyordu). Simdi kolun kendi xy'sinde yukselir.
            cekil_p = torch.cat(
                [ee_pos[:, :2], torch.full_like(ee_pos[:, 2:3], MASA_Z + self.above)],
                dim=-1)
            p_push = torch.where(vardi.unsqueeze(-1), cekil_p, p_push)

            pos = torch.where(m_push.unsqueeze(-1), p_push, pos)
            grip[m_push] = CLOSE      # parmaklar kapali -> tek bir itici yuzey

        return torch.cat([pos, quat, grip], dim=-1)
