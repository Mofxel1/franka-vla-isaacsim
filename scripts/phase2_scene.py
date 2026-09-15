"""FAZ 2 -- iki nesneli sahne ve talimat uretimi.

NEDEN (docs/YOL_HARITASI.md, Faz 2): Faz 1'de her bolumde ayni talimat vardi
("pick up the cube and lift it"). Model dili TAMAMEN yok sayabilir ve hicbir sey
kaybetmez -- VLA'nin V ve A'si calisiyordu, L calismiyordu.

TASARIM KARARLARI (hepsi olculmus bir sebebe dayaniyor):

1. Nesneler `CuboidCfg` ile uretilir, `dex_cube_instanceable.usd` ile DEGIL.
   Instance'lanmis varliga malzeme baglanamiyor. 2026-09-15'te masa renk
   randomizasyonu tam bu yuzden hem hic calismiyordu hem de kosarken collider'i
   bozup kolu kiriyordu (bolumlerin %23'u). Renk SPAWN aninda, simulasyon
   baslamadan veriliyor.

2. Iki kupun konum dagilimi AYNIDIR. Kirmizi hep solda olsaydi model "kirmizi"yi
   degil "sol"u ogrenirdi ve dil yine dekoratif kalirdi. Tek kisit aralarindaki
   minimum mesafe.

3. Hedef nesne bolum basina rastgele secilir ve talimat ona gore yazilir.
   Talimat `eg.attrs["task"]` ile BOLUM BASINA kaydedilir.
"""
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.managers import SceneEntityCfg

CUBE_SIZE = 0.042          # dex_cube 0.8 olcekte ~4.2 cm

# Renkler goruntude NET ayrismali. Render'da dogrulandi: kirmizi ~(228,111,112),
# mavi ~(100,150,240) -- ayirt edilebilir.
NESNELER = [
    {"ad": "red",  "prim": "Object",  "rgb": (0.85, 0.10, 0.10)},
    {"ad": "blue", "prim": "Object2", "rgb": (0.10, 0.20, 0.90)},
]

# Kupun dogabilecegi alan. Faz 1'de olculdu: kup x[0.400,0.598] y[-0.248,0.243]
POZ_X = (0.42, 0.60)
POZ_Y = (-0.18, 0.18)   # bolge cizgisi 0.30; her iki yone de yer kalsin
MIN_ARA = 0.12             # iki kup merkezi arasi en az 12 cm (kup 4.2 cm)


def kup_cfg(nesne):
    """Tek bir renkli kup varligi."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + nesne["prim"],
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0, max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=nesne["rgb"], roughness=0.5, metallic=0.0),
        ),
    )


def ornekle_ciftler(n, dev):
    """n adet (a, b) konum cifti; IKISI DE ayni dagilimdan, en az MIN_ARA uzak.

    Saf fonksiyon: Isaac Sim gerekmeden test edilebilir. Kritik ozellik --
    a ve b'nin MARJINAL dagilimlari ayni olmali. Aksi halde model rengi degil
    konumu ogrenir (ornegin "kirmizi hep solda").
    """
    def ornekle(k):
        x = torch.empty(k, device=dev).uniform_(*POZ_X)
        y = torch.empty(k, device=dev).uniform_(*POZ_Y)
        return torch.stack([x, y], dim=-1)

    a = ornekle(n)
    b = ornekle(n)
    # DIKKAT: cakisma halinde SADECE b'yi yeniden cekmek b'nin dagilimini
    # bozar (b, a'dan uzak yerlere kayar). Bunun yerine IKISINI BIRDEN
    # yeniden cekiyoruz -> ortak dagilim simetrik kalir.
    for _ in range(50):
        yakin = torch.linalg.norm(a - b, dim=-1) < MIN_ARA
        if not bool(yakin.any()):
            break
        k = int(yakin.sum())
        a[yakin] = ornekle(k)
        b[yakin] = ornekle(k)
    return a, b


def reset_iki_nesne(env, env_ids, asset_cfgs=None):
    """Iki kupu AYNI dagilimdan, en az MIN_ARA uzakliga yerlestirir.

    Isaac Lab'in hazir `reset_root_state_uniform` terimi tek varlik icindir ve
    iki nesne arasindaki mesafeyi gozetmez; ust uste dogarlarsa PhysX onlari
    firlatir. Ayrica her nesne icin AYRI terim kullanmak dagilimlari ayirma
    riskini dogurur -- burada ikisi de ayni araliktan cekiliyor.
    """
    n = len(env_ids)
    dev = env.device
    origins = env.scene.env_origins[env_ids]
    a, b = ornekle_ciftler(n, dev)

    z = torch.full((n, 1), 0.055, device=dev)
    quat = torch.zeros((n, 4), device=dev); quat[:, 0] = 1.0
    sifir_hiz = torch.zeros((n, 6), device=dev)

    for isim, xy in (("object", a), ("object2", b)):
        asset: RigidObject = env.scene[isim]
        poz = torch.cat([xy, z], dim=-1) + origins
        asset.write_root_pose_to_sim(torch.cat([poz, quat], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(sifir_hiz, env_ids=env_ids)


def talimat(nesne_adi):
    """Hedef nesnenin adindan talimat metni."""
    return f"pick up the {nesne_adi} cube and lift it"


# ======================================================================
# FAZ 2b -- COK FIILLI TALIMAT URETIMI VE FIILE OZEL BASARI OLCUTLERI
# ======================================================================

from multi_verb_expert import (LIFT, STACK, PLACE, PUSH, FIIL_ADI,
                               BOLGELER, YONLER, CUBE, MASA_Z, BOLGE_Y)

# Talimat sablonlari. Ayni fiil icin tek sablon kullaniliyor -- amac dil
# cesitliligi degil, dilin DAVRANIS SECMESI. Sablon cesitliligi (esanlamlilar)
# ayri bir eksen, sonra eklenebilir.
def talimat_uret(fiil, hedef_ad, diger_ad, yer_ad):
    if fiil == LIFT:
        return f"pick up the {hedef_ad} cube"
    if fiil == STACK:
        return f"put the {hedef_ad} cube on the {diger_ad} cube"
    if fiil == PLACE:
        return f"put the {hedef_ad} cube on the {yer_ad} side"
    if fiil == PUSH:
        return f"push the {hedef_ad} cube to the {yer_ad}"
    raise ValueError(fiil)


def basari_olc(fiil, hedef_iz, diger_iz, parmak_iz, yer_ad):
    """Fiile OZEL basari olcutu.

    hedef_iz/diger_iz: (T,3) bolum boyunca konumlar
    parmak_iz        : (T,)  parmak acikligi toplami
    Tek bir olcut ("z esigi asti mi") kullanmak YANLIS olurdu: PUSH'ta kup hic
    kalkmaz, STACK'te kup indirilir. Her fiil kendi kosuluyla olculur.
    """
    h0, hs = hedef_iz[0], hedef_iz[-1]
    tepe_z = float(hedef_iz[:, 2].max())
    acik_son = bool(parmak_iz[-1] > 0.06)

    if fiil == LIFT:
        return tepe_z > 0.10 and float(hs[2]) > 0.10

    if fiil == STACK:
        d_son = diger_iz[-1]
        xy = float(np.linalg.norm(hs[:2] - d_son[:2]))
        z_beklenen = float(d_son[2]) + CUBE
        return (xy < 0.035 and abs(float(hs[2]) - z_beklenen) < 0.015
                and acik_son)

    if fiil == PLACE:
        merkez = np.asarray(BOLGELER[yer_ad], dtype=np.float32)
        xy = float(np.linalg.norm(hs[:2] - merkez))
        masada = abs(float(hs[2]) - MASA_Z) < 0.015
        return xy < 0.10 and masada and acik_son

    if fiil == PUSH:
        yon = np.asarray(YONLER[yer_ad], dtype=np.float32)
        ilerleme = float(np.dot(hs[:2] - h0[:2], yon))
        # MASADA KALMALI. Ilk surumde bu kosul yoktu ve kupu masadan DUSURMEK
        # "basarili" sayiliyordu (son_z -1.029 = zemin). Olculdu: masanin y
        # yari genisligi ~0.474.
        masada = abs(float(hs[2]) - MASA_Z) < 0.015
        hic_kalkmadi = tepe_z < 0.06        # itildi, KALDIRILMADI
        # bolge cizgisini gectti mi (place ile AYNI hedef bolge)
        vardi = float(hs[1]) * yon[1] > 0.24
        return ilerleme > 0.05 and masada and hic_kalkmadi and vardi

    raise ValueError(fiil)
