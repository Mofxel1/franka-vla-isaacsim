"""
Franka pick-and-lift gosterim toplayici.

Isaac Lab'in scripted state-machine uzmanini kosturur, her adimda RGB goruntuleri
robot durumuyla SENKRON kaydeder ve VLA egitimine uygun HDF5 uretir.

Kullanim:
  python collect_demos.py --headless --num_envs 8 --num_episodes 32 \
      --kit_args="--/rtx/verifyDriverVersion/enabled=false"
"""

import argparse, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka lift gosterim toplayici")
parser.add_argument("--num_envs", type=int, default=8, help="paralel ortam sayisi")
parser.add_argument("--num_episodes", type=int, default=32, help="toplanacak bolum sayisi")
parser.add_argument("--res", type=int, default=224, help="kamera cozunurlugu")
parser.add_argument("--out", type=str, default=None, help="cikti HDF5 yolu")
parser.add_argument("--task_prompt", type=str, default="pick up the cube and lift it")
parser.add_argument("--only_success", action="store_true", help="sadece basarili bolumleri yaz")
parser.add_argument("--max_steps", type=int, default=400, help="bolum basina guvenlik ust siniri")
parser.add_argument("--no_dr", action="store_true", help="domain randomization KAPAT (kamera+isik+masa)")
parser.add_argument("--fix_cam", action="store_true",
                    help="SADECE kamerayi sabitle; isik ve masa randomizasyonu "
                         "acik kalir. Gerekce (2026-09-13): kamera her bolumde "
                         "r +/-0.25 m, aci +/-0.35 rad, z -0.20/+0.25 oynuyor. "
                         "Yani 'kup su pikselde -> kup dunyada surada' esleme "
                         "HER BOLUMDE degisiyor; model once kamerayi cikarmak "
                         "sonra projeksiyonu tersine cevirmek zorunda. 208 "
                         "bolumle bu cok zor. Tek degisken olarak sinanmali.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--side_cam", action="store_true",
                    help="UCUNCU kamera ekle: yan gorus. On kameranin optik ekseni "
                         "x'e paralel oldugu icin x DERINLIK oluyor ve model onu "
                         "okuyamiyor (olculdu: y egimi 0.98, x egimi 0.33). Yan "
                         "kameranin ekseni y boyunca -> x onun icin YANAL. Ikisi "
                         "birlikte ucgenleme saglar. SmolVLA goruntuleri dongude "
                         "isliyor, kamera basina parametre yok -> katman boyutlari "
                         "degismez, mevcut checkpoint'ten devam edilebilir.")
parser.add_argument("--reset_hold", type=int, default=12,
                    help="Sifirlama sonrasi kolu EV POZUNDA tut (kac adim). "
                         "2026-09-15'te bulundu: Isaac Lab sifirlamada eklem "
                         "KONUMLARINI sifirliyor ama surucu HEDEFI onceki "
                         "bolumden kaliyor; kol ona dogru firliyor ve IK "
                         "patliyor. Olculdu: bolumlerin %23'unde 7 eklemin "
                         "HEPSI limit disina cikiyor (eklem 4, gercek araligi "
                         "[-3.07,-0.07], +19 rad'a kadar gidiyor). Uzman REST "
                         "fazinda des_ee_pose=ee_pose yazdigi icin kolu "
                         "duzeltmiyor, sadece ucusunu yankiliyor. "
                         "Patlamayan bolumlerde uzman basarisi %100, "
                         "patlayanlarda %46. 0 = kapali (eski davranis).")
parser.add_argument("--multi_verb", action="store_true",
                    help="FAZ 2b: DORT fiil (lift/stack/place/push). --two_objects "
                         "gerektirir. Tek fiille talimat yalnizca 1 bit tasir "
                         "(hangi kup) ve model dili yok sayip %50 alabilir. "
                         "Dort fiil talimatin DAVRANIS secmesini zorunlu kilar. "
                         "Basari olcutu fiile OZELDIR (push'ta kup hic kalkmaz, "
                         "stack'te indirilir) -- tek olcut kullanmak yanlis olur.")
parser.add_argument("--two_objects", action="store_true",
                    help="FAZ 2: masada IKI renkli kup (kirmizi/mavi) ve bolum "
                         "basina degisen talimat. Hedef nesne rastgele secilir, "
                         "talimat ona gore yazilir ve eg.attrs['task']'e BOLUM "
                         "BASINA kaydedilir. Uzman ve yardimci kafa HEDEF "
                         "nesneyi izler; celdiricinin tepe yuksekligi ayrica "
                         "kaydedilir (yanlis nesneyi kaldirma orani icin).")
parser.add_argument("--stateless_expert", action="store_true",
                    help="Uzmanin fazini ICSEL tutmak yerine HER ADIMDA "
                         "geometriden hesapla. DAgger icin SART: politika "
                         "kavrama pozuna varamayinca fazli makine APPROACH'ta "
                         "takiliyor ve 'tutucuyu kapat' etiketi HIC uretilmiyor "
                         "(olculdu: uzman verisinde %41 kapali kare, DAgger "
                         "verisinde %3.4; 104 bolumun 96'sinda hic kapanmiyor). "
                         "train_fixdag'in %1'inin sebebi buydu.")
parser.add_argument("--dagger_port", type=int, default=0,
                    help="DAgger modu: verilen porttaki politika sunucusundan "
                         "aksiyon al ve ONU uygula, ama ETIKET olarak uzmanin "
                         "ayni durumdan uretecegi komutu kaydet. Boylece veri, "
                         "politikanin GERCEKTEN gezdigi durumlardan toplanir. "
                         "DART'tan farki: DART uzmanin kendi yorungesine gurultu "
                         "ekliyordu (yaklasik), bu tam olarak politikanin "
                         "dagilimini kullanir. Olculen sorun: hedefleme hatasi "
                         "adim 20'de 43mm, adim 60'ta 77mm -- kol dagitim disina "
                         "ciktikca buyuyor.")
parser.add_argument("--dagger_host", type=str, default="127.0.0.1")
parser.add_argument("--randomize_arm", type=float, default=0.0,
                    help="Kolun baslangic eklem acilarini +/- bu kadar radyan "
                         "rastgele kaydir (0 = kapali, onerilen 0.15). Veri setinde "
                         "kol hep AYNI ev pozundan basliyor; model kolun nerede "
                         "oldugunu ortuk biliyor ve sabit bir yorunge uretmeyi "
                         "ogreniyor. Rastgelelesince dogru aksiyon hem kolun hem "
                         "kupun yerine bagli olur, ikisini de GORUNTUDEN cikarmak "
                         "zorunda kalir. Bkz. docs/SONUCLAR.md Test 1.")
parser.add_argument("--action_noise", type=float, default=0.0,
                    help="uzmanin komutuna eklenecek gurultunun std'si (metre). "
                         "Komut BOZULARAK uygulanir ama etikete UZMANIN DOGRU komutu "
                         "yazilir. Boylece kol yorunge disina sapar ve veri 'buradan "
                         "nasil toparlanilir' bilgisini icerir (DART / gurultu enjeksiyonu). "
                         "Dagilim kaymasina karsi. 0.005-0.02 makul.")
parser.add_argument("--env_spacing", type=float, default=8.0,
                    help="ortamlar arasi mesafe (m) - komsu ortamlar kadraja girmesin diye genis")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Kit acildiktan SONRA import edilebilir ---
import torch, numpy as np, h5py, gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets import RigidObjectData
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaac_shutdown import safe_close
from vram_probe import VramProbe
from domain_randomizer import DomainRandomizer

# State machine yerel modulden (orijinali modul seviyesinde AppLauncher calistiriyor)
from pick_lift_sm import PickAndLiftSm  # noqa: E402

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
LIFT_SUCCESS_HEIGHT = 0.10   # m, masa yuzeyinden yukari


def pinhole(focal):
    return sim_utils.PinholeCameraCfg(
        focal_length=focal, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.01, 20.0))


def build_env_cfg():
    cfg: LiftEnvCfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    # Debug marker'lari kapat -- egitim goruntusune sizmamalilar
    cfg.commands.object_pose.debug_vis = False

    # IK cikitisini EKLEM LIMITLERINE kirp. Isaac Lab kirpmiyor ve sifirlama
    # sonrasi taze poz + BAYAT Jacobian birlesimi bolumlerin ~%23'unde kolu
    # fiziksel olarak imkansiz konfigurasyonlara sokuyordu (eklem 4, gercek
    # araligi [-3.07,-0.07], +19 rad'a kadar). Ayrinti: clamped_ik_action.py
    from clamped_ik_action import ClampedDifferentialIKAction
    cfg.actions.arm_action.class_type = ClampedDifferentialIKAction

    # DENENDI, DAHA KOTU YAPTI (2026-09-15): cozucu iterasyonlarini artirmak
    # (pos 8->32, vel 0->4) patlamayi onlemedi, siddetlendirdi:
    # hiz 1941 -> 7562 rad/s, asim 6.66 -> 31.34 rad. GERI ALINDI.
    #
    # DENENIYOR: surucu sertligi. FRANKA_PANDA_HIGH_PD_CFG, IK takibi icin
    # sertligi 80'den 400'e cikariyor. Sert surucu + sifirlama sureksizligi
    # kararsizligin klasik recetesi.
    import os as _os
    if _os.environ.get("NO_SELF_COLL") == "1":
        # SIFIRLAMADA kol isinlanirken ayni karede self-collision tetiklenirse
        # PhysX devasa impuls uretebilir. Varsayilan enabled_self_collisions=True.
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        print("[FIZIK] self-collision KAPALI", flush=True)
    _st = float(_os.environ.get("ARM_STIFFNESS", "0"))
    if _st > 0:
        for _a in ("panda_shoulder", "panda_forearm"):
            cfg.scene.robot.actuators[_a].stiffness = _st
            cfg.scene.robot.actuators[_a].damping = _st / 5.0
        print(f"[FIZIK] kol surucu sertligi -> {_st} (varsayilan 400)", flush=True)
    # FAZ 2: iki renkli kup + ortak dagilimli yerlestirme
    if args_cli.two_objects:
        from isaaclab.managers import EventTermCfg
        from phase2_scene import kup_cfg, reset_iki_nesne, NESNELER
        cfg.scene.object = kup_cfg(NESNELER[0])
        cfg.scene.object2 = kup_cfg(NESNELER[1])
        # Hazir tek-nesne randomizasyonunu KALDIR: iki kupu birlikte, ayni
        # dagilimdan ve minimum mesafeyi gozeterek yerlestiren terimle degistir.
        # (Ayri terimler kullanmak dagilimlari ayirir; ust uste dogan kupleri
        # PhysX firlatir.)
        cfg.events.reset_object_position = EventTermCfg(
            func=reset_iki_nesne, mode="reset", params={})
        print("[FAZ2] iki nesneli sahne: kirmizi + mavi kup", flush=True)

    # Ortamlari birbirinden uzaklastir: komsu robot/masa kadraja girmemeli
    cfg.scene.env_spacing = args_cli.env_spacing

    # Kolun baslangic pozunu rastgelelestir (reset_all'dan SONRA calismali)
    if args_cli.randomize_arm > 0:
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from arm_randomizer import reset_arm_joints_offset
        r = args_cli.randomize_arm
        cfg.events.reset_arm_pose = EventTermCfg(
            func=reset_arm_joints_offset, mode="reset",
            params={"position_range": (-r, r),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])})

    # Ucuncu sahis kamerasi (dunya konumu reset sonrasi nisanlanir)
    cfg.scene.front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb"], spawn=pinhole(18.0),
        width=args_cli.res, height=args_cli.res)

    # Yan kamera (istege bagli): ekseni y boyunca, x'i yanal gorur
    if args_cli.side_cam:
        cfg.scene.side_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/side_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"),
            data_types=["rgb"], spawn=pinhole(18.0),
            width=args_cli.res, height=args_cli.res)

    # Bilek kamerasi: panda_hand uzerine, parmaklara bakar (roll=90 dogrulandi)
    cfg.scene.wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.07, 0.0, -0.03), rot=(0.7071, 0.0, 0.0, 0.7071), convention="ros"),
        data_types=["rgb"], spawn=pinhole(8.0),
        width=args_cli.res, height=args_cli.res)
    return cfg


def to_uint8(t):
    a = t.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a[..., :3]


def main():
    probe = VramProbe().start()
    cfg = build_env_cfg()
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    dev = env.unwrapped.device
    n = env.unwrapped.num_envs

    # Domain randomization + ucuncu sahis kamerayi nisanla
    origins = env.unwrapped.scene.env_origins
    front_cam = env.unwrapped.scene["front_cam"]
    dr = DomainRandomizer(seed=args_cli.seed,
                          jitter_cam=not (args_cli.no_dr or args_cli.fix_cam),
                          jitter_light=not args_cli.no_dr,
                          jitter_table=not args_cli.no_dr)
    side_cam = env.unwrapped.scene["side_cam"] if args_cli.side_cam else None
    dr.apply(front_cam, origins, dev, n, side_cam=side_cam)

    # Uzman state machine
    dt = cfg.sim.dt * cfg.decimation
    if args_cli.multi_verb and not args_cli.two_objects:
        raise SystemExit("--multi_verb, --two_objects gerektirir "
                         "(stack fiili ikinci nesneyi hedefler)")
    if args_cli.multi_verb:
        from multi_verb_expert import (MultiVerbExpert, LIFT, STACK, PLACE, PUSH,
                                       FIIL_ADI, BOLGELER, YONLER)
        from phase2_scene import talimat_uret, basari_olc, NESNELER
        sm = MultiVerbExpert(n, dev)
        print("[UZMAN] COK FIILLI FAZSIZ -- lift/stack/place/push", flush=True)
    elif args_cli.stateless_expert:
        from stateless_expert import StatelessPickSm
        sm = StatelessPickSm(n, dev)
        print("[UZMAN] FAZSIZ -- faz her adimda geometriden hesaplanir", flush=True)
    else:
        sm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=dev)
    actions[:, 3] = 1.0

    # EV POZU: sifirlama sonrasi kolu burada tutacagiz.
    # DIKKAT: eski kod baslangicta `env.step(actions)` yapiyordu; o aksiyon
    # hedef konumu (0,0,0) yani ROBOTUN TABANI demek -- ulasilamaz bir hedef ve
    # DLS cozucusunu zorluyor. Bunun yerine kolun KENDI mevcut pozunu hedef
    # veriyoruz: gecerli, ulasilabilir, kolu yerinde tutar.
    _ee0 = env.unwrapped.scene["ee_frame"]
    home_action = actions.clone()
    home_action[:, :3] = _ee0.data.target_pos_w[..., 0, :] - origins
    home_action[:, 3:7] = _ee0.data.target_quat_w[..., 0, :]
    home_action[:, 7] = 1.0                      # tutucu ACIK
    print(f"[EV POZU] ({home_action[0,0]:+.4f},{home_action[0,1]:+.4f},"
          f"{home_action[0,2]:+.4f}) | sifirlama sonrasi {args_cli.reset_hold} adim tutulacak",
          flush=True)
    hold = np.zeros(n, dtype=int)
    _blown = np.zeros(n, dtype=bool)
    desired_orientation = torch.zeros((n, 4), device=dev)
    desired_orientation[:, 1] = 1.0

    # FAZ 2: her ortam icin HEDEF nesne (0=kirmizi, 1=mavi). Bolum basinda
    # yeniden secilir. Talimat buradan uretilir ve bolum ozniteligine yazilir.
    if args_cli.two_objects:
        from phase2_scene import NESNELER, talimat as _talimat
        _rng2 = np.random.RandomState(args_cli.seed + 7777)
        hedef = _rng2.randint(0, len(NESNELER), size=n)
        if args_cli.multi_verb:
            _FIILLER = [LIFT, STACK, PLACE, PUSH]
            _zorla = os.environ.get("FORCE_VERB")     # hata ayiklama: tek fiil
            if _zorla:
                _ad2f = {v: k for k, v in FIIL_ADI.items()}
                _FIILLER = [_ad2f[_zorla]]
                print(f"[FAZ2] FIIL ZORLANDI: {_zorla}", flush=True)
            _YER_ADLARI = ["left", "right"]
            fiil_arr = _rng2.choice(_FIILLER, size=n)
            yer_arr = _rng2.choice(_YER_ADLARI, size=n)
            # oteki kupun izi -- fiile ozel basari olcutu icin gerekli.
            # HDF5'e YAZILMIYOR: yeni dataset eklemek cevirici tarafinda risk.
            diger_iz = [[] for _ in range(n)]
        # celdiricinin bolum icindeki EN YUKSEK z'si -- yanlis nesneyi
        # kaldirma oranini olcmek icin (dil yok sayilirsa bu oran yukselir)
        celdirici_tepe = np.zeros(n)
    else:
        hedef = None

    # kamera tamponlari reset sonrasi ilk adimda bos olabilir -> bir kez isit.
    # EV POZU komutuyla: kol yerinde kalir.
    for _ in range(2):
        env.step(home_action)

    # --- DAgger istemcisi (istege bagli) ---
    dagger = None
    if args_cli.dagger_port:
        import socket as _socket
        from bridge_protocol import send_msg, recv_msg

        class _DaggerClient:
            def __init__(self, host, port):
                self.sock = _socket.create_connection((host, port))
                print(f"[DAGGER] politika sunucusuna baglanildi: {host}:{port}", flush=True)

            def act(self, front, wrist, state, task, reset, side=None):
                msg = {"front": front, "wrist": wrist, "state": state,
                       "task": task, "reset": reset}
                if side is not None:
                    msg["side"] = side
                send_msg(self.sock, msg)
                return recv_msg(self.sock)["action"]

        dagger = _DaggerClient(args_cli.dagger_host, args_cli.dagger_port)
        print("[DAGGER] politika SURUYOR, uzman ETIKETLIYOR", flush=True)

    buffers = [defaultdict(list) for _ in range(n)]
    episodes, ep_count, step_count = [], 0, 0
    dagger_reset = np.ones(n, dtype=bool)
    t0 = time.time()

    print(f"[TOPLA] {args_cli.num_episodes} bolum hedefi | {n} paralel ortam | {args_cli.res}px | {1/dt:.0f} Hz",
          flush=True)

    with torch.inference_mode():
        while ep_count < args_cli.num_episodes and simulation_app.is_running():
            # --- 1) MEVCUT DURUM s_t oku (goruntu + robot durumu ayni sim adimindan) ---
            front = to_uint8(env.unwrapped.scene["front_cam"].data.output["rgb"])
            wrist = to_uint8(env.unwrapped.scene["wrist_cam"].data.output["rgb"])
            side = to_uint8(side_cam.data.output["rgb"]) if side_cam is not None else None

            robot = env.unwrapped.scene["robot"].data
            joint_pos = robot.joint_pos.clone()          # (n, 9): 7 kol + 2 parmak
            joint_vel = robot.joint_vel.clone()

            ee = env.unwrapped.scene["ee_frame"]
            ee_pos = ee.data.target_pos_w[..., 0, :].clone() - origins
            ee_quat = ee.data.target_quat_w[..., 0, :].clone()

            obj: RigidObjectData = env.unwrapped.scene["object"].data
            obj_pos = obj.root_pos_w - origins
            if args_cli.two_objects:
                # HEDEF nesnenin konumu `obj_pos`'a yazilir; boylece hem uzman
                # hem yardimci kafa "talimatta gecen nesne"yi izler ve asagidaki
                # basari olcutu ("object_pos z esigi asti mi") dogru nesneyi
                # olcer. Celdirici ayri takip edilir.
                obj2_pos = env.unwrapped.scene["object2"].data.root_pos_w - origins
                _m = torch.from_numpy(hedef).to(dev).bool().unsqueeze(-1)
                hedef_pos = torch.where(_m, obj2_pos, obj_pos)
                cel_pos = torch.where(_m, obj_pos, obj2_pos)
                obj_pos = hedef_pos
                celdirici_tepe = np.maximum(celdirici_tepe,
                                            cel_pos[:, 2].cpu().numpy())

            # --- 2) Uzman s_t'den a_t'yi uretsin ---
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            if args_cli.stateless_expert or args_cli.multi_verb:
                # parmak eklemleri: joint_pos[7] + joint_pos[8]
                # ACIK 0.0800 / KUPU TUTARKEN 0.0450 (olculdu) -> esik 0.06
                # FAZSIZ uzman "kavradi mi" kararini BUNDAN veriyor; beslenmezse
                # hep "acik" sanir ve kavrama sonrasi dala HIC gecmez.
                sm.set_fingers(joint_pos[:, 7] + joint_pos[:, 8])
            if args_cli.multi_verb:
                _yer = torch.tensor(
                    [BOLGELER[y] if f in (PLACE,) else YONLER[y]
                     for f, y in zip(fiil_arr, yer_arr)],
                    device=dev, dtype=obj_pos.dtype)
                actions = sm.compute(
                    ee_pos, obj_pos, cel_pos,
                    torch.as_tensor(fiil_arr, device=dev),
                    _yer, desired_orientation)
                for i in range(n):
                    diger_iz[i].append(cel_pos[i].cpu().numpy())
            else:
                actions = sm.compute(
                    torch.cat([ee_pos, ee_quat], dim=-1),
                    torch.cat([obj_pos, desired_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1))

            # --- 3) (s_t, a_t) ciftini kaydet ---
            act_np = actions.detach().cpu().numpy()
            jp_np, jv_np = joint_pos.cpu().numpy(), joint_vel.cpu().numpy()
            ep_np, eq_np, op_np = ee_pos.cpu().numpy(), ee_quat.cpu().numpy(), obj_pos.cpu().numpy()
            for i in range(n):
                b = buffers[i]
                b["observation.images.front"].append(front[i])
                b["observation.images.wrist"].append(wrist[i])
                if side is not None:
                    b["observation.images.side"].append(side[i])
                b["observation.state.joint_pos"].append(jp_np[i])
                b["observation.state.joint_vel"].append(jv_np[i])
                b["observation.state.ee_pos"].append(ep_np[i])
                b["observation.state.ee_quat"].append(eq_np[i])
                b["observation.state.object_pos"].append(op_np[i])
                b["action"].append(act_np[i])

            # --- 4) a_t'yi uygula -> s_{t+1} ---
            # GURULTU ENJEKSIYONU: etikete (adim 3) uzmanin TEMIZ komutu yazildi;
            # burada bozulmus halini uyguluyoruz. Kol yorungeden sapar, bir sonraki
            # karede uzman "buradan toparla" diyen bir komut uretir ve O kaydedilir.
            # Boylece model yorunge disi durumlardan donmeyi ogrenir.
            exec_actions = actions
            if dagger is not None:
                # Politika SURUYOR: kol onun gezdigi durumlara gider.
                # Etiket (adim 3'te kaydedildi) UZMANIN ayni durumdan uretecegi
                # komut -- "buradan nasil toparlanilir" bilgisi budur.
                _st = np.concatenate(
                    [joint_pos.cpu().numpy(), ep_np, eq_np], axis=1).astype(np.float32)
                _a = dagger.act(front, wrist, _st, args_cli.task_prompt,
                                dagger_reset.copy(), side=side)
                dagger_reset[:] = False
                exec_actions = torch.from_numpy(
                    np.ascontiguousarray(_a)).float().to(dev)
            if args_cli.action_noise > 0:
                noise = torch.randn_like(actions) * args_cli.action_noise
                noise[:, 3:] = 0.0        # sadece KONUM bozulur; quat/gripper aynen
                exec_actions = actions + noise
            # SIFIRLAMA TUTUSU: yeni bolumun ilk adimlarinda kolu ev pozunda tut,
            # yoksa onceki bolumden kalan surucu hedefine firliyor ve IK patliyor.
            if args_cli.reset_hold > 0 and hold.any():
                _m = torch.from_numpy(hold > 0).to(dev)
                exec_actions = torch.where(_m.unsqueeze(-1), home_action, exec_actions)
                hold = np.maximum(hold - 1, 0)
            _r0 = env.unwrapped.scene["robot"]; _h0 = _r0.body_names.index("panda_hand")
            _pre_jp = _r0.data.joint_pos[:, :7].clone()
            _pre_bp = (_r0.data.body_pos_w[:, _h0] - origins).clone()
            _pre_jv = _r0.data.joint_vel[:, :7].clone()
            obs, rew, term, trunc, info = env.step(exec_actions)
            dones = term | trunc
            step_count += 1

            if os.environ.get("BLOWUP_PROBE"):
                _r = env.unwrapped.scene["robot"]
                _jv = _r.data.joint_vel[:, :7]
                _bad = (_jv.abs().max(dim=1).values > 10.0).nonzero(as_tuple=False).flatten()
                for _i in _bad.tolist():
                    if _blown[_i]:
                        continue
                    _blown[_i] = True
                    _h = _r.body_names.index("panda_hand")
                    _bp = (_r.data.body_pos_w[:, _h] - origins)[_i].cpu().numpy()
                    _a = exec_actions[_i].cpu().numpy()
                    _dj = (_r.data.joint_pos[_i, :7] - _pre_jp[_i]).cpu().numpy()
                    print(f"[PATLAMA] ortam {_i} bolum-adim {len(buffers[_i]['action'])} "
                          f"|hiz|max={_jv[_i].abs().max().item():9.1f}", flush=True)
                    print(f"          komut  ({_a[0]:+.3f},{_a[1]:+.3f},{_a[2]:+.3f}) "
                          f"quat({_a[3]:+.3f},{_a[4]:+.3f},{_a[5]:+.3f},{_a[6]:+.3f}) grip={_a[7]:+.2f}",
                          flush=True)
                    _pb = _pre_bp[_i].cpu().numpy()
                    print(f"          ADIM ONCESI IK_poz ({_pb[0]:+.3f},{_pb[1]:+.3f},{_pb[2]:+.3f}) "
                          f"|hiz|onceki={_pre_jv[_i].abs().max().item():7.3f}", flush=True)
                    print(f"          ADIM SONRASI       ({_bp[0]:+.3f},{_bp[1]:+.3f},{_bp[2]:+.3f})  "
                          f"eklem degisimi(rad) " + " ".join(f"{x:+.2f}" for x in _dj), flush=True)

            rew_np = rew.detach().cpu().numpy()
            done_np = dones.detach().cpu().numpy()
            for i in range(n):
                buffers[i]["next.reward"].append(float(rew_np[i]))
                buffers[i]["next.done"].append(bool(done_np[i]))

            # --- 5) Biten bolumleri bosalt (reset sonrasi durum SONRAKI bolume ait) ---
            finished = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(finished) > 0:
                for i in finished.tolist():
                    b = buffers[i]
                    if len(b["action"]) > 5:
                        peak_z = float(np.max([p[2] for p in b["observation.state.object_pos"]]))
                        final_z = float(b["observation.state.object_pos"][-1][2])
                        success = peak_z > LIFT_SUCCESS_HEIGHT and final_z > LIFT_SUCCESS_HEIGHT
                        ek = {"_success": success, "_peak_z": peak_z, "_final_z": final_z}
                        if args_cli.multi_verb:
                            # FIILE OZEL olcut. Tek olcut ("z esigi asti mi")
                            # PUSH'ta hep basarisiz, STACK'te hep yanlis olurdu.
                            f_i = int(fiil_arr[i]); y_i = str(yer_arr[i])
                            h_ad = NESNELER[int(hedef[i])]["ad"]
                            d_ad = NESNELER[1 - int(hedef[i])]["ad"]
                            _h_iz = np.asarray(b["observation.state.object_pos"])
                            _d_iz = np.asarray(diger_iz[i])
                            _jp = np.asarray(b["observation.state.joint_pos"])
                            _parmak = _jp[:, 7] + _jp[:, 8]
                            success = bool(basari_olc(f_i, _h_iz, _d_iz, _parmak, y_i))
                            ek["_success"] = success
                            ek["_task"] = talimat_uret(f_i, h_ad, d_ad, y_i)
                            ek["_fiil"] = FIIL_ADI[f_i]
                            ek["_yer"] = y_i
                        if args_cli.two_objects:
                            ad = NESNELER[int(hedef[i])]["ad"]
                            ek.setdefault("_task", _talimat(ad))
                            ek["_hedef"] = ad
                            # YANLIS nesne kaldirildi mi? Dil yok sayilirsa bu
                            # oran yukselir -- basari oranina bakarak anlasilmaz.
                            ek["_celdirici_tepe"] = float(celdirici_tepe[i])
                            ek["_yanlis_kaldirdi"] = bool(
                                celdirici_tepe[i] > LIFT_SUCCESS_HEIGHT)
                        if (not args_cli.only_success) or success:
                            episodes.append({k: np.asarray(v) for k, v in b.items()} | ek)
                            ep_count += 1
                            _ek_log = (f" hedef={ek['_hedef']}"
                                       f"{' [' + ek['_fiil'] + '/' + ek['_yer'] + ']' if '_fiil' in ek else ''}"
                                       f"{' YANLIS-KALDIRDI' if ek['_yanlis_kaldirdi'] else ''}"
                                       if args_cli.two_objects else "")
                            print(f"[TOPLA] bolum {ep_count}/{args_cli.num_episodes} "
                                  f"({len(b['action'])} adim, tepe_z={peak_z:.3f}m, son_z={final_z:.3f}m, "
                                  f"{'BASARILI' if success else 'basarisiz'}{_ek_log})", flush=True)
                    buffers[i] = defaultdict(list)
                    if args_cli.two_objects:
                        # YENI BOLUM -> yeni hedef nesne ve yeni talimat
                        hedef[i] = _rng2.randint(0, len(NESNELER))
                        celdirici_tepe[i] = 0.0
                        if args_cli.multi_verb:
                            fiil_arr[i] = _rng2.choice(_FIILLER)
                            yer_arr[i] = _rng2.choice(_YER_ADLARI)
                            diger_iz[i] = []
                sm.reset_idx(finished)
                dagger_reset[finished.cpu().numpy()] = True
                hold[finished.cpu().numpy()] = args_cli.reset_hold
                _blown[finished.cpu().numpy()] = False

                # ===== BAYAT POZ DUZELTMESI (2026-09-15) =====
                # Isaac Lab sifirlamada eklem durumunu PhysX'e YAZIYOR ama link
                # pozlari (articulation.data.body_pos_w) simulasyon ilerlemeden
                # tazelenmiyor. IK aksiyon terimi mevcut EE pozunu TAM ORADAN
                # okuyor (_compute_frame_pose -> body_pos_w), dolayisiyla
                # sifirlamadan sonraki ILK adimda ONCEKI bolumun bittigi yeri
                # "mevcut poz" saniyor. Hedef ev pozu oldugu icin hata 30-50 cm
                # cikiyor, DLS devasa bir eklem farki uretiyor ve sert PD surucu
                # onu uyguluyor -> cozucu patliyor.
                #
                # KANIT (combo_s901, n=96): onceki bolumun bitis noktasi ev
                # pozundan ne kadar uzaksa patlama o kadar kesin --
                #   <10cm: %18 | >20cm: %38 | >30cm: %100 (11/11)
                #   korelasyon +0.558
                # Hiz izi: kare0 0.000 -> kare1 113 rad/s (limit 2.175)
                #          -> kare2 1931 rad/s
                #
                # sim.forward() PhysX'e ileri kinematigi yeniden hesaplatir
                # (update_articulations_kinematic), boylece bir sonraki
                # process_actions TAZE pozu okur.
                # DIKKAT -- dt SIFIR OLMAMALI. ArticulationData tamponu soyle:
                #     if self._body_state_w.timestamp < self._sim_timestamp:
                #         ...taze oku...
                #     return self._body_state_w.data
                # ve `_sim_timestamp += dt` sadece update(dt) icinde ilerliyor.
                # scene.update(0.0) cagirmak zaman damgasini ILERLETMEZ, tampon
                # "taze" sayilir ve BAYAT veri doner. Ilk denemede bu hataya
                # dusuldu ve duzeltme ise yaramadi (patlama %23 -> %19).
                env.unwrapped.sim.forward()
                env.unwrapped.scene.update(dt)
                if os.environ.get("RESET_DRIVE_TARGET", "0") == "1":
                    # Sifirlama eklem KONUMLARINI sifirliyor; peki SURUCU HEDEFI?
                    # Onceki bolumun son IK ciktisinda kalmis olabilir ve
                    # sifirlanmis kolu oraya cekiyor olabilir.
                    _rb = env.unwrapped.scene["robot"]
                    _ids = finished
                    _rb.set_joint_position_target(
                        _rb.data.default_joint_pos[_ids], env_ids=_ids)
                    _rb.write_data_to_sim()
                if False:
                    _r = env.unwrapped.scene["robot"]; _h = _r.body_names.index("panda_hand")
                    for _i in finished.tolist()[:2]:
                        _bp = (_r.data.body_pos_w[:, _h] - origins)[_i].cpu().numpy()
                        _sp = (ee.data.target_pos_w[..., 0, :] - origins)[_i].cpu().numpy()
                        _jv = _r.data.joint_vel[_i, :7].cpu().numpy()
                        _fg = (_r.data.joint_pos[_i, 7] + _r.data.joint_pos[_i, 8]).item()
                        _oz = (obj.root_pos_w - origins)[_i, 2].item()
                        print(f"[PROBE] ortam {_i} sifirlama sonrasi: "
                              f"IK({_bp[0]:+.3f},{_bp[1]:+.3f},{_bp[2]:+.3f}) "
                              f"sensor({_sp[0]:+.3f},{_sp[1]:+.3f},{_sp[2]:+.3f}) "
                              f"|hiz|max={np.abs(_jv).max():7.3f} parmak={_fg:.4f} kup_z={_oz:.4f}",
                              flush=True)
                # Yeni bolum -> sahneyi yeniden randomize et
                dr.apply(front_cam, origins, dev, n, side_cam=side_cam)
                # NOT (2026-09-03): buraya "isinma" icin env.step(actions) eklendi
                # ve DAHA KOTU cikti -- `actions` onceki bolumun son komutu, yeni
                # sifirlanmis ortam icin alakasiz bir hedef. Kol her bolum basinda
                # savruluyordu (olculdu: ee_pos ilk karelerde z=-0.192'ye kadar,
                # kare0->1 farki 76 -> 84-111). GERI ALINDI.
                # Bayat 0. kare sorunu CEVIRMEDE cozuluyor:
                # convert_to_lerobot.py --skip_first 2

            if step_count > args_cli.max_steps * args_cli.num_episodes:
                print("[TOPLA] guvenlik siniri asildi, duruluyor", flush=True)
                break

    elapsed = time.time() - t0
    peak_vram = probe.stop()

    # --- HDF5 yaz ---
    out = args_cli.out or os.path.expanduser(
        f"~/Projects/franka_vla_data/data/franka_lift_{args_cli.num_episodes}ep_{args_cli.res}px.hdf5")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_succ = 0
    with h5py.File(out, "w") as f:
        f.attrs["fps"] = 1.0 / dt
        f.attrs["task"] = args_cli.task_prompt
        f.attrs["robot"] = "franka_panda"
        f.attrs["env"] = TASK
        f.attrs["image_resolution"] = args_cli.res
        f.attrs["num_episodes"] = len(episodes)
        f.attrs["action_space"] = "ee_pose_abs(pos3+quat4)+gripper1"
        f.attrs["domain_randomization"] = not args_cli.no_dr
        f.attrs["fixed_camera"] = bool(args_cli.fix_cam or args_cli.no_dr)
        f.attrs["stateless_expert"] = bool(args_cli.stateless_expert)
        f.attrs["reset_hold"] = int(args_cli.reset_hold)
        f.attrs["seed"] = args_cli.seed
        f.attrs["action_noise"] = args_cli.action_noise
        g = f.create_group("data")
        for idx, ep in enumerate(episodes):
            succ = ep.pop("_success"); peak = ep.pop("_peak_z"); fin = ep.pop("_final_z")
            n_succ += int(succ)
            eg = g.create_group(f"episode_{idx:06d}")
            eg.attrs["num_samples"] = len(ep["action"])
            eg.attrs["success"] = bool(succ)
            eg.attrs["peak_object_z"] = float(peak)
            eg.attrs["final_object_z"] = float(fin)
            # BOLUM BASINA talimat. Faz 2'de her bolumde farkli.
            # DIKKAT: convert_to_lerobot BUNU okumali, dosya ozniteligini degil.
            eg.attrs["task"] = ep.pop("_task", args_cli.task_prompt)
            if "_fiil" in ep:
                eg.attrs["verb"] = ep.pop("_fiil")
                eg.attrs["place_target"] = ep.pop("_yer")
            if "_hedef" in ep:
                eg.attrs["target_object"] = ep.pop("_hedef")
                eg.attrs["distractor_peak_z"] = float(ep.pop("_celdirici_tepe"))
                eg.attrs["lifted_wrong"] = bool(ep.pop("_yanlis_kaldirdi"))
            for k, v in ep.items():
                comp = dict(compression="gzip", compression_opts=4) if v.ndim == 4 else {}
                eg.create_dataset(k, data=v, **comp)

    size_mb = os.path.getsize(out) / 1e6
    total_frames = sum(len(e["action"]) for e in episodes)
    print("\n" + "=" * 60, flush=True)
    print(f"[TOPLA] YAZILDI: {out}", flush=True)
    print(f"[TOPLA] {len(episodes)} bolum | {total_frames} kare | basari {n_succ}/{len(episodes)}", flush=True)
    print(f"[TOPLA] dosya {size_mb:.1f} MB | sure {elapsed:.1f}s | {total_frames/elapsed:.1f} kare/s", flush=True)
    print(f"[TOPLA] VRAM TEPE: {peak_vram} MiB / 6144 MiB (num_envs={n})", flush=True)
    print(f"[TOPLA] DR raporu: {dr.report()}", flush=True)
    print("=" * 60, flush=True)
    env.close()


main()
safe_close(simulation_app)
