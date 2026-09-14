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
    if args_cli.stateless_expert:
        from stateless_expert import StatelessPickSm
        sm = StatelessPickSm(n, dev)
        print("[UZMAN] FAZSIZ -- faz her adimda geometriden hesaplanir", flush=True)
    else:
        sm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=dev)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((n, 4), device=dev)
    desired_orientation[:, 1] = 1.0

    # kamera tamponlari reset sonrasi ilk adimda bos olabilir -> bir kez isit
    for _ in range(2):
        env.step(actions)

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

            # --- 2) Uzman s_t'den a_t'yi uretsin ---
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            if args_cli.stateless_expert:
                # parmak eklemleri: joint_pos[7] + joint_pos[8]
                # ACIK 0.0800 / KUPU TUTARKEN 0.0450 (olculdu) -> esik 0.06
                sm.set_fingers(joint_pos[:, 7] + joint_pos[:, 8])
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
            obs, rew, term, trunc, info = env.step(exec_actions)
            dones = term | trunc
            step_count += 1

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
                        if (not args_cli.only_success) or success:
                            episodes.append({k: np.asarray(v) for k, v in b.items()} |
                                            {"_success": success, "_peak_z": peak_z, "_final_z": final_z})
                            ep_count += 1
                            print(f"[TOPLA] bolum {ep_count}/{args_cli.num_episodes} "
                                  f"({len(b['action'])} adim, tepe_z={peak_z:.3f}m, son_z={final_z:.3f}m, "
                                  f"{'BASARILI' if success else 'basarisiz'})", flush=True)
                    buffers[i] = defaultdict(list)
                sm.reset_idx(finished)
                dagger_reset[finished.cpu().numpy()] = True
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
            eg.attrs["task"] = args_cli.task_prompt
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
