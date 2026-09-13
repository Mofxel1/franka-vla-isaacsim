"""
SmolVLA politikasini Isaac Sim'de KAPALI DONGUDE calistirir (gercek test).

Ayri bir surecte calisan policy_server.py'a (lerobot ortami) her adimda
gozlem gonderir, aksiyon alir, robotu o aksiyonla surer. Kucuk uzman degil,
EGITILMIS MODEL robotu yonetiyor. Boylece "model gercekten kupu kaldirabiliyor
mu" sorusu cevaplanir -- acik-donguu (tek kare) tahmin dogrulugu degil.

NOT: collect_demos.py'nin kanitlanmis desenini izler -- TEK surekli dongu,
TEK torch.inference_mode() bloku, elle env.reset() YOK. Isaac Lab'in
yonetici-tabanli ortamlari bolum bitince env.step() icinde KENDILIGINDEN
sifirlaniyor. Elle reset cagirmak "inference tensor" hatasina yol aciyordu.

Onceden calistir (ayri terminal):
  conda activate lerobot
  python policy_server.py --ckpt /path/to/checkpoints/004000/pretrained_model

Sonra:
  conda activate isaaclab
  python eval_policy_isaacsim.py --headless --num_episodes 10 \
      --kit_args="--/rtx/verifyDriverVersion/enabled=false"
"""
import argparse, os, socket, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--res", type=int, default=224)
parser.add_argument("--env_spacing", type=float, default=8.0)
parser.add_argument("--legacy_warmup", action="store_true",
                    help="ESKI isinma: env.step(zeros). IK-Abs'ta sifir aksiyon "
                         "gecersiz bir EE hedefi demek ve kolu firlatiyor. "
                         "Sadece A/B karsilastirmasi icin.")
parser.add_argument("--warmup_steps", type=int, default=1,
                    help="Sifirlama+nisanlama sonrasi kac ISINMA adimi atilsin. "
                         "Isaac Sim'in RTX yolu zamansal biriktirme kullaniyor; "
                         "kamera yeni poza atlayinca goruntunun oturmasi birkac kare "
                         "suruyor. Toplamada kamera surekli calistigi icin hep "
                         "yakinsamis; evalde her bolumde sifirlaniyordu. Olculdu: "
                         "canli goruntude benzersiz renk 7031, veri setinde 3157 "
                         "(yakinsamamis render imzasi). Isinma KOLU TUTAN komutla "
                         "yapilir; sifir komut IK-Abs'ta 'EE'yi orijine gotur' demek.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Paralel ortam sayisi. TOPLAMA num_envs=8 ile yapildi; "
                         "eval 1 ortamla calisiyordu. TiledCamera N ortami tek "
                         "dokuya render ediyor ve her ortama ayri lamba konuyor, "
                         "yani goruntu dagitimi N'e bagli. Olculdu: canli x egimi "
                         "0.202 (veri setinde 0.927) -- y ise saglam (0.971). "
                         "Egitimle ayni N kullanmak bu asimetriyi kapatir. "
                         "Sadece 0. ortam kullanilir, digerleri ayni komutu alir.")
parser.add_argument("--side_cam", action="store_true",
                    help="ucuncu (yan) kamerayi ekle -- uc kameralı modeller icin")
parser.add_argument("--no_dr", action="store_true",
                    help="domain randomization KAPAT (eski davranis: sabit kamera, "
                         "varsayilan isik/masa -- egitim dagiliminin DISINDA)")
parser.add_argument("--fix_cam", action="store_true",
                    help="SADECE kamerayi sabitle (isik/masa randomize kalir). "
                         "Veri --fix_cam ile toplandiysa eval de oyle kosmali; "
                         "aksi halde boru hattinin iki ucu yine uyusmaz.")
parser.add_argument("--dr_seed", type=int, default=1234,
                    help="eval sahne randomizasyon tohumu; toplamadakinden farkli "
                         "olmali ki ezberlenmis sahneler degil genelleme olculsun")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--task_prompt", type=str, default="pick up the cube and lift it")
parser.add_argument("--max_total_steps", type=int, default=400,
                    help="guvenlik ust siniri: bolum basina beklenen adim x bolum sayisi civari")
parser.add_argument("--save_gif", action="store_true", default=True)
parser.add_argument("--out_gif", type=str, default=None)
parser.add_argument("--dump_obs", type=str, default=None,
                    help="her bolumun --dump_step'inci gozlemini npz olarak kaydet")
parser.add_argument("--dump_step", type=int, default=0,
                    help="Bolum icinde KACINCI adimin gozlemi kaydedilsin. "
                         "Veri seti olcumleri kare 2'yi kullaniyor (bayat 0. kare "
                         "yuzunden); eval 0. gozlemi kaydediyordu -> iki taraf "
                         "SIFIRLAMADAN BU YANA farkli adim sayisinda kiyaslaniyordu. "
                         "Olculdu: eval goruntusu 4408-7031 benzersiz renk, toplama "
                         "3169-3472 (render bir adim daha oturunca yumusuyor). "
                         "Adil kiyas icin veri seti karesiyle ayni degeri kullan.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np, torch, gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets import RigidObjectData
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from bridge_protocol import send_msg, recv_msg
from isaac_shutdown import safe_close
from domain_randomizer import DomainRandomizer

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
LIFT_SUCCESS_HEIGHT = 0.10


def pinhole(focal):
    return sim_utils.PinholeCameraCfg(
        focal_length=focal, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.01, 20.0))


def to_uint8(t):
    a = t.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a[..., :3]


class PolicyClient:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=30)

    last_cube_pred = None

    def act(self, front, wrist, state, task, reset, side=None):
        msg = {"front": front, "wrist": wrist, "state": state,
               "task": task, "reset": reset}
        if side is not None:
            msg["side"] = side
        send_msg(self.sock, msg)
        resp = recv_msg(self.sock)
        self.last_cube_pred = resp.get("cube_pred")
        return resp["action"]

    def close(self):
        # Sunucuyu KAPATMIYORUZ; sadece baglantiyi biraktik. Boylece ayni
        # sunucuya arka arkaya test kosulabilir (model bir kez yuklenir).
        self.sock.close()


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.commands.object_pose.debug_vis = False
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.scene.front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb"], spawn=pinhole(18.0), width=args_cli.res, height=args_cli.res)
    if args_cli.side_cam:
        cfg.scene.side_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/side_cam",
            offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"),
            data_types=["rgb"], spawn=pinhole(18.0), width=args_cli.res, height=args_cli.res)
    cfg.scene.wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.07, 0.0, -0.03), rot=(0.7071, 0.0, 0.0, 0.7071), convention="ros"),
        data_types=["rgb"], spawn=pinhole(8.0), width=args_cli.res, height=args_cli.res)

    env = gym.make(TASK, cfg=cfg)
    env.reset()  # SADECE BURADA, bir kez
    dev = env.unwrapped.device
    origins = env.unwrapped.scene.env_origins
    front_cam = env.unwrapped.scene["front_cam"]
    wrist_cam = env.unwrapped.scene["wrist_cam"]
    side_cam = env.unwrapped.scene["side_cam"] if args_cli.side_cam else None

    # Egitim verisindeki HER kare randomize edilmis bir sahneden geliyor
    # (rastgele gok isigi + masa ustu lamba + masa rengi + kamera pozu).
    # Burada randomize etmezsek model egitimde hic gormedigi bir aydinlatmayla
    # karsilasir; olculen sey politikanin becerisi degil dagitim kaymasi olur.
    dr = DomainRandomizer(seed=args_cli.dr_seed,
                          jitter_cam=not (args_cli.no_dr or args_cli.fix_cam),
                          jitter_light=not args_cli.no_dr,
                          jitter_table=not args_cli.no_dr)

    def aim_front_cam():
        # Isaac Lab bolum sonunda kamerayi spawn pozuna geri aliyor ve sahne
        # degisikliklerini sifirliyor -> her yeni bolumde yeniden uygulanmali.
        dr.apply(front_cam, origins, dev, env.unwrapped.num_envs, side_cam=side_cam)

    def hold_action():
        """Kolu oldugu yerde tutan komut: mevcut EE pozu + tutucu acik.
        Sifir komut IK-Abs'ta EE'yi orijine cagirir, kolu savurur."""
        ee_ = env.unwrapped.scene["ee_frame"]
        p_ = (ee_.data.target_pos_w[..., 0, :] - origins)
        q_ = ee_.data.target_quat_w[..., 0, :]
        g_ = torch.ones((p_.shape[0], 1), device=dev)
        return torch.cat([p_, q_, g_], dim=-1)

    def warmup(n_steps):
        """Sifirlama/nisanlama sonrasi kamerayi tazele -- FIZIK ADIMI ATMADAN.

        2026-09-12: eski surum `env.step(zeros)` yapiyordu. IK-Abs aksiyon
        uzayinda sifir demek "hedef EE pozu (0,0,0), quaternion (0,0,0,0)"
        demek -- gecersiz bir hedef. IK cozucu sacma eklem hedefi uretiyor ve
        kol tek adimda firliyordu. OLCULDU (dump_step=2):

            ee pozu   EGITIM (+0.375,+0.057,+0.328)
                      CANLI  (+0.284,+0.506,+0.107)   y'de 45cm, z'de 22cm fark
            eklem 4   egitim -2.221  ->  canli +4.036  (6.26 radyan)

        Kol kameranin gordugu en buyuk nesne; egitimde hic gorulmemis bir
        konfigurasyonda oldugu icin goruntunun tamami dagitim disi kaliyordu.
        Isinmanin amaci KAMERAYI tazelemek, kolu oynatmak degil.

        NOT: hold_action() de denendi (2026-09-03) ve daha kotu cikti, cunku
        sifirlamadan hemen sonra ee_frame verisi bayat -- kol onceki bolumun
        pozuna cagriliyordu. Render-only o sorunu da dogurmuyor: hicbir komut
        verilmiyor.

        --legacy_warmup ile eski davranisa donulebilir (A/B icin)."""
        if args_cli.legacy_warmup:
            for _ in range(max(1, n_steps)):
                env.step(torch.zeros(env.action_space.shape, device=dev))
            return
        u = env.unwrapped
        for _ in range(max(1, n_steps)):
            u.sim.render()                 # fizik yok, sadece render
            u.scene.update(0.0)            # sensor tamponlarini bayat isaretle

    client = PolicyClient(args_cli.host, args_cli.port)
    print(f"[TEST] politika sunucusuna baglanildi: {args_cli.host}:{args_cli.port}", flush=True)

    results = []
    gif_frames = []
    peak_z, final_z = 0.0, 0.0
    first_of_episode = True
    step_count = 0
    ep_trace = []          # bu bolumdeki (adim, hedef, ee, kup, tutucu)
    trace_summary = []
    early = {}
    pred_pairs = {}
    pred_err = {}
    obs_dump = []

    with torch.inference_mode():
        aim_front_cam()
        # ISINMA: nisanlama sonrasi ilk render henuz olusmadi, bir adim atmadan
        # kamera okunursa eski kare gelir.
        warmup(args_cli.warmup_steps)

        while len(results) < args_cli.num_episodes and simulation_app.is_running():
            front = to_uint8(front_cam.data.output["rgb"])[0]
            wrist = to_uint8(wrist_cam.data.output["rgb"])[0]
            side = to_uint8(side_cam.data.output["rgb"])[0] if side_cam is not None else None

            # KUPU env.step()'ten ONCE oku.
            # 2026-09-13 BULUNAN HATA: kup env.step()'ten SONRA okunuyordu.
            # Isaac Lab yonetici-tabanli ortami bolum bitince step() ICINDE
            # kendiliginden sifirliyor -> o okuma YENI bolumun taze kupunu
            # veriyordu (hep z=0.055). final_z bu yuzden hicbir zaman 0.10
            # esigini gecemiyordu ve `success` YAPISAL OLARAK imkansizdi.
            # Projedeki butun 0/20 sonuclari bunun eseri. Toplama tarafi dogru
            # olcuyordu (tampona step'ten ONCE yaziyor) -- ayni model orada
            # %22 basarili cikiyordu.
            obj: RigidObjectData = env.unwrapped.scene["object"].data
            cube = (obj.root_pos_w - origins)[0].cpu().numpy()
            z = float(cube[2])
            peak_z = max(peak_z, z)
            final_z = z

            robot = env.unwrapped.scene["robot"].data
            ee = env.unwrapped.scene["ee_frame"]
            ee_pos = (ee.data.target_pos_w[..., 0, :] - origins)[0].cpu().numpy()
            ee_quat = ee.data.target_quat_w[..., 0, :][0].cpu().numpy()
            state = np.concatenate(
                [robot.joint_pos[0].cpu().numpy(), ee_pos, ee_quat]).astype(np.float32)

            if len(ep_trace) == args_cli.dump_step and args_cli.dump_obs:
                _d = {"front": front.copy(), "wrist": wrist.copy(),
                      "state": state.copy(),
                      "cube": cube.copy()}
                # YAN kamera da dokulmeli: 3 kameralı modelin canli x korelasyonu
                # 0.794 -> 0.285 cokuyor ve bunun sebebi yan kameranin CANLI
                # goruntusunun EGITIM goruntusunden farkli olmasi olabilir
                # (on kamerada ayni uyusmazlik 2026-09-04'te bulunmustu).
                # Karsilastirabilmek icin once kaydetmek gerek.
                if side is not None:
                    _d["side"] = side.copy()
                obs_dump.append(_d)
            action = client.act(front, wrist, state, args_cli.task_prompt,
                                first_of_episode, side=side)
            first_of_episode = False

            if len(results) == 0 and args_cli.save_gif:
                gif_frames.append(front)

            act_t = torch.from_numpy(np.ascontiguousarray(action)).float().unsqueeze(0).to(dev)
            if args_cli.num_envs > 1:
                act_t = act_t.expand(args_cli.num_envs, -1).contiguous()
            obs, rew, term, trunc, info = env.step(act_t)
            step_count += 1

            ep_trace.append((len(ep_trace), action[:3].copy(), ee_pos.copy(),
                             cube.copy(), float(action[7])))
            if client.last_cube_pred is not None:
                # ALGI vs EYLEM: modelin kup tahmini gercek kupten ne kadar sapiyor.
                # ISARETLI vektor saklanir (sadece norm degil): canli hata tabani
                # 2026-09-08'de iki farkli modelde de ~60 mm'ye civilenmis cikti,
                # bu SABIT BIR KAYMA'ya isaret ediyor -- isareti kaybedersek
                # kaymayi sacilmadan ayiramayiz.
                _p = np.asarray(client.last_cube_pred, dtype=float)
                pred_err.setdefault(len(ep_trace) - 1, []).append(_p - cube[:2])
                pred_pairs.setdefault(len(ep_trace) - 1, []).append(
                    (_p.copy(), cube[:2].copy()))
            if bool(term[0]) or bool(trunc[0]):
                success = peak_z > LIFT_SUCCESS_HEIGHT and final_z > LIFT_SUCCESS_HEIGHT
                results.append({"peak_z": peak_z, "final_z": final_z, "success": success})
                print(f"[TEST] bolum {len(results)}/{args_cli.num_episodes} | tepe_z={peak_z:.3f}m "
                      f"son_z={final_z:.3f}m | {'BASARILI' if success else 'basarisiz'}", flush=True)
                if len(results) == 1:
                    print("[HAM] bolum 1: adim | hedef(xyz) | ee(xyz) | kup(xy) | tutucu", flush=True)
                    for k in list(range(0, min(30, len(ep_trace)))) + \
                             list(range(30, min(160, len(ep_trace)), 5)):
                        _, tg, ep_, cb, g = ep_trace[k]
                        print(f"[HAM] {k:>2} | ({tg[0]:+.3f},{tg[1]:+.3f},{tg[2]:+.3f}) | "
                              f"({ep_[0]:+.3f},{ep_[1]:+.3f},{ep_[2]:+.3f}) | "
                              f"({cb[0]:+.3f},{cb[1]:+.3f}) | {g:+.2f}", flush=True)
                # ERKEN adimlar: kol henuz sapmamisken model dogru yeri mi hedefliyor?
                # Bu, ALGI hatasini HATA BIRIKMESINDEN ayirir.
                for k in (20, 40, 60):
                    if k < len(ep_trace):
                        _, tgt, eep, cub, _ = ep_trace[k]
                        early.setdefault(k, []).append(
                            (float(np.linalg.norm(tgt[:2] - cub[:2])),
                             float(np.linalg.norm(eep[:2] - cub[:2]))))
                # Tutucunun ILK kapandigi an: model kupu nerede saniyor?
                close_i = next((k for k, (_, _, _, _, g) in enumerate(ep_trace) if g < 0), None)
                if close_i is not None:
                    _, tgt, eep, cub, _ = ep_trace[close_i]
                    trace_summary.append({
                        "ep": len(results), "adim": close_i,
                        "hedef": tgt, "ee": eep, "kup": cub,
                        "hedef_hata": float(np.linalg.norm(tgt[:2] - cub[:2])),
                        "ee_hata": float(np.linalg.norm(eep[:2] - cub[:2]))})
                    print(f"[IZ] bolum {len(results)} kapanis adim={close_i} "
                          f"kup=({cub[0]:.3f},{cub[1]:.3f}) "
                          f"hedef=({tgt[0]:.3f},{tgt[1]:.3f}) "
                          f"ee=({eep[0]:.3f},{eep[1]:.3f}) "
                          f"hedef_hata={np.linalg.norm(tgt[:2]-cub[:2])*1000:.0f}mm "
                          f"ee_hata={np.linalg.norm(eep[:2]-cub[:2])*1000:.0f}mm", flush=True)
                else:
                    print(f"[IZ] bolum {len(results)}: tutucu hic kapanmadi", flush=True)
                ep_trace = []
                peak_z, final_z = 0.0, 0.0
                first_of_episode = True
                # Isaac Lab bu ortami step() icinde KENDILIGINDEN sifirladi;
                # kamerayi yeniden nisanla, bir sonraki okuma dogru olsun.
                aim_front_cam()
                # ISINMA: nisanlama sonrasi render henuz olusmadi. Bu adim
                # olmadan yeni bolumun ILK gozlemi onceki bolumun son karesi
                # olur -- ve n_action_steps kadar adim o bayat kareden uretilir.
                warmup(args_cli.warmup_steps)
                step_count += args_cli.warmup_steps

            if step_count > args_cli.max_total_steps * args_cli.num_episodes:
                print("[TEST] guvenlik siniri asildi, duruluyor", flush=True)
                break

    client.close()

    n_ok = sum(r["success"] for r in results)
    print("\n" + "=" * 60, flush=True)
    if args_cli.dump_obs and obs_dump:
        _arrs = {k: np.stack([o[k] for o in obs_dump])
                 for k in ("front", "wrist", "state", "cube")}
        if all("side" in o for o in obs_dump):
            _arrs["side"] = np.stack([o["side"] for o in obs_dump])
        np.savez_compressed(args_cli.dump_obs, **_arrs)
        print(f"[TEST] dokulen anahtarlar: {sorted(_arrs)}", flush=True)
        print(f"[TEST] {len(obs_dump)} canli gozlem kaydedildi: {args_cli.dump_obs}", flush=True)
    if pred_err:
        print("\n[ALGI] modelin KENDI kup tahmininin hatasi (canli, adim adim):")
        for k in sorted(pred_err):
            if k % 20 == 0 and k <= 160:
                v = np.array(pred_err[k])                 # (n,2) ISARETLI
                a = np.linalg.norm(v, axis=1)
                print(f"[ALGI]   adim {k:>3} (n={len(a)}): medyan {np.median(a)*1000:6.1f}mm  "
                      f"ort {a.mean()*1000:6.1f}mm  |  kayma x{np.median(v[:,0])*1000:+6.1f} "
                      f"y{np.median(v[:,1])*1000:+6.1f} mm")
        # KAYMA mi SACILMA mi? Kaymayi cikarinca ne kaliyor + gercek kupe egim.
        print("\n[ALGI-EGIM] canli algi kalitesi (cevrimdisi 'egim' testinin canli esi):")
        for k in sorted(pred_pairs):
            if k % 20 == 0 and k <= 60:
                pr = np.array([p for p, _ in pred_pairs[k]])
                tr = np.array([t for _, t in pred_pairs[k]])
                d  = pr - tr
                dm = d - np.median(d, axis=0)             # kaymasi cikarilmis
                line = f"[ALGI-EGIM] adim {k:>3}: "
                for j, ax in enumerate("xy"):
                    sl = np.polyfit(tr[:, j], pr[:, j], 1)[0] if tr[:, j].std() > 1e-6 else float("nan")
                    co = np.corrcoef(tr[:, j], pr[:, j])[0, 1] if tr[:, j].std() > 1e-6 else float("nan")
                    line += f"{ax}: egim {sl:+.3f} kor {co:+.3f}  "
                line += f"| kayma cikinca medyan {np.median(np.linalg.norm(dm,axis=1))*1000:5.1f}mm"
                print(line)
    if early:
        print("\n[IZ] ERKEN adimlarda XY hatasi (kol henuz sapmadan):")
        for k in sorted(early):
            a = np.array(early[k])
            print(f"[IZ]   adim {k:>2} (n={len(a)}): hedef<->kup med {np.median(a[:,0])*1000:5.0f}mm  "
                  f"ort {a[:,0].mean()*1000:5.0f}mm   |  ee<->kup med {np.median(a[:,1])*1000:5.0f}mm")
    if trace_summary:
        he = np.array([t["hedef_hata"] for t in trace_summary])
        ee_ = np.array([t["ee_hata"] for t in trace_summary])
        print(f"\n[IZ] tutucu kapanis aninda XY hatasi ({len(he)} bolum):")
        print(f"[IZ]   modelin HEDEFI  <-> kup : ort {he.mean()*1000:.0f}mm  "
              f"med {np.median(he)*1000:.0f}mm  min {he.min()*1000:.0f}mm  max {he.max()*1000:.0f}mm")
        print(f"[IZ]   kolun GERCEK yeri <-> kup: ort {ee_.mean()*1000:.0f}mm  "
              f"med {np.median(ee_)*1000:.0f}mm  min {ee_.min()*1000:.0f}mm  max {ee_.max()*1000:.0f}mm")
        print(f"[IZ]   (kup 4cm; kavrama icin ~<20mm gerekir)\n")
    print(f"[TEST] SONUC: {n_ok}/{len(results)} basarili ({100*n_ok/max(len(results),1):.0f}%)", flush=True)
    print("=" * 60, flush=True)

    if gif_frames:
        out = args_cli.out_gif or os.path.expanduser("~/isaac_captures/policy_rollout.gif")
        from PIL import Image
        imgs = [Image.fromarray(f) for f in gif_frames[::2]]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        print(f"[TEST] gif kaydedildi: {out}", flush=True)

    env.close()


main()
safe_close(simulation_app)
