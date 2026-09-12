"""
Teshis: model ne aksiyon uretiyor, kol nereye gidiyor, kup nerede?

eval_policy_isaacsim.py ile ayni kurulum ama TEK bolum ve her 10 adimda
detayli cikti. Ayrica modele GIDEN goruntuleri diske yazar -- egitim
verisindeki goruntulerle gorsel olarak karsilastirabilmek icin.
"""
import argparse, os, socket, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=224)
parser.add_argument("--env_spacing", type=float, default=8.0)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--task_prompt", type=str, default="pick up the cube and lift it")
parser.add_argument("--steps", type=int, default=120)
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

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
OUT = os.path.expanduser("~/isaac_captures")


def pinhole(f):
    return sim_utils.PinholeCameraCfg(focal_length=f, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.01, 20.0))


def to_uint8(t):
    a = t.detach().cpu().numpy()
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a[..., :3]


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.commands.object_pose.debug_vis = False
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.scene.front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0,0,0), rot=(1,0,0,0), convention="world"),
        data_types=["rgb"], spawn=pinhole(18.0), width=args_cli.res, height=args_cli.res)
    cfg.scene.wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.07,0.0,-0.03), rot=(0.7071,0.0,0.0,0.7071), convention="ros"),
        data_types=["rgb"], spawn=pinhole(8.0), width=args_cli.res, height=args_cli.res)

    env = gym.make(TASK, cfg=cfg)
    env.reset()
    dev = env.unwrapped.device
    origins = env.unwrapped.scene.env_origins
    fc, wc = env.unwrapped.scene["front_cam"], env.unwrapped.scene["wrist_cam"]

    eyes = origins + torch.tensor([[1.5, 0.7, 0.85]], device=dev)
    targets = origins + torch.tensor([[0.30, 0.0, 0.20]], device=dev)
    fc.set_world_poses_from_view(eyes, targets)

    sock = socket.create_connection((args_cli.host, args_cli.port), timeout=60)
    print("[TESHIS] sunucuya baglanildi\n", flush=True)

    from PIL import Image
    with torch.inference_mode():
        env.step(torch.zeros(env.action_space.shape, device=dev))

        print(f"{'adim':>5} | {'MODEL AKSIYONU (hedef EE pozu + gripper)':^44} | {'GERCEK EE':^20} | {'KUP':^20}")
        print("-" * 100)
        for t in range(args_cli.steps):
            front = to_uint8(fc.data.output["rgb"])[0]
            wrist = to_uint8(wc.data.output["rgb"])[0]
            robot = env.unwrapped.scene["robot"].data
            ee = env.unwrapped.scene["ee_frame"]
            ee_pos = (ee.data.target_pos_w[..., 0, :] - origins)[0].cpu().numpy()
            ee_quat = ee.data.target_quat_w[..., 0, :][0].cpu().numpy()
            state = np.concatenate([robot.joint_pos[0].cpu().numpy(), ee_pos, ee_quat]).astype(np.float32)

            send_msg(sock, {"front": front, "wrist": wrist, "state": state,
                             "task": args_cli.task_prompt, "reset": (t == 0)})
            action = recv_msg(sock)["action"]

            obj: RigidObjectData = env.unwrapped.scene["object"].data
            cube = (obj.root_pos_w - origins)[0].cpu().numpy()

            if t % 10 == 0:
                print(f"{t:5d} | pos={np.round(action[:3],3)} quat={np.round(action[3:7],2)} g={action[7]:+.2f} "
                      f"| {np.round(ee_pos,3)} | {np.round(cube,3)}", flush=True)
                Image.fromarray(front).save(f"{OUT}/diag_front_t{t:03d}.png")
                if t == 0:
                    Image.fromarray(wrist).save(f"{OUT}/diag_wrist_t000.png")

            env.step(torch.from_numpy(np.ascontiguousarray(action)).float().unsqueeze(0).to(dev))

    # ozet
    print("\n[TESHIS] modele giden goruntuler ~/isaac_captures/diag_front_t*.png", flush=True)
    # NOT: sunucuyu KAPATMIYORUZ -- ardisik testlerde tekrar tekrar model
    # yuklemek gereksiz (ve yaris durumuna yol aciyor). Sunucuyu elle kapat.
    sock.close()
    env.close()


main()
safe_close(simulation_app)
