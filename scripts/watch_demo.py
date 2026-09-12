"""
Franka'yi PENCEREYLE izle.

Veri kaydetmez, kamera sensoru eklemez -- sadece uzmanin kupu kaldirisini
ekranda gosterir. Ogrenme/gozlem amacli.

Calistir:
    conda activate isaaclab
    cd ~/Projects/franka_vla_data/scripts
    python watch_demo.py                 # 4 robot yan yana
    python watch_demo.py --num_envs 1    # tek robot
"""

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka lift - pencereli izleme")
parser.add_argument("--num_envs", type=int, default=4, help="kac robot yan yana")
parser.add_argument("--env_spacing", type=float, default=2.5, help="robotlar arasi mesafe (m)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False          # <-- PENCERE ACIK

# Surucu surum kontrolunu atla: sürücü 535.309.01 ama Vulkan'in 8-bit minor alani
# tastigi icin Isaac Sim onu "535.53" okuyup eski saniyor. Bu bayrak o yanlis
# kontrolu devre disi birakir. AppLauncher'in kendi kanalindan gecirilmeli --
# dogrudan sys.argv'ye eklenirse argparse tanimadigi argumani reddediyor.
args_cli.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.assets import RigidObjectData
from pick_lift_sm import PickAndLiftSm

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"


def main():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.scene.env_spacing = args_cli.env_spacing
    # NOT: debug oklarini ACIK birakiyoruz -- izlerken hedefi gormek ogretici.
    # Veri toplarken bunlar KAPALI (goruntuye sizmasinlar diye).
    cfg.commands.object_pose.debug_vis = True

    env = gym.make(TASK, cfg=cfg)
    env.reset()
    dev = env.unwrapped.device
    n = env.unwrapped.num_envs
    origins = env.unwrapped.scene.env_origins

    dt = cfg.sim.dt * cfg.decimation
    sm = PickAndLiftSm(dt, n, dev, position_threshold=0.01)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=dev)
    actions[:, 3] = 1.0
    des_quat = torch.zeros((n, 4), device=dev)
    des_quat[:, 1] = 1.0

    print("\n" + "=" * 62)
    print(f"  {n} Franka kolu calisiyor. Kapatmak icin pencereyi kapat.")
    print("  KAMERA: sag tikla-surukle = dondur | orta tus = kaydir | tekerlek = yakinlas")
    print("=" * 62 + "\n", flush=True)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            ee = env.unwrapped.scene["ee_frame"]
            ee_pos = ee.data.target_pos_w[..., 0, :].clone() - origins
            ee_quat = ee.data.target_quat_w[..., 0, :].clone()
            obj: RigidObjectData = env.unwrapped.scene["object"].data
            obj_pos = obj.root_pos_w - origins
            des_pos = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            actions = sm.compute(
                torch.cat([ee_pos, ee_quat], dim=-1),
                torch.cat([obj_pos, des_quat], dim=-1),
                torch.cat([des_pos, des_quat], dim=-1))

            dones = env.step(actions)[-2]
            step += 1
            if step % 50 == 0:   # saniyede bir
                print(f"  t={step/50:5.1f}s   kup yuksekligi: " +
                      "  ".join(f"env{i}={obj_pos[i,2]:.3f}m" for i in range(min(n, 4))), flush=True)
            if dones.any():
                sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    env.close()


main()
simulation_app.close()
