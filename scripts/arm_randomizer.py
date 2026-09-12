"""Kolun BASLANGIC pozunu rastgelelestiren olay terimi.

Neden: veri setinde kol her bolumde AYNI ev pozundan basliyor. Durum vektoru
sifirlanmis olsa bile model kolun nerede oldugunu ortuk olarak biliyor (hep ayni
yer), o yuzden tek bilinmeyen kupun yeri kaliyor -- ve model bunu ogrenmek
yerine sabit bir nokta uretmeyi secti (bkz. docs/SONUCLAR.md Test 1).

Baslangic rastgelelesince sabit bir yorunge uretmek ise yaramaz: dogru aksiyon
hem kolun NEREDE oldugona hem kupun NEREDE oldugona baglidir. Ikisini de
goruntuden cikarmak zorunda kalir.

Isaac Lab'in hazir `reset_joints_by_offset` terimi TUM eklemleri oynatiyor --
tutucu parmaklari dahil, onlar limitlere kirpilinca kol kapali tutucuyla
baslayabiliyor. Bu terim sadece verilen eklemleri oynatir.
"""

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def reset_arm_joints_offset(env, env_ids, position_range, asset_cfg: SceneEntityCfg):
    """Yalnizca asset_cfg.joint_ids ile secilen eklemleri varsayilan pozun
    etrafinda rastgele kaydirir. Hizlar sifirlanir."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    ids = asset_cfg.joint_ids
    if ids is None or ids == slice(None):
        ids = list(range(joint_pos.shape[1]))
    n_j = len(ids) if not isinstance(ids, slice) else joint_pos.shape[1]

    noise = math_utils.sample_uniform(
        position_range[0], position_range[1], (len(env_ids), n_j), joint_pos.device)
    joint_pos[:, ids] += noise

    lim = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(lim[..., 0], lim[..., 1])
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_vel), env_ids=env_ids)
