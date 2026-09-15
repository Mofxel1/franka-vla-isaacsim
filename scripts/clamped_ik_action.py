"""EKLEM LIMITLERINE KIRPAN IK aksiyon terimi.

SORUN (2026-09-15'te olculdu):
Isaac Lab'in `DifferentialInverseKinematicsAction.apply_actions` metodu IK
cikitisini limitlere KIRPMADAN uyguluyor:

    joint_pos_des = self._ik_controller.compute(...)
    self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

Sonuc: bolumlerin ~%23'unde YEDI EKLEMIN HEPSI limit disina cikiyor.
Eklem 4'un gercek araligi [-3.07, -0.07] (dirsek, serbest DONEMEZ); veride
-6.67 ile +19.04 arasi, yani limitin 1095 derece otesi. Kol gorsel olarak
parcalanmis gibi goruluyor (GIF'te fark edildi).

TETIKLEYICI: bolum sifirlamasi. Olculdu --
  - sifirlama ani TERTEMIZ: poz taze, eklemler ev konfigurasyonunda, hiz 0.000
  - SONRAKI TEK ADIMDA eklemler 0.7 rad oynuyor, hiz 113 rad/s (limit 2.175)
  - onceki bolum ev pozundan ne kadar uzakta bittiyse patlama o kadar kesin:
    <10cm %18 | >20cm %38 | >30cm %100 (11/11), korelasyon +0.558

NEDENI: `body_pos_w` okunurken `update_articulations_kinematic()` cagriliyor ve
TAZELENIYOR, ama `jacobian_w` dogrudan `root_physx_view.get_jacobians()` okuyor
-- zaman damgali tampon YOK, kinematik guncelleme YOK. Yani sifirlamadan sonra
IK, TAZE poz ile BAYAT Jacobian'i birlikte kullaniyor. DLS sonumlemesi de cok
dusuk (lambda_val=0.01), bu yuzden tutarsizlik buyuk eklem farkina donusuyor.

Denendi ve YETMEDI: sifirlama sonrasi ev pozunda tutma (--reset_hold),
sim.forward(), scene.update(dt). Ucu de patlamayi onlemedi (%23 -> %17-19).

BU COZUM: sebebi ne olursa olsun IK ciktisi eklem limitlerine kirpilir. Kol
fiziksel olarak imkansiz bir konfigurasyona GIDEMEZ. Faz 3'te gercek Nova 5'e
gecerken bu zaten zorunlu.

Kullanim (env cfg kurulurken):
    from clamped_ik_action import ClampedDifferentialIKActionCfg
    cfg.actions.arm_action.class_type = ClampedDifferentialIKAction
"""
import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class ClampedDifferentialIKAction(DifferentialInverseKinematicsAction):
    """IK cikitisini eklem limitlerine kirpar. Tek fark `apply_actions`."""

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        # --- TEK EKLENEN SEY: limitlere kirp ---
        lim = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        joint_pos_des = torch.clamp(joint_pos_des, min=lim[..., 0], max=lim[..., 1])

        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)
