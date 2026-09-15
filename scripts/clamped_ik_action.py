"""EKLEM LIMITLERINE KIRPAN IK aksiyon terimi.

NE ISE YARAR: Isaac Lab'in `DifferentialInverseKinematicsAction` terimi IK
cikitisini eklem limitlerine KIRPMADAN uyguluyor:

    joint_pos_des = self._ik_controller.compute(...)
    self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

Bu sinif araya tek bir kirpma ekler. Kol fiziksel olarak imkansiz bir eklem
konfigurasyonu HEDEFI alamaz.

DIKKAT -- BU BIR HATA DUZELTMESI DEGIL, BIR EMNIYET KEMERIDIR.
Bu dosya once "bolumlerin ~%23'unde kolun paramparca olmasi" sorununun cozumu
sanilarak yazildi ve buraya "sifirlama sonrasi taze poz + BAYAT Jacobian"
diye bir teshis not edilmisti. O TESHIS YANLISTI ve 2026-09-15'te olcumle
curutuldu:

  - Kirpma patlamayi ONLEMEDI (6/32 sabit kaldi, hiz hala 1941 rad/s).
  - Sifirlama sonrasi ilk karede eklemler varsayilan poza TAM oturuyor
    (sapma 0.0000) ve IK hedefi mevcut pozdan sadece 0.015 rad uzakta.
    Yani ne poz bayat, ne Jacobian bozuk, ne de IK hatali bir hedef uretiyor.
  - Gercek sebep domain randomization'daki `prim.SetInstanceable(False)`
    cagrisiydi: masayi de-instance edip carpisma temsilini yeniden kuruyor,
    bolum sifirlamalarinda articulation'a impuls biniyordu.
    Ayrinti ve olcumler: scripts/domain_randomizer.py + docs/SONUCLAR.md

NEDEN YINE DE DURUYOR: Faz 3'te gercek Dobot Nova 5'e gecilecek. Gercek
donanimda eklem limiti disi bir hedef gondermek fiziksel hasar demektir;
kirpma orada zaten zorunlu. Sim tarafinda da bedeli yok.

Kullanim (env cfg kurulurken):
    from clamped_ik_action import ClampedDifferentialIKAction
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
