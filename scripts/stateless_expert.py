"""
FAZSIZ (stateless) uzman -- DAgger icin.

NEDEN GEREKLI (2026-09-13'te olculdu):
`pick_lift_sm.py` bir DURUM MAKINESI ve fazlari ICSEL:
    REST -> APPROACH_ABOVE -> APPROACH -> GRASP -> LIFT
Faz ancak kol o anki hedefe 1 cm yaklasinca ilerliyor. DAgger'da kolu POLITIKA
suruyor; politika kavrama pozuna varamadigi icin makine APPROACH'ta takiliyor
ve "tutucuyu kapat" komutunu HIC uretmiyor:

    kapali kare orani   tutucuyu HIC kapatmayan bolum
    UZMAN   %41.1              14/104
    DAgger   %3.4              96/104

Birlesik veri setinde sinyal seyreliyor ve DAgger yarisi modele "boyle
durumlarda tutucuyu ACIK tut" ogretiyor -- tam da modelin evalde karsilastigi
durumlar. Sonuc: train_fixdag %1 (en iyi cevrimdisi, en kotu canli).

Bu surum fazi HER ADIMDA GEOMETRIDEN hesaplar. Kol nerede olursa olsun
"buradan ne yapilmali" sorusunun dogru cevabini verir -- DAgger'in vaadi zaten
buydu. Kup elden kacarsa |ee-kup| buyur, "kavradi" kosulu bozulur ve uzman
kendiliginden "geri don, yeniden yaklas" der. Toparlanma davranisi budur.

FARKI DART ile: DART uzmanin KENDI yorungesini bozar, faz normal ilerler ve
"kapat" ornekleri veride kalir -- bu yuzden DART calisiyordu (%31).

Esikler `pick_lift_sm.py`'den ve olculmus veriden:
  - kupun 10 cm ustu (SM offset[:,2] = 0.1)
  - konum esigi 1 cm (SM position_threshold)
  - parmak toplami: ACIK 0.0800, KUPU TUTARKEN 0.0450  -> esik 0.06
"""
import torch

OPEN, CLOSE = 1.0, -1.0


class StatelessPickSm:
    """`PickAndLiftSm` ile ayni arayuz; fazi geometriden hesaplar.

    compute() cagrilmadan once set_fingers() ile parmak acikligi verilmeli.
    """

    def __init__(self, num_envs, device, above=0.10, close_tol=0.015,
                 align_tol=0.02, grasp_tol=0.03, finger_closed=0.06):
        self.num_envs = num_envs
        self.device = device
        self.above = above                  # kupun kac cm ustune yaklasilir
        self.close_tol = close_tol          # bu mesafede tutucu KAPANIR
        self.align_tol = align_tol          # xy'de bu kadar hizaliysa INILIR
        self.grasp_tol = grasp_tol          # kapali + bu mesafe = KAVRAMIS
        self.finger_closed = finger_closed  # parmak toplami bunun altinda = kapali
        self._fingers = torch.full((num_envs,), 0.08, device=device)

    def set_fingers(self, finger_total):
        """finger_total: (n,) iki parmak ekleminin TOPLAMI."""
        self._fingers = finger_total.to(self.device).flatten()[:self.num_envs]

    def reset_idx(self, env_ids=None):
        """API uyumlulugu icin. Fazsiz uzmanda sifirlanacak durum YOK --
        zaten sorunun kaynagi icsel durumdu."""
        return

    def compute(self, ee_pose, object_pose, des_object_pose):
        """ee_pose/object_pose/des_object_pose: (n,7) = pos(3) + quat(w,x,y,z).
        Donen: (n,8) = pos(3) + quat(4) + tutucu(1) -- SM ile ayni duzen."""
        ee_p = ee_pose[:, :3]
        cu_p = object_pose[:, :3]
        go_p = des_object_pose[:, :3]
        des_q = object_pose[:, 3:7]                 # SM de nesnenin quat'ini kullaniyor

        above_p = cu_p.clone()
        above_p[:, 2] = above_p[:, 2] + self.above

        d_cube = torch.linalg.norm(ee_p - cu_p, dim=-1)
        d_xy = torch.linalg.norm(ee_p[:, :2] - cu_p[:, :2], dim=-1)
        closed = self._fingers < self.finger_closed

        grasped = closed & (d_cube < self.grasp_tol)   # kup elde -> KALDIR
        at_grasp = d_cube < self.close_tol             # kavrama pozunda -> KAPAT
        over = (d_xy < self.align_tol) & (ee_p[:, 2] > cu_p[:, 2])  # ustunde -> IN

        pos = above_p.clone()                          # varsayilan: ustune yaklas
        pos = torch.where(over.unsqueeze(-1), cu_p, pos)
        pos = torch.where(at_grasp.unsqueeze(-1), cu_p, pos)
        pos = torch.where(grasped.unsqueeze(-1), go_p, pos)

        grip = torch.full((ee_p.shape[0], 1), OPEN, device=ee_p.device,
                          dtype=ee_p.dtype)
        grip[at_grasp | grasped] = CLOSE

        return torch.cat([pos, des_q, grip], dim=-1)
