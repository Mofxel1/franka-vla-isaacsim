"""
Gorsel domain randomization.

Isaac Lab v2.1'in hazir randomize_* event terimleri yalnizca FIZIK parametrelerini
(kutle, yercekimi, aktuator) rastgeleliyor; isik/doku/kamera icin hazir terim yok.
Bu modul bunlari USD seviyesinde yapar.

Amac: VLA modelinin tek bir isik/aci/renk kombinasyonunu ezberlemesini engellemek.
"""

import numpy as np
import torch


class DomainRandomizer:
    def __init__(self, seed=0, jitter_cam=True, jitter_light=True, jitter_table=True):
        self.rng = np.random.default_rng(seed)
        self.jitter_cam = jitter_cam
        self.jitter_light = jitter_light
        self.jitter_table = jitter_table
        self._table_ok = None      # ilk denemede tespit edilir
        self._stage = None
        self._log = []

    # ---------- USD stage ----------
    @property
    def stage(self):
        if self._stage is None:
            import omni.usd
            self._stage = omni.usd.get_context().get_stage()
        return self._stage

    # ---------- 1) KAMERA POZU ----------
    def randomize_camera(self, cam, origins, device, base_eye=(1.5, 0.7, 0.85),
                         base_tgt=(0.30, 0.0, 0.20)):
        """Ucuncu sahis kamerasini her bolumde biraz oynatir (per-env bagimsiz).

        base_eye/base_tgt ile birden fazla sabit kamera ayni jitter mantigiyla
        nisanlanabilir (ornegin yan kamera)."""
        n = origins.shape[0]
        base_eye = np.asarray(base_eye, dtype=float)
        base_tgt = np.asarray(base_tgt, dtype=float)
        if not self.jitter_cam:
            eye_off = np.tile(base_eye, (n, 1))
            tgt_off = np.tile(base_tgt, (n, 1))
        else:
            # goz: yaricap/aci/yukseklik olarak oynat -> daha dogal cesitlilik
            base_r = float(np.linalg.norm(base_eye[:2]))
            base_th = float(np.arctan2(base_eye[1], base_eye[0]))
            r = base_r + self.rng.uniform(-0.25, 0.25, n)
            th = base_th + self.rng.uniform(-0.35, 0.35, n)
            z = base_eye[2] + self.rng.uniform(-0.20, 0.25, n)
            eye_off = np.stack([r * np.cos(th), r * np.sin(th), z], axis=-1)
            tgt_off = base_tgt + self.rng.uniform(-0.06, 0.06, (n, 3))

        eyes = origins + torch.tensor(eye_off, dtype=torch.float32, device=device)
        targets = origins + torch.tensor(tgt_off, dtype=torch.float32, device=device)
        cam.set_world_poses_from_view(eyes, targets)

    # ---------- 2) ISIK ----------
    def randomize_light(self):
        """DomeLight yogunluk + renk + yonelim. /World/light global oldugu icin
        bolum basina degisir (ortamlar arasi degil)."""
        if not self.jitter_light:
            return
        try:
            from pxr import UsdLux, UsdGeom, Gf
            prim = self.stage.GetPrimAtPath("/World/light")
            if not prim or not prim.IsValid():
                self._note("isik prim'i bulunamadi: /World/light")
                return
            light = UsdLux.DomeLight(prim)
            light.GetIntensityAttr().Set(float(self.rng.uniform(800.0, 5500.0)))
            # notr etrafinda sicak/soguk kayma
            t = self.rng.uniform(-0.18, 0.18)
            base = self.rng.uniform(0.6, 0.95)
            light.GetColorAttr().Set(Gf.Vec3f(float(base + t), float(base), float(base - t)))
            # dome'u dondur -> golge yonu degissin
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddRotateYOp().Set(float(self.rng.uniform(0, 360)))
        except Exception as e:
            self._note(f"isik randomizasyonu basarisiz: {e}")

    # ---------- 3) MASA RENGI ----------
    def randomize_table(self, num_envs):
        """Masaya rastgele renkli UsdPreviewSurface baglar.

        Masa 'table_instanceable.usd' referansi; instance'lanmis geometriye malzeme
        baglamak calismayabilir -> ilk denemede tespit edip raporluyoruz.
        """
        if not self.jitter_table or self._table_ok is False:
            return
        try:
            from pxr import Usd, UsdShade, Sdf, Gf
            ok_any = False
            for i in range(num_envs):
                tp = f"/World/envs/env_{i}/Table"
                prim = self.stage.GetPrimAtPath(tp)
                if not prim or not prim.IsValid():
                    continue
                mpath = f"/World/Looks/RandTable_{i}"
                mtl = UsdShade.Material.Define(self.stage, mpath)
                sh = UsdShade.Shader.Define(self.stage, mpath + "/Shader")
                sh.CreateIdAttr("UsdPreviewSurface")
                c = self.rng.uniform(0.15, 0.85, 3)
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                    float(self.rng.uniform(0.25, 0.95)))
                sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
                    float(self.rng.uniform(0.0, 0.25)))
                mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                # USD instancing: /World/envs/env_N/Table bir instance proxy ise
                # gercek geometri paylasilan prototipte kalir ve baglama islemez.
                # Instancing'i kapatip alttaki Mesh'lere de tek tek baglaniyoruz.
                if prim.IsInstance() or prim.IsInstanceable():
                    prim.SetInstanceable(False)
                api = UsdShade.MaterialBindingAPI.Apply(prim)
                api.Bind(mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                nmesh = 0
                for d in Usd.PrimRange(prim):
                    if d.GetTypeName() == "Mesh":
                        capi = UsdShade.MaterialBindingAPI.Apply(d)
                        capi.Bind(mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                        nmesh += 1
                if i == 0:
                    self._note(f"masa: instance={prim.IsInstance()} baglanan_mesh={nmesh}")
                ok_any = nmesh > 0
            if self._table_ok is None:
                self._table_ok = ok_any
                self._note("masa mesh baglama: " + ("mesh bulundu" if ok_any else "MESH BULUNAMADI")
                           + " (gorsel dogrulama ayrica yapilmali)")
        except Exception as e:
            self._table_ok = False
            self._note(f"masa randomizasyonu basarisiz: {e}")

    # ---------- 4) ORTAM BASINA LOKAL ISIK ----------
    def randomize_env_lights(self, origins):
        """Her ortamin masasi uzerine kendi kuresel lambasi. Global DomeLight tum
        ortamlar icin ortak oldugundan bolum-basi cesitlilik ancak boyle saglanir."""
        if not self.jitter_light:
            return
        try:
            from pxr import UsdLux, UsdGeom, Gf, Sdf
            org = origins.detach().cpu().numpy()
            for i in range(org.shape[0]):
                lp = f"/World/envs/env_{i}/rand_light"
                prim = self.stage.GetPrimAtPath(lp)
                if not prim or not prim.IsValid():
                    light = UsdLux.SphereLight.Define(self.stage, Sdf.Path(lp))
                else:
                    light = UsdLux.SphereLight(prim)
                    prim_x = UsdGeom.Xformable(prim)
                    prim_x.ClearXformOpOrder()
                light.CreateRadiusAttr(float(self.rng.uniform(0.05, 0.25)))
                light.CreateIntensityAttr(float(self.rng.uniform(1.5e4, 2.2e5)))
                c = self.rng.uniform(0.55, 1.0, 3)
                light.CreateColorAttr(Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))
                # masanin uzerinde rastgele konum
                off = np.array([self.rng.uniform(0.0, 0.9),
                                self.rng.uniform(-0.7, 0.7),
                                self.rng.uniform(0.9, 1.8)])
                pos = org[i] + off
                UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            if "lokal isik" not in " ".join(self._log):
                self._note(f"lokal isik: {org.shape[0]} ortam icin olusturuldu")
        except Exception as e:
            self._note(f"lokal isik basarisiz: {e}")

    # ---------- hepsi ----------
    def apply(self, cam, origins, device, num_envs, side_cam=None,
              side_eye=(0.50, -1.45, 0.70), side_tgt=(0.50, 0.0, 0.15)):
        self.randomize_camera(cam, origins, device)
        if side_cam is not None:
            # Yan kamera: optik ekseni y boyunca -> ON kameranin derinlik ekseni
            # olan x, bu goruste YANAL olur. Iki gorus birlikte ucgenleme saglar.
            self.randomize_camera(side_cam, origins, device, side_eye, side_tgt)
        self.randomize_light()
        self.randomize_env_lights(origins)
        self.randomize_table(num_envs)

    def _note(self, msg):
        if msg not in self._log:
            self._log.append(msg)
            print(f"[DR] {msg}", flush=True)

    def report(self):
        return {"masa_baglama": self._table_ok, "notlar": self._log}
