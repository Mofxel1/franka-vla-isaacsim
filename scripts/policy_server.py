"""
SmolVLA politika sunucusu.

Isaac Sim'in (Python 3.10, isaaclab ortami) her adimda gonderdigi gozlemi
alir, egitilmis SmolVLA modeliyle aksiyon uretir, geri gonderir. Ayri
surecte calisir cunku lerobot (Python 3.12+) ile Isaac Sim (Python 3.10)
ayni conda ortaminda bulunamiyor.

Kullanim:
  conda activate lerobot
  python policy_server.py --ckpt /path/to/checkpoints/004000/pretrained_model
"""
import argparse, os, socket, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_protocol import send_msg, recv_msg

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="pretrained_model klasoru")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--n_action_steps", type=int, default=None,
                    help="model tek goruntuden uretilen plandan KAC adim uygulasin. "
                         "Varsayilan 50 (=1 saniye gozu kapali). Dusurmek modelin daha "
                         "sik goruntuye bakip hizasini duzeltmesini saglar. "
                         "Yeniden egitim GEREKTIRMEZ.")
parser.add_argument("--n_samples", type=int, default=1,
                    help="Her planlamada kac ornek cekip ORTALAMASINI alsin. "
                         "SmolVLA akis eslestirme tabanli: her cagri gurultuden "
                         "farkli bir plan ornekliyor. Cevrimdisi olcumlerde K=8 "
                         "ortalama aliniyordu, sunucu ise TEK ornek cekiyordu -- "
                         "olculdu: plan medyani 18.9mm ama ortalama 95.6mm, "
                         "maks 761mm. Yani planlarin cogu iyi, arada bir felaket "
                         "ornek geliyor ve kapali donguda bir kotu parca bolumu "
                         "bitiriyor. Ortalama almak bu ucari kirpar. Onerilen: 8.")
parser.add_argument("--zero_state", action="store_true", default=True,
                    help="Modele durum vektorunu SIFIR gonder. Veri setleri "
                         "--zero_state ile uretiliyor (durum sizinti tasiyordu), "
                         "ama sunucu GERCEK robot durumunu gonderiyordu. Olculdu "
                         "(2026-09-08, train_geo3): sifir 26.4mm vs gercek 31.2mm. "
                         "Kucuk ama bedava; egitim/dagitim tutarsizligini kaldirir. "
                         "Goreli->mutlak cevrimi icin gercek state ayrica kullanilir.")
parser.add_argument("--object_centric", action="store_true",
                    help="Model x,y'yi KUPE gore uretti: mutlak hedef = aksiyon + "
                         "modelin KENDI kup tahmini (yardimci boyutlar 8,9). z mutlak. "
                         "Algi boylece yapisal olarak donguye girer.")
parser.add_argument("--relative_actions", action="store_true",
                    help="model GORELI aksiyon (delta) uretiyorsa ac: cikti,\n"
                         "istemcinin gonderdigi mevcut EE konumuna EKLENIR.")
args = parser.parse_args()

print(f"[SUNUCU] checkpoint yukleniyor: {args.ckpt}", flush=True)
device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

policy = SmolVLAPolicy.from_pretrained(args.ckpt)
policy.eval()
# ONEMLI: from_pretrained modeli checkpoint config'indeki cihaza (genelde cuda)
# yukluyor. Istenen cihaza ACIKCA tasimazsak, --device cpu verildiginde girdiler
# CPU'da model GPU'da kalir ve su hata gelir:
#   "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)..."
policy.to(device)
policy.config.device = device
if args.n_action_steps is not None:
    policy.config.n_action_steps = args.n_action_steps
    print(f"[SUNUCU] n_action_steps -> {args.n_action_steps} "
          f"(chunk_size={policy.config.chunk_size})", flush=True)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=args.ckpt,
    preprocessor_overrides={"device_processor": {"device": device}})
print(f"[SUNUCU] hazir | cihaz={device}", flush=True)
print(f"[SUNUCU] girdi ozellikleri : {list(policy.config.input_features.keys())}", flush=True)
print(f"[SUNUCU] cikti ozellikleri : {list(policy.config.output_features.keys())}", flush=True)
# NOT: bu satir eskiden sadece --relative_actions'i raporluyordu ve
# --object_centric acikken bile "MUTLAK" yaziyordu -- 2026-09-08'de yanlis
# alarma sebep oldu. Artik ucunu de gosteriyor.
_bicim = ("GORELI (delta -> mutlak cevrilir)" if args.relative_actions else
          "NESNE-MERKEZLI (x,y kupe gore; modelin kendi tahminiyle mutlaklastirilir)"
          if args.object_centric else "MUTLAK")
print(f"[SUNUCU] aksiyon bicimi   : {_bicim}", flush=True)
print(f"[SUNUCU] plan ornegi      : {args.n_samples} "
      f"{'(ORTALAMA alinir)' if args.n_samples > 1 else '(tek cekilis)'}", flush=True)


def to_chw_float(img_uint8):
    """(H,W,3) -> (1,3,H,W) float32 [0,1];  (B,H,W,3) -> (B,3,H,W)  (toplu mod)"""
    a = np.ascontiguousarray(img_uint8)
    t = torch.from_numpy(a).float() / 255.0
    if t.ndim == 4:
        return t.permute(0, 3, 1, 2)
    return t.permute(2, 0, 1).unsqueeze(0)


# Kendi aksiyon kuyrugumuz: n_samples>1 iken plani ORTALAYIP kuyruga koyariz.
# (policy.select_action kendi kuyrugunu tutar ama tek ornek ceker.)
_queue = []


def _plan(processed):
    """n_samples kez chunk uret, ORTALA. (chunk_size, action_dim) dondurur."""
    chunks = []
    for _ in range(max(1, args.n_samples)):
        ch = policy.predict_action_chunk(processed)
        chunks.append(postprocessor(ch)[0].cpu().numpy())
    return np.mean(chunks, axis=0)


_bqueue = None      # (B, n_exec, dim) -- toplu mod icin sunucu tarafi plan kuyrugu
_bpos = 0


def handle_batch(obs):
    """DAgger icin TOPLU cikarim: B ortamin gozlemi bir kerede islenir.

    Neden: DAgger'da her ortam BAGIMSIZ veri uretir (eval'de oldugu gibi tek
    ortamin aksiyonunu hepsine yayinlamak yerine). Tek tek sorulursa 8 ortam
    8 kat yavas olur. Burada tek forward gecisiyle B plan uretilir.

    Plan kuyrugu sunucuda tutulur ve HERHANGI bir ortam sifirlandiginda ya da
    kuyruk bittiginde HEPSI icin yeniden planlanir (fazladan hesap, ama
    sifirlanan ortam taze gozlemden plan almis olur -- dogru olan bu).
    """
    global _bqueue, _bpos
    st = np.ascontiguousarray(obs["state"])                 # (B, 16)
    B = st.shape[0]
    reset_any = bool(np.any(obs.get("reset", False)))

    if _bqueue is None or _bpos >= _bqueue.shape[1] or reset_any or _bqueue.shape[0] != B:
        batch = {
            "observation.images.front": to_chw_float(obs["front"]),
            "observation.images.wrist": to_chw_float(obs["wrist"]),
            "observation.state": torch.from_numpy(
                np.zeros_like(st) if args.zero_state else st).float(),
            "task": [obs["task"]] * B,
        }
        if "side" in obs:
            batch["observation.images.side"] = to_chw_float(obs["side"])
        with torch.no_grad():
            policy.reset()
            ch = postprocessor(policy.predict_action_chunk(preprocessor(batch)))
        _bqueue = ch.cpu().numpy()                          # (B, chunk, dim)
        _bpos = 0
        n_exec = args.n_action_steps or _bqueue.shape[1]
        _bqueue = _bqueue[:, :min(n_exec, _bqueue.shape[1])]

    out = _bqueue[:, _bpos].copy()                          # (B, dim)
    _bpos += 1

    if args.object_centric:
        if out.shape[1] < 10:
            raise RuntimeError("--object_centric icin yardimci boyutlar (8,9) sart")
        out[:, 0] += out[:, 8]
        out[:, 1] += out[:, 9]
    elif args.relative_actions:
        out[:, :3] = out[:, :3] + st[:, 9:12]
    if out.shape[1] > 8:
        handle.last_cube_pred = out[:, 8:10].copy()
        out = out[:, :8]
    return out


def handle(obs):
    # Toplu mod: goruntu 4 boyutluysa (B,H,W,C) her ortam kendi aksiyonunu alir
    if np.asarray(obs["front"]).ndim == 4:
        return handle_batch(obs)
    state = np.ascontiguousarray(obs["state"])
    batch = {
        "observation.images.front": to_chw_float(obs["front"]),
        "observation.images.wrist": to_chw_float(obs["wrist"]),
        "observation.state": torch.from_numpy(
            np.zeros_like(state) if args.zero_state else state).float().unsqueeze(0),
        "task": [obs["task"]],
    }
    if "side" in obs:
        batch["observation.images.side"] = to_chw_float(obs["side"])
    processed = preprocessor(batch)
    global _queue
    with torch.no_grad():
        if obs.get("reset", False):
            policy.reset()
            _queue = []
        if args.n_samples <= 1:
            out = postprocessor(policy.select_action(processed))[0].cpu().numpy().copy()
        else:
            if not _queue:
                chunk = _plan(processed)
                n_exec = args.n_action_steps or chunk.shape[0]
                _queue = [chunk[i] for i in range(min(n_exec, chunk.shape[0]))]
            out = _queue.pop(0).copy()
    if args.object_centric:
        # x,y kupe gore -> modelin kendi kup tahminiyle mutlaklastir. z zaten mutlak.
        if out.shape[0] < 10:
            raise RuntimeError("--object_centric icin yardimci boyutlar (8,9) sart; "
                               "veri seti --aux_cube ile uretilmis olmali")
        out[0] += out[8]
        out[1] += out[9]
    elif args.relative_actions:
        # Model delta uretti; ortam MUTLAK EE hedefi bekliyor.
        # state duzeni: [0:9]=eklem acilari, [9:12]=ee_pos, [12:16]=ee_quat
        out[:3] = out[:3] + state[9:12]
    if out.shape[0] > 8:
        # YARDIMCI GOREV boyutlari (kup x,y). Ortam 8 boyut bekliyor.
        # Tani icin loglanir, ortama gonderilmez.
        handle.last_cube_pred = out[8:10].copy()
        out = out[:8]
    return out
handle.last_cube_pred = None


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((args.host, args.port))
srv.listen(1)
print(f"[SUNUCU] dinleniyor: {args.host}:{args.port}", flush=True)

try:
    while True:
        conn, addr = srv.accept()
        print(f"[SUNUCU] baglanti: {addr}", flush=True)
        t_start = time.time()
        n_calls = 0
        try:
            with conn:
                while True:
                    msg = recv_msg(conn)
                    if msg is None:
                        break
                    if msg.get("cmd") == "shutdown":
                        send_msg(conn, {"ok": True})
                        raise SystemExit
                    action = handle(msg)
                    resp = {"action": action}
                    if handle.last_cube_pred is not None:
                        # Modelin KENDI kup tahmini (yardimci boyutlar 8,9).
                        # Kapali donguda algiyi eylemden ayirmak icin loglanir.
                        resp["cube_pred"] = handle.last_cube_pred
                    send_msg(conn, resp)
                    n_calls += 1
        except (ConnectionResetError, BrokenPipeError):
            pass
        print(f"[SUNUCU] baglanti kapandi | {n_calls} cagri | {time.time()-t_start:.1f}s", flush=True)
except (KeyboardInterrupt, SystemExit):
    print("\n[SUNUCU] kapatiliyor", flush=True)
