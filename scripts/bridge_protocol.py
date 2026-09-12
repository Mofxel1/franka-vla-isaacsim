"""Isaac Sim (py3.10, Kit'in kendi eski numpy'si) <-> LeRobot politika
sunucusu (py3.12, numpy 2.x) arasinda basit uzunluk-onekli protokol.

DIKKAT: numpy dizilerini DOGRUDAN pickle'lamiyoruz. Sebep: Isaac Sim'in Kit
calisma zamani kendi paketledigi eski bir numpy surumunu (1.26.0) devreye
sokuyor ve bu surum, numpy 2.x'in urettigi pickle formatini (numpy._core
modul yoluna referans verir) okuyamiyor -> ModuleNotFoundError. Cozum:
diziyi ham bayt + dtype + shape olarak elle paketlemek; bu, hangi numpy
surumu calisirsa calissin ayni sekilde calisir (np.frombuffer + reshape)."""
import pickle, socket, struct
import numpy as np


def _encode(obj):
    if isinstance(obj, np.ndarray):
        return {"__nd__": True, "dtype": str(obj.dtype), "shape": obj.shape,
                "data": np.ascontiguousarray(obj).tobytes()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_encode(v) for v in obj)
    return obj


def _decode(obj):
    if isinstance(obj, dict):
        if obj.get("__nd__"):
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_decode(v) for v in obj)
    return obj


def send_msg(sock, obj):
    data = pickle.dumps(_encode(obj), protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    header = _recvall(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = _recvall(sock, length)
    if payload is None:
        return None
    return _decode(pickle.loads(payload))
