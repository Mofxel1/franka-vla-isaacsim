"""Gercek model yerine sabit aksiyon donduren sahte sunucu -- SADECE
eval_policy_isaacsim.py'nin Isaac Sim tarafini test etmek icin."""
import sys, os, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_protocol import send_msg, recv_msg
import numpy as np

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 8766)); srv.listen(1)
print("[SAHTE SUNUCU] hazir", flush=True)
conn, _ = srv.accept()
n = 0
while True:
    msg = recv_msg(conn)
    if msg is None: break
    if msg.get("cmd") == "shutdown":
        send_msg(conn, {"ok": True}); break
    # sabit: kolu yerinde tut, gripper acik
    action = np.array([0.45, 0.0, 0.4, 0, 1, 0, 0, 1.0], dtype=np.float32)
    send_msg(conn, {"action": action})
    n += 1
conn.close(); srv.close()
print(f"[SAHTE SUNUCU] bitti, {n} cagri islendi", flush=True)
