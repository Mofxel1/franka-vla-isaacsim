"""Surecin GERCEK VRAM kullanimini nvidia-smi ile ornekler (Kit+PhysX+RTX dahil)."""
import os, subprocess, threading, time

class VramProbe:
    def __init__(self, interval=0.25):
        self.pid, self.interval = os.getpid(), interval
        self.peak, self._run = 0, False
    def _loop(self):
        while self._run:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    p, m = [x.strip() for x in line.split(",")]
                    if int(p) == self.pid:
                        self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(self.interval)
    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self
    def stop(self):
        self._run = False
        return self.peak
