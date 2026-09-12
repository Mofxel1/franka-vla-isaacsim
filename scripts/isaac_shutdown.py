"""Isaac Sim kapanis kilidi icin watchdog. close() 30sn icinde donmezse zorla cik."""
import os, threading

def safe_close(simulation_app, timeout=30.0):
    t = threading.Thread(target=simulation_app.close, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"[SHUTDOWN] close() {timeout}s icinde donmedi -> zorla cikis", flush=True)
    os._exit(0)
