"""
TEK KOMUTLA BÜTÜN TESTLER.

    .venv/Scripts/python.exe -m tests.run_all
    .venv/Scripts/python.exe -m tests.run_all -q      # sadece özet

Neden ayrı bir koşucu: testler pytest'e bağımlı DEĞİL (her biri kendi main()'i
olan düz modül) — bu, kurulumu hafif tutar ama "hepsini koştur" komutunu da
ortadan kaldırıyordu. Bu dosya o boşluğu doldurur, yeni bağımlılık getirmez.

Her test ayrı bir alt-süreçte koşar: biri çökerse diğerleri etkilenmez ve
Windows konsol kodlaması (cp1254) yüzünden çıkan UnicodeEncodeError'lar
gerçek hatalarla karışmaz (alt süreçlere UTF-8 verilir).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def discover() -> list[str]:
    return sorted(f[:-3] for f in os.listdir(HERE)
                  if f.startswith("test_") and f.endswith(".py"))


def main() -> int:
    quiet = "-q" in sys.argv
    mods = discover()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    print(f"{len(mods)} test modülü bulundu.\n")

    for i, m in enumerate(mods, 1):
        started = time.time()
        p = subprocess.run([sys.executable, "-m", f"tests.{m}"],
                           cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        dt = time.time() - started
        ok = p.returncode == 0
        mark = "ok  " if ok else "FAIL"
        print(f"[{i:2d}/{len(mods)}] {mark} {m:38s} {dt:6.1f}s")
        if not ok:
            failed.append((m, (p.stdout + p.stderr).strip()))
        elif not quiet and p.stdout.strip():
            for line in p.stdout.strip().splitlines()[-2:]:
                print(f"          {line}")

    print(f"\n{'='*70}")
    if failed:
        for m, out in failed:
            print(f"\n--- BAŞARISIZ: {m} ---\n{out[-2500:]}")
        print(f"\nSONUÇ: {len(mods)-len(failed)}/{len(mods)} geçti, "
              f"{len(failed)} BAŞARISIZ ({time.time()-t0:.0f}s)")
        return 1
    print(f"SONUÇ: {len(mods)}/{len(mods)} test geçti ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
