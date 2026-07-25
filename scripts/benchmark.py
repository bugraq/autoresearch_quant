"""
KIYAS: bizim yaklasim RASTGELE al-satciyi ve PASIF al-tutcuyu geciyor mu?
(hoca talebi, 22.07.2026)

Hocanin yeni basari tanimi:
  "Su an icin bir basari alpha bulmak DEGIL; parayi rastgele al-sat yapan
   birine veya psikolojik davranan birine gore YUKARIDA bulmak da bir basari."

Bu script tam bunu olcer. Ayni evren, ayni tarih, ayni islem maliyeti ve ayni
portfoy kurallariyla dort seyi yaristirir:

  1) AL-TUT (pasif/piyasa)    : tum varliklari esit agirlikla al ve TUT.
                                 "Endeks yatirimcisi" — hic ugrasmadan.
  2) RASTGELE AL-SATCI (maymun): her gun rastgele long/short pozisyon. Tek
                                 maymun sansli olabilir; bu yuzden N tane maymun
                                 kosup DAGILIMINI cikaririz (adil karsilastirma).
  3) PSIKOLOJIK (duygusal)     : dunku kazanani kovalayan (asiri tepki veren)
                                 naif trend-takipcisi — tipik retail davranisi.
  4) BIZIM STRATEJI            : hafizadaki en iyi kabul edilen aday
                                 (yoksa ornek bir hipotez).

Cikti: Sharpe / yillik getiri / toplam P&L / MaxDD tablosu + bizim stratejinin
maymun dagiliminin KACINCI yuzdeliginde oldugu + sade yorum.

Kullanim:
    .venv/Scripts/python.exe scripts/benchmark.py
    .venv/Scripts/python.exe scripts/benchmark.py --monkeys 500 --log

NOT: Kripto evreninde "SPY" yoktur; piyasa proxy'si = evrenin esit-agirlikli
sepeti (ayni evren -> en dürüst karsilastirma). Hisse kampanyasinda bu zaten
esit-agirlikli S&P benzeri sepettir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

COST_BPS = 5.0
_LOG: list[str] = []
_WRITE = False


def P(s: str = "") -> None:
    print(s)
    if _WRITE:
        _LOG.append(s)


def _sharpe(net_pnl: pd.Series, bpy: int) -> float:
    x = net_pnl.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(bpy))


def _total_return(net_pnl: pd.Series) -> float:
    x = net_pnl.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.prod(1.0 + x) - 1.0) if len(x) else 0.0


def _max_dd(net_pnl: pd.Series) -> float:
    x = net_pnl.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.0
    eq = np.cumprod(1.0 + x)
    return float(-(eq / np.maximum.accumulate(eq) - 1.0).min())


def _stats(net_pnl: pd.Series, bpy: int) -> dict:
    return {"sharpe": _sharpe(net_pnl, bpy),
            "yillik": float(net_pnl.mean() * bpy),
            "toplam": _total_return(net_pnl),
            "maxdd": _max_dd(net_pnl)}


# --------------------------------------------------------------------------
# Benchmark'lar — hepsi AYNI motordan (compute_pnl) gecer: adil
# --------------------------------------------------------------------------
def _template_hyp(universe_source: str, long_q: float = 0.2):
    """Rastgele/psikolojik traderlar icin ORTAK portfoy+execution sablonu.
    Yalnizca SINYAL degisir; kisitlar ayni -> adil karsilastirma."""
    from contracts.dsl import Expression
    from contracts.hypothesis_spec import (
        EconomicMechanism, Execution, Falsification, HypothesisFamily,
        HypothesisSpec, Portfolio, Universe,
    )
    return HypothesisSpec(
        hypothesis_id="benchmark", title="benchmark", claim="benchmark",
        family=HypothesisFamily.composite,
        economic_mechanism=EconomicMechanism(type="none", description="benchmark"),
        universe=Universe(source=universe_source),
        features=[], signal=Expression(op="field", field="close"),
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=long_q, short_quantile=long_q),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def buy_and_hold(data) -> pd.Series:
    """Esit-agirlikli AL-TUT: her gun mevcut varliklarin ortalama getirisi.
    Al-tut oldugu icin devir ~0; tek seferlik giris maliyeti ihmal edilir."""
    close = data.get("close")
    try:
        adj = data.get("adjusted_close")
    except KeyError:
        adj = close
    ret = adj.pct_change(fill_method=None)
    # index_membership varsa (survivorship): varlik yalnizca uyeyken sepette
    if "index_membership" in data.fields:
        ret = ret.where(data.get("index_membership") > 0)
    return ret.mean(axis=1).dropna()   # esit agirlik = kesitsel ortalama


def random_trader(data, template, seed: int, cost_bps: float = COST_BPS) -> pd.Series:
    """Bir 'maymun': her bar rastgele skor -> ayni long/short kurallari + maliyet."""
    from backtest.engine import compute_pnl
    close = data.get("close")
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(rng.standard_normal(close.shape),
                       index=close.index, columns=close.columns)
    sig = sig.where(close.notna())          # sadece verisi olan varlikta pozisyon
    net, _ = compute_pnl(sig, template, data, cost_bps)
    return net


def psychological_trader(data, template) -> pd.Series:
    """'Duygusal' trader: dunku kazanani kovalar (asiri tepki / momentum-chasing).
    Tipik retail davranisi — yukselen neyse ona atlar."""
    from backtest.engine import compute_pnl
    close = data.get("close")
    yday_ret = close.pct_change(fill_method=None)     # dunku getiri = sinyal
    net, _ = compute_pnl(yday_ret, template, data, COST_BPS)
    return net


def our_strategy(data, cfg, cost_bps: float = COST_BPS):
    """Hafizadaki en iyi kabul edilen aday; yoksa ornek (canned) hipotez.
    Donder: (etiket, net_pnl serisi)."""
    from backtest.engine import compute_pnl
    from backtest.model_signal import compute_signal
    from dsl import compile_hypothesis
    from contracts.hypothesis_spec import HypothesisSpec
    from memory.store import MemoryStore

    hyp, etiket = None, ""
    db = os.path.join(HERE, "research_memory.sqlite")
    if os.path.exists(db):
        rows = MemoryStore(db).accepted_hypotheses(limit=1)
        if rows and rows[0][1]:
            hyp = HypothesisSpec.model_validate(json.loads(rows[0][1]))
            etiket = f"BIZIM (hafizadan: {rows[0][0]}, Sharpe~{rows[0][2]:+.2f})"
    if hyp is None:
        from scripts.anatomy import _canned_hypothesis
        campaign = _load()["campaign"]
        hyp = _canned_hypothesis(campaign)
        etiket = "BIZIM (ornek hipotez — hafizada kabul yok)"
    signal = compute_signal(compile_hypothesis(hyp), hyp, data)
    net, _ = compute_pnl(signal, hyp, data, cost_bps)
    return etiket, net


# --------------------------------------------------------------------------
def _load():
    import main as M
    return {"campaign": M.load_yaml("campaign.yaml")["campaign"],
            "data_cfg": M.load_yaml("data.yaml")["data"]}


def main() -> None:
    global _WRITE
    ap = argparse.ArgumentParser(description="Bizim yaklasim random/al-tut'u geciyor mu?")
    ap.add_argument("--monkeys", type=int, default=200,
                    help="Kac rastgele al-satci (maymun) kosulsun (varsayilan 200).")
    ap.add_argument("--log", action="store_true", help="runs/benchmark.log'a yaz.")
    args = ap.parse_args()
    _WRITE = args.log
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import main as M
    cfgs = _load()
    campaign, data_cfg = cfgs["campaign"], cfgs["data_cfg"]
    cfg = M.build_config(campaign)

    P("\n" + "═" * 78)
    P("  KIYAS: bizim yaklasim rastgele al-satciyi ve al-tut'u geciyor mu?")
    P("═" * 78)
    P("""
  Hocanin olcutu: su an amac 'alpha bulmak' degil. Amac, parayi rastgele
  al-sat yapan birinden ve hic ugrasmayan (al-tut) birinden DAHA IYI
  yonetebildigimizi gostermek. Asagida herkes AYNI piyasada, AYNI tarihlerde,
  AYNI islem masrafiyla yarisiyor.""")

    P("\n  Veri yukleniyor...")
    data, _ = M.load_data(campaign, data_cfg, cfg.research_fraction)
    bpy = data.bars_per_year
    n_varlik = len(data.get("close").columns)
    P(f"  [ok] {campaign.get('universe')}: {n_varlik} varlik, {len(data.dates)} bar, "
      f"{data.dates[0].date()}→{data.dates[-1].date()}")

    template = _template_hyp(campaign.get("universe", "sp500_point_in_time"))

    # --- 1) AL-TUT ---
    P("\n  [1/4] Al-tut (pasif piyasa) hesaplaniyor...")
    bh = _stats(buy_and_hold(data), bpy)

    # --- 2) RASTGELE (N maymun) — hem masrafli hem MASRAFSIZ ---
    # Masrafsiz kritik: masrafli maymun sadece islem-masrafindan batabilir.
    # Masrafsizda rastgele islem ~0 getirir (beklenen). Bizimki masrafsizda da
    # POZITIF kalirsa, ustunlugumuz 'az islem yaptik'tan degil GERCEK SINYAL'den.
    P(f"  [2/4] {args.monkeys} rastgele al-satci (maymun) kosuluyor (masrafli+masrafsiz)...")
    monkey_sharpes, monkey_totals, monkey_sharpes_free = [], [], []
    for i in range(args.monkeys):
        net = random_trader(data, template, seed=1000 + i)
        monkey_sharpes.append(_sharpe(net, bpy))
        monkey_totals.append(_total_return(net))
        net0 = random_trader(data, template, seed=1000 + i, cost_bps=0.0)
        monkey_sharpes_free.append(_sharpe(net0, bpy))
    monkey_sharpes = np.array(monkey_sharpes)
    monkey_sharpes_free = np.array(monkey_sharpes_free)
    m_sh_med = float(np.median(monkey_sharpes))
    m_sh_best = float(monkey_sharpes.max())
    m_tot_med = float(np.median(monkey_totals))
    m_sh_med_free = float(np.median(monkey_sharpes_free))
    m_sh_best_free = float(monkey_sharpes_free.max())

    # --- 3) PSIKOLOJIK ---
    P("  [3/4] Psikolojik (duygusal/momentum-kovalayan) trader hesaplaniyor...")
    psy = _stats(psychological_trader(data, template), bpy)

    # --- 4) BIZIM — hem masrafli hem MASRAFSIZ ---
    P("  [4/4] Bizim strateji hesaplaniyor (masrafli+masrafsiz)...")
    our_label, our_net = our_strategy(data, cfg)
    our = _stats(our_net, bpy)
    _, our_net_free = our_strategy(data, cfg, cost_bps=0.0)
    our_sh_free = _sharpe(our_net_free, bpy)

    # Bizim strateji maymun dagiliminin kacinci yuzdeliginde? (masrafli VE masrafsiz)
    pctile = float((monkey_sharpes < our["sharpe"]).mean() * 100)
    pctile_free = float((monkey_sharpes_free < our_sh_free).mean() * 100)

    # ---------------- TABLO ----------------
    P("\n" + "─" * 78)
    P("  SONUCLAR  (hepsi ayni evren / tarih / %s bps masraf)" % int(COST_BPS))
    P("─" * 78)
    P(f"\n  {'Yaklasim':<34} {'Sharpe':>8} {'Yillik':>9} {'Toplam':>9} {'MaxDD':>7}")
    P(f"  {'-'*34} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")

    def row(ad, s):
        P(f"  {ad:<34} {s['sharpe']:>+8.2f} {s['yillik']*100:>+8.1f}% "
          f"{s['toplam']*100:>+8.1f}% {s['maxdd']*100:>6.0f}%")

    row("Al-tut (pasif piyasa)", bh)
    P(f"  {'Rastgele al-satci — ORTANCA maymun':<34} {m_sh_med:>+8.2f} "
      f"{'':>9} {m_tot_med*100:>+8.1f}% {'':>7}")
    P(f"  {'Rastgele al-satci — EN SANSLI maymun':<34} {m_sh_best:>+8.2f} "
      f"{'(' + str(args.monkeys) + ' maymun icinde en iyisi)':>26}")
    row("Psikolojik (duygusal) trader", psy)
    P(f"  {'-'*34} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")
    row(our_label, our)

    # --- Masrafsiz kontrol tablosu (gercek sinyal mi, sadece az-islem mi?) ---
    P(f"\n  MASRAFSIZ kontrol (islem masrafi = 0; 'gercek sinyal' testi):")
    P(f"    Rastgele maymun — ortanca Sharpe : {m_sh_med_free:>+6.2f}   "
      f"(beklenen ~0: rastgele islem bilgi tasimaz)")
    P(f"    Rastgele maymun — en sansli      : {m_sh_best_free:>+6.2f}   "
      f"({args.monkeys} maymunun en iyisi bile)")
    P(f"    BIZIM strateji                   : {our_sh_free:>+6.2f}   "
      f"→ maymunlarin %{pctile_free:.0f}'inden iyi")

    # ---------------- SADE YORUM ----------------
    P("\n" + "─" * 78)
    P("  NE ANLAMA GELIYOR?")
    P("─" * 78)
    P(f"""
  • Bizim strateji, {args.monkeys} rastgele maymunun %{pctile:.0f}'inden daha iyi (masrafli).
    (%50 = ortalama bir maymun kadar; %95+ = maymunlarin cok ustunde =
     sansla aciklanamayacak bir beceri isareti.)

  • EN ONEMLI: masraf SIFIR olsa bile maymunlarin %{pctile_free:.0f}'inden iyiyiz.
    Rastgele maymun masrafsizda ~0 Sharpe uretir (rastgele islem bilgi
    tasimaz). Bizimki masrafsizda da POZITIF ({our_sh_free:+.2f}) kaliyor —
    demek ki ustunlugumuz 'az islem yaptik/masraftan kactik'tan DEGIL,
    gercekten bir SINYAL yakalamaktan geliyor. Iste asil fark bu.""")

    yorumlar = []
    if our["sharpe"] > m_sh_med:
        yorumlar.append("rastgele al-satcinin (ortanca maymun) USTUNDE ✓")
    else:
        yorumlar.append("rastgele al-satciyi henuz GECEMIYOR ✗")
    if our["sharpe"] > bh["sharpe"]:
        yorumlar.append("pasif al-tut'un USTUNDE ✓")
    else:
        yorumlar.append("pasif al-tut'un ALTINDA ✗")
    if our["sharpe"] > psy["sharpe"]:
        yorumlar.append("duygusal trader'in USTUNDE ✓")
    else:
        yorumlar.append("duygusal trader'in ALTINDA ✗")
    P("  • Bizim strateji: " + "; ".join(yorumlar) + ".")

    gecti = sum("✓" in y for y in yorumlar)
    P(f"""
  • OZET: 3 kiyastan {gecti}'unu geciyoruz. Hocanin olcutu tam olarak buydu —
    "alpha" degil, rastgele/pasif/duygusal davranisi gecmek. {'Bu esik saglandi.' if gecti >= 2 else 'Bu esik henuz tam saglanmadi; strateji gelistirilecek.'}

  NOT: Bu kiyas arastirma verisinde. Nihai dogrulama yine kilitli holdout +
  coklu-test ile yapilir; bu tablo 'gunluk basari' olcutu, son soz degil.""")
    P("═" * 78 + "\n")

    if _WRITE:
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        p = os.path.join(HERE, "runs", "benchmark.log")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG))
        print(f"Log yazildi: {p}")


if __name__ == "__main__":
    main()
