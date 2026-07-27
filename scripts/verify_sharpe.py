"""
SHARPE DOĞRULAMA — "gerçekten doğru mu?" (hoca talebi, 22.07.2026)

Motorun ürettiği Sharpe'ı ÜÇ bağımsız yolla hesaplar ve karşılaştırır:

  A) Motor        : backtest.engine.fold_metrics()        (pandas)
  B) Elle/NumPy   : bu dosyada, sıfırdan, pandas'sız      (bağımsız hesap)
  C) Excel        : ara adımlar tabloya yazılır; Excel'de
                    =ORTALAMA(...)/=STDSAPMA(...)*KAREKÖK(252) ile elle kontrol

Ayrıca PnL zincirinin HER adımını (sinyal -> ağırlık -> getiri -> brüt -> maliyet
-> net) tek bir gün için açıp gösterir; sayıların nereden geldiği gözle görülür.

Kullanım:
    .venv/Scripts/python.exe scripts/verify_sharpe.py

Çıktı: konsol raporu + runs/sharpe_verification.xlsx (yoksa .csv)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import _build_weights, _execution_prices, compute_pnl, fold_metrics
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from data import gen_cross_sectional_momentum
from dsl import compile_hypothesis
from backtest import evaluate_signal

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
COST_BPS = 5.0


def _line(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def build_case():
    """Küçük, deterministik bir örnek: 8 varlık x 260 gün, momentum sinyali."""
    # 8 varlık: tek gün tablosu tek ekrana sığsın. 60g pencere: veriye gömülü
    # momentum ufkuyla aynı (tests/test_backtest.py ile tutarlı).
    data = gen_cross_sectional_momentum(n_sec=8, n_days=750, seed=1)
    signal_expr = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60,
                   inputs=[Expression(op="field", field="close")])])
    hyp = HypothesisSpec(
        hypothesis_id="hyp_verify", title="60g momentum",
        claim="Son 20 gunun kazananlari kisa vadede kazanmaya devam eder.",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(
            type="underreaction", description="Yatirimci haberlere gec tepki verir."),
        universe=Universe(source="sp500_point_in_time"),
        features=[], signal=signal_expr,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())
    return hyp, data


# --------------------------------------------------------------------------
# 1) ZİNCİRİ TEK GÜN ÜZERİNDE AÇ — sayı nereden geliyor?
# --------------------------------------------------------------------------
def trace_one_day(hyp, data, graph) -> None:
    _line("1) PnL ZINCIRI — TEK GUN UZERINDE ACIK HESAP")

    signal = evaluate_signal(graph, data)
    weights = _build_weights(signal, hyp.portfolio)
    exec_px = _execution_prices(data, "open")
    exec_ret = exec_px.pct_change(fill_method=None)

    # Sinyalin dolu olduğu, ortalarda bir gün seç
    valid = signal.dropna(how="all").index
    t = valid[len(valid) // 2]
    ti = data.dates.get_loc(t)
    t1, t2 = data.dates[ti + 1], data.dates[ti + 2]

    print(f"\nSecilen gun t = {t.date()}   (t+1 = {t1.date()}, t+2 = {t2.date()})")
    print("\nBeyan: signal_time=close_t, trade_time=open_t_plus_1, holding=1 gun")
    print("Yani  : sinyal t'nin KAPANISINDA hesaplanir, pozisyon t+1 ACILISINDA")
    print("        kurulur ve open(t+1) -> open(t+2) getirisini kazanir.\n")

    tbl = pd.DataFrame({
        "sinyal_t (cs_rank)": signal.loc[t],
        "agirlik w_t": weights.loc[t],
        "open(t+1)": exec_px.loc[t1],
        "open(t+2)": exec_px.loc[t2],
        "getiri r(t+1->t+2)": exec_ret.loc[t2],
    })
    tbl["katki = w_t * r"] = tbl["agirlik w_t"] * tbl["getiri r(t+1->t+2)"]
    print(tbl.round(6).to_string())

    manual_gross = float(tbl["katki = w_t * r"].sum())
    print(f"\n  ELLE brut PnL (sutun toplami)          = {manual_gross:+.8f}")

    # Motorun aynı gün için ürettiği brüt PnL
    engine_gross = float((weights.shift(2) * exec_ret).sum(axis=1).loc[t2])
    print(f"  MOTORUN brut PnL (weights.shift(2))    = {engine_gross:+.8f}")
    print(f"  FARK                                   = {abs(manual_gross - engine_gross):.2e}")
    assert abs(manual_gross - engine_gross) < 1e-12, "Hizalama uyusmuyor!"
    print("  [OK] Motor, beyan edilen islem anini birebir uyguluyor.")

    # Maliyet
    turn_t = float((weights - weights.shift(1)).abs().sum(axis=1).loc[t])
    cost = turn_t * COST_BPS / 1e4
    print(f"\n  Devir (turnover) t                     = {turn_t:.6f}")
    print(f"  Maliyet = devir * {COST_BPS:.0f}bps            = {cost:.8f}")
    print(f"  NET PnL = brut - maliyet               = {manual_gross - cost:+.8f}")
    return signal, weights, exec_ret


# --------------------------------------------------------------------------
# 2) SHARPE — üç bağımsız hesap
# --------------------------------------------------------------------------
def verify_sharpe(hyp, data, graph, signal):
    _line("2) SHARPE — UC BAGIMSIZ HESAP")

    net_pnl, turnover_t = compute_pnl(signal, hyp, data, COST_BPS)
    bpy = getattr(data, "bars_per_year", 252)

    # --- A) Motor
    fm = fold_metrics(net_pnl, turnover_t, "fold_0", "verify", bars_per_year=bpy)
    sharpe_engine = fm.sharpe

    # --- B) Elle / saf NumPy (pandas'a hic dokunmadan)
    x = np.array([float(v) for v in net_pnl.to_numpy()], dtype=float)
    n = len(x)
    mean = sum(x) / n                                   # ortalama
    var = sum((xi - mean) ** 2 for xi in x) / (n - 1)   # ORNEKLEM varyansi (ddof=1)
    std = math.sqrt(var)
    sharpe_manual = mean / std * math.sqrt(bpy)

    # Populasyon varyansi (ddof=0) ile ne olurdu? — yaygin hata kaynagi
    var0 = sum((xi - mean) ** 2 for xi in x) / n
    sharpe_ddof0 = mean / math.sqrt(var0) * math.sqrt(bpy)

    print(f"\n  Gozlem sayisi n                = {n}")
    print(f"  Gunluk ortalama getiri         = {mean:+.10f}")
    print(f"  Gunluk std sapma (ddof=1)      = {std:.10f}")
    print(f"  Yilliklastirma  sqrt({bpy})    = {math.sqrt(bpy):.6f}")
    print(f"\n  A) MOTOR    (pandas)           = {sharpe_engine:+.10f}")
    print(f"  B) ELLE     (saf NumPy/math)   = {sharpe_manual:+.10f}")
    print(f"     FARK                        = {abs(sharpe_engine - sharpe_manual):.2e}")

    ok = abs(sharpe_engine - sharpe_manual) < 1e-9
    print(f"  {'[OK] Motorun Sharpe hesabi DOGRU.' if ok else '[HATA] Sharpe uyusmuyor!'}")

    print(f"\n  (Bilgi) ddof=0 kullansaydik    = {sharpe_ddof0:+.10f}"
          f"   -> fark {abs(sharpe_ddof0 - sharpe_manual):.2e}")
    print("  pandas .std() varsayilani ddof=1 (ornek), numpy .std() ddof=0 (populasyon).")
    print("  Ikisi karistirilirsa Sharpe kayar; motor pandas .std() (ddof=1) kullaniyor.")

    # --- Yillik getiri / volatilite / MaxDD elle
    ann_ret = mean * bpy
    ann_vol = std * math.sqrt(bpy)
    equity = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(equity)
    mdd = float(-(equity / peak - 1.0).min())
    print(f"\n  Yillik getiri  elle = {ann_ret:+.6f}   motor = {fm.annualized_return:+.6f}")
    print(f"  Yillik vol     elle = {ann_vol:.6f}   motor = {fm.volatility:.6f}")
    print(f"  Max drawdown   elle = {mdd:.6f}   motor = {fm.max_drawdown:.6f}")
    print(f"  Sharpe = yillik getiri / yillik vol = {ann_ret / ann_vol:+.6f}  (tutarlilik)")

    return net_pnl, turnover_t, {
        "n": n, "mean": mean, "std": std, "bars_per_year": bpy,
        "sharpe_engine": sharpe_engine, "sharpe_manual": sharpe_manual,
        "ann_return": ann_ret, "ann_vol": ann_vol, "max_drawdown": mdd,
    }


# --------------------------------------------------------------------------
# 3) EXCEL ÇIKTISI — hoca kendi eliyle kontrol edebilsin
# --------------------------------------------------------------------------
def export(net_pnl, turnover_t, stats) -> None:
    _line("3) EXCEL CIKTISI — ELLE KONTROL ICIN")
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.DataFrame({
        "tarih": net_pnl.index,
        "net_getiri": net_pnl.to_numpy(),
    })
    df["devir"] = turnover_t.reindex(net_pnl.index).to_numpy()
    df["kumulatif_deger"] = (1.0 + df["net_getiri"]).cumprod()

    bpy = stats["bars_per_year"]
    ozet = pd.DataFrame({
        "olcut": ["gozlem (n)", "ortalama", "std sapma (ddof=1)", "bar/yil",
                  "SHARPE (motor)", "SHARPE (elle)", "yillik getiri", "yillik vol",
                  "max drawdown"],
        "deger": [stats["n"], stats["mean"], stats["std"], bpy,
                  stats["sharpe_engine"], stats["sharpe_manual"],
                  stats["ann_return"], stats["ann_vol"], stats["max_drawdown"]],
        "EXCEL'de nasil dogrularsin": [
            "=BAGDAS_SAY(B:B)  /  =COUNT(B:B)",
            "=ORTALAMA(veri!B:B)  /  =AVERAGE(...)",
            "=STDSAPMA(veri!B:B)  /  =STDEV(...)   <- ORNEKLEM, ddof=1",
            f"gunluk hisse=252, kripto=365, 8h=1095  (burada {bpy})",
            f"=ORTALAMA(veri!B:B)/STDSAPMA(veri!B:B)*KAREKOK({bpy})",
            "ayni formul (motorla ayni cikmali)",
            f"=ORTALAMA(veri!B:B)*{bpy}",
            f"=STDSAPMA(veri!B:B)*KAREKOK({bpy})",
            "=1-MIN(veri!D:D/MAKS_ONCEKI)  (veya D sutunu grafigi)",
        ],
    })

    xlsx = os.path.join(OUT_DIR, "sharpe_verification.xlsx")
    try:
        with pd.ExcelWriter(xlsx) as w:
            ozet.to_excel(w, sheet_name="ozet", index=False)
            df.to_excel(w, sheet_name="veri", index=False)
        print(f"\n  Yazildi: {xlsx}")
        print("  -> 'ozet' sayfasindaki formulleri 'veri' sayfasina uygulayip")
        print("     motorun sayisiyla karsilastirabilirsin.")
    except Exception as e:  # openpyxl yoksa CSV'ye dus
        csv1 = os.path.join(OUT_DIR, "sharpe_verification_veri.csv")
        csv2 = os.path.join(OUT_DIR, "sharpe_verification_ozet.csv")
        df.to_csv(csv1, index=False)
        ozet.to_csv(csv2, index=False)
        print(f"\n  (Excel yazilamadi: {e})")
        print(f"  Yazildi: {csv1}\n  Yazildi: {csv2}")


def main() -> None:
    # Windows konsolu (cp1254) özel karakterlerde patlamasın.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    _line("SHARPE DOGRULAMA RAPORU — motor gercekten dogru mu?")
    hyp, data = build_case()
    graph = compile_hypothesis(hyp)
    print(f"\nOrnek     : {hyp.title} ({hyp.hypothesis_id})")
    print(f"Veri      : {len(data.dates)} gun x {len(data.get('close').columns)} varlik "
          f"(sentetik, seed=1, momentum gomulu)")
    print(f"Portfoy   : {hyp.portfolio.type}, long/short %{hyp.portfolio.long_quantile*100:.0f}")
    print(f"Maliyet   : {COST_BPS:.0f} bps | Yilliklastirma: {data.bars_per_year} bar/yil")

    signal, _, _ = trace_one_day(hyp, data, graph)
    net_pnl, turnover_t, stats = verify_sharpe(hyp, data, graph, signal)
    export(net_pnl, turnover_t, stats)
    _line("BITTI")


if __name__ == "__main__":
    main()
