"""
Walk-forward değerlendirme (Doküman 9.1).

Stratejilerimizin eğitilen parametresi yoktur (yapı LLM+DSL ile sabit), bu
yüzden walk-forward burada: AYNI stratejiyi ardışık zaman dilimlerinde ayrı
ayrı değerlendirip TUTARLILIĞA bakmaktır. Bir strateji sadece bir dönemde
değil, çoğu fold'da pozitif olmalı (rejimler arası kararlılık).

Falsification.minimum_positive_walk_forward_folds bu orana karşı kontrol edilir.
"""
from __future__ import annotations

import numpy as np

from contracts.backtest_result import BacktestResult, CostBreakdown
from contracts.hypothesis_spec import HypothesisSpec
from contracts.strategy_graph import StrategyGraph
from data.synthetic import MarketData
from backtest.engine import compute_pnl, fold_metrics


def _information_coefficient(signal, data: MarketData, horizon: int) -> dict:
    """IC/RankIC/ICIR — modelin SAYISAL tahmin kalitesi (Sharpe'tan AYRI, Qlib/RD-Agent).

    Günlük kesitsel korelasyon: sinyal_t ile t→t+horizon getirisi arasında.
    IC = ortalama Pearson, RankIC = ortalama Spearman (sıra), ICIR = IC/std (kararlılık).
    Yüksek Sharpe + ~0 IC = şans; yüksek Sharpe + anlamlı IC = gerçek öngörü.
    """
    import numpy as np
    from backtest.model_signal import forward_return
    fwd = forward_return(data, horizon).reindex_like(signal)
    ic_t = signal.corrwith(fwd, axis=1)                       # satır (gün) bazında Pearson
    rank_ic_t = signal.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)  # Spearman
    ic_t = ic_t.dropna()
    ic = float(ic_t.mean()) if len(ic_t) else 0.0
    std = float(ic_t.std())

    # DIRECTIONAL ACCURACY (ML accuracy'sinin finans analoğu): model "hangi hisse
    # kesitsel ORTALAMANIN üstünde olacak" tahmininde ne kadar haklı? Kesitsel
    # demean ile piyasa drifti çıkarılır (long-short için doğru ölçü). Baseline %50.
    # NOT: finansta %52-55 bile güçlüdür; ~%97 görürsen SIZINTI şüphelen.
    sig_c = signal.sub(signal.mean(axis=1), axis=0)
    fwd_c = fwd.sub(fwd.mean(axis=1), axis=0)
    hit = (np.sign(sig_c) == np.sign(fwd_c))
    mask = sig_c.notna() & fwd_c.notna() & (sig_c != 0) & (fwd_c != 0)
    n_hit = int(mask.to_numpy().sum())
    dir_acc = float(hit.where(mask).to_numpy()[mask.to_numpy()].mean()) if n_hit else 0.5

    return {
        "ic": ic,
        "rank_ic": float(rank_ic_t.dropna().mean()) if rank_ic_t.notna().any() else 0.0,
        "icir": float(ic / std) if std > 0 else 0.0,
        "dir_acc": dir_acc,
    }


def run_walk_forward(graph: StrategyGraph, hyp: HypothesisSpec, data: MarketData,
                     n_folds: int = 5, cost_bps: float = 5.0, seed: int = 42,
                     signal=None) -> BacktestResult:
    if signal is None:
        from backtest.model_signal import compute_signal
        signal = compute_signal(graph, hyp, data)   # model-farkında (dsl_formula|model)
    net_pnl, turnover_t = compute_pnl(signal, hyp, data, cost_bps)

    idx = net_pnl.index
    chunks = np.array_split(np.arange(len(idx)), n_folds)
    folds = []
    for k, ch in enumerate(chunks):
        if len(ch) < 20:   # çok kısa fold'u atla
            continue
        seg = net_pnl.iloc[ch[0]:ch[-1] + 1]
        folds.append(fold_metrics(seg, turnover_t, f"fold_{k}", "validation",
                                  bars_per_year=getattr(data, "bars_per_year", 252)))

    positive_frac = (sum(1 for f in folds if f.sharpe > 0) / len(folds)) if folds else 0.0

    ic_metrics = _information_coefficient(signal, data, hyp.execution.holding_period_days)

    return BacktestResult(
        hypothesis_id=hyp.hypothesis_id,
        per_fold_metrics=folds,
        net_returns=[float(x) for x in net_pnl.to_numpy()],
        cost_breakdown=CostBreakdown(spread=float((turnover_t.shift(1) * (cost_bps/1e4)).sum())),
        exposures={"positive_fold_fraction": positive_frac, "n_folds": len(folds),
                   "model_type": hyp.model.type, **ic_metrics},
        data_version=getattr(data, "version", "synthetic-v0"),
        engine_version="v0.2-walk-forward",
        seed=seed,
    )
