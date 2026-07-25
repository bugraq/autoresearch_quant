"""
Backtest motoru ÇAPRAZ-DOĞRULAMA — motorun doğruluğunu bağımsız hesapla kanıtlar.

`backtest/engine.py` compute_pnl ile `backtest_service/reference.py` reference_backtest
AYNI stratejiyi AYRI kodlarla hesaplar. Sonuçlar tutmalı (çift kayıt muhasebesi).
Motorun tüm proje sonucunun dayandığı bileşen olduğu için bu doğrulama kritiktir.
"""
from __future__ import annotations

import numpy as np

from contracts.hypothesis_spec import HypothesisSpec
from dsl import compile_hypothesis
from backtest.engine import compute_pnl
from backtest.evaluator import evaluate_signal
from backtest_service.reference import reference_backtest
from data.synthetic import gen_cross_sectional_momentum, gen_short_term_reversal

TRADING_DAYS = 252


def _spec(signal, family="momentum"):
    return HypothesisSpec.model_validate({
        "hypothesis_id": "cv", "title": "capraz", "claim": "c", "family": family,
        "economic_mechanism": {"type": "x", "description": "y"},
        "universe": {"source": "sp500_point_in_time"},
        "features": [], "signal": signal, "model": {"type": "dsl_formula"},
        "portfolio": {"type": "cross_sectional_long_short", "long_quantile": 0.2,
                      "short_quantile": 0.2, "weighting": "equal"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": 1, "rebalance": "daily"},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })


def _engine_sharpe(net):
    return float(net.mean() / net.std() * np.sqrt(TRADING_DAYS)) if net.std() > 0 else 0.0


def _check(signal_expr, data, family):
    hyp = _spec(signal_expr, family)
    graph = compile_hypothesis(hyp)
    signal = evaluate_signal(graph, data)

    engine_net, _turn = compute_pnl(signal, hyp, data, cost_bps=5.0)      # MOTOR
    ref_sharpe, ref_net = reference_backtest(signal, data, 0.2, 0.2, 5.0)  # BAĞIMSIZ

    idx = engine_net.index.intersection(ref_net.index)
    assert len(idx) > 100, "karşılaştırılacak ortak gün az"
    a = engine_net.reindex(idx).to_numpy()
    b = ref_net.reindex(idx).to_numpy()
    # Günlük getiri serileri BİREBİR aynı olmalı (aynı spesifikasyon, ayrı kod)
    assert np.allclose(a, b, atol=1e-10, equal_nan=True), \
        f"net getiri serileri tutmuyor! maks fark={np.nanmax(np.abs(a-b)):.2e}"
    eng_sh = _engine_sharpe(engine_net)
    assert abs(eng_sh - ref_sharpe) < 1e-6, \
        f"Sharpe tutmuyor: motor={eng_sh:.4f} referans={ref_sharpe:.4f}"
    return eng_sh, ref_sharpe


def test_crossval_momentum():
    data = gen_cross_sectional_momentum(n_sec=40, n_days=900, seed=1, drift_spread=0.003)
    sig = {"op": "cross_sectional_rank",
           "inputs": [{"op": "return", "window": 20, "inputs": [{"op": "field", "field": "close"}]}]}
    eng, ref = _check(sig, data, "momentum")
    print(f"  [ok] momentum: motor Sharpe={eng:+.3f} = referans {ref:+.3f} (birebir)")


def test_crossval_reversal():
    data = gen_short_term_reversal(n_sec=40, n_days=900, seed=2)
    sig = {"op": "cross_sectional_rank", "inputs": [
        {"op": "negate", "inputs": [
            {"op": "return", "window": 5, "inputs": [{"op": "field", "field": "close"}]}]}]}
    eng, ref = _check(sig, data, "reversal")
    print(f"  [ok] reversal: motor Sharpe={eng:+.3f} = referans {ref:+.3f} (birebir)")


def main():
    test_crossval_momentum()
    test_crossval_reversal()
    print("OK — motor bağımsız referansla ÇAPRAZ-DOĞRULANDI (iki hesap birebir aynı).")


if __name__ == "__main__":
    main()
