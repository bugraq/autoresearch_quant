"""
Backtest Servisi testleri (hocanın ilk hedefi).

Doğrulanan: (1) formül / istatistiksel / ML modellerinin HEPSİ tek çağrıdan geçer;
(2) sızıntılı strateji ÇALIŞTIRILMAZ (sayı üretilmez); (3) rapor tahmin kalitesi +
performans + kökeni taşır; (4) servis LLM/orchestrator'dan BAĞIMSIZ (servis sınırı).
"""
from __future__ import annotations

import subprocess
import sys

from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import gen_cross_sectional_momentum
from backtest_service import BacktestServiceError, LeakageError, run

_DATA = gen_cross_sectional_momentum(n_sec=40, n_days=800, seed=1, drift_spread=0.003)


def _spec(model_type: str, trade_time: str = "open_t_plus_1") -> HypothesisSpec:
    return HypothesisSpec.model_validate({
        "hypothesis_id": f"hyp_{model_type}", "title": "servis testi",
        "claim": "geçmiş getiri geleceği öngörür", "family": "momentum",
        "economic_mechanism": {"type": "trend", "description": "momentum devamı"},
        "universe": {"source": "sp500_point_in_time"},
        "features": [
            {"name": "mom20", "expression":
                {"op": "return", "window": 20, "inputs": [{"op": "field", "field": "close"}]}},
            {"name": "mom60", "expression":
                {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}},
        ],
        "signal": {"op": "cross_sectional_rank",
                   "inputs": [{"op": "feature_ref", "name": "mom20"}]},
        "model": {"type": model_type},
        "portfolio": {"type": "cross_sectional_long_short", "long_quantile": 0.2,
                      "short_quantile": 0.2, "weighting": "equal"},
        "execution": {"signal_time": "close_t", "trade_time": trade_time,
                      "holding_period_days": 5, "rebalance": "weekly"},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })


def test_all_model_kinds_one_interface():
    """Hoca tarifi: 'formül de olabilir, istatistiksel veya ML modeli de olabilir'."""
    kinds = {
        "formül": ["dsl_formula"],
        "istatistiksel": ["linear_regression", "ridge", "naive_bayes"],
        "ML": ["random_forest", "gradient_boosting"],
    }
    for kind, models in kinds.items():
        for mt in models:
            r = run(_spec(mt), _DATA)
            assert r.model_type == mt
            assert r.sharpe is not None, f"{mt}: Sharpe yok"
            assert r.ic > 0.05, f"{mt}: gömülü momentum bulunamadı (IC={r.ic:.3f})"
            print(f"  [ok] {kind:14s} {mt:20s} IC={r.ic:+.3f} Sharpe={r.sharpe:+.2f}")


def test_leaky_strategy_is_refused():
    """SIZINTILI stratejiye sonuç ÜRETİLMEZ — yanlış sayı, sayı yokluğundan kötüdür."""
    try:
        run(_spec("linear_regression", trade_time="close_t"), _DATA)
    except LeakageError as e:
        assert "temporal_leakage" in str(e)
        print("  [ok] sızıntılı strateji reddedildi (backtest çalıştırılmadı)")
        return
    raise AssertionError("sızıntılı strateji çalıştı — güvenlik kapısı BOZUK!")


def test_report_has_quality_and_performance():
    """Rapor: tahmin kalitesi (IC) + performans (Sharpe) + köken AYRI AYRI."""
    r = run(_spec("linear_regression"), _DATA)
    assert 0.0 <= r.directional_accuracy <= 1.0
    assert r.n_folds > 0 and r.engine_version and r.seed == 42
    s = r.summary()
    assert "Tahmin kalitesi" in s and "Performans" in s and "Köken" in s
    print(f"  [ok] rapor tam: IC={r.ic:+.3f} acc=%{r.directional_accuracy*100:.1f} "
          f"Sharpe={r.sharpe:+.2f} köken={r.engine_version}")


def test_service_is_standalone():
    """SERVİS SINIRI: backtest_service, LLM/orchestrator/agents import ETMEMELİ.

    'Servis servis gitmek iyi olur' — servis tek başına, başka projeden de
    çağrılabilir olmalı. Ayrı süreçte import edip modül ağacını denetliyoruz.
    """
    code = (
        "import sys; import backtest_service; "
        "bad=[m for m in sys.modules if m.split('.')[0] in ('llm','orchestrator','agents','openai')]; "
        "print('BAD:'+','.join(sorted(bad)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("BAD:")]
    assert line, f"alt süreç çalışmadı: {out.stderr[:200]}"
    bad = [x for x in line[0][4:].split(",") if x]
    assert not bad, f"servis bağımsız DEĞİL — şunları import ediyor: {bad}"
    print("  [ok] servis bağımsız (llm/orchestrator/agents import etmiyor)")


def main():
    test_all_model_kinds_one_interface()
    test_leaky_strategy_is_refused()
    test_report_has_quality_and_performance()
    test_service_is_standalone()
    print("OK — backtest servisi testleri geçti (model-agnostik + sızıntı kapısı + bağımsız).")


if __name__ == "__main__":
    main()
