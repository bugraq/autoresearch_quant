"""
Model katmanı testleri (yeni yön: hipotez → MODEL → backtest).

Doğrulanan: (1) model gömülü sinyali ÖĞRENİR (pozitif IC/Sharpe); (2) rastgele
veride SIZDIRMAZ / alpha uydurmaz (IC ~0); (3) dsl_formula varsayılanı bozulmadı.
"""
from __future__ import annotations

from contracts.hypothesis_spec import HypothesisSpec
from dsl import compile_hypothesis
from backtest.walk_forward import run_walk_forward
from data.synthetic import gen_cross_sectional_momentum, gen_random


def _hyp(model_type: str) -> HypothesisSpec:
    return HypothesisSpec.model_validate({
        "hypothesis_id": "hyp_model", "title": "model test",
        "claim": "geçmiş getiri geleceği öngörür", "family": "momentum",
        "economic_mechanism": {"type": "trend", "description": "momentum devamı"},
        "universe": {"source": "sp500_point_in_time"},
        "features": [
            {"name": "mom20", "expression":
                {"op": "return", "window": 20, "inputs": [{"op": "field", "field": "close"}]}},
            {"name": "mom60", "expression":
                {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}},
        ],
        # dsl_formula modunda sinyal budur; model modunda YOK SAYILIR (X = features).
        "signal": {"op": "cross_sectional_rank", "inputs": [{"op": "feature_ref", "name": "mom20"}]},
        "model": {"type": model_type},
        "portfolio": {"type": "cross_sectional_long_short",
                      "long_quantile": 0.2, "short_quantile": 0.2, "weighting": "equal"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": 5, "rebalance": "weekly"},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })


def _run(model_type, data):
    h = _hyp(model_type)
    return run_walk_forward(compile_hypothesis(h), h, data)


def test_dsl_formula_default_unchanged():
    """model bloğu verilmezse varsayılan dsl_formula — mevcut davranış korunur."""
    h = HypothesisSpec.model_validate({
        "hypothesis_id": "h", "title": "t", "claim": "c", "family": "momentum",
        "economic_mechanism": {"type": "x", "description": "y"},
        "universe": {"source": "sp500_point_in_time"},
        "features": [], "signal": {"op": "return", "window": 20,
                                   "inputs": [{"op": "field", "field": "close"}]},
        "portfolio": {"type": "cross_sectional_long_short"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": 5},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })
    assert h.model.type == "dsl_formula", "model verilmeyince varsayılan dsl_formula olmalı"
    r = run_walk_forward(compile_hypothesis(h), h, gen_cross_sectional_momentum(
        n_sec=40, n_days=1000, seed=1, drift_spread=0.003))
    assert r.aggregate_sharpe() > 0.5, "dsl_formula momentum'da pozitif olmalı"
    print(f"  [ok] dsl_formula varsayılanı çalışıyor: Sharpe={r.aggregate_sharpe():+.2f}")


def test_model_learns_momentum():
    """linear_regression + naive_bayes gömülü momentumu öğrenmeli (pozitif IC)."""
    data = gen_cross_sectional_momentum(n_sec=40, n_days=1000, seed=1, drift_spread=0.003)
    for mt in ("linear_regression", "naive_bayes"):
        r = _run(mt, data)
        e = r.exposures
        assert e["ic"] > 0.05, f"{mt}: IC {e['ic']:.3f} çok düşük — momentum öğrenilmedi"
        assert r.aggregate_sharpe() > 0.5, f"{mt}: Sharpe {r.aggregate_sharpe():.2f} düşük"
        assert e["dir_acc"] > 0.52, f"{mt}: dir_acc {e['dir_acc']:.3f} baseline'a yakın"
        assert e["model_type"] == mt
        print(f"  [ok] {mt} momentum öğrendi: Sharpe={r.aggregate_sharpe():+.2f} "
              f"IC={e['ic']:+.3f} dir_acc={e['dir_acc']*100:.1f}%")


def test_model_no_leakage_on_random():
    """SAF gürültüde model IC ~0 vermeli — sızıntı yok, alpha uydurmuyor."""
    data = gen_random(n_sec=40, n_days=1000, seed=7)
    for mt in ("linear_regression", "naive_bayes"):
        e = _run(mt, data).exposures
        assert abs(e["ic"]) < 0.05, f"{mt}: rastgele veride IC {e['ic']:.3f} — SIZINTI şüphesi!"
        assert abs(e["dir_acc"] - 0.5) < 0.03, \
            f"{mt}: rastgele veride dir_acc {e['dir_acc']:.3f} — %50'den sapma SIZINTI şüphesi!"
        print(f"  [ok] {mt} rastgele veride sızdırmıyor: IC={e['ic']:+.3f} dir_acc={e['dir_acc']*100:.1f}%")


def _feat(name, op, w, field="close"):
    return {"name": name, "expression":
            {"op": op, "window": w, "inputs": [{"op": "field", "field": field}]}}


def _model_hyp(features, model="linear_regression", family="composite"):
    return HypothesisSpec.model_validate({
        "hypothesis_id": "h", "title": "t", "claim": "c", "family": family,
        "economic_mechanism": {"type": "x", "description": "y"},
        "universe": {"source": "sp500_point_in_time"},
        "features": features, "model": {"type": model},
        "signal": {"op": "cross_sectional_rank",
                   "inputs": [{"op": "feature_ref", "name": features[0]["name"]}]},
        "portfolio": {"type": "cross_sectional_long_short",
                      "long_quantile": 0.2, "short_quantile": 0.2, "weighting": "equal"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": 5, "rebalance": "weekly"},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })


def test_novelty_distinguishes_model_features():
    """Farklı feature'lı iki model hipotezi DUPLICATE sanılmamalı (placeholder aynı)."""
    from backtest import compute_signal
    from memory.similarity import NoveltyIndex
    data = gen_cross_sectional_momentum(n_sec=40, n_days=1000, seed=1, drift_spread=0.003)
    hA = _model_hyp([_feat("mom60", "return", 60), _feat("vol20", "volatility", 20)])
    hB = _model_hyp([_feat("rev5", "return", 5), _feat("liq", "zscore", 60, "dollar_volume")])
    nov = NoveltyIndex()
    nov.add(hA, compute_signal(compile_hypothesis(hA), hA, data))
    assert nov.check_structural(hB) is None, "farklı feature'lı model hipotezi duplicate sanıldı"
    assert nov.check_structural(hA) is not None, "aynı model hipotezi duplicate yakalanmadı"
    print("  [ok] novelty model feature-setini ayırt ediyor (placeholder yanıltmıyor)")


def test_critic_accepts_model_composite():
    """Model modunda birleştirme modeldedir; critic sinyal-yapısına bakıp REVISE etmemeli."""
    from agents.quant_critic import DummyCritic
    h = _model_hyp([_feat("mom60", "return", 60), _feat("vol20", "volatility", 20)],
                   family="composite")
    dec = DummyCritic().review(h)
    assert dec.decision.value == "accept", f"critic model composite'i {dec.decision.value} yaptı"
    print("  [ok] critic model composite'i kabul ediyor (placeholder sinyale takılmıyor)")


def test_robustness_model_mode():
    """Sağlamlık testleri model modunda çökmeden çalışmalı (param perturbasyonu = re-fit)."""
    from evaluation.robustness import run_robustness
    data = gen_cross_sectional_momentum(n_sec=40, n_days=1000, seed=1, drift_spread=0.003)
    h = _model_hyp([_feat("mom60", "return", 60), _feat("vol20", "volatility", 20)])
    rob = run_robustness(compile_hypothesis(h), h, data, cost_bps=1.0)
    assert rob.robust, "model momentum robust çıkmalıydı"
    print(f"  [ok] robustness model modu: robust={rob.robust} param_min={rob.param_min_sharpe:+.2f}")


def main():
    test_dsl_formula_default_unchanged()
    test_model_learns_momentum()
    test_model_no_leakage_on_random()
    test_novelty_distinguishes_model_features()
    test_critic_accepts_model_composite()
    test_robustness_model_mode()
    print("OK — model katmanı testleri geçti (öğrenme + sızıntı-güvenliği + loop entegrasyonu).")


if __name__ == "__main__":
    main()
