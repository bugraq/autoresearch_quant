"""
Etkileşim benchmark'ı testleri — araştırma VERİMLİLİĞİNİ ayırt eden ortam.

Neden var: tek-faktörlü momentum benchmark'ında random-search 11 backtest'te
LLM'lerle AYNI tavanı buluyordu (Sharpe ~+7.9) → deney ayırt etmiyordu.
Etkileşim benchmark'ında alpha bir ÇARPIMDA gizli; tek faktör ~0 alır, yalnızca
BİRLEŞTİREN bulur → "kim daha iyi arıyor" ölçülebilir hale gelir.

Bu testler o AYIRT EDİCİLİĞİ korur: biri sinyali bozarsa/kolaylaştırırsa yakalanır.
"""
from __future__ import annotations

from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import gen_interaction_alpha
from backtest_service import run

_DATA = gen_interaction_alpha(n_sec=40, n_days=1400, seed=3)

_MOM = {"name": "mom", "expression":
        {"op": "return", "window": 20, "inputs": [{"op": "field", "field": "close"}]}}
_VOL = {"name": "vz", "expression":
        {"op": "zscore", "window": 60, "inputs": [{"op": "field", "field": "volume"}]}}


def _ref(n):
    return {"op": "feature_ref", "name": n}


def _rank(x):
    return {"op": "cross_sectional_rank", "inputs": [x]}


def _spec(name, signal, feats):
    return HypothesisSpec.model_validate({
        "hypothesis_id": name, "title": name, "claim": "test", "family": "composite",
        "economic_mechanism": {"type": "interaction", "description": "hacimle teyitli momentum"},
        "universe": {"source": "sp500_point_in_time"},
        "features": feats, "signal": signal, "model": {"type": "dsl_formula"},
        "portfolio": {"type": "cross_sectional_long_short", "long_quantile": 0.2,
                      "short_quantile": 0.2, "weighting": "equal"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": 5, "rebalance": "weekly"},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })


def test_single_factors_are_useless():
    """Momentum ve hacim TEK BAŞLARINA öngörücü OLMAMALI (matematik: E[.]=0)."""
    for name, feat in (("momentum", _MOM), ("hacim", _VOL)):
        r = run(_spec(f"tek_{name}", _rank(_ref(feat["name"])), [feat]), _DATA)
        assert abs(r.ic) < 0.03, \
            f"tek başına {name} öngörüyor (IC={r.ic:+.3f}) — benchmark ayırt etmiyor!"
        print(f"  [ok] tek başına {name:9s} işe yaramıyor: IC={r.ic:+.3f} (~0 beklenir)")


def test_interaction_is_the_alpha():
    """Yalnızca momentum × hacim ÇARPIMI gerçek alpha'yı bulmalı."""
    sig = _rank({"op": "multiply", "inputs": [_ref("mom"), _ref("vz")]})
    r = run(_spec("etkilesim", sig, [_MOM, _VOL]), _DATA)
    assert r.ic > 0.08, f"etkileşim alpha'sı bulunamadı (IC={r.ic:+.3f})"
    assert r.sharpe > 1.0, f"etkileşim Sharpe'ı düşük ({r.sharpe:+.2f})"
    print(f"  [ok] momentum × hacim GERÇEK alpha: IC={r.ic:+.3f} Sharpe={r.sharpe:+.2f}")


def test_model_can_learn_interaction():
    """ML modeli (ağaç) etkileşimi özellikleri BİRLEŞTİREREK yakalayabilmeli.

    Lineer model çarpımı doğrudan temsil EDEMEZ (etkileşim terimi yok); ağaç
    tabanlı model yakalayabilir. Bu, model katmanının değerini gösterir.
    """
    r = run(_spec("rf", _rank(_ref("mom")), [_MOM, _VOL]).model_copy(
        update={"model": HypothesisSpec.model_validate({
            **_spec("x", _rank(_ref("mom")), [_MOM, _VOL]).model_dump(),
            "model": {"type": "random_forest"}}).model}), _DATA)
    print(f"  [ok] random_forest etkileşimde: IC={r.ic:+.3f} Sharpe={r.sharpe:+.2f}")
    assert r.ic > 0.0, "ağaç modeli etkileşimden hiçbir şey çıkaramadı"


def main():
    test_single_factors_are_useless()
    test_interaction_is_the_alpha()
    test_model_can_learn_interaction()
    print("OK — etkileşim benchmark'ı ayırt edici (tek faktör ~0, birleştiren bulur).")


if __name__ == "__main__":
    main()
