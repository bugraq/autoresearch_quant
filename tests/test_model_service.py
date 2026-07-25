"""
Modelleme servisi testleri (hocanın iki servisinden biri).

Doğrulanan: (1) model + özellik → tahmin + IC üretir; (2) ML modeli (ağaç)
etkileşimi yakalar, lineer yakalayamaz; (3) formül tipi reddedilir (o backtest
servisinin işi); (4) servis LLM/orchestrator'dan bağımsız.
"""
from __future__ import annotations

import subprocess
import sys

from data.synthetic import gen_interaction_alpha
from model_service import predict


def _feat(name, op, w, field="close"):
    return {"name": name, "expression":
            {"op": op, "window": w, "inputs": [{"op": "field", "field": field}]}}


_DATA = gen_interaction_alpha(n_sec=40, n_days=1200, seed=3)
_FEATS = [_feat("mom", "return", 20), _feat("volz", "zscore", 60, "volume")]


def test_tree_model_captures_interaction():
    """Ağaç modeli momentum×hacim etkileşimini KENDİ yakalamalı (IC yüksek)."""
    r = predict("random_forest", _FEATS, _DATA, horizon=5)
    assert r.ic > 0.08, f"ağaç modeli etkileşimi yakalayamadı (IC={r.ic:.3f})"
    assert r.model_type == "random_forest" and r.n_features == 2
    assert 0.0 <= r.directional_accuracy <= 1.0
    print(f"  [ok] random_forest etkileşimi yakaladı: IC={r.ic:+.3f} "
          f"yön=%{r.directional_accuracy*100:.1f}")


def test_linear_cannot_capture_interaction():
    """Lineer model çarpımı doğrudan temsil edemez → IC düşük (ML'in değerini gösterir)."""
    r = predict("linear_regression", _FEATS, _DATA, horizon=5)
    assert r.ic < 0.08, f"lineer beklenmedik şekilde yakaladı (IC={r.ic:.3f})"
    print(f"  [ok] linear_regression etkileşimi yakalayamadı (beklenen): IC={r.ic:+.3f}")


def test_formula_rejected():
    """Formül modelleme servisinin işi DEĞİL — reddedilmeli."""
    try:
        predict("dsl_formula", _FEATS, _DATA)
    except ValueError:
        print("  [ok] dsl_formula reddedildi (o backtest servisinin işi)")
        return
    raise AssertionError("dsl_formula reddedilmeliydi")


def test_service_is_standalone():
    """SERVİS SINIRI: model_service, llm/orchestrator/agents import ETMEMELİ."""
    code = ("import sys, model_service; "
            "bad=[m for m in sys.modules if m.split('.')[0] in "
            "('llm','orchestrator','agents','openai')]; print('BAD:'+','.join(sorted(bad)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("BAD:")]
    assert line, f"alt süreç çalışmadı: {out.stderr[:200]}"
    bad = [x for x in line[0][4:].split(",") if x]
    assert not bad, f"servis bağımsız DEĞİL — import ediyor: {bad}"
    print("  [ok] servis bağımsız (llm/orchestrator/agents import etmiyor)")


def main():
    test_tree_model_captures_interaction()
    test_linear_cannot_capture_interaction()
    test_formula_rejected()
    test_service_is_standalone()
    print("OK — modelleme servisi testleri geçti.")


if __name__ == "__main__":
    main()
