"""
İKİ SERVİS DEMOSU — hocanın "modelleme ve backtest yapan iki servis" isteği.

    python services_demo.py

1) MODELLEME SERVİSİ : model + özellik → walk-forward tahmin + kalite (IC)
2) BACKTEST SERVİSİ  : bir strateji (formül ya da model) → getiri/Sharpe/maliyet

Kontrollü veride (alpha = momentum × hacim etkileşiminde) koşulur; ground-truth
bilindiği için servislerin GERÇEKTEN çalıştığı doğrulanabilir.
"""
from __future__ import annotations

import sys

# Windows konsolu (cp1254) "→ × √" gibi karakterlerde UnicodeEncodeError
# ile PATLAR (bu betik hocaya canli gosteriliyor). main.py'deki korumanin aynisi.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import gen_interaction_alpha


def _feat(name, op, w, field="close"):
    return {"name": name, "expression":
            {"op": op, "window": w, "inputs": [{"op": "field", "field": field}]}}


def main() -> None:
    data = gen_interaction_alpha(n_sec=40, n_days=1400, seed=3)
    ozellikler = [_feat("momentum", "return", 20),
                  _feat("hacim_z", "zscore", 60, "volume")]

    print("=" * 66)
    print("VERİ: kontrollü benchmark — gerçek alpha momentum × hacim etkileşiminde")
    print("=" * 66)

    # ── 1) MODELLEME SERVİSİ ──────────────────────────────────────────
    print("\n### 1) MODELLEME SERVİSİ  (model + özellik → tahmin + IC)\n")
    from model_service import predict
    for model in ("linear_regression", "random_forest", "gradient_boosting"):
        rap = predict(model, ozellikler, data, horizon=5)
        print(rap.summary())
        print()
    print("Okuma: lineer model çarpımı doğrudan temsil EDEMEZ (IC düşük); ağaç "
          "modelleri (random forest/GBM) etkileşimi KENDİ yakalar (IC yüksek).")

    # ── 2) BACKTEST SERVİSİ ───────────────────────────────────────────
    print("\n" + "=" * 66)
    print("### 2) BACKTEST SERVİSİ  (strateji → getiri/Sharpe/maliyet)\n")
    from backtest_service import run

    def spec(title, signal, model, feats):
        return HypothesisSpec.model_validate({
            "hypothesis_id": "demo", "title": title, "claim": "demo", "family": "composite",
            "economic_mechanism": {"type": "test", "description": "demo"},
            "universe": {"source": "sp500_point_in_time"},
            "features": feats, "signal": signal, "model": {"type": model},
            "portfolio": {"type": "cross_sectional_long_short", "long_quantile": 0.2,
                          "short_quantile": 0.2, "weighting": "equal"},
            "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                          "holding_period_days": 5, "rebalance": "weekly"},
            "falsification": {"minimum_oos_sharpe": 0.3},
        })

    ref = lambda n: {"op": "feature_ref", "name": n}
    rank = lambda x: {"op": "cross_sectional_rank", "inputs": [x]}

    # (a) FORMÜL modeli: elle yazılmış momentum × hacim
    formul = spec("Elle-formül: momentum × hacim",
                  rank({"op": "multiply", "inputs": [ref("momentum"), ref("hacim_z")]}),
                  "dsl_formula", ozellikler)
    print(run(formul, data).summary())
    print()
    # (b) ML modeli: aynı özellikler, random forest birleştirsin
    ml = spec("ML: random forest (aynı 2 özellik)",
              rank(ref("momentum")), "random_forest", ozellikler)
    print(run(ml, data).summary())

    print("\n" + "=" * 66)
    print("SONUÇ: iki servis de çalışıyor. Modelleme servisi 'model öngörüyor mu' "
          "(IC) der; backtest servisi 'para kazandırır mı' (Sharpe) der. Bir model "
          "(formül YA DA ML) → tek çağrıyla backtest — hocanın istediği tam bu.")


if __name__ == "__main__":
    main()
