"""
MODELLEME SERVİSİ — hocanın adını koyduğu iki servisten biri.

Hoca (15.07): "modelleme ve backtest yapan İKİ servis ... bir modeli alıp
(formül/istatistik/ML) ..." Bu, o iki servisten MODELLEME olanıdır (backtest
olanı: backtest_service/).

Görevi tek başına: **model + özellikler + veri → walk-forward tahmin + tahmin
kalitesi (IC).** Backtest'ten BAĞIMSIZ kullanılabilir — "bu özelliklerle bu model
geleceği öngörebiliyor mu?" sorusunu, portföy/işlem maliyeti hiç devreye girmeden
cevaplar.

    from model_service import predict
    rapor = predict("random_forest", ozellikler, veri, horizon=5)
    print(rapor.summary())     # IC, RankIC, yön isabeti

SIZINTI GÜVENLİĞİ: model her walk-forward diliminde YALNIZCA geçmişe fit edilir,
embargo ile (purged) — geleceği yapısal olarak göremez.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import MarketData
from dsl import compile_hypothesis
from models import SUPPORTED_MODELS


@dataclass
class ModelReport:
    """Modelleme servisinin çıktısı: tahminler + tahmin kalitesi."""

    model_type: str
    n_features: int
    horizon: int
    ic: float             # tahmin ile gerçek getirinin kesitsel korelasyonu
    rank_ic: float        # sıra korelasyonu (daha sağlam)
    icir: float           # IC kararlılığı (ortalama/std)
    directional_accuracy: float   # yön isabeti (%50 = yazı-tura)
    predictions: pd.DataFrame     # tarih × varlık tahmin paneli (walk-forward OOS)

    def summary(self) -> str:
        return (f"Model: {self.model_type} ({self.n_features} özellik, "
                f"ufuk {self.horizon} gün)\n"
                f"  Tahmin kalitesi : IC={self.ic:+.3f}  RankIC={self.rank_ic:+.3f}  "
                f"ICIR={self.icir:+.2f}  yön isabeti=%{self.directional_accuracy*100:.1f}\n"
                f"  (IC>0.03 anlamlı sinyal · ~0 öngörü yok · finansta yön %55 güçlü)")


def predict(model_type: str, features: "list[dict]", data: MarketData,
            horizon: int = 5) -> ModelReport:
    """Model + özellikler + veri -> tahmin + kalite. Servisin tek giriş noktası.

    Args:
        model_type: linear_regression | ridge | naive_bayes | random_forest |
                    gradient_boosting  (formül DEĞİL — o backtest servisinin işi)
        features:   [{"name": ..., "expression": <DSL>}] — modelin girdisi (X)
        data:       MarketData
        horizon:    hedef = kaç gün ilerinin getirisi (tahminin ufku)

    Raises:
        ValueError: desteklenmeyen model tipi ya da özellik yok.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Bilinmeyen model: {model_type!r} "
                         f"(desteklenen: {sorted(SUPPORTED_MODELS)}). "
                         f"Elle-formül için backtest_service kullan.")
    if not features:
        raise ValueError("En az bir özellik (X) gerekir.")

    # Minimal hipotez kur (model modu): signal PLACEHOLDER, model X'ten öğrenir.
    spec = HypothesisSpec.model_validate({
        "hypothesis_id": "model_service", "title": f"{model_type} tahmini",
        "claim": "özelliklerden ileriki getiri tahmini", "family": "composite",
        "economic_mechanism": {"type": "model", "description": "walk-forward tahmin"},
        "universe": {"source": "sp500_point_in_time"},
        "features": features,
        "signal": {"op": "cross_sectional_rank",
                   "inputs": [{"op": "feature_ref", "name": features[0]["name"]}]},
        "model": {"type": model_type},
        "portfolio": {"type": "cross_sectional_long_short"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": int(horizon), "rebalance": "daily"},
        "falsification": {"minimum_oos_sharpe": 0.3},
    })

    from backtest.model_signal import evaluate_model_signal
    from backtest.walk_forward import _information_coefficient

    graph = compile_hypothesis(spec)
    preds = evaluate_model_signal(graph, spec, data)     # walk-forward OOS tahmin
    ic = _information_coefficient(preds, data, horizon)
    return ModelReport(
        model_type=model_type, n_features=len(features), horizon=int(horizon),
        ic=ic["ic"], rank_ic=ic["rank_ic"], icir=ic["icir"],
        directional_accuracy=ic["dir_acc"], predictions=preds)
