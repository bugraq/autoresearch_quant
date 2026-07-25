"""
Modelleme Servisi — model + özellik + veri → walk-forward tahmin + kalite (IC).

    from model_service import predict
    rapor = predict("random_forest", ozellikler, veri, horizon=5)
    print(rapor.summary())

Hocanın adını koyduğu iki servisten biri (diğeri: backtest_service).
Backtest'ten bağımsız çalışır; LLM/orchestrator import etmez.
"""
from model_service.service import ModelReport, predict

__all__ = ["predict", "ModelReport"]
