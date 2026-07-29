"""
Model sinyali — hipotezin MODEL kutusu (dsl_formula değilse) burada çalışır.

Akış: features (DSL) = X, hedef = ileriki getiri (y). Model WALK-FORWARD fit edilir:
zaman N bloğa bölünür; blok i, YALNIZCA kendisinden ÖNCEKİ veriyle eğitilmiş modelle
tahmin edilir. Böylece üretilen tahmin paneli tamamen ÖRNEKLEM-DIŞIDIR (OOS) — sinyal
budur. Downstream (compute_pnl, fold, IC) sinyali klasik yolla aynı işler.

SIZINTI KONTROLÜ (kritik):
  1) features geçmişe bakar (static validator zaten zorluyor).
  2) Hedef y_t = t→t+horizon getirisi (gelecek). Ama yalnızca EĞİTİMDE kullanılır ve
     EMBARGO ile: test bloğu başlamadan `horizon` bar öncesine kadarki örnekler eğitime
     girer — böylece bir eğitim hedefi test dönemine SARKMAZ (purged walk-forward).
  3) Tahmin günü t için model, t'den önce fit edilmiştir; geleceği görmez.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from contracts.hypothesis_spec import HypothesisSpec
from contracts.strategy_graph import StrategyGraph
from data.synthetic import MarketData
from models import fit_predict


def compute_signal(graph: StrategyGraph, hyp: HypothesisSpec, data: MarketData,
                   liveness_out=None) -> pd.DataFrame:
    """MODEL-farkında sinyal dispatcher — her yer bunu çağırır.

    dsl_formula → DSL sinyal ifadesi (klasik, koşul-canlılık ölçümü destekli).
    model tipi → walk-forward eğitilmiş OOS tahmin paneli (liveness N/A: koşul yok).
    """
    from backtest.evaluator import evaluate_signal
    if hyp.model.type != "dsl_formula":
        return evaluate_model_signal(graph, hyp, data)
    return evaluate_signal(graph, data, liveness_out=liveness_out)


def _feature_panels(graph: StrategyGraph, data: MarketData) -> "dict[str, pd.DataFrame]":
    """Graph'ı çalıştırıp her feature'ın panelini (tarih×varlık) çıkar."""
    from backtest.evaluator import _eval_node, Value
    vals: "dict[str, Value]" = {}
    for node in graph.nodes:
        vals[node.node_id] = _eval_node(node, vals, data)
    feats: "dict[str, pd.DataFrame]" = {}
    for name, nid in graph.feature_node_ids.items():
        v = vals[nid]
        if isinstance(v, pd.DataFrame):
            feats[name] = v.replace([np.inf, -np.inf], np.nan)
    return feats


def forward_return(data: MarketData, horizon: int) -> pd.DataFrame:
    """Hedef: t→t+horizon (ileriki) getiri paneli. GELECEK — sadece eğitim hedefi.

    DÜZELTİLMİŞ fiyattan (adjusted_close): motor PnL'i temettü+split dahil
    düzeltilmiş fiyattan hesaplıyor. Hedefi HAM close'tan almak, modele
    "temettüsüz getiriyi tahmin et" dedirtip sonucu "temettülü getiri" ile
    ölçmek olurdu — hisse senedinde sistematik bir uyumsuzluk (kriptoda fark
    yok: temettü/split yok). Aynı tutarsızlık IC ölçümünü de bozardı.
    """
    try:
        px = data.get("adjusted_close")
    except KeyError:
        px = data.get("close")            # eski/sentetik veri: düzeltme yok
    h = max(1, int(horizon))
    return px.shift(-h) / px - 1.0


def evaluate_model_signal(graph: StrategyGraph, hyp: HypothesisSpec, data: MarketData,
                          n_refits: int = 6, min_train_rows: int = 200) -> pd.DataFrame:
    """MODEL kutusunu walk-forward çalıştırıp OOS tahmin panelini (=sinyal) döndürür."""
    feats = _feature_panels(graph, data)
    if not feats:
        raise ValueError("Model modu en az bir feature ister (X boş).")

    horizon = max(1, int(hyp.execution.holding_period_days or 1))
    close = data.get("close")
    dates = close.index
    fwd = forward_return(data, horizon)

    # Uzun (tidy) forma çevir: satır = (tarih, varlık), sütun = her feature + hedef.
    feat_names = list(feats.keys())
    long = pd.DataFrame({n: feats[n].stack(future_stack=True) for n in feat_names})
    y = fwd.stack(future_stack=True).reindex(long.index)
    row_dates = long.index.get_level_values(0)

    X_ok = long.notna().all(axis=1)                 # tüm feature'lar tanımlı
    pred = pd.Series(np.nan, index=long.index, dtype=float)

    blocks = np.array_split(np.arange(len(dates)), n_refits)
    for blk in blocks:
        if len(blk) == 0:
            continue
        test_start_date = dates[blk[0]]
        # EMBARGO: hedefi test'e sarkan son `horizon` bar eğitimden düşülür.
        embargo_i = max(0, blk[0] - horizon)
        train_cutoff = dates[embargo_i]

        train_mask = X_ok & y.notna() & (row_dates < train_cutoff)
        test_mask = X_ok & (row_dates >= test_start_date) & \
            (row_dates <= dates[blk[-1]])
        if int(train_mask.sum()) < min_train_rows or int(test_mask.sum()) == 0:
            continue  # ilk blok(lar): eğitim yetersiz -> NaN (işlem yok)

        X_tr = long.loc[train_mask, feat_names].to_numpy()
        y_tr = y.loc[train_mask].to_numpy()
        X_te = long.loc[test_mask, feat_names].to_numpy()
        yhat = fit_predict(hyp.model.type, X_tr, y_tr, X_te, hyp.model.params)
        pred.loc[test_mask] = yhat

    signal = pred.unstack()                         # (tarih × varlık) panel
    return signal.reindex(index=dates, columns=close.columns)
