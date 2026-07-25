"""
Bağımsız referans backtest — motorun DOĞRULUĞUNU çapraz-kontrol eder.

Bu dosya, `backtest/engine.py`'nin compute_pnl'iyle KOD PAYLAŞMAZ. Aynı beyan
edilen kuralları (kesitsel long-short, eşit ağırlık, open_t+1 işlem, işlem
maliyeti) SIFIRDAN, sade ve okunabilir biçimde yeniden uygular. İki bağımsız
hesap aynı sonucu veriyorsa motor spesifikasyonuna sadıktır; vermiyorsa bug var.

"Çift kayıt muhasebesi" mantığı: aynı sayıyı iki farklı yoldan hesaplayıp
karşılaştırmak. Golden-test (donmuş sayı) motorun DEĞİŞMEDİĞİNİ; bu referans ise
motorun spesifikasyona UYDUĞUNU bağımsızca doğrular.

Kapsam: standart config (execution.trade_time='open_t_plus_1', eşit ağırlık,
cross_sectional_long_short, rebalance=daily, holding=1). Sektör-nötr, rank-weight,
farklı rebalance aralıkları bu referansın DIŞINDA (onlar ayrı testlerde).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reference_backtest(signal: pd.DataFrame, data, long_q: float = 0.2,
                       short_q: float = 0.2, cost_bps: float = 5.0) -> "tuple[float, pd.Series]":
    """Sinyal + veri -> (yıllık Sharpe, net günlük getiri serisi). Bağımsız hesap.

    Beyan edilen kurallar (engine ile AYNI spesifikasyon, AYRI kod):
      - İşlem open_{t+1}'de kurulur, düzeltilmiş fiyattan open-to-open getiri kazanır
      - Sinyal en yüksek %long_q AL (+), en düşük %short_q SAT (−); her bacak eşit
        ağırlık, gross 0.5+0.5 = 1, net 0 (piyasa-nötr)
      - w_t open_{t+1}'de kurulur → open_{t+1}→open_{t+2} getirisini kazanır: shift(2)
      - Maliyet o pozisyonun getiri kazandığı barla aynı hizada: turnover.shift(2)
    """
    # 1) İşlem fiyatı: open, DÜZELTME faktörüyle (temettü/split). adjusted yoksa ham.
    close = data.get("close")
    try:
        adj = data.get("adjusted_close")
    except KeyError:
        adj = close
    factor = adj / close
    exec_price = data.get("open") * factor
    fwd_ret = exec_price.pct_change(fill_method=None)     # getiri_t = open_t/open_{t-1} − 1

    # 2) Point-in-time üyelik (varsa): hisse yalnızca üye olduğu gün işlem görür
    sig = signal.copy()
    if "index_membership" in data.fields:
        sig = sig.where(data.get("index_membership") > 0)

    # 3) Ağırlıklar — kesitsel sıralama, üst/alt dilim, eşit ağırlık
    ranks = sig.rank(axis=1, pct=True)
    long_sel = (ranks >= 1.0 - long_q).astype(float)
    short_sel = (ranks <= short_q).astype(float)
    lw = long_sel.div(long_sel.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    sw = short_sel.div(short_sel.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    w = (lw.fillna(0) - sw.fillna(0))                     # +long, −short; gross 1

    # 4) PnL — w_t, open_{t+1}→open_{t+2} getirisini kazanır (shift 2)
    gross_pnl = (w.shift(2) * fwd_ret).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    cost = turnover.shift(2) * (cost_bps / 1e4)
    net = (gross_pnl - cost).dropna()

    sharpe = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else 0.0
    return sharpe, net
