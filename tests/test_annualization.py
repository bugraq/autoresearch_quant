"""Yıllıklaştırma ölçeği testleri — hard gate'in adaletini korur.

BULUNAN GERÇEK BUG: TRADING_DAYS=252 (hisse yılı) üç modülde SABİT gömülüydü.
Kripto 7/24 açık (365 bar/yıl) -> Sharpe sqrt(365/252)=1.20 kat DÜŞÜK
hesaplanıyordu. Bu yalnız raporlamayı değil HARD GATE'i bozar: gerçek yıllık
Sharpe'ı 0.55 olan strateji 0.46 görünüp min_acceptance_sharpe=0.5'e takılır ve
HAKSIZ yere reddedilir. 8h barda sapma ~%48'e çıkar (bar getirisi ~1/3, std
~1/sqrt(3)) — düzeltilmeden 8h bara geçmek her şeyi elerdi.

Doğru mimari: ölçek VERİDEN gelir (MarketData.bars_per_year), hesap katmanı
sabit varsaymaz.
"""
import numpy as np
import pandas as pd

from backtest.engine import fold_metrics
from data.synthetic import (
    BARS_PER_YEAR_CRYPTO_8H,
    BARS_PER_YEAR_CRYPTO_DAILY,
    BARS_PER_YEAR_EQUITY_DAILY,
    MarketData,
    split_by_fraction,
)
from evaluation.statistics import bootstrap_sharpe_ci


def _pnl(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return (pd.Series(rng.normal(0.001, 0.01, n), index=idx),
            pd.Series(0.1, index=idx))


def test_sharpe_scales_with_bars_per_year():
    """Aynı getiri serisi, farklı frekans -> Sharpe sqrt(oran) kadar ölçeklenir."""
    pnl, turn = _pnl()
    eq = fold_metrics(pnl, turn, "f", "v", bars_per_year=BARS_PER_YEAR_EQUITY_DAILY)
    cr = fold_metrics(pnl, turn, "f", "v", bars_per_year=BARS_PER_YEAR_CRYPTO_DAILY)
    beklenen = (BARS_PER_YEAR_CRYPTO_DAILY / BARS_PER_YEAR_EQUITY_DAILY) ** 0.5
    assert abs(cr.sharpe / eq.sharpe - beklenen) < 1e-9
    # Kripto günlüğünü 252 ile ölçmek Sharpe'ı ~%17 DÜŞÜK gösterir (bug buydu)
    assert eq.sharpe < cr.sharpe
    print(f"  [ok] Sharpe ölçeği: 252->{eq.sharpe:.3f}, 365->{cr.sharpe:.3f} "
          f"(oran {beklenen:.3f})")


def test_gate_injustice_is_real():
    """BUG'IN SOMUT KANITI: kripto stratejisi 365 ile eşiği geçer, 252 ile takılır."""
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # Yıllık Sharpe'ı 365 ölçeğinde ~0.55 olacak seri kur
    hedef, sd = 0.55, 0.01
    mu = hedef * sd / np.sqrt(BARS_PER_YEAR_CRYPTO_DAILY)
    pnl = pd.Series(rng.normal(mu, sd, n), index=idx)
    pnl = (pnl - pnl.mean()) / pnl.std() * sd + mu     # momentleri sabitle
    turn = pd.Series(0.1, index=idx)
    dogru = fold_metrics(pnl, turn, "f", "v", bars_per_year=BARS_PER_YEAR_CRYPTO_DAILY).sharpe
    yanlis = fold_metrics(pnl, turn, "f", "v", bars_per_year=BARS_PER_YEAR_EQUITY_DAILY).sharpe
    GATE = 0.5
    assert dogru > GATE > yanlis, f"doğru={dogru:.3f}, yanlış={yanlis:.3f}"
    print(f"  [ok] gate adaleti: doğru ölçek {dogru:.2f} GEÇER, "
          f"yanlış ölçek {yanlis:.2f} takılır (haksız red)")


def test_split_preserves_bars_per_year():
    """Bölmek frekansı değiştirmez; taşınmazsa holdout FARKLI ölçekle ölçülür
    ve araştırma/holdout karşılaştırması sessizce elmayla-armut olur."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    md = MarketData(fields={"close": pd.DataFrame(1.0, index=idx, columns=["A"])},
                    bars_per_year=BARS_PER_YEAR_CRYPTO_8H)
    res, hold = split_by_fraction(md, 0.7)
    assert res.bars_per_year == BARS_PER_YEAR_CRYPTO_8H
    assert hold.bars_per_year == BARS_PER_YEAR_CRYPTO_8H
    print("  [ok] split_by_fraction bars_per_year'ı araştırma+holdout'a taşıyor")


def test_default_is_equity_daily():
    """Geriye uyumluluk: frekans belirtilmezse hisse günlüğü (mevcut davranış)."""
    md = MarketData(fields={"close": pd.DataFrame({"A": [1.0]})})
    assert md.bars_per_year == BARS_PER_YEAR_EQUITY_DAILY == 252
    print("  [ok] varsayılan 252 (hisse günlüğü) — geriye uyumlu")


def test_bootstrap_ci_scales():
    """CI de yıllıklaştırılmış Sharpe cinsinden -> ölçekten etkilenir."""
    pnl, _ = _pnl(n=200, seed=3)
    lo_e, hi_e = bootstrap_sharpe_ci(list(pnl), n_boot=200, seed=1,
                                     bars_per_year=BARS_PER_YEAR_EQUITY_DAILY)
    lo_c, hi_c = bootstrap_sharpe_ci(list(pnl), n_boot=200, seed=1,
                                     bars_per_year=BARS_PER_YEAR_CRYPTO_DAILY)
    assert hi_c > hi_e and lo_c > lo_e
    print(f"  [ok] bootstrap CI ölçekleniyor: 252=[{lo_e:.2f},{hi_e:.2f}] "
          f"365=[{lo_c:.2f},{hi_c:.2f}]")


def main():
    test_sharpe_scales_with_bars_per_year()
    test_gate_injustice_is_real()
    test_split_preserves_bars_per_year()
    test_default_is_equity_daily()
    test_bootstrap_ci_scales()
    print("OK — yıllıklaştırma ölçeği testleri geçti.")


if __name__ == "__main__":
    main()
