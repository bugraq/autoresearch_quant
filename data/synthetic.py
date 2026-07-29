"""
Sentetik veri üreteci — bilinen özelliklere sahip veri (Doküman 23.1).

Amaç: motoru sınamak. Motor, veri içine GÖMÜLÜ gerçek sinyali bulabilmeli
(momentum/reversal) ve TAMAMEN RASTGELE veride sahte alpha üretmemeli.

MarketData: alan adı -> DataFrame(index=tarih, columns=varlık). Wide panel.
Bu, ileride gerçek point-in-time veri adaptörüyle aynı arayüzü paylaşacak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


#: Yıllıklaştırma sabitleri — bar frekansı başına yıldaki bar sayısı.
#  Sharpe = ortalama/std * sqrt(bars_per_year) olduğundan bu sabit YANLIŞSA
#  Sharpe ölçeği bozulur ve hard gate (min_acceptance_sharpe) HAKSIZ karar verir.
BARS_PER_YEAR_EQUITY_DAILY = 252    # borsa kapalı günler hariç
BARS_PER_YEAR_CRYPTO_DAILY = 365    # kripto 7/24 açık
BARS_PER_YEAR_CRYPTO_8H = 365 * 3   # 8 saatlik bar (funding periyoduyla senkron)


@dataclass
class MarketData:
    fields: dict[str, pd.DataFrame]
    # ticker -> GICS sektör (varsa). Gerçek sektör-nötralizasyonu için gerekir;
    # None ise neutralize_sector / portfolio.sector_neutral piyasa-nötre düşer
    # (dürüstçe belgelendi — sentetik/plain veride sektör yok).
    sectors: "dict[str, str] | None" = None
    # Yılda kaç bar? Sharpe/volatilite/devir YILLIKLAŞTIRMASI bunu kullanır.
    # Veri kendi frekansını bilir; hesap katmanı sabit varsaymamalı. Yanlış değer
    # doğrudan hard gate'i bozar: kripto günlüğü 252 ile ölçmek Sharpe'ı ~%20
    # DÜŞÜK gösterir (sqrt(365/252)=1.20) -> gerçekte 0.55 olan strateji 0.46
    # görünüp haksızca reddedilir. 8h barda sapma ~%48'e çıkar.
    bars_per_year: int = BARS_PER_YEAR_EQUITY_DAILY

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.fields:
            raise KeyError(f"Veri alanı yok: {name}")
        return self.fields[name]

    @property
    def dates(self) -> pd.Index:
        return next(iter(self.fields.values())).index


def _prices_from_returns(returns: np.ndarray, dates, tickers) -> MarketData:
    """Getirilerden fiyat paneli ve türev alanları kur."""
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    close = pd.DataFrame(prices, index=dates, columns=tickers)
    rng = np.random.default_rng(0)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, size=prices.shape),
                          index=dates, columns=tickers)
    return MarketData(fields={
        "close": close,
        "adjusted_close": close,
        "open": close.shift(1).bfill(),   # basitleştirme: açılış ~ önceki kapanış
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": volume,
        "dollar_volume": close * volume,
        "market_cap": close * 1e7,
    })


def split_by_fraction(md: MarketData, research_frac: float = 0.7) -> tuple[MarketData, MarketData]:
    """Zaman çizgisini araştırma / KİLİTLİ holdout olarak böl.

    Araştırma ajanı yalnızca ilk parçayı görür; holdout (son parça) ayrı bir
    servise kalır ve asla LLM'e/araştırmaya sızmaz (Doküman 2.2, 10.3).
    """
    n = len(md.dates)
    cut = int(n * research_frac)
    # bars_per_year TAŞINMALI: bölmek frekansı değiştirmez. Taşınmazsa holdout
    # varsayılana (252) düşer ve araştırmadan FARKLI ölçekle Sharpe hesaplanır —
    # yani araştırma/holdout karşılaştırması sessizce elmayla-armut olurdu.
    research = MarketData(fields={k: v.iloc[:cut].copy() for k, v in md.fields.items()},
                          sectors=md.sectors, bars_per_year=md.bars_per_year)
    holdout = MarketData(fields={k: v.iloc[cut:].copy() for k, v in md.fields.items()},
                         sectors=md.sectors, bars_per_year=md.bars_per_year)
    return research, holdout


def concat_market(history: MarketData, future: MarketData) -> MarketData:
    """İki dilimi zaman ekseninde birleştir (history ÖNCE, future SONRA).

    Neden gerekir (holdout ISINMA sorunu): holdout dilimi tek başına
    değerlendirilirse strateji, kilitli dönemin BAŞINDA geçmişsiz kalır —
    120 barlık bir rolling pencere holdout'un ilk 120 barını, walk-forward
    eğitilen bir ML modeli ise ilk ~%17'sini NaN bırakır. Daha kötüsü: model
    holdout'un KENDİ İÇİNDE yeniden eğitilir, yani sınav, araştırmada kabul
    edilen modeli değil BAŞKA bir modeli ölçer.

    Doğru bilgi akışı: geçmiş (araştırma dilimi) -> gelecek (holdout). Bu
    SIZINTI DEĞİLDİR; ters yön (holdout -> araştırma) sızıntı olurdu ve burada
    yapısal olarak imkânsızdır (history her zaman önce gelir).
    """
    if set(history.fields) != set(future.fields):
        raise ValueError("concat_market: alan kümeleri farklı "
                         f"({sorted(set(history.fields) ^ set(future.fields))})")
    if history.bars_per_year != future.bars_per_year:
        raise ValueError("concat_market: bar frekansları farklı "
                         f"({history.bars_per_year} vs {future.bars_per_year})")
    fields = {}
    for k, hv in history.fields.items():
        fv = future.fields[k]
        cols = list(dict.fromkeys([*hv.columns, *fv.columns]))   # birleşim, sıra korunur
        fields[k] = pd.concat([hv.reindex(columns=cols), fv.reindex(columns=cols)])
    return MarketData(fields=fields, sectors=history.sectors or future.sectors,
                      bars_per_year=history.bars_per_year)


def gen_random(n_sec=20, n_days=750, seed=0) -> MarketData:
    """Öngörülemez rastgele yürüyüş — hiçbir alpha OLMAMALI."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.02, size=(n_days, n_sec))
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    return _prices_from_returns(returns, dates, tickers)


def gen_cross_sectional_momentum(n_sec=20, n_days=750, seed=0,
                                 drift_spread=0.0008) -> MarketData:
    """
    Kalıcı kesitsel momentum: her varlığın gizli bir drift'i var. Geçmiş
    getiri, gelecekteki getiriyi kesitsel olarak öngörür. 'Geçmiş getiriye
    göre sırala, kazananı tut' stratejisi POZİTİF Sharpe vermeli.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, drift_spread, size=n_sec)          # varlık başına gizli drift
    noise = rng.normal(0.0, 0.02, size=(n_days, n_sec))
    returns = mu[None, :] + noise
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    return _prices_from_returns(returns, dates, tickers)


def gen_interaction_alpha(n_sec=40, n_days=1400, seed=0, beta=0.5,
                          noise=0.02) -> MarketData:
    """ETKİLEŞİM alpha'sı — araştırma VERİMLİLİĞİNİ ölçmek için zor benchmark.

    Neden gerekli: gen_cross_sectional_momentum'da sinyal TEK faktörlü ve o kadar
    kolay ki rastgele arama bile birkaç denemede buluyor (ölçüldü: random-search
    11 backtest'te Sharpe +7.9). Herkes tavanı buluyorsa "kim daha iyi arıyor"
    sorusu CEVAPSIZ kalır — benchmark ayırt etmiyor demektir.

    Buradaki gerçek alpha bir ÇARPIMDA gizli:
        getiri_t = beta * (momentum_z_{t-1} * hacim_z_{t-1}) + gürültü

    Kritik özellik: momentum ve hacim TEK BAŞLARINA öngörücü DEĞİL —
        E[getiri | momentum] = beta * momentum_z * E[hacim_z] = 0
        E[getiri | hacim]    = beta * hacim_z * E[momentum_z] = 0
    (ikisi de kesitsel standardize edildiği için ortalamaları 0). Yani tek-faktörlü
    stratejiler ~0 alır; sinyali bulmak için iki faktörü BİRLEŞTİRMEK şart.
    Ekonomik karşılığı: "hacimle teyitli momentum".
    """
    rng = np.random.default_rng(seed)
    # Hacim: gözlenebilir, dışsal (getiriden bağımsız üretilir)
    volume = rng.lognormal(mean=14.0, sigma=0.5, size=(n_days, n_sec))
    log_vol = np.log(volume)

    def _z(x):  # kesitsel standardize (satır bazında)
        return (x - x.mean()) / (x.std() + 1e-9)

    returns = np.zeros((n_days, n_sec))
    warmup = 25
    returns[:warmup] = rng.normal(0.0, noise, size=(warmup, n_sec))
    for t in range(warmup, n_days):
        mom = returns[t - 20:t].sum(axis=0)        # son 20 günün getirisi
        inter = _z(_z(mom) * _z(log_vol[t - 1]))   # ETKİLEŞİM (gecikmeli, gözlenebilir)
        returns[t] = beta * noise * inter + rng.normal(0.0, noise, size=n_sec)

    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    md = _prices_from_returns(returns, dates, tickers)
    # Üretilen GERÇEK hacmi koy (etkileşimin ikinci bacağı gözlenebilir olmalı)
    vol_df = pd.DataFrame(volume, index=dates, columns=tickers)
    md.fields["volume"] = vol_df
    md.fields["dollar_volume"] = md.fields["close"] * vol_df
    return md


def gen_short_term_reversal(n_sec=20, n_days=750, seed=0, phi=0.25) -> MarketData:
    """
    Kısa vadeli reversal: r_{t} = -phi * r_{t-1} + gürültü. 'Dünün getirisini
    tersine çevir' (negate) stratejisi POZİTİF Sharpe vermeli.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.02, size=(n_days, n_sec))
    returns = np.zeros_like(noise)
    returns[0] = noise[0]
    for t in range(1, n_days):
        returns[t] = -phi * returns[t - 1] + noise[t]
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_sec)]
    return _prices_from_returns(returns, dates, tickers)
