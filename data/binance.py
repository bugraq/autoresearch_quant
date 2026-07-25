"""
Binance perpetual futures — funding rate + OHLCV (ÜCRETSİZ, anahtarsız).

NEDEN: dört domain (S&P large-cap, kripto, fundamentals, small-cap) denendi,
dördü de null. Hepsinin ortak yanı AYNI ham malzeme: fiyat + hacim. Small-cap
maliyet ablasyonu kanıtladı ki maliyet SIFIRKEN bile sinyal yok → sorun piyasa
değil, bakılan BİLGİ. Fiyat/hacim herkesin elinde.

FUNDING RATE = hisselerde OLMAYAN bilgi. Perpetual futures'ta spot fiyata
yakınsamayı sağlayan periyodik ödeme: funding pozitifse long'lar short'lara öder
(long'lar kalabalık ve kaldıraçlı) → likidasyon riski birikir → sonraki getiri
düşme eğilimi. Bu bir POZİSYONLANMA/duygu sinyali ve fiyat-hacim panelinde
GÖRÜNMEZ. Mekanizması sağlam (kalabalık kaldıraçlı taraf tasfiye edilir).

ZAMAN HİZALAMASI (sızıntının kritik noktası)
--------------------------------------------
Funding 8 saatte bir ödenir (00:00 / 08:00 / 16:00 UTC). Günlük bar [00:00,
23:59] içinde bu üç ödeme de gerçekleşir → günün TOPLAM funding'i ancak gün
KAPANIŞINDA bilinir. Bu yüzden günlük funding alanının info_tick'i close_t'dir
(FIELD_BASE_TICK["funding_rate"] = 1) — open_t'de bilindiğini varsaymak
geleceğe bakmak olurdu. Ödemeler yalnızca gerçekleştikleri güne yazılır;
ileri taşıma (ffill) YAPILMAZ.

SURVIVORSHIP (bu modülün asıl kazancı)
--------------------------------------
yfinance yalnızca BUGÜN yaşayan coinleri verir → LUNA, FTT gibi çöken coinler
panelde yoktur ve backtest yapay olarak kazananları seçer. Small-cap'te
öğrenildi: bu yanlılık araştırma+holdout'u AYNI yönde bozar → HOLDOUT ONU
YAKALAYAMAZ. Binance ise delist olmuş sembollerin geçmişini SİLMEZ
(doğrulandı: LUNAUSDT funding+fiyat 2022-05-13 çöküşüne kadar geliyor).
Bu yüzden evren = yaşayanlar + DEAD_SYMBOLS (bilinen delist'ler).

KALAN DÜRÜST SINIR: exchangeInfo yalnız yaşayanları listeler; ölü sembol havuzu
elle derlenir → EKSİK olabilir (tam liste için Binance delist duyuru arşivi
gerekir). Yine de yfinance'e göre büyük iyileşme; sayı yüklemede raporlanır.
"""
from __future__ import annotations

import json
import os
import time
import pandas as pd

from data.synthetic import BARS_PER_YEAR_CRYPTO_8H, BARS_PER_YEAR_CRYPTO_DAILY, MarketData

_BASE = "https://fapi.binance.com"
_UA = "agentic-quant-research/0.1 (akademik staj projesi)"
_CACHE = os.path.join(os.path.dirname(__file__), "binance_cache")
_LIMIT = 1000            # Binance sayfa başına azami kayıt
_SLEEP = 0.10            # nezaket: ~10 istek/sn


class SymbolNotFound(Exception):
    """Binance bu sembolü tanımıyor (HTTP 400) — 'veri yok', arıza DEĞİL."""


_session = None


def _get_session():
    """Tek bir keep-alive oturumu. ÖNEMLİ: her istekte yeni TCP bağlantısı açmak
    (urllib'in yaptığı) yüzlerce sembolde Binance'in bağlantıyı zorla kapatmasına
    yol açıyordu (WinError 10054). Session bağlantıyı yeniden kullanır."""
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers.update({"User-Agent": _UA})
    return _session


def _get(path: str, params: dict) -> list | dict:
    """Binance GET; geçici arızalarda (bağlantı kesilmesi, rate-limit) artan
    beklemeyle yeniden dener. Kalıcı 'sembol yok' (400) ayrı tipe çevrilir ki
    çağıran onu geçici arızayla karıştırmasın."""
    import requests

    sess = _get_session()
    url = f"{_BASE}{path}"
    last: Exception | None = None
    for attempt in range(5):
        try:
            r = sess.get(url, params=params, timeout=30)
            if r.status_code == 400:
                raise SymbolNotFound(params.get("symbol", "?"))
            if r.status_code in (429, 418):        # rate-limit / geçici ban
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e                                # bağlantı kesildi: bekle, tekrar dene
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Binance erişilemedi ({path} {params.get('symbol','')}): {last}")


def _ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


# Bilinen delist/settle olmuş USDT perpetual'lar (survivorship kapatma).
# exchangeInfo bunları VERMEZ ama geçmiş verileri Binance'te DURUYOR.
# Eksik olabilir — dürüstçe belgelendi (yüklemede kaç tanesinin verisi geldiği raporlanır).
DEAD_SYMBOLS = [
    "LUNAUSDT",    # Terra çöküşü (2022-05) — kripto tarihinin en büyük sıfırlanması
    "FTTUSDT",     # FTX çöküşü (2022-11)
    "SRMUSDT", "RAYUSDT",           # FTX/Alameda ekosistemi
    "ANCUSDT", "MIRUSDT",           # Terra ekosistemi
    "BTCSTUSDT", "COCOSUSDT",       # manipülasyon/delist
    "SCUSDT", "RGTUSDT", "TLMUSDT", "CVCUSDT", "BTSUSDT",
    "HNTUSDT", "TOMOUSDT", "CTKUSDT", "AGIXUSDT", "OCEANUSDT",
    "FTMUSDT", "MATICUSDT",         # yeniden adlandırma (S / POL) — eski sembol öldü
    "WAVESUSDT", "DGBUSDT", "KEYUSDT", "NULSUSDT", "REEFUSDT",
    "STORMUSDT", "DODOUSDT", "BTCDOMUSDT",
]


def live_symbols() -> list[str]:
    """Şu an listelenen USDT perpetual sembolleri (TRADING + SETTLING)."""
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, "exchange_info.json")
    if os.path.exists(path):
        info = json.load(open(path, encoding="utf-8"))
    else:
        info = _get("/fapi/v1/exchangeInfo", {})
        json.dump(info, open(path, "w", encoding="utf-8"))
    return [s["symbol"] for s in info["symbols"]
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") in ("TRADING", "SETTLING")]


def universe(include_dead: bool = True) -> list[str]:
    """Evren = yaşayanlar + bilinen ölüler (survivorship düzeltmesi)."""
    syms = live_symbols()
    if include_dead:
        syms = syms + [s for s in DEAD_SYMBOLS if s not in syms]
    return sorted(set(syms))


def _paged(path: str, symbol: str, start: str, end: str, tkey, **extra) -> list:
    """Sayfalı geçmiş çekimi (Binance tek istekte en fazla 1000 kayıt verir).

    `extra`: endpoint'e özel zorunlu parametreler (klines `interval` ister).
    """
    out: list = []
    cur, end_ms = _ms(start), _ms(end)
    while cur < end_ms:
        batch = _get(path, {"symbol": symbol, "startTime": cur,
                            "endTime": end_ms, "limit": _LIMIT, **extra})
        if not batch:
            break
        out.extend(batch)
        last = batch[-1][tkey] if isinstance(batch[-1], dict) else batch[-1][0]
        nxt = int(last) + 1
        if nxt <= cur:            # ilerlemiyorsa sonsuz döngüyü kes
            break
        cur = nxt
        time.sleep(_SLEEP)
        if len(batch) < _LIMIT:   # son sayfa
            break
    return out


def _cached(name: str, fetch) -> list:
    """Cache'li çekim. YALNIZCA 'bu sembol/veri yok' (HTTP 400) boş olarak
    cache'lenir; ağ/rate-limit hataları YÜKSELTİLİR — yoksa geçici bir arıza
    kalıcı boş cache'e dönüşür ve sessizce eksik veriyle araştırma yapılır."""
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, name)
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    try:
        data = fetch()
    except SymbolNotFound:        # sembol yok = gerçekten boş; cache'lenebilir
        data = []
    json.dump(data, open(path, "w", encoding="utf-8"))
    return data


def funding_series(symbol: str, start: str, end: str,
                   interval: str = "1d") -> pd.Series:
    """Funding'i bar frekansına indir.

    ZAMAN HİZALAMASI (sızıntının kalbi) — funding 00/08/16 UTC'de ödenir ve
    ödeme, KENDİNDEN ÖNCEKİ 8 saatlik periyoda aittir:
      * interval='1d': barın [00:00, 24:00) içindeki ödemeler TOPLANIR; toplam
        ancak gün kapanışında bilinir -> info_tick = close_t.
      * interval='8h': bar [00:00, 08:00) ve onun funding'i 08:00'de ödenir =
        TAM BAR KAPANIŞI. Yani her bara BİR ödeme düşer, bilgi EZİLMEZ ve yine
        kapanışta bilinir -> info_tick = close_t (aynı kural, daha keskin veri).
    Günlük barda 3 ödemeyi toplamak bilginin çoğunu eziyordu; 8h bar bunu çözer.

    Ödeme olmayan bar NaN kalır — ffill YAPILMAZ (funding bir OLAYDIR, durum
    değil; ileri taşımak olmayan bir ödemeyi varmış gibi gösterirdi).
    """
    recs = _cached(f"{symbol}_funding_{start}_{end}.json",
                   lambda: _paged("/fapi/v1/fundingRate", symbol, start, end, "fundingTime"))
    if not recs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(recs)
    ts = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    rate = df["fundingRate"].astype(float)

    # ÖDEME -> PERİYOT eşlemesi. Binance damgaları TAM SAAT DEĞİL (ölçüldü:
    # 00:00:00.006, 08:00:00.009) — bu yüzden saat aritmetiğiyle oynamak yerine
    # tanımı doğrudan uyguluyoruz: t'de ödenen funding [t-8s, t) periyoduna aittir.
    # floor('8h') milisaniye gecikmesini temizler, -8s periyot başına götürür.
    period_start = ts.dt.floor("8h") - pd.Timedelta(hours=8)
    # 8h bar: periyot başı = barın kendisi (bar [00:00,08:00) <- ödeme 08:00'de).
    # 1d  bar: periyodu içeren GÜN. Böylece bir günün funding'i o günün
    #          [00:00,24:00) aralığına ait 3 ödemedir (08:00, 16:00, ertesi 00:00)
    #          ve hepsi gün kapanışında bilinir. (Eski normalize() ödemeyi ait
    #          OLMADIĞI güne yazıyordu: 00:00'daki ödeme aslında ÖNCEKİ gecenin.)
    bar = period_start if interval == "8h" else period_start.dt.normalize()
    return rate.groupby(bar).sum()


def ohlcv_frame(symbol: str, start: str, end: str,
                interval: str = "1d") -> pd.DataFrame:
    """OHLCV (UTC bar açılış etiketiyle). interval: '1d' | '8h' | '4h' | '1h'."""
    recs = _cached(f"{symbol}_klines_{interval}_{start}_{end}.json",
                   lambda: _paged("/fapi/v1/klines", symbol, start, end, 0,
                                  interval=interval))  # interval ZORUNLU (yoksa HTTP 400)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    ts = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["date"] = ts.dt.normalize() if interval == "1d" else ts
    df = df.drop(columns=["open_time"]).set_index("date").astype(float)
    return df[~df.index.duplicated(keep="last")]


class BinanceAdapter:
    """Binance USDT perpetual evreni: OHLCV + funding_rate.

    Evren survivorship-düzeltilmiş (yaşayanlar + bilinen ölüler). Bir coin
    yalnızca verisi olduğu günlerde işlem görür (`index_membership`), böylece
    LUNA gibi çöküp delist olanlar çöküşe kadar panelde durur, sonra düşer.
    """

    _BARS_PER_YEAR = {"1d": BARS_PER_YEAR_CRYPTO_DAILY, "8h": BARS_PER_YEAR_CRYPTO_8H}

    def __init__(self, start: str, end: str, symbols: "list[str] | None" = None,
                 include_dead: bool = True, max_symbols: "int | None" = None,
                 interval: str = "1d") -> None:
        self.start, self.end = str(start), str(end)
        self.interval = str(interval)
        if self.interval not in self._BARS_PER_YEAR:
            raise ValueError(
                f"Desteklenmeyen bar aralığı: {interval!r}. Yıllıklaştırma sabiti "
                f"bilinmeyen bir frekansta Sharpe ölçeği bozulur (hard gate haksız "
                f"karar verir). Desteklenen: {sorted(self._BARS_PER_YEAR)}")
        self.symbols = symbols or universe(include_dead=include_dead)
        if max_symbols:
            self.symbols = self.symbols[:int(max_symbols)]

    def load(self) -> MarketData:
        opens, highs, lows, closes, vols, fundings = {}, {}, {}, {}, {}, {}
        empty: list[str] = []
        no_funding: list[str] = []
        for i, sym in enumerate(self.symbols, 1):
            bars = ohlcv_frame(sym, self.start, self.end, self.interval)
            if bars.empty:
                empty.append(sym)
                continue
            opens[sym], highs[sym] = bars["open"], bars["high"]
            lows[sym], closes[sym] = bars["low"], bars["close"]
            vols[sym] = bars["volume"]
            f = funding_series(sym, self.start, self.end, self.interval)
            if f.empty:
                no_funding.append(sym)
            else:
                fundings[sym] = f
            if i % 50 == 0:
                print(f"  [binance] {i}/{len(self.symbols)} sembol indirildi", flush=True)

        if not closes:
            raise RuntimeError("Binance hiç veri döndürmedi (ağ/rate-limit?).")

        close = pd.DataFrame(closes).sort_index()
        idx = close.index
        def _al(d: dict) -> pd.DataFrame:
            return pd.DataFrame(d).reindex(idx)

        close = _al(closes)
        volume = _al(vols)
        funding = _al(fundings).reindex(columns=close.columns)

        # index_membership: coin YALNIZCA fiyat verisi olan günlerde işlem görür.
        # LUNA gibi çökenler delist gününden sonra otomatik düşer (survivorship
        # düzeltmesinin çalıştığı yer).
        memb = close.notna().astype(float)

        fields = {
            "open": _al(opens), "high": _al(highs), "low": _al(lows),
            "close": close, "adjusted_close": close,   # perpetual: split/temettü yok
            "volume": volume,
            "dollar_volume": close * volume,
            "index_membership": memb,
            # YENİ BİLGİ: günün toplam funding'i (ffill YOK — ödeme bir olaydır)
            "funding_rate": funding,
        }
        dead_loaded = [s for s in DEAD_SYMBOLS if s in close.columns]
        print(f"  [binance] {len(close.columns)} sembol × {len(idx)} bar "
              f"({self.interval}) | funding'i olan: {len(fundings)} | "
              f"ölü coin yüklendi: {len(dead_loaded)}/{len(DEAD_SYMBOLS)}")
        if empty:
            print(f"  [binance] veri gelmeyen {len(empty)} sembol (örnek: {empty[:5]})")
        if no_funding:
            print(f"  [binance] UYARI: {len(no_funding)} sembolde funding YOK "
                  f"(o semboller funding alanında NaN): {no_funding[:5]}")
        # Yıllıklaştırma ölçeği veriyle birlikte taşınır: kripto 7/24 -> günlük 365
        # (hisse 252 DEĞİL), 8h bar -> 1095. Yanlışsa hard gate haksız karar verir.
        return MarketData(fields=fields, sectors=None,
                          bars_per_year=self._BARS_PER_YEAR[self.interval])
