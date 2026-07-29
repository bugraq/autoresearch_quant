"""
SEC EDGAR fundamentals — ÜCRETSİZ + gerçek point-in-time temel veri.

Neden: fiyat/hacim sinyalleri kalabalık ve rejime-bağımlı (holdout eliyor). Value
(book-to-market) ve quality (ROE) faktörleri onlarca yıl out-of-sample DAYANMIŞ
(Fama-French); ekonomik mekanizması var → rejimler arası kalıcı olma şansı yüksek.

EDGAR companyfacts API her mali kalemi GERÇEK DOSYALAMA TARİHİYLE (`filed`) verir.
Point-in-time'ın anahtarı bu: bir gün için yalnızca O GÜNE KADAR DOSYALANMIŞ değeri
kullanırız → bilanço/kazanç açıklanmadan önce onu bilmek (look-ahead) YAPISAL olarak
imkânsız. yfinance ~1.5 yıl verirken EDGAR ~20 yıl + dosyalama tarihi verir.

Nezaket: SEC User-Agent ister ve ~10 istek/sn sınırı var. Ham veri cache'lenir.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

_UA = "agentic-quant research (yusufbugra4416@gmail.com)"
_CACHE = os.path.join(os.path.dirname(__file__), "edgar_cache")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def ticker_to_cik(tickers: "list[str]") -> "dict[str, str]":
    """Ticker -> 10 haneli CIK (SEC company_tickers.json, cache'li)."""
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, "company_tickers.json")
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
    else:
        data = _get("https://www.sec.gov/files/company_tickers.json")
        json.dump(data, open(path, "w", encoding="utf-8"))
    lookup = {row["ticker"].upper(): f"{int(row['cik_str']):010d}"
              for row in data.values()}
    return {t: lookup[t.upper()] for t in tickers if t.upper() in lookup}


def _concept(cik: str, concept: str) -> "list[dict] | None":
    """us-gaap kavramının (filed, end, val) kayıtları. Cache'li; yoksa None."""
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"{cik}_{concept}.json")
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
    else:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        try:
            d = _get(url)
            time.sleep(0.12)   # SEC ~10 istek/sn
        except Exception:      # noqa: BLE001 — bu şirkette bu kavram yoksa
            json.dump({}, open(path, "w", encoding="utf-8"))
            return None
        json.dump(d, open(path, "w", encoding="utf-8"))
    # Değer USD'de (defter/kâr) VEYA shares'te (hisse sayısı) raporlanır — ikisini de dene.
    u = d.get("units", {}) if d else {}
    return u.get("USD") or u.get("shares") or None


#: Yıllık sayılan dönem uzunluğu (gün). Mali yıllar 52/53 hafta olabildiği için
#: tam 365 aranmaz.
_ANNUAL_MIN, _ANNUAL_MAX = 300, 400


def _pit_series(records: "list[dict]", dates: pd.DatetimeIndex,
                annual_only: bool = False) -> pd.Series:
    """Kayıtları POINT-IN-TIME günlük seriye çevir: her gün, O GÜNE KADAR
    DOSYALANMIŞ EN GÜNCEL DÖNEMİN değeri.

    İKİ HATA BURADA KAPATILDI (ölçüldü, bkz. aşağıdaki gerçek örnek):

    1) AYNI DOSYALAMADA BİRDEN ÇOK DÖNEM. Bir 10-K, cari dönemi VE önceki
       yılların karşılaştırmalarını AYNI `filed` tarihiyle raporlar. Eski kod
       `sorted((filed, val))` + `keep="last"` yapıyordu; yani aynı tarihteki
       kayıtlardan EN BÜYÜK DEĞERİ seçiyordu — en güncel dönemi değil.
       Gerçek örnek (CIK 0000002488, filed 2011-02-18, 4 kayıt):
           2007 -> 3.230.000.000   <- eski kod BUNU seçiyordu
           2008 ->   127.000.000
           2009 ->   648.000.000
           2010 -> 1.013.000.000   <- doğrusu (en güncel dönem)
       3.2 kat yanlış ve 3 yıl bayat. 175 dosyalamanın 64'ü çok kayıtlıydı.
       Düzeltme: sıralama (filed, END) — değere göre DEĞİL.

    2) DÖNEM UZUNLUĞU KARIŞIMI (annual_only). Akış kalemleri (net kâr) hem
       3-aylık hem 12-aylık raporlanır (ölçüldü: 144 çeyrek / 42 yıllık /
       26 yarım / 26 dokuz-aylık). Bunları aynı havuzda kullanmak, kesitsel
       sıralamada bir şirketin ÇEYREK kârını başkasının YILLIK kârıyla
       yarıştırır — elma-armut. annual_only=True yalnız ~yıllık dönemleri alır.
       (Stok kalemleri — özsermaye, hisse sayısı — anlıktır; onlarda kapalı.)

    filed > date olan kayıt o günde BİLİNMEZ: look-ahead yapısal olarak imkânsız.
    """
    rows = []
    for r in records:
        if annual_only:
            start, end = r.get("start"), r.get("end")
            if not start or not end:
                continue                     # anlık kalem: yıllık filtresi anlamsız
            gun = (pd.Timestamp(end) - pd.Timestamp(start)).days
            if not (_ANNUAL_MIN <= gun <= _ANNUAL_MAX):
                continue
        try:
            rows.append((pd.Timestamp(r["filed"]),
                         pd.Timestamp(r.get("end") or r["filed"]),
                         float(r["val"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return pd.Series(np.nan, index=dates)

    # (filed, END) — aynı dosyalamada EN GÜNCEL DÖNEM kazanır (değer DEĞİL).
    rows.sort(key=lambda x: (x[0], x[1]))
    filed = pd.DatetimeIndex([r[0] for r in rows])
    vals = pd.Series([r[2] for r in rows], index=filed)
    vals = vals[~vals.index.duplicated(keep="last")]
    # her tarih için: <= tarih olan son dosyalamanın değeri (as-of)
    return vals.reindex(vals.index.union(dates)).ffill().reindex(dates)


def pit_panel(tickers: "list[str]", dates: pd.DatetimeIndex, concept: str,
              ciks: "dict[str,str] | None" = None,
              annual_only: bool = False) -> pd.DataFrame:
    """Bir us-gaap kavramının POINT-IN-TIME günlük paneli (tarih × ticker).

    annual_only: AKIŞ kalemlerinde (net kâr) zorunlu — çeyrek/yıllık karışımı
    kesitsel sıralamayı elma-armut yapar (bkz. _pit_series).
    """
    ciks = ciks or ticker_to_cik(tickers)
    out = {}
    for t in tickers:
        cik = ciks.get(t.upper())
        recs = _concept(cik, concept) if cik else None
        out[t] = (_pit_series(recs, dates, annual_only) if recs
                  else pd.Series(np.nan, index=dates))
    return pd.DataFrame(out, index=dates)


# Value/quality için gereken kavramlar (fallback'li: şirketler farklı etiket kullanır)
BOOK_EQUITY = ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
NET_INCOME = ["NetIncomeLoss", "ProfitLoss"]
SHARES = ["CommonStockSharesOutstanding", "CommonStockSharesIssued"]


def _first_available(tickers, dates, concepts, ciks,
                     annual_only: bool = False) -> pd.DataFrame:
    """Kavram listesinden ilk dolu olanı kullan (ticker bazında birleştir)."""
    panel = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for concept in concepts:
        p = pit_panel(tickers, dates, concept, ciks, annual_only)
        panel = panel.where(panel.notna(), p)   # boş olanları doldur
    return panel


def fundamentals(tickers: "list[str]", dates: pd.DatetimeIndex,
                 price: pd.DataFrame) -> "dict[str, pd.DataFrame]":
    """PIT value/quality faktörleri -> {book_to_market, roe} panelleri.

    book_to_market = defter özsermayesi / piyasa değeri (piyasa değeri = fiyat×hisse)
    roe            = net kâr / defter özsermayesi
    Hepsi dosyalama-tarihi hizalı (look-ahead yok). Kavramı olmayan hisse NaN
    (işlem görmez).

    NET KÂR: TTM toplama YAPILMAZ; bunun yerine yalnızca ~YILLIK dönemler
    kullanılır (annual_only). Eski docstring "4-çeyrek toplanır" diyordu ama
    kodda öyle bir toplama YOKTU — çeyreklik ve yıllık kayıtlar aynı havuzda
    yarışıyordu. Yıllık filtre daha basit ve kesitsel olarak tutarlıdır
    (Fama-French value/quality de yıllık muhasebe verisiyle çalışır).
    """
    ciks = ticker_to_cik(tickers)
    book = _first_available(tickers, dates, BOOK_EQUITY, ciks)
    # NET KÂR bir AKIŞ kalemidir: yalnız ~yıllık dönemler alınır. Aksi hâlde
    # bir şirketin ÇEYREK kârı başkasının YILLIK kârıyla yarışır (elma-armut)
    # ve kesitsel ROE sıralaması anlamsızlaşır.
    ni = _first_available(tickers, dates, NET_INCOME, ciks, annual_only=True)
    shares = _first_available(tickers, dates, SHARES, ciks)

    mktcap = price.reindex_like(book) * shares
    book_to_market = (book / mktcap.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    roe = (ni / book.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return {"book_to_market": book_to_market, "roe": roe}


def add_fundamentals(md) -> None:
    """Yüklenmiş MarketData'ya PIT value/quality alanlarını EKLE (yerinde).

    md.fields'e book_to_market ve roe panelleri eklenir; kripto gibi CIK'i olmayan
    semboller NaN kalır (o semboller bu alanlarda işlem görmez — sorun değil).
    Kesitsel long-short için ham değer yeterli (motor sıralar).
    """
    close = md.get("close")
    f = fundamentals(list(close.columns), close.index, close)
    md.fields["book_to_market"] = f["book_to_market"].reindex_like(close)
    md.fields["roe"] = f["roe"].reindex_like(close)
