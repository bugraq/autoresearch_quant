"""
EDGAR point-in-time testleri — temel veri DOĞRU dönemden mi geliyor?

İki hata gerçek cache verisiyle ölçülüp kapatıldı:

1) AYNI DOSYALAMADA BİRDEN ÇOK DÖNEM. Bir 10-K cari dönemi VE önceki yılların
   karşılaştırmalarını aynı `filed` tarihiyle raporlar. Eski kod
   `sorted((filed, val))` + `keep="last"` ile aynı tarihteki kayıtlardan EN
   BÜYÜK DEĞERİ seçiyordu. Gerçek örnek (CIK 0000002488, filed 2011-02-18):
   2007 özsermayesi (3.23 milyar) seçiliyordu, 2010'unki (1.01 milyar) yerine
   — 3.2 kat yanlış, 3 yıl bayat. 175 dosyalamanın 64'ü çok kayıtlıydı.

2) DÖNEM UZUNLUĞU KARIŞIMI. Net kâr hem 3-aylık hem 12-aylık raporlanıyor
   (ölçüm: 144 çeyrek / 42 yıllık / 26 yarım / 26 dokuz-aylık). Karışık
   havuz, kesitsel sıralamada bir şirketin ÇEYREK kârını başkasının YILLIK
   kârıyla yarıştırır.

Üçüncüsü hâlâ garanti altında: `filed > tarih` olan kayıt o günde BİLİNMEZ
(look-ahead yapısal olarak imkânsız) — bu da test ediliyor.
"""
import numpy as np
import pandas as pd

from data.edgar import _pit_series


def _r(filed: str, end: str, val: float, start: "str | None" = None) -> dict:
    d = {"filed": filed, "end": end, "val": val}
    if start:
        d["start"] = start
    return d


def test_ayni_dosyalamada_EN_GUNCEL_DONEM_secilir():
    """ASIL TEST: en büyük değer değil, en güncel dönem kazanmalı."""
    kayitlar = [                       # hepsi AYNI gün dosyalandı (10-K)
        _r("2011-02-18", "2007-12-29", 3_230_000_000),   # en BÜYÜK, en ESKİ
        _r("2011-02-18", "2008-12-27",   127_000_000),
        _r("2011-02-18", "2009-12-26",   648_000_000),
        _r("2011-02-18", "2010-12-25", 1_013_000_000),   # en GÜNCEL -> doğru
    ]
    s = _pit_series(kayitlar, pd.DatetimeIndex(["2011-03-01"]))
    assert s.iloc[0] == 1_013_000_000, (
        f"en güncel dönem yerine {s.iloc[0]:,.0f} seçildi — aynı dosyalamadaki "
        f"kayıtlar hâlâ DEĞERE göre sıralanıyor olabilir")
    print("  [ok] aynı dosyalamada en GÜNCEL dönem seçiliyor (en büyük değer değil)")


def test_look_ahead_yapisal_olarak_imkansiz():
    """filed > tarih olan kayıt o günde BİLİNEMEZ."""
    kayitlar = [_r("2020-02-15", "2019-12-31", 100.0),
                _r("2021-02-15", "2020-12-31", 999.0)]   # GELECEKTE dosyalandı
    d = pd.DatetimeIndex(["2020-06-01", "2021-06-01"])
    s = _pit_series(kayitlar, d)
    assert s.iloc[0] == 100.0, "açıklanmamış veri kullanıldı (look-ahead!)"
    assert s.iloc[1] == 999.0, "açıklandıktan sonra güncel değer kullanılmıyor"
    print("  [ok] look-ahead yok: dosyalanmadan önceki değer görünmüyor")


def test_dosyalama_oncesi_NaN():
    s = _pit_series([_r("2020-02-15", "2019-12-31", 100.0)],
                    pd.DatetimeIndex(["2019-06-01", "2020-06-01"]))
    assert np.isnan(s.iloc[0]), "ilk dosyalamadan ÖNCE değer üretildi"
    assert s.iloc[1] == 100.0
    print("  [ok] ilk dosyalamadan önce NaN (uydurma değer yok)")


def test_annual_only_ceyrekleri_eler():
    kayitlar = [
        _r("2020-05-01", "2020-03-31",  50.0, start="2020-01-01"),   # ~çeyrek
        _r("2020-05-01", "2020-03-31", 200.0, start="2019-04-01"),   # ~yıllık
    ]
    d = pd.DatetimeIndex(["2020-06-01"])
    karisik = _pit_series(kayitlar, d, annual_only=False).iloc[0]
    yillik = _pit_series(kayitlar, d, annual_only=True).iloc[0]
    assert yillik == 200.0, f"yıllık filtre çeyreği elemedi: {yillik}"
    assert karisik in (50.0, 200.0)      # filtresiz: hangisi geldiği belirsiz
    print("  [ok] annual_only çeyreklik dönemleri eliyor")


def test_annual_only_hicbiri_yillik_degilse_NaN():
    """Yıllık kayıt yoksa uydurma yapılmaz — çeyreğe düşülmez."""
    kayitlar = [_r("2020-05-01", "2020-03-31", 50.0, start="2020-01-01")]
    s = _pit_series(kayitlar, pd.DatetimeIndex(["2020-06-01"]), annual_only=True)
    assert np.isnan(s.iloc[0]), "yıllık yokken çeyreklik değere düşüldü"
    print("  [ok] yıllık kayıt yoksa NaN (çeyreğe düşmüyor)")


def test_stok_kalemi_annual_only_ile_kaybolmaz():
    """Özsermaye/hisse sayısı ANLIKTIR (start yok); yıllık filtre onları elemez.

    fundamentals() bunlarda annual_only KULLANMAZ; bu test o sözleşmeyi korur:
    yanlışlıkla açılırsa panel tamamen boşalır ve sessizce NaN döner.
    """
    anlik = [_r("2020-02-15", "2019-12-31", 100.0)]        # start YOK
    d = pd.DatetimeIndex(["2020-06-01"])
    assert _pit_series(anlik, d, annual_only=False).iloc[0] == 100.0
    assert np.isnan(_pit_series(anlik, d, annual_only=True).iloc[0]), (
        "anlık kalemde annual_only açılırsa panel boşalır — fundamentals() "
        "onu stok kalemlerinde KULLANMAMALI")
    print("  [ok] stok kalemi sözleşmesi korunuyor (annual_only yalnız akışta)")


def test_bozuk_kayit_cokertmez():
    kayitlar = [{"filed": "2020-02-15", "end": "2019-12-31", "val": None},
                {"filed": "yok", "end": "2019-12-31", "val": 1.0},
                _r("2020-02-15", "2019-12-31", 100.0)]
    s = _pit_series(kayitlar, pd.DatetimeIndex(["2020-06-01"]))
    assert s.iloc[0] == 100.0, "bozuk kayıtlar sağlam olanı bozdu"
    assert np.isnan(_pit_series([], pd.DatetimeIndex(["2020-06-01"])).iloc[0])
    print("  [ok] bozuk/boş kayıtlar çökertmiyor")


def test_fundamentals_net_karda_yillik_kullaniyor():
    """Sözleşme testi: akış kalemi yıllık, stok kalemi anlık okunmalı."""
    import inspect

    from data import edgar
    src = inspect.getsource(edgar.fundamentals)
    assert "NET_INCOME, ciks, annual_only=True" in src, \
        "net kâr yıllık filtreyle çekilmiyor (çeyrek/yıllık karışımı geri geldi)"
    assert "BOOK_EQUITY, ciks)" in src and "SHARES, ciks)" in src, \
        "stok kalemlerine yanlışlıkla yıllık filtre uygulanmış olabilir"
    print("  [ok] fundamentals: net kâr yıllık, özsermaye/hisse anlık")


def main() -> None:
    test_ayni_dosyalamada_EN_GUNCEL_DONEM_secilir()
    test_look_ahead_yapisal_olarak_imkansiz()
    test_dosyalama_oncesi_NaN()
    test_annual_only_ceyrekleri_eler()
    test_annual_only_hicbiri_yillik_degilse_NaN()
    test_stok_kalemi_annual_only_ile_kaybolmaz()
    test_bozuk_kayit_cokertmez()
    test_fundamentals_net_karda_yillik_kullaniyor()
    print("OK — EDGAR point-in-time testleri geçti (doğru dönem, look-ahead yok).")


if __name__ == "__main__":
    main()
