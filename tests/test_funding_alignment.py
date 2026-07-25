"""Funding -> bar hizalama testleri (ağsız; Binance kayıt biçimi taklit edilir).

BULUNAN GERÇEK BUG: Binance funding damgaları TAM SAAT DEĞİL (gerçek veriden
ölçüldü: 00:00:00.006, 08:00:00.009, 16:00:00.000). İlk hizalama '1 nanosaniye
geri al + floor' yapıyordu; bu milisaniye gecikmesini yenemeyip ödemeleri
TUTARSIZ dağıtıyordu: tam saatte ödenen 16:00 kaydı bir önceki bara, milisaniye
gecikmeli 08:00 kaydı kendi barına gidiyordu -> bir bar iki ödeme yutuyor, bir
bar boş kalıyordu.

DOĞRUSU tanımın kendisi: t anında ödenen funding [t-8s, t) periyoduna aittir.
  8h bar: periyot başı = barın etiketi (bar [00:00,08:00) <- ödeme 08:00'de)
  1d  bar: periyodu içeren gün (bir günün funding'i = 08:00 + 16:00 + ertesi 00:00)

Bu ayrıca info_tick=close_t kuralını korur: bir barın funding'i o barın
KAPANIŞINDA ödenir/bilinir, açılışında değil.
"""
import json
import os
import tempfile

import pandas as pd

import data.binance as B


def _fake_records():
    """Binance'in gerçek biçimi: ms damgalar, tam saatte DEĞİL (ölçülmüş sapma)."""
    rows = [
        ("2022-01-01 00:00:00.006", 0.0001),   # [12-31 16:00, 01-01 00:00) periyodu
        ("2022-01-01 08:00:00.009", 0.0002),   # [01-01 00:00, 08:00)  periyodu
        ("2022-01-01 16:00:00.000", 0.0004),   # [01-01 08:00, 16:00)  periyodu
        ("2022-01-02 00:00:00.028", 0.0008),   # [01-01 16:00, 02-01 00:00) periyodu
        ("2022-01-02 08:00:00.000", 0.0016),   # [01-02 00:00, 08:00)  periyodu
    ]
    return [{"fundingTime": int(pd.Timestamp(t, tz="UTC").timestamp() * 1000),
             "fundingRate": str(r)} for t, r in rows]


def _patched(interval, tmpdir):
    """funding_series'i ağ olmadan, sahte cache üzerinden çalıştır."""
    old = B._CACHE
    B._CACHE = tmpdir
    try:
        path = os.path.join(tmpdir, "TESTUSDT_funding_2022-01-01_2022-01-03.json")
        json.dump(_fake_records(), open(path, "w", encoding="utf-8"))
        return B.funding_series("TESTUSDT", "2022-01-01", "2022-01-03", interval)
    finally:
        B._CACHE = old


def test_8h_one_payment_per_bar():
    """Her 8h bara TAM BİR ödeme düşer; bilgi ezilmez, bar atlanmaz."""
    with tempfile.TemporaryDirectory() as d:
        s = _patched("8h", d)
    beklenen = {
        "2021-12-31 16:00": 0.0001,   # 00:00'da ödendi -> ÖNCEKİ akşamın periyodu
        "2022-01-01 00:00": 0.0002,   # 08:00'de ödendi -> bar [00:00,08:00)
        "2022-01-01 08:00": 0.0004,   # 16:00'da ödendi -> bar [08:00,16:00)
        "2022-01-01 16:00": 0.0008,   # ertesi 00:00'da ödendi -> bar [16:00,24:00)
        "2022-01-02 00:00": 0.0016,
    }
    got = {str(k.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")): round(v, 6)
           for k, v in s.items()}
    assert got == beklenen, f"\ngelen  : {got}\nbeklenen: {beklenen}"
    assert len(s) == len(set(s.index)), "her bar bir kez görünmeli"
    print("  [ok] 8h: her bara tam bir ödeme, doğru periyoda hizalı")


def test_16h_bar_not_swallowed():
    """REGRESYON: eski hizalamada 16:00 barı kayboluyor, 08:00 iki ödeme yutuyordu."""
    with tempfile.TemporaryDirectory() as d:
        s = _patched("8h", d)
    saatler = sorted({k.hour for k in s.index})
    assert saatler == [0, 8, 16], f"16:00 barı kaybolmuş olabilir: {saatler}"
    print("  [ok] 16:00 barı yerinde (eski bug'ın regresyon testi)")


def test_daily_groups_correct_period():
    """Bir günün funding'i = o güne ait 3 ödeme (08:00 + 16:00 + ertesi 00:00).
    Eski normalize() 00:00'daki ödemeyi (ÖNCEKİ gecenin) o güne yazıyordu."""
    with tempfile.TemporaryDirectory() as d:
        s = _patched("1d", d)
    g = {str(k.strftime("%Y-%m-%d")): round(v, 6) for k, v in s.items()}
    # 01-01 günü: 0.0002 (08:00) + 0.0004 (16:00) + 0.0008 (ertesi 00:00)
    assert g["2022-01-01"] == 0.0014, g
    # 00:00:00.006'daki ödeme 12-31 akşamına ait -> 01-01'e YAZILMAMALI
    assert g["2021-12-31"] == 0.0001, g
    print("  [ok] günlük: ödemeler ait oldukları güne yazılıyor")


def test_daily_equals_sum_of_8h():
    """Tutarlılık: aynı veri, iki frekans -> günlük toplam == 8h barların toplamı."""
    with tempfile.TemporaryDirectory() as d:
        s8 = _patched("8h", d)
    with tempfile.TemporaryDirectory() as d:
        s1 = _patched("1d", d)
    toplam8 = s8.groupby(s8.index.normalize()).sum()
    ortak = s1.index.intersection(toplam8.index)
    assert len(ortak) > 0
    for k in ortak:
        assert abs(s1[k] - toplam8[k]) < 1e-12, f"{k}: {s1[k]} != {toplam8[k]}"
    print("  [ok] günlük == 8h toplamı (iki yol aynı veriyi veriyor)")


def main():
    test_8h_one_payment_per_bar()
    test_16h_bar_not_swallowed()
    test_daily_groups_correct_period()
    test_daily_equals_sum_of_8h()
    print("OK — funding hizalama testleri geçti.")


if __name__ == "__main__":
    main()
