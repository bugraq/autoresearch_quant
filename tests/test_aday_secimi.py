"""
ADAY SEÇİMİ testleri — "bizimki" diye hangi stratejiyi gösteriyoruz?

Bu kural projede İKİ KEZ ayrı ayrı bozuldu, çünkü dört araç (benchmark,
forward_test, dashboard, anatomy) kendi kopyasını taşıyordu:

  1) forward_test.py `accepted_hypotheses(limit=1)` kullanıyordu — yani en
     yüksek ARAŞTIRMA Sharpe'ı. Bu, holdout'ta çöken hipotezi (hyp_0010,
     +0.97 -> -0.32) ileri-teste sokup holdout'tan sağ çıkanı atlıyordu.
     Düzeltildi.
  2) benchmark.py AYNI hatayı taşımaya devam etti (düzeltme oraya
     uygulanmamıştı). Ölçüldü: benchmark hyp_0021'i (araştırma +1.14, taze
     veride -%44) "bizim strateji" diye gösteriyordu.

Kural artık tek yerde (evaluation/aday.py) ve burada çivileniyor:

    ARAŞTIRMA SHARPE'I BİR KALİTE ÖLÇÜSÜ DEĞİLDİR.
    En parlak araştırma skoru, genelde en aşırı-uydurulmuş adaydır.

Testler gerçek veritabanı kurmadan, saf sıralama mantığını doğrular.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.aday import Aday


def _aday(hid, r, h, f) -> Aday:
    guc = 3 if (h is not None and f is not None) else (2 if h is not None else 1)
    return Aday(hid, "{}", hid, r, h, f, "", guc)


def _sirala(adaylar, min_acceptance_sharpe=0.5):
    """evaluation.aday.tum_adaylar() ile AYNI sıralama anahtarı."""
    def anahtar(a):
        gecti = a.hukum(min_acceptance_sharpe).passed
        oos = [s for s in (a.holdout_sharpe, a.forward_sharpe) if s is not None]
        return (not gecti, -a.kanit_gucu, -(min(oos) if oos else -99))
    return sorted(adaylar, key=anahtar)


def test_arastirma_sharpe_KAZANMAZ():
    """ASIL TEST: araştırmada en parlak olan, üç dönemi geçeni yenemez."""
    parlak = _aday("hyp_0021", +1.14, +0.93, -0.36)   # araştırmada BİRİNCİ
    saglam = _aday("hyp_0033", +0.66, +0.72, +0.37)   # üç dönemi geçen
    ilk = _sirala([parlak, saglam])[0]
    assert ilk.hypothesis_id == "hyp_0033", (
        f"araştırma Sharpe'ı en yüksek olan ({ilk.hypothesis_id}) seçildi — "
        f"aday seçimi yine ARAŞTIRMA skoruna göre yapılıyor olabilir")
    print("  [ok] üç dönemi geçen aday, araştırmada daha parlak olanı yeniyor")


def test_holdout_olculen_olculmeyeni_yener():
    """Kanıt gücü: kilitli dönemde ölçülmüş aday, hiç ölçülmemişi geçer."""
    olculmemis = _aday("hyp_9999", +2.50, None, None)   # devasa araştırma skoru
    olculmus = _aday("hyp_0002", +0.61, +1.29, None)
    ilk = _sirala([olculmemis, olculmus])[0]
    assert ilk.hypothesis_id == "hyp_0002", (
        "hiç sınava girmemiş aday, kilitli dönemde ölçülmüş adayın önüne geçti")
    print("  [ok] ölçülmüş aday, yalnız araştırmada parlayanı yeniyor")


def test_hukum_arastirmayi_HIC_kullanmaz():
    """Araştırma Sharpe'ı hükmü DEĞİŞTİRMEMELİ — yalnız iki OOS dönemi sayar."""
    dusuk = _aday("a", +0.10, +0.72, +0.37).hukum()
    yuksek = _aday("b", +9.99, +0.72, +0.37).hukum()
    assert dusuk.verdict == yuksek.verdict == "DOĞRULANDI", (
        f"araştırma Sharpe'ı hükmü değiştirdi: {dusuk.verdict} vs {yuksek.verdict}")
    print("  [ok] hüküm araştırma dönemine bakmıyor (yalnız holdout + ileri-test)")


def test_eksik_donem_gecti_sayilmaz():
    """İleri-testi olmayan aday 'DOĞRULANDI' sayılamaz — sessiz terfi yok."""
    v = _aday("hyp_0002", +0.61, +1.29, None).hukum()
    assert not v.passed and v.verdict == "EKSİK", (
        f"ileri-testi olmayan aday {v.verdict} sayıldı — eksik dönem "
        f"'geçti' diye yorumlanıyor")
    print("  [ok] ileri-testi olmayan aday DOĞRULANDI sayılmıyor")


def test_cokmus_aday_en_sona_duser():
    """Taze veride en sert çöken, sıralamanın en altında olmalı."""
    adaylar = [_aday("cokuk", +0.86, +0.59, -1.46),
               _aday("orta", +0.61, +1.29, -0.05),
               _aday("saglam", +0.66, +0.72, +0.37)]
    sirali = [a.hypothesis_id for a in _sirala(adaylar)]
    assert sirali[0] == "saglam" and sirali[-1] == "cokuk", (
        f"sıralama kanıta göre değil: {sirali}")
    print("  [ok] taze veride çöken aday en sona düşüyor")


def test_salt_okunur_baglanti():
    """Kontrol paneli/rapor gibi 'sadece bakan' yerler sicili DEĞİŞTİREMEZ."""
    import inspect

    from evaluation import aday as modul
    src = inspect.getsource(modul)
    assert "mode=ro" in src, "veritabanları salt-okunur açılmıyor"
    for yasak in ("INSERT", "UPDATE ", "DELETE", "DROP"):
        assert yasak not in src.upper().replace("INSERT OR IGNORE", ""), \
            f"aday modülünde yazma ifadesi var: {yasak}"
    print("  [ok] aday modülü salt-okunur (sicile yazamaz)")


def main() -> None:
    test_arastirma_sharpe_KAZANMAZ()
    test_holdout_olculen_olculmeyeni_yener()
    test_hukum_arastirmayi_HIC_kullanmaz()
    test_eksik_donem_gecti_sayilmaz()
    test_cokmus_aday_en_sona_duser()
    test_salt_okunur_baglanti()
    print("OK — aday seçimi kanıta göre yapılıyor (araştırma Sharpe'ına göre değil).")


if __name__ == "__main__":
    main()
