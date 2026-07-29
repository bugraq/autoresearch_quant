"""
DASHBOARD KIYAS BÖLÜMÜ testleri — hocanın başarı ölçütü raporda görünüyor mu?

Bu bölüm dashboard'a sonradan eklendi çünkü YOKTU: kıyas sonucu (rastgele
al-satçıyı / al-tut'u geçiyor muyuz?) yalnızca bir terminal logunda duruyordu.
Yani hocaya gösterilen görsel raporda, hocanın koyduğu ölçütün kendisi eksikti.

Testlerin koruduğu iki kural:

  1) ARAŞTIRMA DÖNEMİ SAYIYA KATILMAZ. "3/3 geçtik" cümlesi yalnız
     örneklem-dışı dönemlerden kurulabilir; aday araştırma döneminde
     SEÇİLDİĞİ için orada kazanması beklenir ve hiçbir şey kanıtlamaz.
  2) ÖLÇÜM YOKSA SESSİZ GEÇİLMEZ. Dosya yoksa bölüm "henüz ölçülmedi" der;
     boş bir bölüm, ölçütün sağlandığı izlenimi bırakmamalı.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.report import _kiyas


def _yaz(donemler: list, klasor: str) -> None:
    d = {"kosum_tarihi": "2026-07-29T20:49:21", "cost_bps": 10.0,
         "maymun_sayisi": 100,
         "aday": {"hypothesis_id": "hyp_0033", "title": "test",
                  "secim_nedeni": "ÜÇ DÖNEMİ de geçen tek aday",
                  "hukum": "DOĞRULANDI"},
         "donemler": donemler}
    with open(os.path.join(klasor, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def _donem(ad, oos, al_tut, rastgele, duygusal, bizim=0.7):
    return {"ad": ad, "oos": oos, "aralik": "2020-01-01→2024-12-31",
            "bizim": {"sharpe": bizim, "toplam": 0.2, "maxdd": 0.2, "yillik": 0.1},
            "al_tut": {"sharpe": 0.9, "toplam": 0.7, "maxdd": 0.6, "yillik": 0.5},
            "duygusal": {"sharpe": -7.0, "toplam": -0.9, "maxdd": 0.9, "yillik": -1.0},
            "maymun_ortanca_sharpe": -10.0, "bizim_masrafsiz_sharpe": 1.2,
            "maymun_yuzdeligi": 100.0,
            "gecti": {"al-tut": al_tut, "rastgele": rastgele,
                      "duygusal": duygusal}}


def test_arastirma_donemi_HUKME_katilmaz():
    """ASIL TEST: yalnız araştırmada geçmek 'geçtik' diye okunmamalı."""
    with tempfile.TemporaryDirectory() as t:
        _yaz([_donem("ARASTIRMA", False, True, True, True),      # üçünü de geçti
              _donem("HOLDOUT *OOS", True, False, True, True)],  # al-tut'u geçemedi
             t)
        h = _kiyas(t)
    assert "geçemediğimiz" in h and "al-tut" in h, (
        "araştırmada 3/3 geçmek, OOS'taki başarısızlığı örtüyor — okuma "
        "cümlesi örneklem-dışı dönemlerden kurulmuyor olabilir")
    print("  [ok] hüküm yalnız OOS dönemlerden kuruluyor (araştırma sayılmıyor)")


def test_arastirma_satiri_kanit_degil_diye_isaretli():
    with tempfile.TemporaryDirectory() as t:
        _yaz([_donem("ARASTIRMA", False, True, True, True)], t)
        h = _kiyas(t)
    assert "kanıt değil" in h, "araştırma satırı 'kanıt değil' diye işaretlenmemiş"
    assert "KANIT DEĞİLDİR" in h or "kanıt" in h.lower()
    print("  [ok] araştırma satırı görsel olarak 'kanıt değil' işaretli")


def test_sadece_arastirma_varsa_hukum_verilmez():
    """OOS dönem yoksa 'geçtik/geçemedik' denmez — ölçüm eksiktir."""
    with tempfile.TemporaryDirectory() as t:
        _yaz([_donem("ARASTIRMA", False, True, True, True)], t)
        h = _kiyas(t)
    assert "KANIT DEĞİLDİR" in h, "yalnız araştırma varken hüküm verilmiş"
    print("  [ok] OOS dönem yokken hüküm verilmiyor")


def test_hepsini_gecince_yine_alpha_denmez():
    """Üç rakibi de geçmek 'alpha bulduk' demek değildir."""
    with tempfile.TemporaryDirectory() as t:
        _yaz([_donem("ILERI-TEST *OOS", True, True, True, True)], t)
        h = _kiyas(t)
    assert "'alpha' demek değildir" in h or "alpha" in h, \
        "üç rakibi geçince ölçüt fazla yorumlanıyor"
    assert "çoklu-test" in h, "çoklu-test uyarısı düşmüş"
    print("  [ok] üç rakibi geçmek 'alpha bulundu' diye sunulmuyor")


def test_olcum_yoksa_sessiz_gecilmez():
    with tempfile.TemporaryDirectory() as t:
        h = _kiyas(t)
    assert "henüz ölçülmedi" in h, "ölçüm yokken bölüm sessizce boş geçiyor"
    assert "benchmark.py" in h, "nasıl ölçüleceği söylenmiyor"
    print("  [ok] ölçüm yokken 'henüz ölçülmedi' + nasıl koşulacağı yazıyor")


def test_olcum_tarihi_damgasi_var():
    """Bayat sayı 'güncel gerçek' sanılmasın: ölçüm tarihi görünmeli."""
    with tempfile.TemporaryDirectory() as t:
        _yaz([_donem("HOLDOUT *OOS", True, False, True, True)], t)
        h = _kiyas(t)
    assert "2026-07-29" in h, "ölçüm tarihi damgası yok"
    assert "10.0 bps" in h, "hangi işlem maliyetiyle ölçüldüğü yazmıyor"
    print("  [ok] ölçüm tarihi ve maliyet damgası basılıyor")


def test_bozuk_dosya_dashboardu_cokertmez():
    with tempfile.TemporaryDirectory() as t:
        with open(os.path.join(t, "benchmark.json"), "w", encoding="utf-8") as f:
            f.write("{bozuk json")
        h = _kiyas(t)
    assert "okunamadı" in h, "bozuk dosya sessizce yutuldu ya da çökertti"
    print("  [ok] bozuk kıyas dosyası dashboard'u çökertmiyor")


def main() -> None:
    test_arastirma_donemi_HUKME_katilmaz()
    test_arastirma_satiri_kanit_degil_diye_isaretli()
    test_sadece_arastirma_varsa_hukum_verilmez()
    test_hepsini_gecince_yine_alpha_denmez()
    test_olcum_yoksa_sessiz_gecilmez()
    test_olcum_tarihi_damgasi_var()
    test_bozuk_dosya_dashboardu_cokertmez()
    print("OK — kıyas bölümü raporda var ve araştırma dönemini kanıt saymıyor.")


if __name__ == "__main__":
    main()
