"""
Sade anlatım katmanı testleri (evaluation/plain.py).

Buradaki testler biçim değil DÜRÜSTLÜK sınar. Sade dilin en büyük riski,
anlaşılır olayım derken iyimser olmaktır: para kaybeden bir strateji
"maymunu geçtik" diye başarı gibi sunulursa, okuyan anladığını sanır ve
YANLIŞ anlar. Bu yüzden hüküm mantığının sırası kilitlenmiştir:

    1) para kaybı        -> her şeyin önüne geçer
    2) zayıf risk/kazanç -> "henüz güvenilir değil"
    3) istatistik yok    -> "kazandırdı ama onaylanmadı"
    4) al-tut'un altı    -> açıkça söylenir
"""
from evaluation.plain import (
    TERIMLER, buyuk, durust_hukum, esik_yorumu, para_dili, strateji_karnesi,
)


def test_para_kaybi_asla_basari_sayilmaz():
    """-%20 kaybeden strateji, maymundan iyi olsa bile 'başarılı' denemez."""
    baslik, gerekce = durust_hukum(toplam_getiri=-0.202, sharpe=-0.16,
                                   al_tut_getiri=2.673)
    assert baslik == "PARA KAZANDIRMADI", f"beklenmeyen hüküm: {baslik}"
    metin = " ".join(gerekce)
    assert "PARA KAYBETTİ" in metin
    assert "daha az kaybetmek" in metin, "kıyas tuzağı uyarısı kayboldu"
    assert "işlem yapmamak daha kârlıydı" in metin, "al-tut karşılaştırması yok"
    print("  [ok] para kaybı hükmü: 'başarı' kelimesi geçmiyor")


def test_istatistik_onaylamazsa_hukum_asagi_ceker():
    """Kâr var ama FDR geçmediyse 'umut verici' denmez — çıktı çelişmemeli."""
    b_onaysiz, g = durust_hukum(1.21, 0.97, fdr_gecti=False)
    b_onayli, _ = durust_hukum(1.21, 0.97, fdr_gecti=True)
    assert b_onaysiz == "KAZANDIRDI AMA İSTATİSTİK ONAYLAMADI"
    assert b_onayli != b_onaysiz, "FDR durumu hükmü hiç değiştirmiyor"
    assert "tesadüf" in " ".join(g)
    print(f"  [ok] FDR geçmedi -> '{b_onaysiz}'")


def test_zayif_sharpe_kar_etse_bile_uyarir():
    baslik, _ = durust_hukum(toplam_getiri=0.03, sharpe=0.21)
    assert "ZAYIF" in baslik, f"zayıf Sharpe uyarısı yok: {baslik}"
    print(f"  [ok] zayıf Sharpe -> '{baslik}'")


def test_al_tutun_altinda_kalmak_soylenir():
    baslik, gerekce = durust_hukum(toplam_getiri=0.10, sharpe=0.80,
                                   al_tut_getiri=0.45, fdr_gecti=True)
    assert "AL-TUT" in baslik, f"al-tut karşılaştırması yutuldu: {baslik}"
    assert "long-short" in " ".join(gerekce), "dürüst nüans (piyasa riski) yok"
    print(f"  [ok] al-tut altı -> '{baslik}'")


def test_negatif_sharpe_para_kaybediyor_der():
    assert esik_yorumu("sharpe", -0.16) == "PARA KAYBEDİYOR"
    assert "çok iyi" in esik_yorumu("sharpe", 3.0)
    assert "hatanın işareti" in esik_yorumu("sharpe", 3.0), \
        "şüphe uyandıracak kadar yüksek Sharpe uyarısı kayboldu"
    assert "öngörü YOK" in esik_yorumu("ic", 0.001)
    print("  [ok] eşik yorumları: negatif=kayıp, aşırı yüksek=şüphe")


def test_para_dili_somut_rakam_verir():
    s = para_dili(-0.202)
    assert "KAYIP" in s and "79.800" in s, s
    assert "kazanç" in para_dili(0.5)
    print(f"  [ok] para dili: {s}")


def test_turkce_buyuk_harf():
    assert buyuk("risk başına kazanç") == "RİSK BAŞINA KAZANÇ"
    assert buyuk("en dip anındaki kayıp") == "EN DİP ANINDAKİ KAYIP"
    print("  [ok] Türkçe büyük harf (i->İ, ı->I)")


def test_her_terimin_karsiligi_ve_aciklamasi_var():
    for k, (ad, aciklama) in TERIMLER.items():
        assert ad and not ad.isupper(), f"{k}: trader karşılığı yok"
        assert len(aciklama) > 30, f"{k}: açıklama fazla kısa"
        assert k.lower() not in ad.lower(), \
            f"{k}: 'karşılık' teknik terimin kendisi ({ad}) — çeviri değil bu"
    print(f"  [ok] {len(TERIMLER)} terimin sade karşılığı + açıklaması var")


def test_karne_her_sayinin_yaninda_yorum_basar():
    out = strateji_karnesi(sharpe=0.97, max_dd=0.28, turnover=137.0,
                           toplam=1.21, hit=0.51, ic=0.02)
    assert "idare eder" in out and "sert" in out and "çok yüksek" in out
    assert "birim para koysaydın" in out, "somut para karşılığı basılmıyor"
    for satir in out.splitlines():
        assert len(satir) < 100, f"satır taşmış: {satir!r}"
    print("  [ok] karne: her metriğin yanında iyi/kötü yorumu var")


def main() -> None:
    test_para_kaybi_asla_basari_sayilmaz()
    test_istatistik_onaylamazsa_hukum_asagi_ceker()
    test_zayif_sharpe_kar_etse_bile_uyarir()
    test_al_tutun_altinda_kalmak_soylenir()
    test_negatif_sharpe_para_kaybediyor_der()
    test_para_dili_somut_rakam_verir()
    test_turkce_buyuk_harf()
    test_her_terimin_karsiligi_ve_aciklamasi_var()
    test_karne_her_sayinin_yaninda_yorum_basar()
    print("OK — sade anlatım testleri geçti (sade dil, iyimser dil DEĞİL).")


if __name__ == "__main__":
    main()
