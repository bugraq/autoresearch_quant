"""
Üç-dönem hükmü testleri (evaluation/three_period.py).

Bu modülün varlık sebebi ÖLÇÜLMÜŞ bir olaydır: kampanya v4'ün kabul edilen üç
adayı da kilitli dönemi GEÇTİ (+0.93 / +0.59 / +1.29), üçü de taze veride
ÇÖKTÜ (−0.36 / −1.46 / −0.05). Yalnız holdout'a bakılsaydı "alpha bulduk"
denirdi. Testler bu tuzağın geri gelmesini engeller.
"""
from evaluation.three_period import ThreePeriodVerdict, final_verdict, verdict_table

ESIK = 0.5

# Gerçek koşudan (29.07.2026) — (hid, araştırma, holdout, ileri-test)
GERCEK = [
    ("hyp_0033", 0.66, 0.72, 0.37),    # üç dönemde de pozitif
    ("hyp_0021", 1.14, 0.93, -0.36),   # holdout geçti, taze veride çöktü
    ("hyp_0025", 0.86, 0.59, -1.46),
    ("hyp_0002", 0.61, 1.29, -0.05),
]


def test_holdout_gecip_ileri_testte_cokene_GUVENILIR_DENMEZ():
    """ASIL TEST: 'holdout geçti' tek başına yeterli sayılamaz."""
    for hid, r, h, f in GERCEK[1:]:
        v = final_verdict(r, h, f, ESIK)
        assert not v.passed, f"{hid}: taze veride çöktü ama GÜVENİLİR sayıldı"
        assert v.verdict == "REJİM-BAĞIMLI", f"{hid}: {v.verdict}"
        assert "rejim" in " ".join(v.reasons).lower() or \
               "REJİM" in v.verdict, "rejim bağımlılığı açıklanmıyor"
    print("  [ok] holdout'u geçip taze veride çökenler GÜVENİLİR sayılmıyor")


def test_uc_donemde_de_ayakta_kalan_dogrulanir():
    hid, r, h, f = GERCEK[0]
    v = final_verdict(r, h, f, ESIK)
    assert v.passed and v.verdict == "DOĞRULANDI", v.verdict
    metin = " ".join(v.reasons)
    assert "alpha bulundu" in metin.lower(), \
        "abartı uyarısı yok — 'DOĞRULANDI' tek başına alpha sanılabilir"
    assert "ÖLMEDİ" in metin or "ölmedi" in metin, \
        "doğru çerçeve ('henüz ölmedi') yazılmamış"
    print(f"  [ok] {hid} DOĞRULANDI — ama 'alpha bulundu değil' uyarısıyla")


def test_olculmemis_donem_sessizce_gecti_sayilmaz():
    """Eksik dönem = EKSİK hüküm. Bu modülün var olma sebebi tam da bu."""
    for h, f in ((0.9, None), (None, 0.4), (None, None)):
        v = final_verdict(0.8, h, f, ESIK)
        assert v.verdict == "EKSİK" and not v.passed, \
            f"holdout={h}, ileri={f} -> {v.verdict} (sessizce geçti sayıldı)"
    print("  [ok] ölçülmemiş dönem 'geçti' sayılmıyor (EKSİK)")


def test_her_iki_oos_de_batarsa_coktu():
    v = final_verdict(1.5, -0.2, -0.8, ESIK)
    assert v.verdict == "ÇÖKTÜ" and not v.passed
    assert "uydurulmuş" in " ".join(v.reasons), "aşırı uyum teşhisi yok"
    print("  [ok] iki OOS de batınca ÇÖKTÜ")


def test_holdout_zayif_ileri_iyi_de_rejim_bagimlidir():
    """Çelişen OOS dönemleri her iki yönde de güvensizliktir."""
    v = final_verdict(0.9, 0.20, 0.80, ESIK)
    assert v.verdict == "REJİM-BAĞIMLI" and not v.passed
    print("  [ok] ters yön (holdout zayıf, ileri iyi) de REJİM-BAĞIMLI")


def test_arastirma_sharpe_hukmu_DEGISTIRMEZ():
    """Araştırma dönemi kanıt değildir; hükme girmemeli."""
    a = final_verdict(0.60, 0.90, 0.40, ESIK)
    b = final_verdict(9.99, 0.90, 0.40, ESIK)
    assert a.verdict == b.verdict == "DOĞRULANDI", (a.verdict, b.verdict)
    c = final_verdict(9.99, 0.90, -0.40, ESIK)
    assert not c.passed, "yüksek araştırma Sharpe'ı çöküşü örtbas etti"
    print("  [ok] araştırma Sharpe'ı hükmü değiştirmiyor (kanıt değil)")


def test_esik_kampanyadan_gelir():
    """Kabul eşiği kampanyanındır; farklı eşik farklı hüküm vermeli."""
    assert final_verdict(0.8, 0.55, 0.30, 0.5).passed
    assert not final_verdict(0.8, 0.55, 0.30, 0.9).passed
    print("  [ok] kampanya eşiği hükme yansıyor")


def test_tablo_tum_adaylari_ve_hukmu_basar():
    t = verdict_table(GERCEK, ESIK)
    for hid, *_ in GERCEK:
        assert hid in t, f"{hid} tabloda yok"
    assert "DOĞRULANDI" in t and "REJİM-BAĞIMLI" in t
    assert verdict_table([], ESIK).strip().startswith("(")
    for satir in t.splitlines():
        assert len(satir) < 100, f"satır taşmış: {satir!r}"
    print("  [ok] tablo tüm adayları hükmüyle basıyor")


def main() -> None:
    test_holdout_gecip_ileri_testte_cokene_GUVENILIR_DENMEZ()
    test_uc_donemde_de_ayakta_kalan_dogrulanir()
    test_olculmemis_donem_sessizce_gecti_sayilmaz()
    test_her_iki_oos_de_batarsa_coktu()
    test_holdout_zayif_ileri_iyi_de_rejim_bagimlidir()
    test_arastirma_sharpe_hukmu_DEGISTIRMEZ()
    test_esik_kampanyadan_gelir()
    test_tablo_tum_adaylari_ve_hukmu_basar()
    print("OK — üç-dönem hükmü testleri geçti (tek holdout yeterli sayılmıyor).")


if __name__ == "__main__":
    main()
