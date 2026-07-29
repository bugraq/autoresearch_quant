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


# ===========================================================================
# KIMLIK CAKISMASI — hypothesis_id kampanyalar arasi TEKIL DEGIL
# ===========================================================================
# --fresh sayaci sifirlar (yeni kampanya yine hyp_0001'den baslar) ama holdout
# audit'i kampanyalar arasi YASAR (kilitli donem proje capinda sonlu kaynak).
# Gercek ornek: v2'nin hyp_0033'u uc donemi gecti; v4'un hyp_0033'u bambaska
# bir hipotez ve reddedildi. Kimlige guvenerek join yapmak YANLIS hipotezi
# raporlar. Cozum: icerik parmak izi.


def _spec(hid: str, window: int, title: str = "t"):
    from contracts.hypothesis_spec import HypothesisSpec
    return HypothesisSpec.model_validate({
        "hypothesis_id": hid, "title": title, "claim": "c", "family": "momentum",
        "economic_mechanism": {"type": "t", "description": "d"},
        "universe": {"source": "sp500_point_in_time"}, "features": [],
        "signal": {"op": "cross_sectional_rank", "inputs": [
            {"op": "return", "window": window,
             "inputs": [{"op": "field", "field": "close"}]}]},
        "portfolio": {"type": "cross_sectional_long_short"},
        "execution": {"signal_time": "close_t", "trade_time": "open_t_plus_1",
                      "holding_period_days": 5},
        "falsification": {}})


def test_ayni_kimlik_farkli_strateji_ayirt_edilir():
    from holdout.service import hypothesis_fingerprint
    v2 = _spec("hyp_0033", 60)
    v4 = _spec("hyp_0033", 20)          # AYNI kimlik, FARKLI strateji
    assert hypothesis_fingerprint(v2) != hypothesis_fingerprint(v4), (
        "iki farkli strateji ayni parmak izini uretti — kimlik cakismasi "
        "tespit edilemez, yanlis hipotez raporlanabilir")
    print("  [ok] ayni kimlikli FARKLI stratejiler ayirt ediliyor")


def test_ayni_strateji_farkli_kimlik_ve_baslikta_ayni_iz():
    """Parmak izi ICERIGE bakar: yeniden adlandirma onu degistirmemeli."""
    from holdout.service import hypothesis_fingerprint
    a = _spec("hyp_0001", 60, "orijinal baslik")
    b = _spec("bambaska_kimlik", 60, "tamamen farkli baslik")
    assert hypothesis_fingerprint(a) == hypothesis_fingerprint(b), (
        "ayni strateji farkli kimlik/baslikla farkli iz uretti")
    print("  [ok] ayni strateji, farkli kimlik/baslik -> ayni parmak izi")


def test_parmak_izi_holdout_kaydina_yaziliyor():
    import os
    import tempfile

    from data import gen_cross_sectional_momentum, split_by_fraction
    from holdout import HoldoutService
    from holdout.service import hypothesis_fingerprint

    fd, db = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd); os.remove(db)
    _r, holdout = split_by_fraction(gen_cross_sectional_momentum(seed=1), 0.7)
    svc = HoldoutService(holdout, audit_path=db, cost_bps=1.0)
    hyp = _spec("hyp_0001", 20)
    svc.evaluate(hyp)
    kayit = svc.audit_log()[0]
    svc.close()
    import sqlite3
    c = sqlite3.connect(db)
    (iz,) = c.execute("SELECT hypothesis_hash FROM holdout_access").fetchone()
    c.close()
    assert iz == hypothesis_fingerprint(hyp),         f"audit'e yazilan parmak izi yanlis: {iz}"
    print("  [ok] parmak izi holdout kaydina yaziliyor")


def main() -> None:
    test_holdout_gecip_ileri_testte_cokene_GUVENILIR_DENMEZ()
    test_uc_donemde_de_ayakta_kalan_dogrulanir()
    test_olculmemis_donem_sessizce_gecti_sayilmaz()
    test_her_iki_oos_de_batarsa_coktu()
    test_holdout_zayif_ileri_iyi_de_rejim_bagimlidir()
    test_arastirma_sharpe_hukmu_DEGISTIRMEZ()
    test_esik_kampanyadan_gelir()
    test_tablo_tum_adaylari_ve_hukmu_basar()
    test_ayni_kimlik_farkli_strateji_ayirt_edilir()
    test_ayni_strateji_farkli_kimlik_ve_baslikta_ayni_iz()
    test_parmak_izi_holdout_kaydina_yaziliyor()
    print("OK — üç-dönem hükmü testleri geçti (tek holdout yeterli sayılmıyor).")


if __name__ == "__main__":
    main()
