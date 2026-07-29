"""
Holdout ISINMA testleri — kilitli dönem sınavı doğru şeyi mi ölçüyor?

İki hata bu testlerle kapatıldı:

  1) ISINMA KAYBI: holdout dilimi tek başına değerlendirilince rolling
     pencereler ve walk-forward eğitilen ML modeli kilitli dönemin BAŞINDA
     geçmişsiz kalır. Gerçek kripto koşusunda holdout'un %17'si sinyalsizdi.

  2) MODELİN YENİDEN EĞİTİLMESİ (asıl mesele): ML modu holdout'un İÇİNDE
     yeniden fit ediliyordu. Yani sınav, araştırmada kabul edilen modeli
     değil BAŞKA bir modeli ölçüyordu. Gerçek koşuda bu, kabul edilmiş bir
     hipotezin holdout Sharpe'ını -0.36 (kaldı) yerine +0.72 (geçti)
     gösterecek kadar büyük bir farktı.

Düzeltme: HoldoutService(history=araştırma_dilimi). Bilgi akışı tek yönlü
(geçmiş -> gelecek) olduğu için SIZINTI DEĞİLDİR; ters yön yapısal olarak
imkânsızdır (concat_market history'yi her zaman öne koyar).
"""
import os
import tempfile

import numpy as np

from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, ModelSpec, Portfolio, Universe,
)
from contracts.dsl import Expression, NamedFeature
from data import concat_market, gen_cross_sectional_momentum, split_by_fraction
from holdout import HoldoutService


def _tmp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.remove(path)
    return path


def _hyp(model_type: str, window: int = 60) -> HypothesisSpec:
    ret = Expression(op="return", window=window,
                     inputs=[Expression(op="field", field="close")])
    return HypothesisSpec(
        hypothesis_id=f"warm_{model_type}_{window}", title="ısınma", claim="c",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="t", description="d"),
        universe=Universe(source="sp500_point_in_time"),
        features=[NamedFeature(name="mom", expression=ret)],
        signal=Expression(op="cross_sectional_rank",
                          inputs=[Expression(op="feature_ref", name="mom")]),
        model=ModelSpec(type=model_type),
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=5),
        falsification=Falsification())


def test_concat_market_zaman_sirasini_korur():
    """concat_market history'yi ÖNE koyar — ters yön (sızıntı) imkânsız."""
    data = gen_cross_sectional_momentum(seed=1)
    research, holdout = split_by_fraction(data, 0.7)
    full = concat_market(research, holdout)
    assert len(full.dates) == len(data.dates)
    assert full.dates.is_monotonic_increasing, "birleşim zaman sırasını bozdu"
    assert full.dates[0] == research.dates[0]
    assert full.dates[len(research.dates)] == holdout.dates[0], \
        "holdout, history'nin ÖNÜNE geçmiş (sızıntı yönü!)"
    assert full.bars_per_year == data.bars_per_year, "yıllıklaştırma ölçeği kayboldu"
    print("  [ok] concat_market: sıra korunuyor, ölçek taşınıyor")


def test_isinma_kilitli_donemin_basini_kurtarir():
    """Uzun pencere: ısınmasız holdout'un başı sinyalsiz; ısınmalı %100 kapsar."""
    data = gen_cross_sectional_momentum(seed=1)
    research, holdout = split_by_fraction(data, 0.7)
    hyp = _hyp("dsl_formula", window=60)

    soguk = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0)
    r_soguk = soguk.evaluate(hyp)
    soguk.close()

    sicak = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0,
                           history=research)
    r_sicak = sicak.evaluate(hyp)
    sicak.close()

    assert r_soguk.coverage < 0.99, \
        f"60 barlık pencere ısınmasız tam kapsamamalıydı (kapsama {r_soguk.coverage:.2f})"
    assert r_sicak.coverage > r_soguk.coverage, "ısınma kapsamayı artırmadı"
    assert r_sicak.coverage > 0.99, f"ısınmalı kapsama eksik: {r_sicak.coverage:.2f}"
    print(f"  [ok] dsl_formula kapsama: %{r_soguk.coverage*100:.0f} -> "
          f"%{r_sicak.coverage*100:.0f}")


def test_model_holdout_icinde_yeniden_egitilmiyor():
    """ML modu: ısınmasız holdout'un başı eğitime yanar; ısınmalı yanmaz.

    Ayrıca iki Sharpe'ın FARKLI çıkması, sınavın gerçekten başka bir modeli
    ölçüyor olduğunun kanıtıdır (regresyon koruması).
    """
    data = gen_cross_sectional_momentum(seed=1)
    research, holdout = split_by_fraction(data, 0.7)
    hyp = _hyp("linear_regression", window=20)

    soguk = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0)
    r_soguk = soguk.evaluate(hyp)
    soguk.close()

    sicak = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0,
                           history=research)
    r_sicak = sicak.evaluate(hyp)
    sicak.close()

    assert r_soguk.coverage < r_sicak.coverage, (
        "ML modunda ısınmasız holdout, başını eğitime yakmalıydı "
        f"(soğuk {r_soguk.coverage:.2f} vs sıcak {r_sicak.coverage:.2f})")
    assert not np.isclose(r_soguk.sharpe, r_sicak.sharpe), (
        "İki Sharpe aynı çıktı — model ya hiç eğitilmiyor ya da ısınma "
        "bağlanmamış (bu testin varlık sebebi bu farkı korumaktır).")
    print(f"  [ok] model kapsama: %{r_soguk.coverage*100:.0f} -> "
          f"%{r_sicak.coverage*100:.0f} | Sharpe {r_soguk.sharpe:+.3f} -> "
          f"{r_sicak.sharpe:+.3f} (sınav artık ARAŞTIRMADAKİ modeli ölçüyor)")


def test_one_shot_hala_gecerli():
    """Isınma eklendi diye one-shot koruması gevşememeli."""
    from holdout import HoldoutError
    data = gen_cross_sectional_momentum(seed=1)
    research, holdout = split_by_fraction(data, 0.7)
    svc = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0,
                         history=research)
    hyp = _hyp("dsl_formula", window=20)
    svc.evaluate(hyp)
    try:
        svc.evaluate(hyp)
    except HoldoutError:
        print("  [ok] one-shot korunuyor (ikinci değerlendirme reddedildi)")
    else:
        raise AssertionError("one-shot ihlali: aynı aday ikinci kez değerlendirildi")
    finally:
        svc.close()


# ===========================================================================
# GEÇERSİZ KILMA — hatalı bir değerlendiriciyle üretilmiş sonucu düzeltmek
# ===========================================================================
# Neden gerekli: yukarıdaki ısınma hatası, ZATEN KAYDEDİLMİŞ holdout
# sonuçlarını da yanlış yapmıştı. One-shot kilidi (doğru olarak) ikinci
# değerlendirmeye izin vermiyordu; tek çıkış yolu kaydı SİLMEKTİ — bilimsel
# kaydı silmek ise en kötü seçenek. Çözüm: append-only geçersiz kılma.


def test_gerekcesiz_gecersiz_kilma_reddedilir():
    """Gerekçesiz sıfırlama, sınavı fiilen ortadan kaldırır — yasak."""
    from holdout import HoldoutError
    data = gen_cross_sectional_momentum(seed=1)
    _r, holdout = split_by_fraction(data, 0.7)
    svc = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0)
    for bos in ("", "   ", None):
        try:
            svc.invalidate(bos)
        except HoldoutError:
            pass
        else:
            raise AssertionError(f"gerekçe {bos!r} ile geçersiz kılma KABUL edildi")
    svc.close()
    print("  [ok] gerekçesiz geçersiz kılma reddedildi")


def test_gecersiz_kilma_silmez_ve_yeniden_kosmaya_izin_verir():
    """Eski kayıt gerekçesiyle DURUR; aynı hipotez yeniden değerlendirilebilir."""
    import sqlite3

    data = gen_cross_sectional_momentum(seed=1)
    research, holdout = split_by_fraction(data, 0.7)
    db = _tmp_db()
    hyp = _hyp("linear_regression", window=20)

    # 1) ESKİ (ısınmasız) değerlendirici ile bir sonuç üret
    eski = HoldoutService(holdout, audit_path=db, cost_bps=1.0)
    r1 = eski.evaluate(hyp)
    eski.close()

    # 2) Geçersiz kıl + YENİ (ısınmalı) değerlendirici ile yeniden koş
    yeni = HoldoutService(holdout, audit_path=db, cost_bps=1.0, history=research)
    n = yeni.invalidate("degerlendirici hatasi: isinma yok (test)")
    assert n == 1, f"geçersiz kılınan kayıt sayısı {n}, beklenen 1"
    r2 = yeni.evaluate(hyp)          # one-shot ARTIK engellemiyor
    kayit = yeni.audit_log()
    yeni.close()

    assert len(kayit) == 2, f"eski kayıt silinmiş olmalı DEĞİL: {len(kayit)} satır"
    durumlar = [k[4] for k in kayit]
    assert durumlar == ["invalidated", "active"], durumlar
    assert kayit[0][7], "geçersiz kılma gerekçesi saklanmamış"
    assert kayit[1][5] == "v2-warmup", f"değerlendirici sürümü yok: {kayit[1][5]}"
    assert r1.sharpe != r2.sharpe, "yeniden koşu aynı sayıyı verdi (ısınma bağlı mı?)"
    print(f"  [ok] geçersiz kılma: eski {r1.sharpe:+.3f} (saklandı) -> "
          f"yeni {r2.sharpe:+.3f} (aktif), kayıt silinmedi")


def test_gecersiz_kayit_kotayi_doldurmaz():
    """Düzeltme, düzeltmenin kendisi yüzünden imkânsızlaşmamalı."""
    data = gen_cross_sectional_momentum(seed=1)
    research, holdout = split_by_fraction(data, 0.7)
    db = _tmp_db()
    svc = HoldoutService(holdout, audit_path=db, cost_bps=1.0, max_candidates=1)
    svc.evaluate(_hyp("dsl_formula", window=20))
    assert svc._count() == 1
    svc.invalidate("test gerekcesi")
    assert svc._count() == 0, "geçersiz kayıt hâlâ kotayı dolduruyor"
    svc.evaluate(_hyp("dsl_formula", window=30))    # kota açıldı
    svc.close()
    print("  [ok] geçersiz kayıt kotayı doldurmuyor")


def test_secili_hipotez_gecersiz_kilinabilir():
    """hypothesis_ids verilirse yalnız onlar geçersiz olur (toptan değil)."""
    data = gen_cross_sectional_momentum(seed=1)
    _r, holdout = split_by_fraction(data, 0.7)
    svc = HoldoutService(holdout, audit_path=_tmp_db(), cost_bps=1.0)
    a, b = _hyp("dsl_formula", 20), _hyp("dsl_formula", 30)
    svc.evaluate(a); svc.evaluate(b)
    n = svc.invalidate("yalniz a", hypothesis_ids=[a.hypothesis_id])
    assert n == 1, f"{n} kayıt geçersiz kılındı, beklenen 1"
    aktif = {k[0] for k in svc.audit_log(only_active=True)}
    assert aktif == {b.hypothesis_id}, aktif
    svc.close()
    print("  [ok] seçili geçersiz kılma yalnız hedefi etkiliyor")


def test_eski_audit_dosyasi_tasinir():
    """UNIQUE kısıtlı ESKİ audit dosyası, kayıt kaybetmeden yeni şemaya geçer."""
    import sqlite3

    data = gen_cross_sectional_momentum(seed=1)
    _r, holdout = split_by_fraction(data, 0.7)
    db = _tmp_db()
    c = sqlite3.connect(db)                    # ESKİ şema (v1)
    c.execute("CREATE TABLE holdout_access (id INTEGER PRIMARY KEY AUTOINCREMENT,"
              " hypothesis_id TEXT UNIQUE NOT NULL, sharpe REAL, passed INTEGER,"
              " accessed_at TEXT)")
    c.executemany("INSERT INTO holdout_access (hypothesis_id, sharpe, passed) "
                  "VALUES (?,?,?)", [("eski_1", -1.06, 0), ("eski_2", 0.68, 1)])
    c.commit(); c.close()

    svc = HoldoutService(holdout, audit_path=db, cost_bps=1.0)
    kayit = svc.audit_log()
    assert len(kayit) == 2, f"taşımada kayıt kayboldu: {len(kayit)}"
    assert {k[0] for k in kayit} == {"eski_1", "eski_2"}
    assert all(k[4] == "active" for k in kayit), "eski kayıtlar aktif kalmalı"
    assert svc.invalidate("tasima sonrasi test") == 2
    svc.close()

    sql = sqlite3.connect(db).execute(
        "SELECT sql FROM sqlite_master WHERE name='holdout_access'").fetchone()[0]
    assert "UNIQUE" not in sql.upper(), "UNIQUE kısıtı düşürülmemiş"
    print("  [ok] eski audit dosyası kayıpsız taşındı, UNIQUE düştü")


def main() -> None:
    test_concat_market_zaman_sirasini_korur()
    test_isinma_kilitli_donemin_basini_kurtarir()
    test_model_holdout_icinde_yeniden_egitilmiyor()
    test_one_shot_hala_gecerli()
    test_gerekcesiz_gecersiz_kilma_reddedilir()
    test_gecersiz_kilma_silmez_ve_yeniden_kosmaya_izin_verir()
    test_gecersiz_kayit_kotayi_doldurmaz()
    test_secili_hipotez_gecersiz_kilinabilir()
    test_eski_audit_dosyasi_tasinir()
    print("OK — holdout ısınma testleri geçti (sınav doğru modeli ölçüyor).")


if __name__ == "__main__":
    main()
