"""
Alan hizalaması testleri — kampanya kısıtları ile YÜKLENEN veri uyuşmazlığı.

Gerçek koşuda patladı: `compare.py` değerlendirme ortamını compare.yaml ile
sentetik benchmark'a çeviriyor, ama `allowed_fields` kripto kampanyasından
(`funding_rate` dahil) geliyordu. LLM'e olmayan bir alan sunulunca üretilen
her hipotez `KeyError: 'Veri alanı yok: funding_rate'` ile çöktü:
5 yarışmacının 4'ü hiç hipotez üretemeden düştü ve "LLM karşılaştırması"
(agent.py menü [3]) anlamsız bir tablo bastı.

Düzeltme run_campaign içinde, çünkü her kampanya yolu oradan geçer.
Sessizce düzeltmez: neyin neden düştüğünü yüksek sesle söyler.
"""
import numpy as np
import pandas as pd

from data.synthetic import MarketData, gen_cross_sectional_momentum
from orchestrator.loop import CampaignConfig, align_allowed_fields


def _veri() -> MarketData:
    return gen_cross_sectional_momentum(seed=1)


def test_veride_olmayan_alan_dusurulur():
    data = _veri()
    cfg = CampaignConfig(allowed_fields=["close", "volume", "funding_rate", "roe"])
    izinli = align_allowed_fields(cfg, data)
    assert "funding_rate" not in izinli, "olmayan alan LLM'e sunulmaya devam ediyor"
    assert "roe" not in izinli
    assert "close" in izinli and "volume" in izinli, "var olan alanlar da düştü"
    print(f"  [ok] olmayan alanlar düştü, kalan: {izinli}")


def test_hepsi_uyumluysa_hicbir_sey_degismez():
    data = _veri()
    alanlar = ["close", "open", "volume"]
    cfg = CampaignConfig(allowed_fields=list(alanlar))
    assert align_allowed_fields(cfg, data) == alanlar, "uyumlu config değiştirildi"
    print("  [ok] uyumlu config'e dokunulmuyor (sıra korunuyor)")


def test_hicbiri_yoksa_sessizce_devam_etmez():
    """Tamamen uyumsuz config = kurulum hatası; sessizce bos kampanya kosmak
    saatlerce anlamsiz cikti uretirdi. Yuksek sesle patlamali."""
    bos = MarketData(fields={"close": pd.DataFrame(np.ones((5, 2)))})
    cfg = CampaignConfig(allowed_fields=["funding_rate", "book_to_market"])
    try:
        align_allowed_fields(cfg, bos)
    except ValueError as e:
        assert "allowed_fields" in str(e), f"hata mesaji yol gostermiyor: {e}"
        print("  [ok] tümüyle uyumsuz config ValueError ile durduruluyor")
    else:
        raise AssertionError("hiçbir alan yokken kampanya sessizce başladı")


def test_run_campaign_hizalamayi_uyguluyor():
    """Kanıt: düzeltme yardımcı fonksiyonda değil, GERÇEK yolda devrede."""
    from memory import MemoryStore
    from orchestrator.loop import run_campaign
    import os
    import tempfile

    class _SahteSaglayici:
        """Hiç hipotez üretmez; amacımız yalnız hizalamanın koşmasi."""
        total_prompt_tokens = 0
        total_completion_tokens = 0

        def next(self, ctx):
            raise RuntimeError("uretim yok (test)")

    fd, db = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.remove(db)
    mem = MemoryStore(db)
    cfg = CampaignConfig(allowed_fields=["close", "funding_rate"], max_experiments=1)
    run_campaign(_SahteSaglayici(), _veri(), mem, cfg)
    mem.close()
    assert cfg.allowed_fields == ["close"], \
        f"run_campaign hizalamayı uygulamadı: {cfg.allowed_fields}"
    print("  [ok] run_campaign hizalamayı gerçekten uyguluyor")


def main() -> None:
    test_veride_olmayan_alan_dusurulur()
    test_hepsi_uyumluysa_hicbir_sey_degismez()
    test_hicbiri_yoksa_sessizce_devam_etmez()
    test_run_campaign_hizalamayi_uyguluyor()
    print("OK — alan hizalaması testleri geçti (olmayan alan LLM'e sunulmuyor).")


if __name__ == "__main__":
    main()
