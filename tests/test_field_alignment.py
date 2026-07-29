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


# ===========================================================================
# BASELINE'LAR DA KAMPANYA KISITINA UYAR (adil kiyas)
# ===========================================================================
# Static validator ufuk kisitini uygulamaya baslayinca olculdu: baseline'lar
# holding_period_days'i SABIT [1,5,10]'dan seciyordu; kripto kampanyasinin
# izinli ufuklari [5,10,20,60,90,120] oldugu icin holding=1 kisiti ihlal
# ediyordu. Sonuc: random-search uretimlerinin %47'si, GP'nin %27'si backtest'e
# BILE GIRMEDEN disallowed_horizon ile eleniyordu. Yani LLM'i kiyasladigimiz
# alt-cita (Deney A / MVP kriter 9) sakat kaliyordu — "ustunluk" kismen
# rakibin diskalifiye edilmesinden gelirdi.


def test_baseline_holding_kampanya_ufkuna_uyar():
    from baselines._common import allowed_holdings
    assert allowed_holdings(None) == [1, 5, 10], "kisitsiz varsayilan degisti"
    # kesisim korunur (kisa tutma tercihi)
    assert allowed_holdings([5, 10, 20, 60, 90, 120]) == [5, 10]
    # kesisim bossa en kisa uc ufuk (uretim imkansiz kalmasin)
    assert allowed_holdings([30, 60, 90, 120]) == [30, 60, 90]
    print("  [ok] allowed_holdings kampanya ufuklarina uyuyor")


def test_baseline_uretimleri_statik_kapidan_gecer():
    """ASIL TEST: uc baseline de kampanya kisitlari altinda uretim yapabilmeli."""
    from collections import Counter

    from baselines import (
        BayesianOptProvider, GPHypothesisProvider, RandomHypothesisProvider,
    )
    from contracts.research_context import GenerationMode, ResearchContext
    from dsl import compile_hypothesis, validate

    ufuklar = [5, 10, 20, 60, 90, 120]      # kripto kampanyasi (1 YOK)
    ctx = ResearchContext(
        campaign_goal="t", universe_description="u",
        allowed_fields=["open", "high", "low", "close", "volume", "dollar_volume"],
        allowed_operators=["return", "rolling_mean", "zscore", "volatility",
                           "cross_sectional_rank", "multiply", "negate"],
        allowed_horizons=ufuklar, allowed_rebalance=["daily", "weekly"],
        allowed_portfolio_types=["cross_sectional_long_short"],
        generation_mode=GenerationMode.new, experiments_remaining=40)

    for ad, saglayici in (("random", RandomHypothesisProvider(seed=1)),
                          ("gp", GPHypothesisProvider(seed=1)),
                          ("bayesopt", BayesianOptProvider(seed=1))):
        red = Counter()
        n = 25
        for _ in range(n):
            h = saglayici.next(ctx)
            dec = validate(compile_hypothesis(h), h, allowed_horizons=ufuklar)
            if dec.decision.value != "accept":
                red[dec.issues[0].type if dec.issues else "?"] += 1
        assert not red, (
            f"{ad} baseline'inin {sum(red.values())}/{n} uretimi kampanya "
            f"kisitina takildi ({dict(red)}) — alt-cita sakat, kiyas adil degil")
        print(f"  [ok] {ad}: {n} uretimin hepsi kampanya kisitlarindan gecti")


def main() -> None:
    test_veride_olmayan_alan_dusurulur()
    test_hepsi_uyumluysa_hicbir_sey_degismez()
    test_hicbiri_yoksa_sessizce_devam_etmez()
    test_run_campaign_hizalamayi_uyguluyor()
    test_baseline_holding_kampanya_ufkuna_uyar()
    test_baseline_uretimleri_statik_kapidan_gecer()
    print("OK — alan/ufuk hizalamasi testleri gecti (kisitlar GERCEKTEN uygulaniyor).")


if __name__ == "__main__":
    main()
