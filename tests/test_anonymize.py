"""
LLM memorization önlemi testi.

Anonimleştirme AÇIKKEN (varsayılan) LLM'e giden hiçbir prompta ticker adı
veya tarih aralığı sızmamalı — LLM eğitim verisinden dönemin kazananlarını
ezbere bilir (parametre-içi look-ahead). Kapalıyken (ablation) sızar.
"""
from agents.hypothesis_generator import _build_system_prompt, _build_user_prompt
from contracts.research_context import GenerationMode
from memory import MemoryStore
from orchestrator.loop import ANONYMOUS_UNIVERSE, CampaignConfig, _build_context

_DESC = "50 büyük ABD hissesi (AAPL, MSFT, NVDA vb.), Yahoo Finance, 2015-2023"
_SECRETS = ["AAPL", "MSFT", "NVDA", "2015", "2023"]


def _ctx(anonymize: bool, anonymous_description: str | None = None):
    cfg = CampaignConfig(universe_description=_DESC, anonymize_universe=anonymize,
                         anonymous_description=anonymous_description)
    memory = MemoryStore(":memory:")
    ctx = _build_context(cfg, memory, remaining=5, mode=GenerationMode.new, parent=None)
    memory.close()
    return ctx


def test_anonymize_on_hides_tickers_and_dates():
    ctx = _ctx(anonymize=True)
    assert ctx.universe_description == ANONYMOUS_UNIVERSE
    full_prompt = _build_system_prompt(ctx) + _build_user_prompt(ctx)
    for secret in _SECRETS:
        assert secret not in full_prompt, f"prompta sızdı: {secret}"
    print("  [ok] anonimleştirme açık: prompta ticker/tarih sızmıyor")


def test_anonymize_off_is_ablation():
    ctx = _ctx(anonymize=False)
    assert "AAPL" in ctx.universe_description
    print("  [ok] anonimleştirme kapalı (ablation): gerçek tarif gidiyor")


def test_custom_anonymous_description_used_and_clean():
    """Kampanya kendi anonim tarifini verebilir (evren large-cap değilse şart),
    ama o tarif de gerçek tarifin sırlarını (ticker/tarih) taşımamalı."""
    custom = ("Küçük ölçekli hisse evreni; düşük likidite, yüksek işlem maliyeti. "
              "Hangi piyasa/şirket/tarih olduğu verilmiyor.")
    ctx = _ctx(anonymize=True, anonymous_description=custom)
    assert ctx.universe_description == custom, "kampanyanın anonim tarifi kullanılmalı"
    full_prompt = _build_system_prompt(ctx) + _build_user_prompt(ctx)
    for secret in _SECRETS:
        assert secret not in full_prompt, f"özel anonim tarifle prompta sızdı: {secret}"
    print("  [ok] kampanyaya özel anonim tarif: kullanılıyor + ticker/tarih sızmıyor")


def test_config_anonymous_description_has_no_secrets():
    """Gerçek configs/campaign.yaml'daki anonim tarif denetlenir: canlı kampanyada
    LLM'e giden metin bu — içine ticker/tarih kaçarsa memorization önlemi delinir."""
    import io
    import os
    import re

    import yaml

    path = os.path.join(os.path.dirname(__file__), "..", "configs", "campaign.yaml")
    campaign = yaml.safe_load(io.open(path, encoding="utf-8"))["campaign"]
    desc = campaign.get("anonymous_description")
    if not desc:
        print("  [ok] config'te özel anonim tarif yok (varsayılan kullanılacak)")
        return
    assert not re.search(r"\b(19|20)\d{2}\b", desc), f"anonim tarife YIL sızmış: {desc!r}"
    # Ticker benzeri 2-5 harfli BÜYÜK kelimeler (OHLCV gibi terimler hariç)
    allowed = {"OHLCV", "LLM", "DSL", "ABD", "USD", "ETF", "IPO", "ROE", "AL", "SAT"}
    suspects = [w for w in re.findall(r"\b[A-Z]{2,5}\b", desc) if w not in allowed]
    assert not suspects, f"anonim tarife ticker benzeri sızmış olabilir: {suspects}"
    print("  [ok] configs/campaign.yaml anonim tarifi temiz (yıl/ticker yok)")


def main():
    test_anonymize_on_hides_tickers_and_dates()
    test_anonymize_off_is_ablation()
    test_custom_anonymous_description_used_and_clean()
    test_config_anonymous_description_has_no_secrets()
    print("OK — memorization önlemi testleri geçti.")


if __name__ == "__main__":
    main()
