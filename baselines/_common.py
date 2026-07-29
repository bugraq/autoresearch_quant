"""
Baseline'ların ORTAK kampanya-kısıtı yardımcıları.

Neden var: baseline'lar (random / GP / BayesOpt) pencere uzunluklarını zaten
`context.allowed_horizons`'tan çekiyordu, ama `holding_period_days`'i SABİT
`[1, 5, 10]` listesinden seçiyordu. Kampanya izinli ufukları farklıysa (kripto:
[5,10,20,60,90,120]) `holding=1` kampanya kısıtını ihlal eder.

Bu, static validator ufuk kısıtını uygulamaya başlayınca ölçüldü: random-search
üretimlerinin **%47'si**, GP'nin **%27'si** backtest'e bile girmeden
`disallowed_horizon` ile eleniyordu. Yani LLM'i kıyaslayacağımız alt-çıta
(Deney A / MVP kabul kriteri 9) sakat kalıyordu — LLM'in "üstünlüğü" kısmen
rakibin diskalifiye edilmesinden gelirdi.

Kural: baseline da kampanyanın kurallarına UYAR. Adil kıyas, aynı kısıtlar
altında yapılan kıyastır.
"""
from __future__ import annotations

#: Kampanya ufuk kısıtı yokken kullanılacak varsayılan tutma süreleri.
DEFAULT_HOLDINGS = [1, 5, 10]


def allowed_holdings(allowed_horizons: "list[int] | None") -> "list[int]":
    """Kampanyanın izin verdiği tutma süreleri (holding_period_days).

    - Kısıt yoksa: varsayılan kısa süreler.
    - Kısıt varsa: varsayılanlarla kesişim (kısa tutmayı korur).
    - Kesişim boşsa: izinli ufukların EN KISA üçü (ör. kripto -> [5,10,20]).
      Boş liste dönmek üretimi imkânsız kılardı; en kısa ufuklar niyete
      (kısa tutma) en yakın meşru seçenektir.
    """
    if not allowed_horizons:
        return list(DEFAULT_HOLDINGS)
    izinli = sorted({int(h) for h in allowed_horizons})
    kesisim = [h for h in DEFAULT_HOLDINGS if h in izinli]
    return kesisim or izinli[:3]
