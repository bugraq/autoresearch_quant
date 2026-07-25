# İki Servis: Modelleme + Backtest

*Hoca (15.07.2026): "En önemli servislerden biri **modelleme ve backtest yapan
iki servis**. Bir modeli alıp (formül / istatistiksel / ML) backtest yapabileceğimiz
servisi geliştir."*

Bu belge o iki servisi anlatır. İkisi de **tek başına** çağrılabilir, LLM'den ve
araştırma döngüsünden **bağımsızdır**, başka bir projeden de kullanılabilir.

Hızlı bakış için: **`python services_demo.py`** (ikisini birlikte çalıştırır).

---

## Servis 1 — Modelleme (`model_service/`)

**Soru:** *"Bu özelliklerle bu model geleceği ÖNGÖREBİLİYOR mu?"*
(Portföy / işlem maliyeti hiç devreye girmez — saf tahmin gücü.)

```python
from model_service import predict

ozellikler = [
    {"name": "momentum", "expression":
        {"op": "return", "window": 20, "inputs": [{"op": "field", "field": "close"}]}},
    {"name": "hacim_z", "expression":
        {"op": "zscore", "window": 60, "inputs": [{"op": "field", "field": "volume"}]}},
]
rapor = predict("random_forest", ozellikler, veri, horizon=5)
print(rapor.summary())
```

**Girdi:** model tipi + özellikler (X) + veri + ufuk (kaç gün ilerinin getirisi).
**Çıktı (`ModelReport`):** `ic`, `rank_ic`, `icir`, `directional_accuracy`, `predictions`.

**Desteklenen modeller:** `linear_regression`, `ridge`, `naive_bayes`,
`random_forest`, `gradient_boosting` *(ileride LSTM — Qlib havuzu)*.

**Sızıntı güvenliği:** model her walk-forward diliminde YALNIZCA geçmişe fit edilir
(embargo'lu, purged) — geleceği yapısal olarak göremez.

**Metrik — IC (Information Coefficient):** tahmin ile gerçek getirinin kesitsel
korelasyonu. `>0.03` = anlamlı sinyal · `~0` = öngörü yok. *(Sharpe'tan bağımsız:
"yüksek Sharpe + sıfır IC" = şans işareti.)*

---

## Servis 2 — Backtest (`backtest_service/`)

**Soru:** *"Bu strateji PARA KAZANDIRIR mı?"* (işlem maliyeti sonrası, gerçekçi.)

```python
from backtest_service import run

rapor = run(strateji, veri)     # strateji: formül YA DA ML modeli
print(rapor.summary())
```

**Girdi:** `HypothesisSpec` (model + özellikler + portföy + execution kuralları) + veri.
**Çıktı (`BacktestReport`):** tahmin kalitesi (IC/RankIC/ICIR/yön isabeti) **+**
performans (Sharpe / maks. düşüş / turnover / fold tutarlılığı) **+** köken (seed,
motor sürümü — tekrar üretilebilirlik).

**Ayırt edici özellik:** **sızıntılı stratejiyi ÇALIŞTIRMAYI REDDEDER** (`LeakageError`).
Piyasadaki hazır motorların hiçbiri bunu yapmıyor — backtest'i koşup yanlış sayıyı
döndürüyorlar. İlkemiz: *yanlış bir sayı, sayı yokluğundan kötüdür.*

**Doğruluk:** motor, kod paylaşmayan bağımsız bir referansla çapraz-doğrulandı —
sentetik + gerçek S&P 500'de iki hesap **birebir aynı** (`test_engine_crossvalidation.py`).

---

## İkisi birlikte: model → backtest

```
                 MODELLEME SERVİSİ            BACKTEST SERVİSİ
özellikler ───►  model tahmin üretir   ───►  strateji para kazandırır mı?
                 "öngörüyor mu?" (IC)         "kâr eder mi?" (Sharpe)
```

Aynı model **ikisinden de** geçebilir: önce modelleme servisi "öngörü var mı?"
(ucuz, hızlı ön-eleme), sonra geçenler backtest servisinde tam sınanır.

---

## Demo sonucu (kontrollü benchmark — alpha momentum×hacim etkileşiminde)

| Servis | Model | Sonuç |
|---|---|---|
| Modelleme | linear_regression | IC **-0.01** (çarpımı temsil edemez) |
| Modelleme | random_forest | IC **+0.17** (etkileşimi KENDİ yakalar) |
| Modelleme | gradient_boosting | IC **+0.16** |
| Backtest | formül (elle momentum×hacim) | Sharpe **+4.83** |
| Backtest | random_forest | Sharpe **+4.43** |

**Okuma:** Ağaç modelleri, "çarp" demeden iki özelliğin etkileşimini kendi keşfediyor
— ML katmanının somut değeri. Lineer model bunu yapamıyor (beklenen). Kontrollü
veride ground-truth bilindiği için servislerin GERÇEKTEN çalıştığı doğrulanır.
