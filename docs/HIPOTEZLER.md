# Altı Hipotez — üç dilde

Bunlar **sistemin gerçekten ürettiği** hipotezler; örnek olsun diye yazılmadı.
Her biri üç katmanda anlatılıyor:

- **① Sade** — konuyu hiç bilmeyen birinin anlayacağı dil
- **② Teknik** — bizim çalışma dilimiz
- **③ Makine dili** — sistemin gerçekten işlediği DSL (LLM bunu üretiyor)

Altısı bilerek seçildi: biri **geçti**, biri **aşırı uydurma**, biri **kırılgan**,
biri **kural ihlali**, biri **ölü kural**, biri **kopya**. Yani sadece "sistem ne
buldu"yu değil, **neyi nasıl elediğini** de gösteriyorlar.

> **Not — LLM serbest kod yazmıyor.** Metin olan tek şey *iddia cümlesi* (insan
> için gerekçe). Makinenin işlediği şey aşağıdaki tipli JSON ağacı; pydantic
> doğruluyor, statik denetleyici sızıntıya bakıyor, sonra derleniyor.
> Aşağıdaki JSON'lar okunabilirlik için boş varsayılan alanlardan arındırıldı;
> yapı birebir aynı.

Ortak kurulum (hepsinde): kripto perpetual evreni, kesitsel long-short,
en iyi %20 al / en kötü %20 sat, işlem maliyeti 10 bps, sinyal kapanışta
üretilir **ertesi açılışta** işlenir (`close_t → open_t_plus_1`).

---

## 1. `hyp_0033` — GEÇEN TEK ADAY ✅

**① Sade.** Vadeli piyasada kaldıraçlı pozisyon tutmanın bir bedeli var: *funding*.
Bu bedel çok yükseldiyse, kalabalık aynı yöne yığılmış demektir. Kalabalığın
yığıldığı yer genelde tehlikelidir. Bu fikir dört şeye birden bakıyor:
**kalabalık ne kadar yığılmış**, **son 60 günde ne kadar yükselmiş**, **ne kadar
kolay alınıp satılıyor**, ve **gün içinde alıcı mı satıcı mı baskın**. Bunları
tek tek değil **birlikte** değerlendiriyor: "kalabalık aşırı yığılmış **ama**
yükseliş zayıf **ve** likidite düşükse" tehlike sinyali.

**② Teknik.** 4 feature → random forest → tahmin edilen 10 günlük ileri getiri →
kesitsel sıralama → long/short %20. Haftalık denge.

**③ Makine dili.**
```json
funding_crowd = {"op": "cross_sectional_rank", "inputs": [{"op": "field", "field": "funding_rate"}]}
mom_60d       = {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
liq_z         = {"op": "zscore", "window": 120, "inputs": [{"op": "field", "field": "dollar_volume"}]}
intraday_buy  = {"op": "close_location", "window": 20}

model     = random_forest
execution = {"signal_time": "close_t", "trade_time": "open_t_plus_1",
             "holding_period_days": 10, "rebalance": "weekly"}
```

**Karne:**

| araştırma | HOLDOUT (kilitli) | İLERİ-TEST (taze) | hüküm |
|---|---|---|---|
| +0.66 | **+0.72** | **+0.45** | **DOĞRULANDI** |

İleri-testte 575 günde **+%11**; aynı dönemde al-tut −%70. **Ama** çoklu-test
düzeltmesini geçmiyor (DSR 0.20). Doğru okuma: *"alpha bulduk" değil, "bu aday
henüz ölmedi."*

---

## 2. `hyp_0021` — AŞIRI UYDURMANIN DERS KİTABI ❌

**① Sade.** Aynı ailede bir fikir: momentum + gün içi alım baskısı + oynaklık
rejimi + funding kalabalığı. Geçmiş veride **en parlak** görünen hipotez buydu.

**② Teknik.** Yapı `hyp_0033`'e çok benzer; farkı oynaklığı 60 günle ölçüp
gün içi baskıyı 20 günlük ortalamayla yumuşatması.

**③ Makine dili.**
```json
mom60        = {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
close_loc_20 = {"op": "rolling_mean", "window": 20, "inputs": [{"op": "close_location", "window": 20}]}
vol60        = {"op": "volatility", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
funding_z60  = {"op": "zscore", "window": 60, "inputs": [{"op": "field", "field": "funding_rate"}]}
```

**Karne:**

| araştırma | HOLDOUT | İLERİ-TEST | hüküm |
|---|---|---|---|
| **+1.14** (en yüksek) | +0.93 | **−0.36** | **REJİM-BAĞIMLI** |

Taze veride **−%44**. Ders şu: **en yüksek in-sample Sharpe, en sert çöken oldu.**
"En iyi Sharpe'ı seç" sezgisi bu veride **ters** çalışıyor. Bu yüzden sistem
adayı araştırma skoruna göre değil, **kanıta** göre seçiyor.

---

## 3. `hyp_0015` — SAYI İYİ AMA KIRILGAN ❌

**① Sade.** "Yükselen, gün içinde alıcısı güçlü, sakin ve likit olanlar
kazanır." Kâğıt üstünde çok iyi görünüyordu.

**② Teknik.** Araştırma Sharpe **+1.32** — kabul eşiğinin (0.50) çok üstünde.
Kapıyı geçti ama **sağlamlık testinde** elendi: pencereler ±%20 oynatıldığında
Sharpe **−0.47**'ye düştü. Yani sonuç seçilen pencerelere yapışık; gerçek bir
kural değil, o pencerelerin şansı.

**③ Makine dili.**
```json
mom60          = {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
close_loc_20   = {"op": "close_location", "window": 20}
vol60          = {"op": "volatility", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
dollar_vol_z60 = {"op": "zscore", "window": 60, "inputs": [{"op": "field", "field": "dollar_volume"}]}
```

**Sonuç:** `robustness_rejected` — `perm_p=0.00, cost2x=0.87, param_min=-0.47`

Backtest'i geçen bir strateji **yine de** reddedilebiliyor. Kapı tek katman değil.

---

## 4. `hyp_0005` — BACKTEST'E HİÇ GİRMEDİ ❌

**① Sade.** Fikir makuldü ama LLM, kampanyanın **izin verdiği tutma süreleri
dışında** bir süre seçti (7 gün). Kural ihlali, o yüzden hiç test edilmedi.

**② Teknik.** `execution.holding_period_days = 7`; izinli küme
`[5, 10, 20, 60, 90, 120]`. Statik denetleyici **backtest'ten önce** reddetti.

> Bu kısıt bir dönem **beyan ediliyor ama uygulanmıyordu** — prompt LLM'e
> "zorunlu" diyor, doğrulayıcı hiç bakmıyordu. Kapatıldı ve teste bağlandı.

**③ Makine dili.**
```json
execution = {"signal_time": "close_t", "trade_time": "open_t_plus_1",
             "holding_period_days": 7,        // ← REDDEDİLDİ
             "rebalance": "weekly"}
```

**Sonuç:** `static_rejected` — *"holding_period_days=7 bu kampanyada izinli değil"*

Bütçe boşa harcanmadı: hipotez veriye **hiç dokunmadan** elendi.

---

## 5. `hyp_0023` — KURAL VAR AMA ÖLÜ ⚠️

**① Sade.** En "akıllı" görünen fikir: *"Piyasa çalkantılıysa kalabalığın tersine
git; sakinse trendi takip et."* Yani duruma göre strateji değiştiren bir kural.

**② Teknik.** Burada model `dsl_formula` — yani mantık açıkça yazılı, bir
`conditional` düğümü var. **Ama** koşul (`high_vol`) hücrelerin **%0.0**'ında
tetikleniyor. Rejim koşullaması **fiilen ölü**: kural yazılmış, hiç çalışmıyor.
Strateji her zaman tek daldan gidiyor, ama raporda "rejim-duyarlı" gibi görünüyor.

**③ Makine dili.**
```json
vol_rank         = {"op": "cross_sectional_rank", "inputs": [{"op": "feature_ref", "name": "vol60"}]}
high_vol         = {"op": "greater_than", "inputs": [{"op": "feature_ref", "name": "vol_rank"},
                                                     {"op": "const", "value": 0.5}]}
funding_reversal = {"op": "negate", "inputs": [{"op": "feature_ref", "name": "funding_z60"}]}
high_vol_signal  = {"op": "multiply", "inputs": [{"op": "feature_ref", "name": "funding_reversal"},
                                                 {"op": "feature_ref", "name": "close_loc_20"}]}
low_vol_signal   = {"op": "multiply", "inputs": [{"op": "feature_ref", "name": "mom60"},
                                                 {"op": "feature_ref", "name": "close_loc_20"}]}

signal = {"op": "cross_sectional_rank",
          "inputs": [{"op": "conditional",
                      "inputs": [{"op": "feature_ref", "name": "high_vol"},        // koşul
                                 {"op": "feature_ref", "name": "high_vol_signal"}, // doğruysa
                                 {"op": "feature_ref", "name": "low_vol_signal"}]}]} // yanlışsa
```

**Sonuç:** `degenerate_conditional` → **revizyona** gönderildi (silinmedi).

Bu, gözle yakalanamayacak bir hata: kod çalışıyor, sayı üretiyor, ama
**anlattığı şeyi yapmıyor.**

---

## 6. `hyp_0006` — AYNI FİKRİN KILIK DEĞİŞTİRMİŞ HÂLİ ⚠️

**① Sade.** LLM'in en sık yaptığı şey: bir fikri beğenince etrafında dönüp
durmak. Bu hipotez `hyp_0002`'nin neredeyse aynısı — tek fark oynaklığı 20 gün
yerine 60 günle ölçmesi. Başlığı bile bunu itiraf ediyor: *"daha kararlı rejim
tespiti için 60 günlük oynaklık."*

**② Teknik.** Özgünlük skoru **0.22**, eşik 0.25. `hyp_0002`'ye **%78 benzer**.
Backtest'e **sokulmadan** reddedildi ve LLM'den yeni fikir istendi.

**③ Makine dili.**
```json
funding_z   = {"op": "zscore", "window": 60, "inputs": [{"op": "field", "field": "funding_rate"}]}
vol60       = {"op": "volatility", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
mom60       = {"op": "return", "window": 60, "inputs": [{"op": "field", "field": "close"}]}
close_loc_5 = {"op": "close_location", "window": 5}
```

**Sonuç:** `low_originality` — *"özgünlük 0.22 eşik-altı; hyp_0002'ye %78 benzer"*

**Neden önemli:** aynı fikri 10 kez denemek, çoklu-test düzeltmesinde 10 deneme
sayılır ve gerçek bir bulguyu istatistiksel olarak öldürür. Bir kampanyada
24 slotun **11'i** bu şekilde yandığı ölçüldü; sebebi bulunup düzeltildi.

---

## Altısı birlikte ne anlatıyor?

| # | hipotez | nerede elendi / ne oldu | hangi tehlikeyi yakalıyor |
|---|---|---|---|
| 1 | `hyp_0033` | **üç dönemi de geçti** | — (tek hayatta kalan) |
| 2 | `hyp_0021` | taze veride çöktü | **aşırı uydurma** |
| 3 | `hyp_0015` | sağlamlık testi | **kırılganlık** (pencereye yapışık) |
| 4 | `hyp_0005` | statik denetim | **kural ihlali** (backtest öncesi) |
| 5 | `hyp_0023` | ölü koşul | **yazılan ≠ çalışan** |
| 6 | `hyp_0006` | özgünlük | **tekrar** (çoklu-test enflasyonu) |

Sistemin değeri "hipotez üretmesi" değil — **ürettiğini eleyebilmesi.**
Üretmek kolay; elemek zor.

---

## Bir uyarı: hipotez numaraları tekil değil

`hyp_0033` numarası **iki farklı hipoteze** ait: yukarıdaki (v2 kampanyası,
geçen aday) ve v4 kampanyasındaki *"Funding extremity with intraday exhaustion
reversal"* (reddedilmiş, bambaşka bir fikir). Numaralar kampanya içinde
sıfırlandığı için çakışıyorlar.

Bu yüzden aday sicili numaraya değil **içerik parmak izine** göre tutuluyor
(`213a3da7…`). Bu belgedeki `hyp_0033`, sicildeki v2 adayıdır.
