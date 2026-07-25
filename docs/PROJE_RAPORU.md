# LLM Tabanlı Otonom Quant Araştırmacısı — Teknik Rapor

---

## 1. Proje nedir?

Bir insan quant araştırmacısının yaptığı işi **kapalı bir döngüde otomatikleştiren** bir sistem:

> Hipotez üret → stratejiye derle → sızıntısız backtest et → istatistiksel değerlendir → kabul/red/geliştir → hafızaya yaz → öğren → yeni hipotez üret → tekrarla.

**Kritik nokta:** Projenin çıktısı "kârlı bir strateji" değil, **kendini kandırmayan, tekrar üretilebilir bir araştırma sisteminin kendisi.** Öneri dokümanının 22. bölümü bunu açıkça söyler. Bu, raporun sonundaki sonuçların neden "null" olduğunu ve bunun neden bir başarı olduğunu anlamak için şarttır.

Sistem, literatürde **"autoresearch"** (Karpathy döngüsü) denen paradigmanın finans uyarlamasıdır — NVIDIA'nın ve Zhengyao Jiang'ın Temmuz 2026'da paylaştığı örneklerle aynı aile, farklı alan.

---

## 2. Temel tasarım kararları (ve nedenleri)

### 2.1. Üç katmanın kesin ayrımı
Sistemin kalbi budur:

| Katman | Sorumluluk | Neden |
|---|---|---|
| **LLM** | Ekonomik mekanizma, hipotez, yapısal değişiklik, deney yorumu | Yaratıcılık ve alan bilgisi burada |
| **Deterministik sistem** | Veri erişimi, zaman tutarlılığı, derleme, backtest, istatistik, holdout | Denetlenebilirlik ve tekrar üretilebilirlik burada |
| **Sayısal optimizasyon** | Sürekli parametreler (pencere uzunlukları, eşikler) | LLM sayı aramada kötü; optimizer iyi |

**Neden bu ayrım?** LLM'in serbest Python yazıp çalıştırmasına izin verilseydi, sistem backtest motorundaki açıkları sömürerek sahte alpha üretirdi (literatürde *reward hacking*; `dietmarwo/autoresearch-trading` bunu gerçek bir vakada gösterdi). LLM yalnızca **doğrulanabilir bir DSL ağacı** üretir.

### 2.2. Point-in-time veri zorunluluğu
Bugünün S&P 500 listesini geçmişe uygulamak **survivorship bias**'tır (iflas edenler görünmez → getiriler şişer). Sistem, Wikipedia'nın endeks değişiklik tarihçesinden **her tarih için gerçek üye kümesini** kurar; pencerede bir gün bile üye olmuş ~700 ticker indirilir ve hisse **yalnızca üye olduğu günlerde** işlem görür. Delist olanlar veri setinden çıkarılmaz; pozisyondan **-%30 şokla** çıkar (CRSP delisting return'ün tutucu vekili).

### 2.3. Kilitli holdout (tek atış)
Araştırma ajanı nihai test dönemini **göremez**. Holdout ayrı bir servistir, LLM'i import bile etmez, aynı hipotezi iki kez değerlendirmez (UNIQUE audit tablosu). Sebep: test sonucunu görüp strateji ayarlamak, test verisini eğitim verisine çevirir.

### 2.4. Çoklu-test farkındalığı mimarinin parçası
**Başarısızlar dahil her deney kaydedilir.** Sadece kazananları saymak, yapılan gerçek deneme sayısını gizler ve istatistiği geçersiz kılar. FDR / Deflated Sharpe hesapları bu tam kayıt üzerinden yapılır.

### 2.5. Tek metrik optimize edilmez
Yalnızca Sharpe optimize etmek Goodhart etkisi ve overfitting üretir. Sistem hard gate + çok amaçlı Pareto + çoklu-test kullanır.

---

## 3. Sistem mimarisi — klasör klasör

### `contracts/` — İstasyonlar arası akan 5 obje (sözleşmeler)
Sistemin omurgası. Kutuların içi değişebilir, **sözleşmeler sabittir** (modülerliğin temeli).

| Dosya | Ne taşır |
|---|---|
| `research_context.py` | LLM'e giden **her şey** (prompt'un kendisi): hedef, evren, izinli operatörler, geçmiş deneyler, dersler |
| `hypothesis_spec.py` | LLM'in **yapılandırılmış** çıktısı: iddia, ekonomik mekanizma, features, sinyal, **model**, portföy, execution, yanlışlanma kriterleri |
| `strategy_graph.py` | HypothesisSpec'in derlenmiş, deterministik hali (operatör DAG'ı + sızıntı etiketleri) |
| `backtest_result.py` | Fold metrikleri, net getiriler, maliyet dökümü, IC/accuracy, reproducibility bilgisi |
| `decision.py`, `review.py` | Kabul/red/revizyon kararı ve bağımsız reviewer raporları |

**Neden Pydantic şeması?** LLM serbest metin üretirse doğrulanamaz. Şema, geçersiz çıktıyı daha derlemeden reddeder (Doküman 17.2).

### `dsl/` — LLM'in konuşabildiği tek dil
| Dosya | İş |
|---|---|
| `operators.py` | Operatör kaydı: tip, arite, zaman yönü, pencere sınırları. **Yarım-bar "tick" zaman modeli** burada (`open_t=0, close_t=1, open_t+1=2...`) |
| `compiler.py` | HypothesisSpec → StrategyGraph (deterministik; aynı girdi hep aynı graph) |
| `static_validator.py` | **Çalıştırmadan önce** sızıntı denetimi |

**En kritik kontrol:** her düğümün "bilgi anı" (`info_tick`) ağaçta aşağıdan yukarı yayılır, sonra tek bir eşitsizlik: `signal.info_tick < execution.info_tick`. Bu, look-ahead bias'ı **yapısal olarak** imkânsız kılar. Ayrıca: dejenere koşul tespiti (iki dalı aynı olan sahte `conditional`), izinsiz veri alanı reddi, şema=çalıştırılan kontrolü.

### `data/` — Veri katmanı
| Dosya | İş |
|---|---|
| `pit_universe.py` | Point-in-time S&P 500 üyeliği + GICS sektörleri (survivorship düzeltmesi) |
| `adapter.py` | Veri kaynağı soyutlaması: sentetik / yfinance / sp500_pit **aynı arayüzü** paylaşır (`load() → MarketData`) |
| `synthetic.py` | Bilinen özellikli veri üreteci (momentum / reversal / saf rastgele) |

**`synthetic.py` neden var?** Gerçek veride **doğru cevabı bilmiyoruz** — Sharpe 0.8 gerçek keşif mi, motordaki bug mu, ayırt edilemez. Sentetik veride cevabı biz koyarız: momentum gömeriz → motor bulmalı; saf gürültü veririz → motor **hiçbir şey** bulmamalı. Bu, bir backtest motorunu doğrulamanın tek yoludur (Doküman 23.1). **Araştırma gerçek veride yapılır; sentetik yalnızca kalibrasyon aletidir.**

### `backtest/` — Deterministik çekirdek
| Dosya | İş |
|---|---|
| `evaluator.py` | StrategyGraph → sinyal paneli (tarih × varlık) |
| `engine.py` | Vectorized motor: **beyan edilen** execution gerçekten uygulanır (trade_time → bar offset + faz), düzeltilmiş fiyatlardan getiri, işlem maliyeti, portföy kuralları, sektör-nötr, evren filtreleri |
| `walk_forward.py` | K dilimde ayrı değerlendirme + fold tutarlılığı + **IC / RankIC / ICIR / directional accuracy** |
| `model_signal.py` | **MODEL kutusu**: features = X, hedef = ileriki getiri, walk-forward + embargo ile fit, tahmin = sinyal |

**"Şema = çalıştırılan şey" ilkesi:** hipotez `open_t_plus_1`'de işlem diyorsa motor gerçekten orada işlem yapar. Beyan ile uygulama ayrışırsa tüm rapor yalan olur.

### `models/` — Model havuzu *(yeni: hipotez → MODEL → backtest)*
`registry.py`: sklearn `LinearRegression` + `GaussianNB`. Model, DSL özelliklerini **sürekli bir tahmin skoruna** çevirir; skor sıralanıp portföye girer.
**Sızıntı güvenliği:** model her walk-forward diliminde **yalnızca geçmişe** fit edilir; test bloğuna sarkan eğitim hedefleri **embargo** ile atılır (purged walk-forward).

### `evaluation/` — Değerlendirme ve istatistiksel yönetişim
| Dosya | İş |
|---|---|
| `hard_gate.py` | Deterministik red kapısı (sabit eşikler; LLM gameleyemez) |
| `robustness.py` | Permutation testi, maliyet 2x, parametre perturbasyonu, bir bar ek gecikme |
| `statistics.py` | PSR, **Deflated Sharpe Ratio**, moving-block bootstrap CI, Benjamini–Hochberg FDR (scipy'siz) |
| `multiple_testing.py` | Tüm denemeleri hesaba katan rapor; parametre varyantlarını parent'a katlar |
| `pareto.py` | Çok amaçlı sıralama (Sharpe alt sınırı / drawdown / turnover) |
| `improvement.py` | **İyileşme eğrisi**: best-so-far + self-contained SVG |

### `memory/` — Araştırma hafızası (3 katman)
| Dosya | Katman | İş |
|---|---|---|
| `store.py` | **Episodic** | SQLite: her deneyin tam kaydı (hipotez, karar, metrik, seed, prompt/output hash, lineage) |
| `semantic.py` | **Semantic** | "Hangi faktör ailesi umut verici/zayıf" |
| `procedural.py` | **Procedural** | "Hangi araştırma hamlesi (revizyon/ters-çevirme/birleştirme) işe yarıyor" |
| `similarity.py` | — | **3 seviye tekrar kontrolü**: metinsel (leksikal cosine), yapısal (AST Jaccard), davranışsal (sinyal korelasyonu) |

**Neden tekrar kontrolü?** Aynı fikri N kez denemek hem bütçe yer hem çoklu-test muhasebesini bozar. Yapısal kontrol **backtest'ten önce** çalışır → bedava bütçe koruması.

### `agents/` — LLM ve denetçi ajanlar
| Dosya | Rol |
|---|---|
| `hypothesis_generator.py` | ResearchContext → prompt → LLM → HypothesisSpec (+ tek onarım denemesi) |
| `quant_critic.py` | **Bağımsız** eleştirmen: ekonomik mekanizma tutarlı mı, etiket sinyalle uyuşuyor mu (backtest'ten ÖNCE → bütçe korur) |
| `backtest_auditor.py` | Deterministik denetçi: sızıntı, survivorship, maliyet, likidite |
| `statistical_reviewer.py` | Deterministik denetçi: FDR, güven aralıkları, fold tutarlılığı |
| `literature.py` | Klasik anomali corpus'u (hipotez üreticiye tohum) |

**Üç bağımsız reviewer** (Doküman 15): üreten LLM kendi stratejisini onaylamaz.

### `llm/` — Değiştirilebilir model kutusu
`openai_client.py`: tek OpenAI-uyumlu istemci (OpenRouter bugün, vLLM yarın — sadece `base_url` değişir). API anahtarı **asla** koda girmez.
`providers.py`: `HypothesisProvider.next(context) → HypothesisSpec` arayüzü sabit; dummy/random/GP/Bayesian/LLM hepsi aynı arayüz.

### `orchestrator/` — Döngünün kendisi
`loop.py`: her iterasyonda bir hipotezi pipeline'dan geçirir; üretim modunu seçer (yeni / revizyon / ters-çevirme / birleştirme), bütçeyi yönetir.
`budget.py`: **Thompson Sampling bandit** — araştırma bütçesini strateji aileleri arasında dağıtır (başarılıya daha çok, belirsizde keşfe devam).

**Neden basit Python loop?** MVP için LangGraph/Temporal/Ray gereksiz karmaşıklık. Sözleşmeler sabit olduğu sürece ileride değiştirilebilir.

### `optimization/` — Sayısal arama
`parameter_search.py`: yapı sabit tutulur, **pencereler** aranır. Doküman 27'nin en vurgulu ayrımı: LLM yapıyı, optimizer sayıyı arar. **Dürüst sayım:** optimizer'ın yaptığı her backtest bir denemedir ve çoklu-test muhasebesine girer.

### `baselines/` — "LLM gerçekten daha iyi mi?" (Deney A)
`random_search.py` (kör örnekleme), `genetic_programming.py` (fitness'a göre evrim), `bayesian_opt.py` (TPE).
**Neden şart?** Akademik iddia ancak baseline karşılaştırmasıyla savunulur. Random-search alt çıtadır: LLM ondan iyi değilse LLM'in katkısı yoktur.

### `holdout/`, `dashboard/`, `tests/`, `configs/`
- `holdout/service.py`: kilitli veri dışarı verilmez, tek atış, ayrı audit log, LLM import etmez.
- `dashboard/report.py`: tek dosyalık statik HTML (funnel, leaderboard, çoklu-test, lineage, reviewer raporları, düz Türkçe strateji anlatımı).
- `tests/`: 32 paket — sızıntı (8 test), golden backtest (regresyon), property testleri, model katmanı, benzerlik, holdout…
- `configs/`: `campaign.yaml` (evren, bütçe, risk kısıtları, izinli operatörler), `data.yaml` (veri kaynağı), `models.yaml` (model card).

**Neden config?** Model/veri değiştirmek = tek satır YAML. Kod değişmez (tak-çalıştır).

---

## 4. Bir deneyin hayatı (uçtan uca akış)

```
ResearchContext (hedef + hafıza + dersler)
   ↓ [LLM]
HypothesisSpec (iddia + mekanizma + features + model + execution)
   ↓ derle
StrategyGraph  →  Static Validator (SIZINTI kontrolü) ─── ret ──→ kayıt
   ↓ geçti
Tekrar kontrolü (metinsel/yapısal) ─── duplicate ──→ kayıt (bütçe korunur)
   ↓ yeni
Quant Critic (ekonomik ön-eleme) ─── revise ──→ kayıt
   ↓ geçti
Sinyal üret  →  [MODEL kutusu ya da DSL formülü]
   ↓
Walk-forward backtest (K fold + IC + accuracy)
   ↓
Hard Gate (Sharpe / drawdown / turnover / fold tutarlılığı) ─── ret ──→ kayıt
   ↓ geçti
Sağlamlık (permutation, maliyet 2x, parametre, ek gecikme) ─── ret ──→ kayıt
   ↓ geçti
Parametre optimizasyonu (her deneme sayıma girer)
   ↓
KABUL → hafıza → çoklu-test raporu → (holdout adayı)
```

Her aşamada karar **kaydedilir** — başarısızlar dahil.

---

## 5. Sonuçlar (gerçek veri: point-in-time S&P 500, 2015–2023, ~700 ticker)

### 5.1. Ham sayılar
| | |
|---|---|
| Toplam kayıtlı deney | **405** |
| Backtest edilen | **275** |
| Kabul edilen | **11** |
| Reddedilen | 268 |
| Tekrar (duplicate) elenen | 115 |
| Revizyon istenen | 11 |
| Farklı strateji **yapısı** | **161** |

**Elenme nedenleri:** hard gate 120, tekrar 115, parametre-arama denemesi 144, critic 8, statik/sızıntı 3, dejenere koşul 3, derleme hatası 1.

### 5.2. En iyi kabul edilen stratejiler
| Hipotez | Sharpe | MaxDD | Turnover | Strateji |
|---|---|---|---|---|
| hyp_0221 | 0.97 | %8 | 43 | Volatilite + duyarlılık koşullu kısa-vade reversal |
| hyp_0214 | 0.96 | %8 | 40 | Orta-volatilite + negatif duyarlılıkta reversal |
| hyp_0178 | 0.96 | %8 | 43 | Volatilite-ayarlı kısa-vade reversal |
| hyp_0259 | 0.96 | %8 | 43 | Rejim-koşullu kısa-vade reversal |
| hyp_0251 | 0.95 | %8 | 45 | Çok-rejimli volatilite + duyarlılık |

Tutma süresi 5–10 gün (**swing**), long+short, günlük frekans.

### 5.3. **Ana bulgu: çoklu-test sonrası hiçbir strateji hayatta kalmıyor**
| | |
|---|---|
| Ham backtest | 275 |
| **Birincil (tekil) strateji** | **118** |
| **FDR'ı geçen** | **0** |
| En yüksek Deflated Sharpe | **0.032** (eşik: 0.95) |

**Okuma:** En iyi stratejinin ham Sharpe'ı 0.97 ve ham p-değeri ~0.011 — tek başına bakılsa "anlamlı" görünür. Ancak **118 birincil strateji tarandığı** hesaba katılınca Deflated Sharpe 0.032'ye düşer: gözlenen performans, bu kadar aramanın **şans eseri** üreteceği seviyeden ayırt edilemiyor.

> **Sonuç: Etkin, likit large-cap piyasada yalnızca fiyat/hacim verisiyle doğrulanmış (istatistiksel olarak savunulabilir) alpha bulunamamıştır.**

Bu bir başarısızlık değil, sistemin **tasarlandığı gibi çalışmasıdır**. Kötü bir sistem "Sharpe 0.97 buldum!" der; bu sistem "118 deneme yaptım, bu sonuç şans olabilir, güvenilir değil" der. **Holdout bilinçli olarak açılmamıştır** — tek atışlık kaynak, DSR'yi geçemeyen adaylar için harcanmaz.

### 5.4. Sistemin doğru çalıştığının kanıtları (kontrollü ortam)
Gerçek veride "sinyal yok" sonucunun **motordaki bir hatadan** kaynaklanmadığı, bilinen-cevaplı sentetik veriyle doğrulandı:

| Test | Sonuç |
|---|---|
| Gömülü momentum → model bulmalı | **IC +0.21**, directional accuracy **%55.6**, Sharpe +7.5 ✓ |
| Saf rastgele → hiçbir şey bulmamalı | IC **~0.00**, accuracy **%50.2** ✓ (sahte alpha yok, sızıntı yok) |
| Model, elle yazılmış formülü geçiyor mu | Evet (+7.5 vs +6.4; iki özelliği optimal birleştiriyor) |
| Sızıntı testleri (8 senaryo) | Hepsi yakalanıyor ✓ |
| Golden backtest (regresyon) | Referans Sharpe'lar sabit ✓ |

**Directional accuracy** = ML'deki accuracy'nin finans analoğu. Not: finansta %55 yön isabeti **güçlüdür**; bir trading modelinde %97 accuracy görülmesi başarı değil, **sızıntı işaretidir** — sistemin tüm denetim katmanı tam bunu yakalamak için vardır.

### 5.5. Maliyet
Kampanya başına **~$0.15** (≈200k token, DeepSeek V3 + GPT-4o-mini critic).

---

## 6. Yol boyunca elde edilen metodolojik bulgular

1. **Mode collapse ve model gücü.** gpt-4o-mini tek fikre saplandı (219 backtest → 27 yapı). DeepSeek V3'e geçince yapı çeşitliliği **3.6 kat** arttı (0.12 → 0.43 yapı/hipotez), aileler dengelendi. **Ama null sonuç değişmedi** — bu, sonucun modelin zayıflığından değil **piyasanın gerçeğinden** kaynaklandığını gösterir. (Ablation değeri yüksek bir bulgu.)
2. **Ters çevirme deneyi.** Yüksek turnover'lı bir strateji (-2.01 Sharpe) ters çevrilince **yine -2.01** verdi (+2.01 değil). Kayıp yönden değil **işlem maliyetinden**. Sistem sadece kabul/red değil, gerçek ekonomik içgörü üretiyor.
3. **Sahte rejim-koşullaması.** LLM iki dalı aynı olan `conditional` yazarak "rejim-koşullu" etiketi alıyordu; dejenere-koşul tespiti eklendi.
4. **Parametre varyantları çoklu-testi şişiriyordu.** Korelasyonlu pencere varyantları bağımsız deneme değildir; artık parent'a katlanıyor (Doküman 10.1).
5. **Maliyet-getiri hizalaması hatası** bulundu ve düzeltildi (maliyet, pozisyon getiri kazanmadan bir bar önce düşüyordu).

---

## 7. Literatürdeki yeri

En yakın çalışma: **Microsoft RD-Agent(Q) + Qlib** — faktör ve modeli birlikte optimize eden, IC/ICIR metrikleriyle çalışan, Thompson bandit kullanan autoresearch sistemi. Ayrıca AlphaAgent (KDD'25), FactorMiner, AlphaMemo, Hubble.

**Bizim ayrımımız:** Bu sistemlerin hiçbiri **ciddi çoklu-test kontrolü (FDR / Deflated Sharpe)** ve **kilitli tek-atış holdout yönetişimi** yapmıyor. Faktör üretiminde güçlüler; "kendini kandırmama" katmanında zayıflar. Akademik katkı tam buradadır:

> *Point-in-time ve çoklu-test farkındalığına sahip, bütün başarısız deneyleri kaydeden, doğal dildeki ekonomik hipotezi yanlışlanabilir yapısal bir tanıma dönüştüren ve nihai test verisine araştırma ajanının erişemediği otonom quant araştırma sistemi.*

Ek olarak: NVIDIA'nın ajanı kendi eğitim ortamını kurar; **bizimki bilinçli olarak kuramaz** — finansta ajanın kendi backtest'ini yazması reward hacking'dir. Bu kısıt, alan gereği bir güçtür.

---

## 8. Sıradaki adımlar

1. **İyileşme eğrileri** (devam ediyor): best-so-far Sharpe/IC vs deney sayısı; LLM vs random/GP/Bayesian eğrilerinin yan yana karşılaştırması (Doküman 26 araştırma verimliliği). Altyapı kuruldu; yarı-sentetik kontrollü benchmark'ta eğrinin tırmandığı doğrulandı.
2. **Model katmanının derinleştirilmesi**: hiperparametre optimizasyonu, açık rejim tespiti; ileride Qlib üzerinden LSTM/Transformer.
3. **IC'nin karar kapısına katılması**: yüksek Sharpe + sıfır IC = şans işareti.
4. **Fundamentals + haber** (Faz 2, point-in-time açıklanma tarihiyle).
5. **Değerlendirilmiş ama ertelenmiş**: Qlib'i motor olarak benimsemek (fizibilite testi yapıldı: Windows'ta çalışıyor, pandas verimizi doğrudan alıyor; basit modeller için sklearn yeterli olduğundan şimdilik gerekmedi).

---

## 9. Özet

- Sistem **uçtan uca çalışıyor**: 405 deney otonom olarak üretildi, test edildi, kaydedildi, öğrenildi.
- **Doğrulanmış alpha bulunamadı** — ve bu, disiplinli bir sistemin etkin piyasada vermesi gereken **dürüst cevaptır**.
- Sistemin doğruluğu, bilinen-cevaplı sentetik veriyle **bağımsız olarak kanıtlandı** (gömülü sinyali buluyor, gürültüde uydurmuyor).
- Güçlü bir üretici modelle tekrarlandığında sonuç değişmedi → bulgu **modelin değil piyasanın** gerçeği.
- Asıl teslim edilen değer: **tekrar üretilebilir, denetlenebilir, kendini kandırmayan bir araştırma altyapısı.**
