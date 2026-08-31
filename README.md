# Agentic Quant — LLM Tabanlı Otonom Quant Araştırmacısı

Finansal araştırma sürecini kapalı bir döngüde otomatikleştiren sistem:

```
hipotez üret → stratejiye derle → sızıntısız backtest et → istatistiksel
değerlendir → kabul/red/geliştir → hafızaya yaz → yeni hipotez
```

**En hızlı başlangıç:** `agent.bat`'a çift tıkla. Panel açılır, önce mevcut
sonucu gösterir, sonra menüden seçersin. Komut ezberlemek yok.

---

## 1. Sonuç — bir bakışta

> **Sistem çalışıyor ve kendini kandırmayı reddediyor.** Kabul edilen üç aday
> kilitli dönemi geçti; sistem "buldum" demedi, üçüncü bağımsız bir dönemde
> üçünü de çürüttü. Tek aday hâlâ ayakta — ama **"alpha bulundu" değil.**

### Üç-dönem karnesi

| hipotez | araştırma | HOLDOUT *(kilitli)* | İLERİ-TEST *(taze)* | hüküm |
|---|---|---|---|---|
| **hyp_0033** | +0.66 | **+0.72** | **+0.31** | **DOĞRULANDI** |
| hyp_0002 | +0.61 | +1.29 | −0.05 | REJİM-BAĞIMLI |
| hyp_0021 | **+1.14** | +0.93 | −0.36 | REJİM-BAĞIMLI |
| hyp_0025 | +0.86 | +0.59 | −1.46 | REJİM-BAĞIMLI |

**Üç bulgu:**

1. **Tek kilitli dönem yeterli kanıt değil.** Holdout tek bir *rejim
   çekilişidir*; onu geçmek "genelliyor" demek değildir.
2. **Çoklu-test katmanı haklıydı, holdout yanılttı.** DSR 0.20 / 0.09 —
   istatistik "anlamlı değil" dedi, holdout "geçti" dedi, ileri-test
   istatistiği doğruladı. **Hiçbir katman tek başına yetmiyor.**
3. **En yüksek in-sample Sharpe en sert çöktü** (+1.14 → −%44). *"En iyi
   Sharpe'ı seç"* sezgisi bu veride **ters** çalışıyor.

### Başarı ölçütü: rastgeleyi / al-tut'u geçiyor muyuz?

Herkes **aynı evrende, aynı tarihlerde, aynı 10 bps maliyetle** yarışıyor:

| dönem | al-tut | rastgele al-satçı | duygusal trader |
|---|---|---|---|
| araştırma *(kanıt değil)* | ✗ | ✓ | ✓ |
| **HOLDOUT** *(kilitli)* | ✗ | ✓ | ✓ |
| **İLERİ-TEST** *(taze)* | ✓ | ✓ | ✓ |

- Rastgele ve duygusal trader'ı **her dönemde** geçiyoruz — ama bu **düşük bir
  eşik**: ikisi de işlem masrafından batıyor (ortanca maymun −%94).
- Al-tut'u **boğa piyasasında geçemiyoruz, ayı piyasasında geçiyoruz.**
  2025→bugün al-tut **−%76 Sharpe**, biz **+0.31**.
- En derin düşüşümüz al-tut'un **üçte biri** (%20 vs %76) — long-short piyasa
  riski taşımıyor.

### İleri-test sicili (aday hâlâ izleniyor)

Holdout **tek-atıştır ve tükenir**; ileri-test dönemi her gün büyür. Ölçümler
üzerine yazılmaz, **eklenir**:

| ölçüm | Sharpe | biriken getiri |
|---|---|---|
| 29.07.2026 | +0.37 | +%8.0 |
| 30.07.2026 | +0.45 | +%10.5 |
| **18.08.2026** | **+0.31** | **+%6.6** |

Aday ayakta ve hâlâ `DOĞRULANDI`, **ama zayıflıyor** — son 19 günün taze
verisi net negatif geldi. Bu bir zafer turu değil, **süregelen bir sınav**.

### LLM'in katkısı ölçüldü

*"LLM gerçekten arıyor mu, rastgele denemekle aynı mı?"* Aynı veri, aynı
bütçe, tek fark hipotez üreticisi:

| üretici | doğru yapıyı deneme oranı | ilk isabet |
|---|---|---|
| **LLM** | **%86** | **1. hipotez** |
| Bayes optimizasyonu (TPE) | %50 | 2. |
| rastgele arama | %25 | 4. |
| genetik programlama | %22 | 2. |

### ⚠ Dürüst sınırlar

- **"Alpha bulduk" demiyoruz.** Ayakta kalan aday çoklu-test düzeltmesini
  geçmiyor (DSR 0.20). Doğru ifade: **"tek aday henüz ölmedi."**
- **Tek varlık sınıfı** — bu sonuçlar kripto perpetual'dan. Hisse tarafı
  (PIT S&P 500 + EDGAR temel veri) kurulu ama bu sonuçları üretmedi.
- **Kâğıt üstünde** — gerçek emir, slipaj modeli, borsa mikroyapısı yok.
- **Kilitli dönem aşınıyor** — her kullanımda tükeniyor. İleri-test
  tükenmiyor; gücü bu.

---

## 2. Kurulum

Python 3.10+ gerekiyor.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Doğrulama (ikisi de LLM/internet gerektirmez):

```bash
.venv/Scripts/python.exe -m tests.run_all
```

**48/48 test geçmeli.** Geçiyorsa kurulum tamamdır.

### API anahtarı — sadece YENİ kampanya koşacaksan

Aşağıdakilerin **hiçbiri** anahtar istemez: kontrol paneli, karne, kıyas,
dashboard, ileri-test, holdout, Sharpe doğrulama, testler. Yalnızca **yeni
hipotez üretmek** (menü [5]/[6]) LLM çağırır.

Gerekirse kök dizine `.env` koy:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Örnek dosya: `.env.example`. Model seçimi `configs/models.yaml`'da.

---

## 3. Nasıl çalıştırılır — kontrol paneli

```bash
agent.bat          # Windows: çift tıkla
python agent.py    # veya böyle
```

Panel açılışta **önce durumu** gösterir (en güçlü aday + üç-dönem karnesi +
hüküm), komut listesi sonra gelir. Durum yalnız SQLite okur — saniyeler
sürer, veri indirmez, hiçbir şeye yazmaz.

### Sonuçlara bak — *hızlı, veri indirmez, anahtar istemez*

| # | ne yapar |
|---|---|
| **1** | **Karne** — her adayın araştırma / holdout / ileri-test notu + hüküm |
| **2** | **Kıyas** — rastgele / al-tut / duygusal trader'ı geçiyor muyuz (her dönemde ayrı) |
| **3** | **Dashboard** — tarayıcıda görsel rapor (`dashboard.html`) |
| **4** | **Tek fikrin anatomisi** — bir stratejiyi doğuşundan kararına izle (sade/teknik) |

> **Hocaya gösterim için başlangıç noktası: [1], sonra [2], sonra [3].**

### Yeni ölçüm yap — *yavaş, veri indirir veya LLM çağırır*

| # | ne yapar |
|---|---|
| **5** | Kampanyayı sürdür (yeni hipotezler üret) — **LLM anahtarı gerekir** |
| **6** | Yeni kampanya, hafızayı sıfırla (aday sicili ve holdout kaydı korunur) |
| **7** | Holdout sınavı — kilitli dönemde tek-atış, LLM'siz |
| **8** | İleri-test — sistemin gördüğü tarihten sonraki taze veri |

> İlk `[8]` koşusu Binance'den ~665 sembol indirir; **1 saati aşabilir**.
> Takılmış değildir. Sonraki koşular cache'ten saniyeler içinde açılır.

### Denetle — *sayılar doğru mu?*

| # | ne yapar |
|---|---|
| **9** | **Sharpe gerçekten doğru mu** — motoru saf NumPy ve Excel'le karşılaştırır |
| **t** | Bütün testleri koş (48) |
| **k** | Üretici karşılaştırması — LLM vs rastgele/GP/Bayes (**varsayılan bedava**) |
| **d** | Durum / ayarlar — aktif evren, model, bütçe, işlem maliyeti |

Menüsüz doğrudan çalıştırmak istersen:

```bash
.venv/Scripts/python.exe main.py --holdout          # kilitli dönem sınavı
.venv/Scripts/python.exe scripts/benchmark.py --ileri --log
.venv/Scripts/python.exe scripts/forward_test.py --log
.venv/Scripts/python.exe scripts/verify_sharpe.py
.venv/Scripts/python.exe scripts/anatomy.py --aday --log
.venv/Scripts/python.exe compare.py --bedava --seeds 3
```

---

## 4. Boru hattı — hipotezden paraya

Üç katman **kesin ayrı**; LLM asla backtest'e veya veriye dokunmaz ve
**serbest Python yazmaz** — yalnızca onaylı bir DSL'de tipli yapı üretir.

| katman | sorumluluk |
|---|---|
| **LLM** | hipotez, ekonomik mekanizma, yapısal değişiklik, yorum |
| **Deterministik sistem** | veri, derleme, backtest, metrik, istatistik, holdout |
| **Sayısal optimizasyon** | sürekli parametreler (pencere, eşik, ağırlık) |

```
ResearchContext → [LLM] → HypothesisSpec → [Compiler] → StrategyGraph
                → [Backtest] → BacktestResult → [Gate+Critic] → Decision → Memory
```

### Beş adım (gerçek bir aday üzerinden: `hyp_0033`)

**1 · Hipotez üretme.** LLM tipli JSON üretir; pydantic doğrular. Metin olan
tek şey *iddia cümlesi* (insan için gerekçe).

**2 · Modele çevirme.** Derleyici → `StrategyGraph`. Statik denetleyici
**sızıntıya** bakar: `signal.info_tick < trade_time.info_tick` sağlanmıyorsa
hipotez backtest'e **hiç girmez**. Model modunda **bütün** feature'lar
denetlenir. `allowed_fields` / `allowed_horizons` burada uygulanır.

**3 · Model eğitme.** 4 feature paneli = **X**; hedef **y** = ileriki 10 günün
getirisi (`adjusted_close.shift(-10)/adjusted_close − 1`).

- Zaman **6 bloğa** bölünür; blok *i* yalnızca **kendisinden önceki** veriyle
  eğitilmiş modelle tahmin edilir (**walk-forward**).
- **Embargo:** test bloğu başlamadan son 10 bar eğitimden düşülür — bir eğitim
  hedefi test dönemine sarkmaz (*purged*).
- `RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=50,
  random_state=42)` — derinlik bilerek **sığ**: finansal veride sinyal/gürültü
  çok düşük, derin ağaç ezberler.

**4 · Backtest.** Tahmin → kesitsel sıralama → en iyi %20 long / en kötü %20
short, eşit ağırlık. Sinyal kapanışta üretilir, **ertesi açılışta** işlenir:

```python
gross_pnl = (weights.shift(2) * getiri).sum(axis=1)
cost_t    = turnover.shift(2) * (10 / 10_000)
net_pnl   = gross_pnl - cost_t
```

`shift(2)` aynı-bar sızıntısını **yapısal olarak** imkânsız kılar.

**5 · Metrikler.** Sharpe, win rate, biriken P&L, MaxDD, turnover ve
**IC/RankIC**. Yıllıklaştırma ölçeği **veriden** gelir (hisse günlük 252,
kripto günlük 365, kripto 8h **1095**) — sabit varsaymak Sharpe'ı ~%20 kaydırır.

> **IC neden ayrı?** Sharpe'tan bağımsızdır: *yüksek Sharpe + sıfır IC = şans
> işareti*. İkisi birlikte okunur.

### Kabul, bitiş değil

`accept` aldıktan sonra sırasıyla: **sağlamlık** (permütasyon, maliyet 2×,
parametre perturbasyonu) → **çoklu-test** (DSR + PSR + bootstrap CI +
Benjamini-Hochberg FDR) → **kilitli holdout** (tek-atış) → **ileri-test**.

Hüküm tek yerden çıkar (`evaluation/three_period.py`):
`DOĞRULANDI` / `REJİM-BAĞIMLI` / `ÇÖKTÜ` / `EKSİK`.
**Ölçülmemiş dönem asla "geçti" sayılmaz.**

---

## 5. Sayıları kendin doğrula

Sisteme güvenmen gerekmiyor; her sayının bağımsız kontrolü var.

| soru | komut | beklenen |
|---|---|---|
| Sharpe doğru mu? | menü **[9]** | motor == NumPy == Excel, fark **4.4e-16** |
| Bir fikir nasıl sayıya döndü? | menü **[4]** teknik | her düğümün paneli, PnL açık hesap |
| Sızıntı var mı? | `python -m tests.test_leakage` | sızıntılı strateji **reddediliyor** |
| Kapı kandırılabilir mi? | `python -m tests.test_hard_gate` | LLM kendi eşiğini gevşetemiyor |
| Hepsi | `python -m tests.run_all` | **48/48** |

`[9]` ayrıca `runs/sharpe_verification.xlsx` üretir — Excel'de kendi
formülünle (`=ORTALAMA(...)/=STDSAPMA(...)*KAREKÖK(1095)`) kontrol edebilirsin.

> **Tuzak:** `pandas.std()` ddof=1, `numpy.std()` ddof=0. Karıştırılırsa
> Sharpe kayar; motor ddof=1 kullanır.

---

## 6. Proje yapısı

```
agent.py / agent.bat   # KONTROL PANELİ — tek giriş noktası
main.py                # kampanya döngüsü + --holdout
compare.py             # üretici karşılaştırması (LLM vs baseline'lar)

contracts/             # istasyonlar arası akan veri objeleri (Pydantic)
configs/               # kampanya, model kartı, veri, karşılaştırma (YAML)
llm/                   # LLM soyutlaması (openrouter / vllm / dummy)
agents/                # LLM rolleri: hypothesis generator, critic, auditor
baselines/             # LLM'siz arayıcılar: random / genetic / bayesopt
dsl/                   # operatörler + compiler + static_validator (sızıntı)
data/                  # veri adaptörü (kripto / PIT S&P), EDGAR temel veri
backtest/              # motor, portföy, maliyet, execution, walk-forward
backtest_service/      # model-agnostik backtest servisi
model_service/         # model eğitim servisi (walk-forward + embargo)
models/                # model havuzu (formül / istatistiksel / ML)
evaluation/            # hard gate, sağlamlık, istatistik, üç-dönem hükmü
memory/                # episodic/semantic/procedural + benzerlik
holdout/               # kilitli dönem servisi + denetim kaydı
orchestrator/          # döngünün kendisi
optimization/          # parametre arama
scripts/               # şeffaflık: anatomi, kıyas, ileri-test, Sharpe doğrulama
dashboard/             # tek dosyalık statik HTML rapor üretici
tests/                 # 48 test dosyası (pytest gerektirmez)
ui/                    # canlı kampanya paneli (rich TUI)
docs/                  # belgeler (aşağıdaki dizin)
runs/                  # üretilen log / rapor / karşılaştırma çıktıları
arsiv/                 # eski kampanya kayıtları ve emekli bileşenler
```

### Kalıcı durum dosyaları

| dosya | ne tutar |
|---|---|
| `research_memory.sqlite` | kampanya hafızası (her deney, karar, gerekçe) |
| `holdout_audit.sqlite` | kilitli dönem denetim kaydı + aday sicili + **ileri-test zaman serisi** |
| `dashboard.html` | son üretilen görsel rapor |

> `--fresh` yalnızca **kampanya hafızasını** sıfırlar. Holdout denetim kaydı ve
> aday sicili **korunur** (parmak izi anahtarlı) — üç dönemden geçmiş bir aday
> kampanya sıfırlanınca kaybolmaz.

---

## 7. Belgeler

| dosya | içerik |
|---|---|
| [`docs/OZET.md`](docs/OZET.md) | **buradan başla** — projenin özeti |
| [`docs/HIPOTEZLER.md`](docs/HIPOTEZLER.md) | sistemin ürettiği **6 gerçek hipotez**, üç dilde (sade / teknik / makine) |
| [`docs/ZOOM_BRIFING.md`](docs/ZOOM_BRIFING.md) | 20 dakikalık sunum planı + hangi ekran ne zaman |
| [`docs/PROJE_RAPORU.md`](docs/PROJE_RAPORU.md) | ayrıntılı proje raporu |
| [`docs/YOL_HARITASI.md`](docs/YOL_HARITASI.md) | ne yapıldı, neredeyiz, sırada ne var |
| [`docs/SERVISLER.md`](docs/SERVISLER.md) | backtest ve model servislerinin tasarımı |
| [`docs/DENETIM_KAYDI.md`](docs/DENETIM_KAYDI.md) | bulunan ve düzeltilen **~30 hata** (geliştirme günlüğü) |
| [`docs/AUTORESEARCH_TARAMA.md`](docs/AUTORESEARCH_TARAMA.md) | hazır autoresearch-quant çözümleri taraması |
| [`docs/EKOSISTEM_ANALIZI.md`](docs/EKOSISTEM_ANALIZI.md) | ekosistem / literatür analizi |

---

## 8. Yapılandırma

Hepsi `configs/` altında, koda gömülü değil:

| dosya | ne ayarlar |
|---|---|
| `campaign.yaml` | evren, tarih aralığı, **izinli alanlar/ufuklar**, bütçe, işlem maliyeti, kabul eşikleri, model tipi |
| `models.yaml` | hangi LLM, hangi sağlayıcı, sıcaklık |
| `data.yaml` | veri kaynağı (kripto / PIT hisse / sentetik) |
| `compare.yaml` | karşılaştırma yarışmacıları ve değerlendirme ortamı |

Kampanya kısıtları **gerçekten uygulanır**: izinli olmayan bir veri alanı veya
pencere kullanan hipotez backtest'e girmeden reddedilir.

### LLM sağlayıcısı değiştirilebilir

`models.yaml` → `provider`: `openrouter` | `vllm` | `openai_compatible` |
`dummy` | `random` | `gp` | `bayesopt`. Kod değişmez — yerel bir vLLM
sunucusuna geçmek tek satırlık ayardır.

---

## 9. Bilinen sınırlar ve sıradaki adımlar

**Sınırlar** (yukarıdaki "Dürüst sınırlar" bölümünün teknik hâli):

- Delisting getirisi modellenmiyor; Yahoo'da verisi olmayan delist ticker'lar
  yüklemede raporlanır ama kurtarılamaz (tam çözüm CRSP ister).
- Karşılaştırmada şu an yalnız **bedava** modeller ölçüldü; ücretli modeller
  `python compare.py` (filtresiz) ile koşulabilir (~$2/koşu).
- İleri-test tek adayı izliyor; çok adaylı sürekli izleme kurulu ama
  kullanılmadı.

**Sıradaki adımlar:**

1. **İleri-testi düzenli koşmak** — sicil zaman serisi olarak birikiyor; en
   ucuz ve en dürüst ilerleme yolu.
2. **Hisse evreninde tekrar** — PIT S&P 500 + EDGAR hazır; sonuç tek varlık
   sınıfına bağlı kalmasın.
3. **Ücretli modellerle karşılaştırma** — "hangi LLM daha iyi arıyor" sorusu.
