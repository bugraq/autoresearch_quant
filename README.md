# Agentic Quant — LLM Tabanlı Otonom Quant Araştırmacısı

Finansal araştırma sürecini kapalı bir döngüde otomatikleştiren sistem:
hipotez üret → stratejiye derle → sızıntısız backtest et → istatistiksel
değerlendir → kabul/red/geliştir → hafızaya yaz → yeni hipotez.

## Temel ilke (3 katman kesin ayrı)
- **LLM** → hipotez, ekonomik mekanizma, yapısal değişiklik, yorum
- **Deterministik sistem** → veri, derleme, backtest, metrik, istatistik, holdout
- **Sayısal optimizasyon** → sürekli parametreler (pencere, eşik, ağırlık)

LLM asla backtest/veriye dokunmaz; serbest Python yazmaz, sadece onaylı bir
DSL ile yapısal strateji tanımı üretir.

## Pipeline akışı (contract'lar)
```
ResearchContext -> [LLM] -> HypothesisSpec -> [Compiler] -> StrategyGraph
-> [Backtest] -> BacktestResult -> [Gate+Critic] -> Decision -> Memory
```

## Yapı
```
contracts/    # istasyonlar arası akan veri objeleri (Pydantic)
configs/      # kampanya, model card, veri, değerlendirme (YAML)
llm/          # LLM soyutlaması (openrouter / vllm / dummy) — değiştirilebilir
agents/       # LLM'i kullanan roller (hypothesis generator, critic, auditor)
dsl/          # operatörler + compiler + static_validator (sızıntı kontrolü)
data/         # asset-class adaptörü (sp500 / kripto), point-in-time
backtest/     # motor, portföy, maliyet, execution, walk-forward
backtest_service/ # model-agnostik backtest servisi (formül/ML tek arayüz)
model_service/    # model eğitim servisi (walk-forward + embargo)
evaluation/   # hard gate, robustness, istatistik (FDR/Deflated Sharpe), pareto
memory/       # episodic/semantic/procedural + similarity (tekrar kontrolü)
orchestrator/ # döngünün kendisi (basit Python loop)
scripts/      # şeffaflık araçları: anatomi, benchmark, Sharpe doğrulama
```

## Kurulum
```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m tests.test_contracts_smoke   # duman testi
```

## Yol haritası (walking skeleton)
Bütün döngüyü önce en aptal haliyle uçtan uca çalıştır, sonra kutuları
tek tek `dummy → gerçek` yap. Baştan gerçek yapılacak iki şey:
**static validator (sızıntı)** ve **reproducibility (seed + versiyon)**.

- [x] Contract'lar (Pydantic modelleri) + duman testi
- [x] DSL operatör kaydı + compiler + static validator (sızıntı kontrolü) — 8 leakage testi geçiyor
- [x] Sentetik veri + vectorized backtest (tek fold) — bilinen sinyali buluyor, sahte alpha yok, property testleri geçiyor
- [x] Hard gate + SQLite hafıza (her deney kaydediliyor)
- [x] Orchestrator loop (dummy LLM) — **iskelet uçtan uca dönüyor** (`python main.py`)
- [x] Gerçek LLM (OpenRouter/OpenAI-uyumlu, `models.yaml` provider switch) — **otonom döngü çalışıyor**
- [x] Memory-güdümlü öğrenme — semantic memory (aile bazında ders), champion/revision modu; LLM momentum'a kilitleniyor, champion'ı geliştiriyor, leaderboard doluyor
- [x] Similarity/novelty (duplicate kontrolü) — yapısal (AST token, backtest öncesi) + davranışsal (işaretli korelasyon); tekrarları eleyip bütçe koruyor (Doküman 14)
- [x] İstatistiksel yönetişim — Deflated Sharpe Ratio + PSR + bootstrap CI + Benjamini-Hochberg FDR (multiple testing raporu; "kabul" != "istatistiksel geçerli")
- [x] Walk-forward (çoklu fold, tutarlılık) + robustness (permutation testi, maliyet 2x, parametre perturbasyonu)
- [x] Holdout servisi (LLM'den ayrı, kilitli dönem, one-shot, audit log, aday kotası)
- [x] Critic ajanı (bağımsız LLM, farklı prompt+düşük sıcaklık, ekonomik mekanizma denetimi)
- [x] DataAdapter (sentetik <-> gerçek tak-çalıştır); yfinance ile gerçek S&P 500 (survivorship uyarısıyla)
- [x] Bandit bütçe tahsisi (Thompson sampling) — araştırma bütçesini aileler arasında başarıya göre dağıtır
- [x] Research dashboard — funnel, leaderboard, hipotez detayı, multiple-testing, holdout, aile perf., soy ağacı
- [x] Reproducibility metadata (model, prompt/output hash, seed) — her deney yeniden üretilebilir (Doküman 17.3/25.5)
- [x] Lineage (soy ağacı) + inversion modu (başarısızı ters çevir) + universe filtreleri + ek-gecikme robustness
- [x] Pareto çok-amaçlı sıralama (Sharpe alt-sınır + drawdown + turnover) + muhafazakâr skor
- [x] Tam kampanya config'i (izin verilen alan/ufuk/operatör, bütçe, risk kısıtları) — hepsi koda bağlı
- [x] Kampanya kalıcılığı (varsayılan devam / `--fresh` sıfırla)
- [x] Motor-şema hizalaması — beyan edilen `trade_time`/`rebalance`/`holding_period`/
  `portfolio.type`/`weighting`/`gross_exposure` motor tarafından GERÇEKTEN uygulanır;
  uygulanamayan beyan static validator'da reddedilir (şema = çalıştırılan şey)
- [x] Getiriler düzeltilmiş fiyattan (temettü+split; adjusted_close, open'a faktör)
- [x] Optimizer denemeleri multiple-testing sayımında (her backtest = 1 deneme;
  `parameter_search` stage'i ile hafızaya yazılır) + min-fold muhafazakâr skor
- [x] market_cap placeholder'ı kaldırıldı (sahte size faktörünü önler)
- [x] LLM memorization önlemi — `anonymize_universe: true` iken prompta ticker/tarih
  gitmez (parametre-içi look-ahead kontrolü); `false` = ablation deneyi
- [x] Random-search baseline (Deney A) — `models.yaml -> provider: random`; aynı
  pipeline, aynı bütçe, ekonomik gerekçe yok; LLM'in katkısı ölçülebilir
- [x] Point-in-time S&P 500 evreni (survivorship düzeltmesi) — Wikipedia değişiklik
  tarihçesinden her tarihteki GERÇEK üye kümesi (`data/pit_universe.py`); pencerede
  üye olmuş ~700 ticker (bugün endekste olmayanlar dahil) indirilir; motor
  `index_membership` maskesiyle hisseyi yalnızca üye olduğu günlerde işleme sokar.
  Kalan dürüst sınırlar: Yahoo'da verisi hiç olmayan delist ticker'lar (yüklemede
  raporlanır) ve delisting return modeli yok — tam çözüm CRSP ister.
- [x] Model karşılaştırma koşucusu (`python compare.py`) — aynı veri/bütçe/kısıtlarla
  N üreticiyi (LLM'ler + random baseline) yarıştırır; araştırma-verimliliği tablosu
  (kabul, tekrar, derleme hatası, en iyi DSR, FDR, token) + `runs/comparison.md`.
  Yarışmacılar `configs/compare.yaml`'da. Critic varsayılan dummy (adalet),
  literatür kapalı (varyans), holdout'a dokunulmaz.
- [x] Revision karantinası — revizyonları 3+ kez duplicate üretmiş champion
  revision için karantinaya alınır (komşuluk tükendi); sıradaki kabule geçilir.
- [x] Kripto adaptörü (Binance perpetual: OHLCV + funding rate; survivorship-düzeltmeli — ölü coinler dahil)
- [x] Model-agnostik backtest servisi (`backtest_service/`) — formül/istatistiksel/ML
  tek arayüzden; sızıntılı stratejiyi çalıştırmayı REDDEDER; bağımsız referansla
  çapraz-doğrulandı (fark 0.00)
- [x] win rate + birikimli P&L metrikleri (fold'lar bileşik)
- [x] Şeffaflık araçları: tek-hipotez anatomisi (sade/teknik), Sharpe elle doğrulama
  (motor==NumPy==Excel), kampanya `--detay` modu (her adım tek tek)
- [x] Kıyas / maymun testi (`scripts/benchmark.py`): random/al-tut/duygusal trader'a
  karşı + masrafsız kontrol (üstünlük gerçek sinyalden mi?)
- [ ] (opsiyonel) lineage grafiği görselleştirmesi
- [ ] forward-test / paper-trading modülü (holdout'un canlı, bitmeyen versiyonu)

### En kolay başlangıç — kontrol paneli
`python agent.py` (veya **`agent.bat`'a çift tıkla**) → menüden seç. Komut
ezberlemek yok:

| # | Menü | Ne yapar |
|---|------|----------|
| 1 | Kampanyayı izle (devam) | Ajanı koştur; görünüm seç: **canlı panel / detaylı akış / sade özet** |
| 2 | Yeni kampanya (sıfırdan) | Hafızayı sıfırlar, baştan başlar (görünüm seçilir) |
| 3 | LLM karşılaştırması | 5 LLM'i aynı bütçeyle yarıştırır (hangi model daha iyi hipotez) |
| 4 | Holdout değerlendirmesi | Kabul edilenleri kilitli dönemde tek-atış sınar |
| 5 | Dashboard (tarayıcı) | Görsel rapor: leaderboard, çoklu-test, holdout, funnel |
| 6 | **Tek fikri baştan sona anlat** | Bir hipotezi doğuşundan kararına — sade veya teknik |
| 7 | **Kıyas (maymun testi)** | random / al-tut / duygusal trader'ı geçiyor muyuz |
| 8 | Durum / ayarlar | Aktif evren, model, bütçe |

*(Eski streamlit web arayüzü emekli — `arsiv/`'de; işlevini agent.py + dashboard karşılıyor.)*

`main.py` çalışınca `dashboard.html` üretilir (offline, tarayıcıda aç). Ayrıca:
`python -m dashboard`

### Şeffaflık ve doğrulama araçları (`scripts/`)
Sistemin "kara kutu" olmadığını gösteren, hocaya sunum için üç araç. Hepsi
tek başına çalışır, `--log` ile çıktıyı `runs/`'a yazar.

- **`anatomy.py`** — BİR yatırım fikrini doğuşundan kararına adım adım açar.
  - `--sade` : konuyu hiç bilmeyenin anlayacağı düz Türkçe (6 adım, her terim açıklamalı, hipotez "kart" olarak).
  - (bayraksız) : teknik mod — LLM'e giden tam prompt, ham cevap, cümle→graf→**sayı** dönüşümü (her düğümün paneli), model eğitimi (embargo tarihleriyle), PnL açık hesap, metrikler, gate.
  - `--canned` : LLM çağırmadan (bedava/hızlı); prompt yine gerçektir.
- **`benchmark.py`** — "başarı" ölçütü: stratejimizi rastgele al-satçı (N maymun),
  pasif al-tut ve duygusal trader ile AYNI koşullarda yarıştırır. **Masrafsız
  kontrol** ile üstünlüğün gerçek sinyalden mi yoksa düşük işlemden mi geldiğini ayırır.
- **`verify_sharpe.py`** — "Sharpe gerçekten doğru mu?" Motoru saf-NumPy ve Excel
  ile karşılaştırır (PnL zincirini tek gün üzerinde açık hesapla), `runs/sharpe_verification.xlsx` üretir.

```bash
.venv/Scripts/python.exe scripts/anatomy.py --sade --log
.venv/Scripts/python.exe scripts/benchmark.py --log
.venv/Scripts/python.exe scripts/verify_sharpe.py
```

### Kampanyayı detaylı izleme
`python main.py --detay` → her deneyin **7 adımı tek tek** basılır (üretim →
derleme → sızıntı denetimi → sinyalin sayıya dönüşü → backtest fold Sharpe'ları
+ IC → gate → sağlamlık). Görünürlük için; çekirdek mantık değişmez.

### Kampanya kalıcılığı (devam vs sıfırla) ve holdout
- **Varsayılan:** `python main.py` mevcut kampanyaya **DEVAM eder** — novelty, champion,
  dersler, çoklu-test sayımı koşular arası birikir; aynı hipotez tekrar üretilmez.
  Bir kampanya = çok deney (Doküman 4.1). Tekrar tekrar çalıştırıp büyütebilirsin.
- **Yeni kampanya:** `python main.py --fresh` hafızayı sıfırlar.
- **Holdout AYRI komuttur:** `python main.py --holdout` — kampanya koşusu kilitli
  döneme ASLA dokunmaz; kabul edilen adaylar ancak kampanya bitti kararıyla, bir kez
  (one-shot, audit log'lu) sınanır (Doküman 10.3 — insan-döngüsü sızıntısını da kapatır).
- Hafıza: `research_memory.sqlite` (episodic), `holdout_audit.sqlite` (one-shot log).

### LLM sağlayıcısı (esnek)
`configs/models.yaml` → `provider: openrouter|vllm|openai_compatible|dummy`. Hepsi
tek OpenAI-uyumlu istemci; geçiş = base_url + model + api_key ortam değişkeni.
API key `.env`'de (`OPENROUTER_API_KEY`), koda/log'a asla girmez.
