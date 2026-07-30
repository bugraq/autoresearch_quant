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
./.venv/Scripts/python.exe -m tests.run_all                # BÜTÜN testler (48)
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
- [x] Üretici karşılaştırma koşucusu (`python compare.py`) — aynı veri/bütçe/kısıtlarla
  N üreticiyi yarıştırır; araştırma-verimliliği tablosu (yapısal isabet, keşif
  hızı, kabul, tekrar, en iyi DSR, FDR, token) + `runs/comparison.md`.
  Critic varsayılan dummy (adalet), literatür kapalı (varyans), holdout'a
  dokunulmaz. Yarışmacılar `configs/compare.yaml`'da ve **iki ayrı soruyu**
  cevaplayan iki gruba ayrılmış:
  - **(A) bilimsel kontrol** — `random-search` / `genetic-programming` /
    `bayesian-opt`. LLM'siz, aynı pipeline, aynı bütçe. *"LLM gerçekten arıyor
    mu?"* **BEDAVA.** `python compare.py --bedava`
  - **(B) model seçimi** — 5 LLM. *"hangi model daha iyi hipotez üretiyor?"*
    **~$2/koşu** (3 ücretli). Koşu başında açıkça uyarılır; `cost` etiketi
    yazılmamış yarışmacı **ücretli varsayılır** (habersiz kredi harcamamak için).
  - Aynı anda **iki koşu engellenir** (`runs/.compare.lock`): yarışmacı
    hafızaları etiket+seed'den türer ve her yarışmacıda silinip yeniden kurulur,
    üst üste binen iki koşu birbirinin ölçümünü **sessizce karıştırır**
    (çökme olmaz — canlı yaşandı).
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
- [x] Holdout ISINMA düzeltmesi — kilitli dönem artık geçmişle (araştırma dilimi)
  ısıtılarak değerlendirilir. Öncesinde: (a) rolling pencereler holdout'un başında
  NaN kalıyordu (gerçek koşuda kapsama %83), (b) daha kritiği, ML modeli
  holdout'un İÇİNDE yeniden eğitiliyordu — yani sınav, araştırmada kabul edilen
  modeli değil BAŞKA bir modeli ölçüyordu. Etki gerçek koşuda büyük:
  hyp_0033 holdout Sharpe -0.36 (kaldı) -> +0.72 (geçti). Bilgi akışı tek yönlü
  (geçmiş->gelecek) olduğu için sızıntı değildir. `tests/test_holdout_warmup.py`
### Gerçek koşu denetimi (senin çalıştıracağın yollar)
Testler değil, **gerçek giriş noktaları** uçtan uca koşuldu: `agent.py`,
`main.py` (gerçek LLM), `main.py --holdout`, `compare.py`, `scripts/*`,
`dashboard.report`. Bulunan ve düzeltilenler:

- [x] **`compare.py` (menü [3]) tamamen kırıktı** — `compare.yaml` değerlendirme
  ortamını sentetiğe çeviriyor ama `allowed_fields` kampanyadan (kripto,
  `funding_rate`) geliyordu: 5 yarışmacının 4'ü
  `KeyError: 'Veri alanı yok: funding_rate'` ile çöktü, tablo anlamsız çıktı.
  İki katmanlı düzeltme: (a) `run_campaign` artık izinli alanları YÜKLENEN
  verininkilerle kesiyor ve neyin düştüğünü yüksek sesle söylüyor
  (`align_allowed_fields`; hiçbiri kalmazsa hata verip duruyor — sessizce
  anlamsız kampanya koşmaz), (b) veri ezildiğinde kampanya ANLATISI da
  nötrleşiyor — yoksa LLM'e "funding_rate'e öncelik ver" denip alan listesinde
  olmuyor ve her hipotez `disallowed_field` ile eleniyordu (çökme değil, sessiz
  sabotaj: tabloda "kötü model" gibi görünüyordu).
  (`tests/test_field_alignment.py`)
- [x] **`scripts/*` maliyeti kampanyadan okuyor** — dördü de `COST_BPS = 5.0`
  sabitliyordu; aktif kampanya 10.0 kullanıyor. Yani kıyas (maymun testi) ve
  ileri-test YARIM maliyetle koşuyor, aynı hipotez kampanyada başka /
  "her şeyi açıklayan" script'te başka (daha iyimser) Sharpe gösteriyordu.
- [x] **`forward_test.py`'da ısınma hatasının ikinci kopyası** — script kendi
  değerlendirme yolunu kullandığı için `holdout/service.py` düzeltmesinden
  yararlanmıyordu; her dönemi izole değerlendirip modeli dönemin İÇİNDE
  yeniden eğitiyordu. Artık her dönem kendinden öncekiyle ısıtılıyor ve iki
  bağımsız yol AYNI sayıyı veriyor (hyp_0010 holdout: **−0.32** her ikisinde).
- [x] `anatomy.py` ön-kayıt eşiği hep `-` basıyordu (`minimum_sharpe` diye bir
  alan yok; doğrusu `minimum_oos_sharpe`) — şeffaflık scriptinde eşik görünmüyordu.
- [x] Sağlamlık reddi artık elma-elma: gate Sharpe'ı fold ORTALAMASI, sağlamlık
  testleri TÜM-SERİ. "0.93 -> -0.17" yazmak düşüşü olduğundan büyük gösteriyordu
  (gerçek maliyet etkisi 0.77 -> -0.17); ikisi de yazılıyor.
- [x] `forward_test.py` süre uyarısı gerçekçileştirildi ("birkaç dakika" -> ilk
  koşuda 1 saati aşabilir): yanlış tahmin, çalışan süreci "takıldı" sanıp
  öldürmeye ve indirmenin baştan başlamasına yol açıyordu.

- [x] **Ödül-hackleme açığı kapatıldı (hard gate)** — `hard_gate.py` docstring'i
  "kabul kapısını LLM'e vermeyiz" diyordu ama walk-forward tutarlılık kontrolü
  `hyp.falsification.minimum_positive_walk_forward_folds **or** min_positive_folds`
  yazıyordu: LLM'in beyanı kampanya eşiğinin YERİNE geçiyordu. `0.1` yazan bir
  hipotez, 5 dönemin 4'ünde para kaybedip birinde patlarken KABUL alabiliyordu.
  Artık kampanya eşiği TABANDIR (`max(kampanya, kendi)`) — hipotez kendini
  yalnızca daha SIKI bağlayabilir; gevşetme girişimi yok sayılır ama
  `weaker_own_threshold` notuyla kabul kaydına yazılır (iz sürülebilsin).
  Gerçek kampanyada bu açık KULLANILMAMIŞ (0 hipotez gevşek eşik beyan etmiş),
  yani mevcut sonuçlar etkilenmiyor — düzeltme önleyicidir.
  (`tests/test_hard_gate.py`, 5 test)
- [x] **Kilitli dönem: gerekçeli geçersiz kılma** (`--holdout-invalidate "gerekçe"`).
  Sorun: ısınma hatası ZATEN KAYDEDİLMİŞ holdout sonuçlarını da yanlış yapmıştı;
  one-shot kilidi (doğru olarak) düzeltmeye izin vermiyordu ve tek çıkış yolu
  kaydı SİLMEKTİ — bilimsel kaydı silmek en kötü seçenek. Çözüm append-only:
  kayıt `invalidated` işaretlenir, **gerekçe + tarih + değerlendirici sürümü**
  kalıcı saklanır, dashboard eski sonucu gerekçesiyle ayrı tabloda gösterir.
  Gerekçe ZORUNLU (gerekçesiz sıfırlama sınavı ortadan kaldırır); geçersiz kayıt
  kotayı doldurmaz; one-shot yalnız AKTİF kayıtlar için işler. Eski audit
  dosyaları (`hypothesis_id UNIQUE`) kayıpsız taşınır.
  **Meşru kullanım:** değerlendirici hatalıysa. **Meşru DEĞİL:** sonucu görüp
  stratejiyi değiştirip yeniden denemek — bu kilitli dönemi araştırma verisine
  çevirir. Denetlenebilirlik caydırıcılıktır. (`tests/test_holdout_warmup.py`)
- [x] **Model modunda sızıntı açığı kapatıldı** — static validator yalnızca
  *sinyal* düğümünün bilgi-anına bakıyordu. Model modunda sinyal, modelin TÜM
  feature'lardan ürettiği tahmindir: sinyal ifadesinin atıf yapmadığı bir
  feature `close_t` bilgisi taşıyıp modele X olarak girebiliyor ve `close_t`'de
  işlem yapılabiliyordu = aynı-bar sızıntısı, üstelik "TEMİZ" damgasıyla.
  Artık model modunda bütün feature'lar denetleniyor (`tests/test_leakage.py`,
  3 yeni test; dsl_formula'da yanlış pozitif üretmiyor).
- [x] Holdout audit'inde **kampanya adayı / elle sonda ayrımı** — kilitli döneme
  kampanya dışı, elle yazılmış sondalar da girmişti ve biri "geçti"; dashboard
  bunu "kilitli dönemi 1/6 geçti" diye gösteriyordu. Artık ayrım kanıta dayalı
  (hafızada kaydı olmayan kimlik = kampanya ürünü değil), başlık yalnız kampanya
  adaylarını sayıyor, sondalar tabloda `elle sonda` etiketiyle ve açıklamasıyla
  görünüyor (gizlenmiyor).
- [x] ML hedefi düzeltilmiş fiyattan (`adjusted_close`) — PnL zaten öyleydi;
  hedefi ham `close`'tan almak modele "temettüsüz getiriyi tahmin et" deyip
  sonucu temettülü getiriyle ölçmekti (hissede sistematik, kriptoda etkisiz).
- [x] Bağımsız çapraz-doğrulama referansı da yıllıklaştırmayı veriden alıyor
  (`backtest_service/reference.py`; sabit 252 kriptoda denetçinin cetvelini bozuyordu).
- [x] Dashboard, metriği olmayan kabul kaydında çökmüyor (tek eksik sayı tüm
  raporu üretilemez yapıyordu).
- [x] **Sade anlatım katmanı** (`evaluation/plain.py`) — ML/ajan bilmeyen, sadece
  alım-satım bilen biri çıktıyı okuyabilsin. Kampanya sonunda **TRADER ÖZETİ**
  (fikirler nerede elendi, geçenlerin karnesi, somut para karşılığı, terim sözlüğü),
  dashboard'ın en üstünde aynı özet, holdout ve çoklu-test tablolarında "bu ne
  diyor" çevirisi. **Dürüstlük kuralı koda gömülü:** sade dil ≠ iyimser dil —
  para kaybı asla "başarı" diye sunulmaz, hüküm önce paraya bakar, çoklu-test
  onaylamadıysa hüküm otomatik aşağı çekilir (`tests/test_plain.py`).
- [x] Kıyas hükmü dürüstleştirildi — "3 kıyastan 2'sini geçtik, eşik sağlandı"
  ifadesi kaldırıldı: ortanca maymun %-96 batarken %-20 batmak başarı değildir.
  Artık önce para, sonra kıyas sayacı (bilgi amaçlı) basılır; yer tutucu hipotez
  ölçülüyorsa bu ayrıca ve büyük harfle uyarılır.
- [x] Yıllıklaştırma tutarlılığı — çoklu-test tablosu ve dashboard artık ölçeği
  VERİDEN alır. Öncesinde 252 varsayıyordu: 8h kripto kampanyasında aynı
  stratejinin Sharpe'ı leaderboard'da 0.97, çoklu-test tablosunda 0.49 görünüyordu.
  (DSR/FDR etkilenmiyordu — per-period; yalnız raporlanan Sharpe ve CI.)
- [ ] (opsiyonel) lineage grafiği görselleştirmesi
- [ ] forward-test / paper-trading modülü (holdout'un canlı, bitmeyen versiyonu)

### En kolay başlangıç — kontrol paneli
`python agent.py` (veya **`agent.bat`'a çift tıkla**) → menüden seç. Komut
ezberlemek yok:

Panel açılışta önce **durum** gösterir (en güçlü aday + üç-dönem karnesi +
hüküm), komut listesi sonra gelir. Durum yalnız SQLite okur — saniyeler sürer,
veri indirmez. Menü, "hangi soruyu cevaplıyor" mantığına göre gruplu:

**Sonuçlara bak** *(hızlı — veri indirmez)*

| # | Menü | Ne yapar |
|---|------|----------|
| 1 | **Karne: üç dönemde ne oldu?** | Her adayın araştırma / kilitli holdout / taze ileri-test notu + hüküm |
| 2 | **Kıyas: rastgeleyi ve al-tut'u geçiyor muyuz?** | Hocanın başarı ölçütü. **Her dönemde ayrı** yarış + geçme matrisi |
| 3 | Dashboard (tarayıcı) | Görsel rapor: kıyas, üç-dönem hükmü, çoklu-test, holdout, huni |
| 4 | Tek fikri baştan sona anlat | **Bulunan aday** (sicilden, üç-dönem karneli) veya yeni fikir; sade/teknik |

**Yeni ölçüm yap** *(yavaş — veri indirir / LLM çağırır)*

| # | Menü | Ne yapar |
|---|------|----------|
| 5 | Kampanyayı sürdür | Ajanı koştur; görünüm seç: **canlı panel / detaylı akış / sade özet** |
| 6 | Yeni kampanya (hafızayı SIFIRLA) | Baştan başlar (**aday sicili ve holdout kaydı korunur**) |
| 7 | Holdout sınavı | Kilitli dönemde tek-atış; geçenler **otomatik ileri-teste** girer |
| 8 | İleri-test | Sistemin gördüğü tarihten SONRAKİ taze veri (rejim-bağımlılığı yakalar) |

**Denetle** *(sayılar doğru mu?)*

| # | Menü | Ne yapar |
|---|------|----------|
| 9 | **Sharpe gerçekten doğru mu?** | Motoru saf-NumPy ve Excel'le karşılaştırır; PnL zincirini gün gün açar |
| t | **Bütün testleri koş** | 48 test: sızıntı, ödül-hackleme, ısınma, hizalama, PIT veri, aday seçimi |
| k | LLM karşılaştırması | 5 modeli aynı veri/bütçeyle yarıştırır |
| d | Durum / ayarlar | Aktif evren, model, bütçe, **işlem maliyeti**, tarih aralığı |
| 0 | Çıkış | |

*(Not: `9` ve `t` daha önce menüde HİÇ yoktu — yalnız bu README'de yazıyordu.
Hocanın açıkça istediği "Sharpe'ı elle doğrula" aracı panelden erişilemiyordu.)*

*(Eski streamlit web arayüzü emekli — `arsiv/`'de; işlevini agent.py + dashboard karşılıyor.)*

`main.py` çalışınca `dashboard.html` üretilir (offline, tarayıcıda aç). Ayrıca:
`python -m dashboard`

### Şeffaflık ve doğrulama araçları (`scripts/`)
Sistemin "kara kutu" olmadığını gösteren, hocaya sunum için üç araç. Hepsi
tek başına çalışır, `--log` ile çıktıyı `runs/`'a yazar.

- **`anatomy.py`** — BİR yatırım fikrini doğuşundan kararına adım adım açar.
  - `--aday` : YENİ fikir üretmez; **sicildeki mevcut adayı** baştan sona anlatır
    (varsayılan: üç dönemden geçmiş olan). Üstte üç-dönem karnesi + hüküm.
    `--aday hyp_0033` ile belirli bir aday seçilir. Hocaya "işte bulunan tek
    aday" demek için olan mod budur.
  - `--sade` : konuyu hiç bilmeyenin anlayacağı düz Türkçe (6 adım, her terim açıklamalı, hipotez "kart" olarak).
  - (bayraksız) : teknik mod — LLM'e giden tam prompt, ham cevap, cümle→graf→**sayı** dönüşümü (her düğümün paneli), model eğitimi (embargo tarihleriyle), PnL açık hesap, metrikler, gate.
  - `--canned` : LLM çağırmadan (bedava/hızlı); prompt yine gerçektir.
- **`benchmark.py`** — "başarı" ölçütü: stratejimizi rastgele al-satçı (N maymun),
  pasif al-tut ve duygusal trader ile AYNI koşullarda yarıştırır. **Masrafsız
  kontrol** ile üstünlüğün gerçek sinyalden mi yoksa düşük işlemden mi geldiğini ayırır.
  - Yarış **her dönemde ayrı** koşar (araştırma / holdout / `--ileri` ile taze).
    Araştırma dönemi kanıt değildir: aday zaten orada seçilmiştir, orada
    kazanması beklenir. Anlamlı olan `*OOS` satırlarıdır.
  - Sonuç `runs/benchmark.json`'a yazılır; dashboard onu okuyup **Kıyas**
    bölümünü basar (ölçüm tarihi damgasıyla — bayat sayı güncel sanılmasın).

  **Ölçülen sonuç (29.07.2026, aday `hyp_0033`, 10 bps):**

  | dönem | al-tut | rastgele | duygusal |
  |---|---|---|---|
  | araştırma *(kanıt değil)* | ✗ | ✓ | ✓ |
  | HOLDOUT *OOS | ✗ | ✓ | ✓ |
  | İLERİ-TEST *OOS | ✓ | ✓ | ✓ |

  Okuma: rastgele ve duygusal trader'ı **her dönemde** geçiyoruz (düşük eşik —
  ikisi de işlem masrafından batar). Al-tut'u **boğa piyasasında geçemiyoruz**,
  **ayı piyasasında geçiyoruz** (2025→bugün: al-tut −%70, biz +%8). Bu, piyasa
  riski taşımayan long-short bir stratejiden beklenen davranıştır — düşüşümüz de
  al-tut'un üçte biri (%20 vs %76).
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
