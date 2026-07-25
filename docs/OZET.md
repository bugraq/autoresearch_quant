# LLM Tabanlı Otonom Quant Araştırmacısı — Proje Özeti

## Bir cümlede
LLM'in finansal araştırma sürecini **kapalı bir döngüde** otomatikleştiren bir
sistem: hipotez üret → stratejiye derle → sızıntısız backtest et → istatistiksel
değerlendir → kabul/red/geliştir → hafızaya yaz → öğren → yeni hipotez. Çıktısı
yalnızca bir strateji değil, **tekrar üretilebilir bir araştırma kaydı**.

## Temel tasarım kararı (sistemin kalbi)
Üç katman kesin biçimde ayrıldı:

| Katman | Sorumluluk |
|---|---|
| **LLM** | Hipotez, ekonomik mekanizma, yapısal strateji değişikliği, yorum |
| **Deterministik sistem** | Veri erişimi, derleme, backtest, metrik, istatistik, holdout |
| **Sayısal optimizasyon** | Bandit ile bütçe tahsisi, parametreler |

LLM **asla** backtest'e veya veriye dokunmaz; serbest Python yazmaz. Sadece
önceden onaylanmış bir **DSL** (domain-specific language) ile yapılandırılmış
strateji tanımı üretir. Bu ayrım; veri sızıntısını, reward hacking'i ve
tekrar-üretilemezliği baştan engeller.

## Araştırma döngüsü (pipeline)
Her hipotez şu istasyonlardan geçer; herhangi birinde elenebilir:

```
LLM üretir → DSL'e derlenir → SIZINTI kontrolü → bağımsız Critic (ekonomik) →
tekrar (novelty) kontrolü → walk-forward backtest → hard gate (+fold tutarlılık) →
sağlamlık testleri → hafıza → öğrenme → bandit bütçe → çoklu-test → HOLDOUT
```

Öne çıkan güvence mekanizmaları:

- **Sızıntısızlık** — Her ifadeye "bu bilgi en erken ne zaman bilinebilir"
  (info_tick) etiketi atanır; `sinyal < işlem zamanı` eşitsizliği zorlanır.
  Sızıntı "test edilerek" değil, DSL'de **ifade edilemez kılınarak** önlenir.
- **Bağımsız Critic** — Üreten LLM kendi stratejisini onaylamaz; ayrı bir LLM
  ekonomik mekanizmayı denetler ("gizli bir faktörün yeniden adı mı?").
- **Multiple testing** — Binlerce backtest sonrası tek yüksek Sharpe anlamsızdır.
  Deflated Sharpe Ratio + FDR + bootstrap ile "kabul" ile "istatistiksel geçerli"
  ayrılır. Her deney (başarısız dahil) sayılır.
- **Kilitli Holdout** — Araştırmadan tamamen ayrı, one-shot, audit log'lu son sınav.
  Araştırma ajanı bu veriye asla erişemez.
- **Öğrenme + bandit** — Sistem geçmiş deneylerden ders çıkarır (hangi aile
  çalışıyor) ve araştırma bütçesini başarılı ailelere Thompson sampling ile dağıtır.

## Hocanın 5 adımı → kodda nerede
Hocanın verdiği çerçeve (hipotez → model → eğitim → backtest → metrik) sistemde
birebir karşılık bulur:

| # | Hoca çerçevesi | Kodda |
|---|---|---|
| 1 | Hipotez üretme | LLM → `HypothesisSpec` (`agents/hypothesis_generator.py`) |
| 2 | Hipotezi modele çevirme | `compile_hypothesis()` → `StrategyGraph` + **statik sızıntı denetimi** (`dsl/`) |
| 3 | Modeli eğitme | purged walk-forward + embargo (`backtest/model_signal.py`) |
| 4 | Backtest | `compute_pnl()` + 5 fold (`backtest/engine.py`) |
| 5 | Metrikler | Sharpe, **win rate, P&L**, MaxDD, IC/RankIC (`backtest_service/`) |
| — | Go to 1 | orchestrator döngüsü + hafıza (`orchestrator/loop.py`) |

**3. adımın cevabı** (hocanın sorusu — "model neyi predict eder, entry/TP/SL mi?"):
Model **ileriki getiriyi** tahmin eder (kesitsel: "hangi varlıklar diğerlerine göre
daha iyi olacak"), entry/take-profit/stop-loss DEĞİL. Çünkü TP/SL bar-içi belirsizlik
nedeniyle vektörize backtest'te dürüst simüle edilemez (sahte alpha kaynağı).

**Bizde fazladan** (hocanın listesinde yok, kritik): 2. adımdaki sızıntı kapısı ve
5'ten sonra gelen **çoklu-test düzeltmesi (DSR/FDR) + kilitli holdout** — "go to 1"
döngüsü 500. denemede şansa çıkanı gerçek sanmasın diye.

## Şeffaflık araçları (kara kutu değil)
Sistemin her adımı gözle izlenebilir (`scripts/`, detay: README):
- **`anatomy.py`** — bir hipotezi doğuşundan kararına açar. `--sade` ile konuyu
  bilmeyenin anlayacağı düz Türkçe; teknik modda LLM prompt'undan sayısallaşmaya
  kadar her adım.
- **`benchmark.py`** — maymun testi (aşağıda).
- **`verify_sharpe.py`** — Sharpe motorunu NumPy ve Excel ile çapraz-doğrular.
- `main.py --detay` — kampanyada her deneyin 7 adımı tek tek.

## En güçlü kanıt: dürüstlük
- **Sentetik veride** (bilinen bir sinyal gömülü) sistem sinyali **buluyor**.
- **Gerçek veride** (S&P 500, kripto) basit stratejilerin hepsi holdout'ta eleniyor —
  sistem dürüstçe "kolay alpha yok" diyor, **sahte bir kazanan uydurmuyor**.

Kötü tasarlanmış bir sistem gerçek veride de "alpha buldum" diye kendini
kandırır. Bu sistemin gerçek veride negatif sonuç üretmesi, tasarımının
sağlam olduğunun kanıtıdır.

**"Başarı" ölçütü (maymun testi):** Alpha bulmak yerine, sistemin rastgele al-satçıyı
(200 maymun), pasif al-tut'u ve duygusal trader'ı geçip geçmediği ölçülür
(`scripts/benchmark.py`). Kritik: **masraf sıfırlanınca da** rastgele al-satçının
üstünde kalıyoruz → üstünlük "az işlem yaptık"tan değil, gerçek sinyalden. Rastgele
al-satçının masrafsızda ~0 Sharpe vermesi, aynı zamanda motorun sahte alpha
üretmediğinin kanıtı.

## Modülerlik (pipeline önce, model tak-çalıştır)
Her şey config'ten sürülür, kod değişmez:
- **LLM sağlayıcısı** — `configs/models.yaml` (OpenRouter bugün, vLLM yarın; aynı kod;
  `provider: random` = LLM'siz random-search baseline, Deney A)
- **Veri kaynağı** — `configs/data.yaml` (sentetik ↔ yfinance ↔ **point-in-time S&P 500**)
- **Kampanya sınırları** — `configs/campaign.yaml` (bütçe, eşikler, operatörler)

## Survivorship düzeltmesi (point-in-time evren)
Bugünün endeks listesini geçmişe uygulamak, "kazananlarla backtest" demektir.
Sistem artık Wikipedia'nın S&P 500 değişiklik tarihçesinden **her tarihteki
gerçek üye kümesini** kurar (2015-2023 penceresinde ~700 farklı ticker; bugün
endekste olmayan ~150 isim dahil) ve bir hisse yalnızca **o tarihte üyeyken**
işlem görebilir. Kalan dürüst sınırlar belgelidir: Yahoo'da verisi hiç olmayan
delist ticker'lar yüklemede raporlanır; delisting return modellenmez (CRSP yok).

## Teknik durum
- ~35 modül, **40+ test paketi** (sızıntı, backtest, istatistik, holdout, critic,
  bandit, dashboard, benchmark, annualization, funding-alignment...), hepsi geçiyor.
- Üretici LLM: bedava `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter;
  ⚠ bedava slug'lar zamanla ücretliye dönebiliyor — koşmadan önce kontrol).
- Kampanya maliyeti ~birkaç sent (bedava modelle ~0). GitHub'da versiyonlu.

---

## Nasıl çalıştırılır (kendin dene)

**En kolay yol — kontrol paneli:** `agent.bat`'a çift tıkla (veya `python agent.py`).
Menüden seç: kampanya izle (canlı/detaylı/sade), LLM karşılaştırması, holdout,
dashboard, **tek fikri baştan sona anlat**, **kıyas (maymun testi)**, durum. Komut
ezberlemek yok. Aşağısı terminal komutlarını tek tek ister.

Terminalde proje klasöründe (`agentic_quant`):

**1. Kurulum (bir kez):**
```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. API anahtarı** — `.env` dosyası (zaten var):
```
OPENROUTER_API_KEY=sk-or-...
```

**3. Kampanyayı çalıştır:**
```
.\.venv\Scripts\python.exe main.py
```
İlk çalıştırma point-in-time S&P 500 verisini indirir (~700 ticker, birkaç
dakika); sonraki koşular `data/` altındaki cache'ten saniyeler içinde açılır.
Terminalde her hipotezin kararını, leaderboard'u, multiple-testing raporunu ve
holdout sonuçlarını görürsün.

**3b. Holdout (kampanya BİTİNCE, bilinçli karar):**
```
.\.venv\Scripts\python.exe main.py --holdout
```
Normal koşu kilitli döneme dokunmaz; kabul edilen adaylar bu komutla BİR KEZ
sınanır (one-shot, audit log). Araştırma-değerlendirme ayrımının gereği.

**3c. Model karşılaştırması (hocanın istediği deney):**
```
.\.venv\Scripts\python.exe compare.py
```
`configs/compare.yaml`'daki yarışmacıları (ücretli/ücretsiz LLM'ler + random
baseline) aynı veri ve bütçeyle koşturur; araştırma-verimliliği tablosunu
terminale ve `runs/comparison.md`'ye yazar.

**4. Dashboard'u aç:**
Çalışma bitince proje klasöründe **`dashboard.html`** oluşur.
Üstüne çift tıkla → tarayıcıda açılır. (Funnel, leaderboard, istatistik, holdout,
aile performansı görsel olarak.) Ayrıca tek başına:
```
.\.venv\Scripts\python.exe -m dashboard.report
```

**5. Şeffaflık araçları (hocaya göster):**
```
.\.venv\Scripts\python.exe scripts\anatomy.py --sade --log     # fikri sade anlat
.\.venv\Scripts\python.exe scripts\benchmark.py --log          # maymun testi
.\.venv\Scripts\python.exe scripts\verify_sharpe.py            # Sharpe doğrula
.\.venv\Scripts\python.exe main.py --detay                     # her adım tek tek
```

**6. Testleri çalıştır:**
```
.\.venv\Scripts\python.exe -m tests.test_leakage
.\.venv\Scripts\python.exe -m tests.test_benchmark
# (tests/ altındaki her test_*.py aynı şekilde)
```

## Ayarlarla oynamak
- **Gerçek veriye geç:** `configs/data.yaml` içinde `source: yfinance` yap.
- **Modeli değiştir:** `configs/models.yaml` içinde `model:` satırını değiştir
  (ücretsiz denemek için `:free` biten modeller).
- **Deney sayısı / eşikler:** `configs/campaign.yaml`.
