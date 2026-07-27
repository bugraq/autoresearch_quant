# Benzer Projeler / Ekosistem Analizi — bizim projeye göre konumlandırma

*Görev (Şener Hoca): "Aşağıdaki kaynakları oku, bizim projeyle alaka seviyelerine
göre kısa yorumlar yaz" + 23 projelik referans listesi. Bu doküman hepsini bizim
sistemin AYIRT EDİCİ EKSENİNDE değerlendirir.*

## Değerlendirme ekseni (bizim ayırt edici 4 özellik)

Onlarca "autoresearch quant" projesi var; hepsi LLM → strateji → backtest → tekrar
yapıyor. Bizi ayıran şey ÜRETİM değil, **kendini kandırmayı önleyen doğrulama**:

1. **Sızıntı önleme yöntemi** — runtime audit mı, compile-time imkansızlık mı?
   Bizde: LLM serbest Python yazmaz; tipli DSL ağacı üretir, `info_tick` eşitsizliği
   sızıntıyı **ifade edilemez** kılar (test etmekle değil, dille).
2. **Kilitli one-shot holdout** — araştırmadan tamamen ayrı, tek-atış, audit-log'lu.
3. **Çoklu-test düzeltmesi** — Deflated Sharpe + BH-FDR (binlerce denemede en iyiyi
   seçmek şans üretir; bunu istatistiksel olarak ayıklamak).
4. **Olgunluk / üstüne inşa edilebilirlik.**

**Ana bulgu:** Ekosistem 2025-2026'da patladı (Karpathy'nin "autoresearch"
kavramı tetikledi). Çoğu proje holdout + çoklu-test'ten YOKSUN → sahte keşif riski.
Ama **2-3 proje artık DSR / permütasyon / sızıntı-doğrulama yapıyor** — yani rekabet
ısınıyor, ayrımımız daralıyor. Hâlâ ayırt edici olan: **tipli DSL (compile-time
sızıntı-imkansızlık) + kilitli one-shot holdout** kombinasyonunu yapan yok.

---

## A. Hocanın "oku ve yorumla" dediği 6 kaynak

| Kaynak | Ne | LLM | Holdout | Çoklu-test | Sızıntı | Bizimle ilişki |
|---|---|---|---|---|---|---|
| **dietmarwo/autoresearch-trading** (21★) | LLM + evrimsel optimizer (BiteOpt), 25-fold walk-forward, numba motor | ✓ | ✗ | ✗ | fold ayrımı (indikatör dış katmanda) | Orta. Parametre aramada güçlü (bizim optimizer'a benzer) ama kilitli holdout ve çoklu-test yok → seçim yanlılığına açık. |
| **TraderAlice/Auto-Quant** (388★) | LLM + FreqTrade, kripto, strateji evrim (create/fork/kill) | ✓ | ✗ | ✗ | zayıf (`@informative` çok-zaman-dilimi, nedensellik zorlanmıyor) | Yüksek benzerlik. "Başarı = döngü çalıştı mı, kârlı strateji buldu mu DEĞİL" felsefesi bizim maymun-testi tutumuna yakın. Ama aynı veride optimize+değerlendirme = holdout yok. |
| **Reddit r/ai_trading** | autoresearch-for-quant tartışması | — | — | — | — | Erişilemedi (Reddit fetch engelli). İçerik muhtemelen yukarıdaki repolara işaret ediyor. |
| **X / varun_mathur** | AI quant gönderisi | — | — | — | — | Erişilemedi (X login/ödeme duvarı). |
| **Faenzi (LinkedIn pulse)** | Karpathy autoresearch → Gemini + Colab, "crucible" (slippage + Black Swan enjeksiyonu) | ✓ | ✗ | ✗ | look-ahead'e karşı sürtünmeli motor | Orta. Ana dersi **"backtests are illusions"** — bizim holdout felsefesiyle birebir. Ama human-in-loop + tek kullanıcı, istatistiksel yönetişim yok. |
| **yllvar / Iqbal Zainal (LinkedIn)** | 4 aşama: ArXiv paper → kod → backtest (SPY/QQQ/BTC) → **1000+ Monte-Carlo/permütasyon** | ✓ | ✗ | ~ permütasyon | RestrictedPython sandbox + look-ahead tespiti | Yüksek. ArXiv literatür (bizim `literature.py`) + permütasyon testi (bizim robustness) var. DSR/FDR ve kilitli holdout yok. |

**A özeti:** Bu 6'sının hiçbiri kilitli one-shot holdout + çoklu-test düzeltmesini
birlikte yapmıyor. En yakını yllvar (permütasyon + literatür). Hepsi "üretim"de
güçlü, "kendini kandırmama"da bizim kadar katı değil.

---

## B. Hocanın 23 projelik listesi — bizim eksende

### Tam bizim işi (LLM autoresearch quant)

| # | Proje | Holdout | Çoklu-test | Sızıntı önleme | Bizimle ilişki / alınacak |
|---|---|---|---|---|---|
| 1 | **RD-Agent** (MS) | ✗ | ✗ | yok | En olgun referans; LLM **serbest Python** yazıyor (bizde tipli DSL). **Qlib'in LSTM havuzu** alınabilir. |
| 2 | **QuantaAlpha** | ✗ | ✗ | yok | Evrimsel faktör döngüsü; bizim originality/novelty ile örtüşür. Doğrulama zayıf. |
| 3 | **AgentQuant** (171★) | ✗ | ✗ | **Peek** (runtime leakage audit) | "Research memory" + rejim hafızası bizimkine yakın. Peek fikri (runtime audit) bizim compile-time DSL'e EK güvence olabilir. |
| 4 | **nhocconan/AutoResearch** (5★) | ✓ train/test | ✗ | **AST validator** (`.shift(-n)` reddi) + 47 sentetik test | **Çok yakın**: funding 8h + maliyet + train2021-24/test2025+ = bizim kripto kampanyanın neredeyse aynısı. Bizde ek: kilitli holdout + DSR/FDR + walk-forward tutarlılık. |
| 5 | **alpha-search** | ~ paper trading | ✗ | walk-forward | Veri→hipotez→sinyal→backtest→hafıza→**paper trading** hattı bize çok yakın; paper-trading aşaması bizim forward-test'e ilham. |
| 6 | **Vibe-Trading** (HKUDS) | ✗ | ✗ | PIT kontrolleri + walk-forward | Ürünleşme (çok-piyasa veri yükleyici, PIT, dışa aktarma) güçlü; doğrulama disiplini bizde daha katı. |
| 7 | **★ zostaff/ai-quant-researcher** (163★) | ~ purge | ✓ **Deflated Sharpe gate** | feature-korelasyon + purge walk-forward | **EN YAKIN İKİZ.** Claude üretiyor + **DSR gate** + adversarial critic + korelasyon-dedup + SQLite trial sayımı — mimarimiz neredeyse birebir. Fark: bizde **tipli DSL + kilitli one-shot holdout**, onda Claude Python yazıyor + ayrı locked holdout yok. |
| 8 | **AlphaPROBE** | ✗ | ✗ | yok | Bayesçi factor retriever + graph çok-ajan üretici + **soy ağacı** — bizim lineage/inversion mekanizmasına referans. |
| 9 | **Alpha Harness** | — | ~ | — | LLM'in alpha keşfinde NEREDE değer kattığını ölçen **ablation** — bizim "LLM vs random baseline" (Deney A) ile aynı bilimsel soru. |
| 10 | **MCTS-LLM Alpha Mining** | ✗ | ✗ | Qlib backtest | LLM + Monte-Carlo Tree Search ile formül üretimi; arama uzayının ağaç temsili ilginç. |

### Akademik baz yöntemler (LLM'siz — kıyas için)

| # | Proje | Yöntem | Bizimle ilişki |
|---|---|---|---|
| 11 | **AlphaGen** (KDD'23) | RL ile formulaic alpha, Qlib | LLM'in karşılaştırılacağı en önemli **baz yöntem** (Deney A'nın RL versiyonu). |
| 12 | **AlphaForge** (AAAI'25) | Faktör keşfi + dinamik kombinasyon | Faktör havuzu / kombinasyon yönetimi referansı (WorldQuant modeli). |
| 13 | **Alpha-GFN** | GFlowNet, faktör DAĞILIMI | Tek en iyi yerine çeşitli faktör dağılımı — bizim özgünlük/çeşitlilik hedefiyle aynı ruh. |
| 14 | **AlphaTransform** | Transformer + RL | LLM'siz otomatik üretim baz yöntemi. |
| 15 | **AlphaEval** | Ortak değerlendirme altyapısı | **Standartlaştırılmış değerlendirme** katmanı örneği — bizim gate/istatistik raporuna kıyas. |

### Altyapı / hafıza / çok-ajan

| # | Proje | Ne | Bizimle ilişki |
|---|---|---|---|
| 16 | **Qlib** (MS) | ML quant platformu (veri/model/backtest) | **En değerli alınacak parça: LSTM model havuzu** → bizim `model_service`'e bağlanabilir. Kendi motorumuzu attırmaz; sadece model katmanı. |
| 17 | **Lumibot** | Backtest + canlı + AI ajan | Sabit-kural / LLM / hibrit stratejiyi aynı döngüde desteklemesi ürün mimarisi referansı. |
| 18 | **AgenticTrading** (Open-Finance-Lab) | Ajan karşılaştırma + paper trading + leaderboard | Bizim compare.py + forward-test'in ürünleşmiş hali; paper-trading referansı. |
| 19 | **FinMem** | Katmanlı LLM hafıza (kısa/orta/uzun) | Bizim episodic/semantic/procedural hafızaya doğrudan referans (ama karar odaklı, keşif değil). |
| 20 | **TradingAgents** (80k★) | Çok-ajan tartışma (analist/boğa/ayı/risk) | Ajan ayrıştırma mimarisi (bizim üretici/critic/auditor) için örnek; ama **araştırma değil karar simülasyonu**. |

### Sızıntı / güvenilir değerlendirme + listeler

| # | Proje | Ne | Bizimle ilişki |
|---|---|---|---|
| 21 | **Look-Ahead-Bench** | LLM'de look-ahead bias ölçüm benchmark'ı | **Doğrudan bizim tezimiz**: memorization + PIT + kilitli dönem. `anonymize_universe` bunun için var. Standart benchmark olarak bizi ölçebilir. |
| 22 | **Awesome-LLM-Quant-Trading-Papers** | Literatür listesi | Takip kaynağı. |
| 23 | **awesome-llm-trading-agents** | Ajan projeleri + **eleştirel akademik notlar** | "Gerçekten trading agent mı, kanıt düzeyi ne" eleştirisi — bizim dürüstlük çizgimizle aynı. |

---

## C. Sonuç — bizim gerçek konumumuz

**1. Yalnız değiliz, ama kalabalıkta ayrımız net.**
Onlarca proje LLM→backtest→tekrar yapıyor. Ezici çoğunluğu (RD-Agent, Auto-Quant,
QuantaAlpha, AgentQuant, dietmarwo, Faenzi…) **kilitli holdout + çoklu-test'ten
yoksun** → araştırmada parlayanı gerçek sanma riski taşıyor. Bizim tüm mimarimiz
tam bunu önlüyor.

**2. Rekabet ısınıyor (dürüst olalım).**
Üç proje artık istatistiksel yönetişime giriyor:
- **zostaff/ai-quant-researcher**: Deflated Sharpe gate + critic + purge — bizim ikizimiz.
- **yllvar**: 1000+ permütasyon testi + literatür.
- **nhocconan**: AST validator + sentetik leakage testleri + funding 8h.

**3. Hâlâ ayırt edici olan:** Şu ÜÇÜNÜ BİRLİKTE yapan başka proje görmedik:
- tipli DSL → sızıntı **compile-time imkansız** (runtime audit değil),
- **kilitli one-shot holdout** (audit-log'lu, LLM'in eline geçmeyen),
- Deflated Sharpe **+ BH-FDR** (sadece DSR değil, ikisi).
Artı bizde artık: **çok-dönem OOS forward-test** (holdout'un canlı hali) ve
**maymun testi** (random/al-tut/duygusal kıyas).

**4. Hocanın "hazırı al, üstüne ekle" sorusuna cevap:**
Hazır bir repoyu **temel almak bizi geriye götürür** — çünkü en değerli parçamızı
(holdout + DSR/FDR + DSL) atıp yeniden yazmak gerekir; hiçbiri bunu bizden iyi
yapmıyor. Doğru hamle **tersi**: bizim çekirdek kalır, onlardan **spesifik parça**
alınır:
- **Qlib → LSTM model havuzu** (bizim en net eksik = derin öğrenme modeli),
- **AgentQuant "Peek" → runtime leakage audit** (compile-time DSL'e ikinci güvence),
- **Look-Ahead-Bench → standart ölçüm** (bizi dışarıdan doğrulamak için).

**5. Karşılaştırma için baz yöntemler:** AlphaGen (RL) ve Alpha Harness (ablation),
"LLM gerçekten random/RL'den iyi mi" sorusunu (Deney A) güçlendirmek için ideal.
