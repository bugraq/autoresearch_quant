# Hazır Autoresearch-Quant Sistemleri — Tarama, Karşılaştırma ve Öneri

*Görev (Şener Hoca, 15.07.2026): "Önce autoresearch quant var mı hazırda, buna bir bak. Varsa neler var, hangilerini kullanabiliriz, bir tablo yap. Pro ve con'ları nelerdir. İlk hedef backtest servisi."*

---

## 0. TL;DR — üç cümlelik hüküm

1. **Evet, hazır autoresearch-quant sistemleri var** ve sayıları hızla artıyor: QuantaAlpha, Microsoft RD-Agent(Q), AgentQuant, dietmarwo/autoresearch-trading, TraderAlice/Auto-Quant.
2. **Hiçbiri "kutudan çıkıp" bizim işi görmüyor**; her birinin ciddi bir eksiği var — ama **parçaları alınabilir** (özellikle backtest motoru + model havuzu).
3. **Ortak ve sistematik boşluk:** hiçbiri **çoklu-test kontrolü (FDR / Deflated Sharpe)** ve **kilitli tek-atış holdout** yapmıyor. Bizim projenin akademik katkısı tam burada duruyor — yani "hazırı kullanalım" ile "bizim katkımız" çelişmiyor: **altyapıyı ödünç al, rigor katmanını biz koyalım.**

---

## 1. Ana tablo — autoresearch-quant sistemleri

| Sistem | Ne yapıyor | Backtest motoru | Veri / Pazar | Doğrulama | Olgunluk | Kullanabilir miyiz? |
|---|---|---|---|---|---|---|
| **RD-Agent(Q)** (Microsoft) | Faktör **+ model** birlikte optimizasyon; 5 birimli kapalı döngü | **Qlib** | Çin A-hisse öncelikli (US destekli) | IC, ICIR, RankIC, ARR, IR, MDD | ⭐ En olgun, kurumsal | **Evet — referans mimari.** Qlib'i motor olarak alabiliriz |
| **QuantaAlpha** | Doğal dil → faktör madenciliği → evrim → doğrulama | **Qlib** | CSI 300 (A-hisse); S&P500 transfer **zayıf** | IC 0.047, RankIC 0.046, yıllık kırılım | Üretime yakın, MIT, Web UI + CLI | **Kısmen** — A-hisseye bağlı; **backtest runner'ı** ayrı kullanılabilir |
| **AgentQuant** | Hisse listesi → LLM hipotez → grid search → turnuva backtest | **Kendi motoru** (look-ahead koruması + market impact) | **yfinance (US!)** + FRED | Bootstrapped Sharpe, warmup enforcement, walk-forward (belirsiz) | Pre-release; 167★, 63 test, CI | **Evet — en yakın "servis"**; pip + CLI + Streamlit |
| **dietmarwo/autoresearch-trading** | LLM yapı üretir, **BiteOpt/CMA-ES parametre arar** | **Kendi Numba çekirdeği** + 99 gösterge | yfinance (hisse/kripto/ETF) | Walk-forward (25+ fold), **stationary bootstrap** | Araştırma demosu, MIT | **Parça olarak** — walk-forward + bootstrap kodu, Numba çekirdeği |
| **TraderAlice/Auto-Quant** | Karpathy autoresearch → FreqTrade stratejileri | **FreqTrade** (olgun) | Kripto | results.tsv + git geçmişi; formal test yok | Deneysel | **FreqTrade'i** motor olarak düşünebiliriz (kripto) |
| **AlphaAgent** (KDD'25) | Idea → Factor (DSL + regülarizasyon) → Eval | Kendi/Qlib | Tushare (A-hisse) | IC, turnover, quantile | Akademik | Fikir olarak (aşırı-uyum regülarizasyonu) |
| **TradingAgents** | Haber/duyarlılık ajanlarıyla **işlem kararı** | — | Çoklu LLM sağlayıcı | Karar günlükleri, yansıtmalı öğrenme | Olgun, popüler | **Hayır** — farklı kategori (alpha araştırması değil, trade kararı) |

---

## 2. Detaylı pro / con

### 2.1. Microsoft **RD-Agent(Q)** + **Qlib** — *en ciddi referans*
Kapalı döngü: Specification → Synthesis → Implementation → Validation → Analysis. **Faktör ve modeli BİRLİKTE** optimize ediyor; hangisini iyileştireceğine **Thompson sampling bandit** karar veriyor (8 boyutlu performans vektörü).

| ✅ PRO | ❌ CON |
|---|---|
| Hocanın "hipotez → **model** → backtest" vizyonunun birebir karşılığı | Çin piyasası öncelikli; US için veri kurulumu gerek |
| **Qlib model havuzu**: lineer → LightGBM/XGBoost → LSTM/GRU/Transformer (YAML tak-çalıştır) | Ağır bağımlılık; öğrenme eğrisi dik |
| IC/ICIR/RankIC standart metrikler | **Çoklu-test yok, kilitli holdout yok** |
| Kurumsal kalite, aktif geliştirme, açık kaynak | RD-Agent'ın kendisi LLM'e kod yazdırıyor → reward-hacking riski |
| **Qlib'i Windows'ta kurup test ettim: çalışıyor** (`pyqlib-0.9.7-cp310-win_amd64`), `StaticDataLoader` ile pandas verimizi doğrudan alıyor | |

**Hüküm:** **Backtest+model servisi için en güçlü aday.** Fizibilitesi kanıtlandı.

---

### 2.2. **AgentQuant** — *"servis"e en yakın olan*
Hisse listesi → rejim analizi (VIX yüzdelik, momentum, SMA) → LLM hipotez → grid search → turnuva backtest → SQLite hafıza.

| ✅ PRO | ❌ CON |
|---|---|
| **yfinance = bizim veri kaynağımız** (US hisseleri) — uyum sorunu yok | **Pre-release** (release yok, 46 commit) |
| Gerçekten **servis**: `pip install -e .`, CLI (`agentquant run`), Streamlit dashboard | Kendi backtest motoru — savaş-test edilmemiş |
| **Look-ahead koruması var** (`WarmupEnforcer`, min 252 bar) | Walk-forward ve çoklu-test **belirsiz/eksik** |
| **Market impact modeli** (√-impact, 5bps), bootstrapped Sharpe (p5) | Holdout açıkça yok |
| 63 test, CI, Python 3.10-3.12 | Rejim tespiti basit yüzdelik eşik |
| Multi-agent (Memory / Regime / Strategy / Critic / Backtest Coordinator) | Lisans belirtilmemiş |

**Hüküm:** **Backtest servisi için ciddi aday** — US/yfinance uyumu bizim için büyük artı. Motorunu inceleyip parça alabiliriz.

---

### 2.3. **dietmarwo/autoresearch-trading** — *bizim tasarımın ikizi*
**Split-brain:** LLM yapıyı yazar (hangi gösterge, al/sat koşulu), **fcmaes/BiteOpt** sayısal parametreleri arar (saniyede 10.000+ değerlendirme). Bu, bizim "LLM yapı / optimizer sayı" ayrımımızın **birebir aynısı** — tasarımımızı bağımsız olarak doğruluyor.

| ✅ PRO | ❌ CON |
|---|---|
| Tasarım felsefesi bizimle aynı (bağımsız doğrulama) | **Servis değil**, CLI araştırma demosu |
| **Numba JIT** backtest çekirdeği (C hızında) + 99 hazır gösterge | **Slippage, komisyon, market impact YOK** |
| **Walk-forward** (365g eğitim / 90g test, 25+ fold) | Execution-order açıkları hâlâ mümkün (kendi dokümanı kabul ediyor) |
| **Stationary bootstrap** (overfit tespiti) | **Holdout yok** (ana döngüde) |
| yfinance — hisse/kripto/ETF hepsi | Tek koşuda tek strateji; portföy yok |
| MIT; walk-forward + bootstrap kodu **kopyalanabilir** | LLM numba tip hataları yapıyor; "tek aileye sıkışma" sorunu (bizdeki mode-collapse ile aynı!) |

**Hüküm:** **Parça alınacak kaynak** — walk-forward/bootstrap kodu ve Numba çekirdeği. Bütünüyle benimsenmez.

---

### 2.4. **QuantaAlpha** — *hazır ama Çin'e bağlı*
| ✅ PRO | ❌ CON |
|---|---|
| MIT, Web UI + CLI + **bağımsız backtest runner** | **A-hisse odaklı**; S&P500 transferi çöküyor (%4.68 → %1.91 ARR) |
| Qlib tabanlı; IC/RankIC yıllık kırılım | HDF5 ön-hesaplama "aşırı zaman alıcı" |
| Trajectory-level evrim (araştırma sürecini evrimleştiriyor) | **İşlem maliyeti / slippage YOK** |
| Yapısal hipotez-kod kısıtları (bizim DSL mantığı) | Çoklu-test / holdout yok |

**Hüküm:** Doğrudan kullanım **zor** (pazar bağımlılığı); mimari fikir + backtest runner örneği olarak değerli.

---

## 3. İlk hedef: **backtest servisi** — motor adayları

Hocanın tarifi: *"bir modeli alıp (formül / istatistiksel / ML), bununla backtest yapabileceğimiz servis."*

| Motor | Model desteği | Gerçekçilik | US/yfinance | PIT / survivorship | Hüküm |
|---|---|---|---|---|---|
| **Qlib** | ⭐ Lineer→LightGBM→LSTM/Transformer havuzu, **IC yerleşik** | Orta (maliyet var) | Kurulum gerek | **PIT DB var** (2022+) | ⭐ **En uygun** — hocanın "formül/istatistik/ML" tarifini birebir karşılıyor |
| **vectorbt** | Yok (sinyal-tabanlı) | Zayıf (partial fill/slippage yok) | ✅ | ✗ | Hızlı parametre taraması için yardımcı |
| **backtrader** | Yok | ⭐ Gerçekçi (event-driven, komisyon, stop) | ✅ | ✗ | 2021'de geliştirme durdu |
| **zipline-reloaded** | Yok | İyi (split/temettü/survivorship) | ✅ | Kısmen | Yavaş, eski Python |
| **NautilusTrader** | Yok | ⭐⭐ Production-parity | ✅ | ✗ | Aşırı ağır; iterasyon yavaş |
| **backtesting.py** | Yok | Zayıf | ✅ | ✗ | Sadece demo |
| **FreqTrade** | Yok | İyi | Kripto odaklı | ✗ | Auto-Quant bunu kullanıyor |
| **Bizim motorumuz** | ✅ (yeni model katmanı: formül/lineer/naive bayes) | Orta-iyi (maliyet, beyan-edilen execution) | ✅ | ⭐ **PIT + delisting şoku + tipli sızıntı kontrolü** | Tek "rigor" sahibi olan |

**Kritik gözlem:** Listedeki hazır motorların **hiçbirinde** point-in-time üyelik + tipli sızıntı kontrolü + çoklu-test yok. Qlib'de PIT var ama tipli sızıntı denetimi yok.

---

## 4. Öneri — ne yapalım?

### Öneri: **Hibrit** (hocanın "baştan yazma, birleştir" tavsiyesinin karşılığı)

| Katman | Karar | Gerekçe |
|---|---|---|
| **Model havuzu** | **Qlib'i benimse** (lineer → LightGBM → LSTM) | Hocanın "formül/istatistiksel/ML modeli" tarifi birebir; yeniden yazmak anlamsız |
| **Backtest motoru** | **Aşama 1:** kendi motorumuz + Qlib ile **çapraz doğrulama**<br>**Aşama 2:** Qlib'e geçiş değerlendirilir | Kendi motorumuz PIT/sızıntı garantilerini taşıyor; onları kaybetmeden Qlib'in sayılarıyla doğrularız |
| **Metrikler** | **IC / ICIR / RankIC** (Qlib standardı) — ✅ zaten eklendi | Ortak dil; literatürle karşılaştırılabilir |
| **Rigor katmanı** | **Bizde kalır** (FDR, Deflated Sharpe, kilitli holdout, PIT) | **Hiçbir hazır sistemde yok = akademik katkımız** |
| **Parça ödünç** | dietmarwo'dan walk-forward/bootstrap deseni; AgentQuant'tan market-impact + warmup deseni | Tekerleği yeniden icat etmeyelim |

### Neden "hepsini al, kendimizi sil" değil?
Çünkü hazır sistemlerin **ortak açığı** tam da bizim tezimiz:
- QuantaAlpha, dietmarwo, Auto-Quant: **işlem maliyeti/slippage bile yok** ya da eksik
- Hepsinde: **çoklu-test yok, kilitli holdout yok**
- dietmarwo kendi dokümanında **execution-order açığı** olduğunu kabul ediyor

Yani hazırı olduğu gibi alsak, **sahte alpha üreten** bir sistem devralmış oluruz. Doğru hamle: **altyapıyı ödünç al, dürüstlük katmanını koru.**

---

## 5. Sıradaki adım (hocanın "adım adım" talebi)

1. ✅ **Bu tablo** (Görev 1 — tamam)
2. ⬜ **Backtest servisi** (Görev 2 — ilk hedef):
   - Qlib'i US/yfinance verimizle uçtan uca koştur (fizibilite **kanıtlandı**, şimdi entegrasyon)
   - Servis arayüzü: `backtest(model, veri, dönem) → metrikler (IC + Sharpe + maliyet sonrası)`
   - Kendi motorumuzun sayılarıyla **çapraz doğrula** (aynı strateji, iki motor, aynı sonuç mu?)
3. ⬜ AgentQuant'ın motorunu incele (US/yfinance uyumu nedeniyle en yakın örnek)
4. ⬜ NVIDIA "Agent Skills" videosu (50 dk) → tekrar-kullanılabilir beceri kütüphanesi deseni

---

## Kaynaklar
- [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) · [RD-Agent-Quant (arXiv)](https://arxiv.org/html/2505.15155v2) · [microsoft/qlib](https://github.com/microsoft/qlib)
- [QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha)
- [OnePunchMonk/AgentQuant](https://github.com/OnePunchMonk/AgentQuant)
- [dietmarwo/autoresearch-trading](https://github.com/dietmarwo/autoresearch-trading)
- [TraderAlice/Auto-Quant](https://github.com/TraderAlice/Auto-Quant)
- [RndmVariableQ/AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent)
- [TauricResearch/TradingAgents](https://github.com/tauricresearch/tradingagents)
