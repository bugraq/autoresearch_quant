# Yol Haritası — Otonom Quant Araştırmacısı

*Bu dosya PUSULADIR. Ne yapıyoruz, neredeyiz, sırada ne var — hepsi burada.
Yeni bir fikir çıkarsa doğrudan yapmayız; aşağıdaki **PARK** listesine yazarız.
Aynı anda tek iş. Kafamız dağılmaz.*

---

## 🎯 Amaç (değişmez)
LLM'in **hipotez** ürettiği → bir **modelle (formül/istatistik/ML)** tahmine
çevrildiği → **backtest** edildiği → iyiyse geliştirilip (exploit) kötüyse yenisi
denendiği (explore), **kendini kandırmayan** otonom bir quant araştırma sistemi.

Senin cümlenle: **LLM → ML modeli → backtest → iyiyse devam / değilse yeni.**

---

## 🧭 Hocanın çerçevesi (15.07): "İKİ SERVİS"
> *"En önemli servislerden biri **modelleme ve backtest** yapan iki servis. Bir
> modeli alıp (formül/istatistik/ML) backtest yapabileceğimiz servisi geliştir.
> Baştan yazma — hazırı kullan/incele."*

Buna göre önceliğimiz **büyük araştırma döngüsü değil**, hocanın adını koyduğu
**iki servisi sağlam ve DOĞRULANMIŞ** hale getirmek. Autoresearch kampanyası
(büyük döngü) çekirdeği zaten çalışıyor; o Faz 4'e, servisler oturunca.

## 📍 Şu an NEREDEYİZ (2026-07-24 — GÜNCEL)
**FAZ 7 — OTONOM + ŞEFFAF PLATFORM. Hocanın "basit/şeffaf" madde seti TAMAMLANDI; sırada PAKETLEME.**

Çerçeve pivotu (kullanıcı, 2026-07): alpha avı 5 domainde null verdi; piyasa da
"alpha winter"da. Karar: *"alpha bul" takıntısını bırak → sistemi otonom, geleceğe
dönük ve DÜZGÜN yap.* Tez artık: **"şu an kolay alpha yok (titizce gösterdim); ama
alpha ortaya çıktığında onu otonom bulup kendini kandırmadan doğrulayacak bir
araştırma PLATFORMU kurdum."**

**Hocanın "basit/şeffaf" talepleri (2026-07-22) — HEPSİ KAPANDI:**
- ✅ Sharpe elle doğrulama (`scripts/verify_sharpe.py`): motor == NumPy == Excel
- ✅ win rate + birikimli P&L metrikleri (eksikti — hit_rate hiç dolmuyordu)
- ✅ tek-hipotez anatomisi (`scripts/anatomy.py`): teknik + **sade** (konuyu bilmeyen anlar)
- ✅ kıyas/maymun testi (`scripts/benchmark.py`): random/al-tut/duygusal + masrafsız kontrol
- ✅ kampanya `--detay` modu: her deneyin 7 adımı tek tek
- ✅ repo taraması (RD-Agent/AlphaAgent/Qlib...): "Qlib'i katman olarak al" önerisi

**SIRADA (paketleme — hoca sunum paketi):** README (✅ güncellendi) + bu pusula
(✅) + OZET/PROJE_RAPORU güncelle + (öneri) forward-test modülü = holdout'un canlı,
bitmeyen versiyonu.

---


## ★ ÜÇ-DÖNEM TESTİ (29.07.2026) — tek holdout YETMİYOR

Kampanya v4, kilitli dönemde **3/3 geçti**. Sadece ona bakılsaydı "alpha bulduk"
denirdi. Üçüncü, bağımsız bir örneklem-dışı dönem (2025→bugün, sistemin HİÇ
görmediği taze veri) bunu yalanladı:

| hipotez | kampanya | araştırma | HOLDOUT (23-24) | İLERİ-TEST (25→) | ileri toplam |
|---|---|---|---|---|---|
| hyp_0033 | v2 | +0.66 | +0.72 | **+0.31** | **+%6.6** |
| hyp_0021 | v4 | +1.14 | +0.93 | −0.36 | −44% |
| hyp_0025 | v4 | +0.86 | +0.59 | −1.46 | −80% |
| hyp_0002 | v4 | +0.61 | +1.29 | −0.05 | −29% |

*(İleri-test sütunu EN SON ölçümdür ve her koşuda değişir — dönem büyüyor.
hyp_0033'ün ölçüm serisi: 29.07 +0.37 → 30.07 +0.45 → 18.08 **+0.31**.
Yani aday ayakta ama **zayıflıyor**; bkz. dashboard "İleri-Test Sicili".)*

**Üç bulgu:**

1. **Tek kilitli dönem yeterli kanıt değildir.** v4'ün üç adayı da holdout'u
   geçti, üçü de taze veride çöktü. Holdout (2023-24) tek bir REJİM çekilişidir;
   onu geçmek "genelliyor" demek değil, "o rejimde de tuttu" demektir.

2. **Çoklu-test katmanı HAKLIYDI, holdout YANILTTI.** Kabul edilenlerin DSR'ı
   0.20 ve 0.09'du — istatistik "anlamlı değil" diyordu. Holdout "geçti" dedi.
   İleri-test istatistiği doğruladı. Yani katmanlar birbirini denetliyor ve
   *hiçbiri tek başına yeterli değil* (Faz 5 dersinin genişlemiş hâli).

3. **Yüksek in-sample Sharpe = daha sert çöküş.** En yüksek araştırma Sharpe'lı
   aday (hyp_0021, +1.14) en sert düştü (−44%); en düşükler (hyp_0002 +0.61,
   hyp_0033 +0.66) en az. Klasik aşırı-uyum imzası. "En iyi Sharpe'ı seç"
   sezgisi burada tam ters yönde çalışıyor.

**Tek hayatta kalan:** hyp_0033 (v2) üç dönemde de pozitif (+0.66 / +0.72 /
+0.31). ABARTILMAMALI: +%6.6/1.5 yıl zayıftır, DSR'ı 0.20'dir (çoklu-test
süzgecini geçmedi) ve çok sayıda deneme içinden çıkmıştır. Doğru ifade
"alpha bulundu" değil, **"tek aday henüz ölmedi"**dir.

**Metodolojik sonuç (bu projenin katkısı):** kilitli holdout tek başına
yetmiyor; BİRDEN FAZLA bağımsız OOS rejimi gerekiyor. Sistem bunu kendi
verisiyle ampirik olarak gösterdi.

### Arka plan: FAZ 6 — ARAMA KALİTESİ (tamamlanan)
**Özgünlük → kombinasyon → uzun-koşu.**

5 domain denendi (S&P large-cap, kripto v1, fundamentals, small-cap, kripto funding).
Funding en umut vericisiydi. *(29.07.2026 DÜZELTME: "hepsi null" hükmü hatalı bir
holdout değerlendiricisinden geliyordu — funding hipotezlerinden biri düzeltilmiş
sınavdan +0.72 ile geçti. Ayrıntı: Faz 5 başındaki düzeltme notu.)*
Literatür taraması (WorldQuant, AlphaAgent arXiv:2502.16789, alpha-decay makaleleri)
şunu netleştirdi: **sorun "tek güçlü sinyal" hedefinin KENDİSİ.** Kimse öyle çalışmıyor;
WorldQuant ~4M zayıf sinyali birleştiriyor, "başarı" IC~0.02 (kıl payı ama tutarlı).

AlphaAgent (bizim birebir akademik ikizimiz: LLM→formül→backtest) tek en önemli
dersi verdi: **LLM iyi-bilinen faktörlere (momentum) yaslanıp homojen, çabuk çürüyen
(crowding) sinyaller üretiyor; çözüm özgünlüğü ÜRETİM+KABUL hedefine gömmek
(novelty-regularization).** Bizim ÜSTÜNLÜĞÜMÜZ: onlarda kilitli holdout + DSR/FDR YOK;
bizde var → daha katı doğrularız. Portföy testi bunu doğruladı: düşük korelasyon bile
holdout'u kurtarmadı çünkü sinyaller özgün değildi (hepsi aynı döneme overfit).

**Faz 6 adımları (öncelik sırasıyla — özgünlük kombinasyonun ÖN KOŞULU):**
1. ⏳ **Özgünlük zorlaması** (AlphaAgent'ın kanıtlı katkısı): novelty binary "duplicate-at"tan
   → sürekli özgünlük skoruna; LLM üretimine "en benzer faktör" geri bildirimi; skoru
   kabul/Pareto kriterine kat. *(similarity.py originality_score/nearest EKLENDİ+test; sıradaki: prompt + kabul entegrasyonu.)*
2. ⬜ **Hedef/metrik + kombinasyon**: tek Sharpe-0.5 yerine çok zayıf+özgün sinyal → birleştir;
   IC-tabanlı gerçekçi eşik; portföyü gate+DSR+holdout'tan geçir (WorldQuant modeli).
3. ⬜ **Uzun-koşu modu** (kullanıcının vizyonu): gece/günlerce sürekli keşif, biriktir, raporla.
4. ⬜ **Yeni bilgi**: on-chain (borsa akışları) / sentiment — funding'in devamı, taze holdout.

DÜRÜST ÇERÇEVE: bunlar alpha GARANTİSİ değil (AlphaAgent bile IC 0.02); "en mantıklı
yol" + sistemin kendisi tez-değerli (otonom, kendini-kandırmayan araştırma ajanı).

---
### Arşiv — FAZ 5 (kripto funding, tamamlandı)
Faz 5'in tezi: **fiyat/hacim herkesin elinde; yeni bilgi lazım.** Kripto'nun gerçek
üstünlüğü ucuzluğu değil, hisselerde OLMAYAN verisi: **funding rate** (perpetual
futures pozisyonlanması). Mekanizma sağlam: funding pozitifse long'lar kalabalık
ve kaldıraçlı → likidasyon riski → sonraki getiri düşük. Fiyat/hacimde görünmez.

---

## Fazlar

### ✅ FAZ 0 — Çekirdek autoresearch döngüsü (BİTTİ)
LLM hipotez → DSL → sızıntı kontrolü → backtest → hard gate → sağlamlık →
hafıza → öğren → tekrar. Exploration/exploitation (bandit + revizyon/ters-çevirme/
birleştirme). Çoklu-test (FDR/DSR) + kilitli holdout. 34 test paketi.

### ✅ FAZ 1 — Model katmanı + Backtest servisi (BİTTİ)
- `models/` : formül · lineer · ridge · naive bayes · random forest · GBM
- `backtest_service/` : `run(model, veri) → IC/Sharpe/yön isabeti`, sızıntılıyı reddeder
- Directional accuracy (ML accuracy'sinin finans analoğu)
- Kullanıcı dostu Streamlit arayüzü (`streamlit run streamlit_app.py`)

### ✅ FAZ 2 — Backtest servisini DOĞRULA (BİTTİ)
1. ✅ **Bağımsız referans backtest** (`backtest_service/reference.py`) — motorla kod
   paylaşmaz, aynı spec'i sıfırdan uygular.
2. ✅ **Kalıcı test** (`test_engine_crossvalidation.py`).
3. Sonuç: sentetik + GERÇEK S&P 500 → motor = referans, fark **0.00** (birebir).
   Motor doğru. *NOT: Qlib IC cross-check YAPILMADI — Qlib'in IC'si `pandas groupby
   korelasyon`dan ibaret, bizimkiyle matematiksel AYNI; ağır kurulum marjinal.
   Qlib'in asıl değeri model havuzu (LSTM) → Faz 4.*

### ✅ FAZ 3 — İki servisi temiz paketle (BİTTİ)
- ✅ **Modelleme servisi** (`model_service/`): model+özellik+veri → tahmin + IC.
- ✅ **Backtest servisi** (`backtest_service/`).
- ✅ **Demo** (`services_demo.py`) + **belge** (`docs/SERVISLER.md`, hocaya sunum).

### 👉 FAZ 4 — Model-ağırlıklı kampanya (BAŞLADI, sürüyor)
- ✅ Model-ağırlıklı demo koşusu yapıldı (`--fresh`, bütçe 10): **model kullanımı
  %0.7 → %83**. Kayıtta hat DOLU: hyp_0001 linear_regression → backtest → red;
  hyp_0002 random_forest → red; … + explore (hyp_0004 ters çevrildi). Gerçek etkin
  piyasada HEPSİ red (dürüst null; exploit=champion revizyonu accept olmadığı için
  fire etmedi — sinyalli veride fire eder, bkz. services_demo). Model koşusu
  `research_memory_modelrun.sqlite`'ta saklı; ana DB (405 kayıt) korundu.
- ✅ KANONİK model kampanyası (bütçe 24, bedava tencent/hy3 + sabit random_forest,
  gerçek S&P 500): 26 deney, ~%85 RF, $0, 0 rate-limit. 0 kabul (etkin piyasa null);
  en iyi DSR 0.59 (az deneme=az ceza ama <0.95, FDR geçen 0). dashboard.html güncel.
  Eski 405-kayıt arşivlendi (research_memory_arsiv_405.sqlite).
- ✅ SACRED HOLDOUT uçtan uca demo edildi (sinyalli veri, bedava hy3 + RF): 6 kabul
  → kilitli tek-atış holdout. GÜZEL DERS: en iyi araştırma-Sharpe'ı (hyp_0018, 2.08)
  holdout'ta ÇÖKTÜ (0.48=overfit); hyp_0016 GENELLEDİ (1.13→1.13=gerçek sinyal).
  Holdout sahte-parlağı gerçek sinyalden ayırıyor. (research_memory_holdout_demo.sqlite
  + dashboard_holdout_demo.html saklandı; ana DB kanonik'e döndü.)
- ⬜ Kalan (opsiyonel): temiz sunum figürleri; Qlib LSTM havuzu; UX cilası (PARK).

### 👉 FAZ 5 — ALPHA AVI (ŞU AN BURADAYIZ)

**Denenen 4 domain — hepsi fiyat/hacim, hepsi null:**

| # | Domain | Sonuç | Ders |
|---|--------|-------|------|
| 1 | S&P 500 large-cap | 0 kabul, en iyi DSR 0.59 | Etkin piyasada kolay alpha yok |
| 2 | **Kripto** (~28 coin) | DSR **0.98 GEÇTİ**, holdout **eledi** (1.45→−0.16) | **Çoklu-test yeterli değil; rejim overfitting'i yalnız holdout yakalar** |
| 3 | Fundamentals (value/quality, EDGAR PIT) | value hipotezi holdout'ta çöktü (0.66→−0.79) | Value primi de kurtarmadı |
| 4 | Small-cap PIT (S&P 600, 25bps) | 0 kabul; **maliyet ablasyonu: 0bps'te bile ortalama 0.05** | **Sinyal zaten yok — maliyet olmayanı gömüyor, gizli alphayı yemiyor** |
| 5 | **Kripto funding** (yeni bilgi + survivorship kapalı + doğru corpus + rate-limit çözülmüş) | araştırmada 0.92 / 4 kabul, holdout **hepsini eledi** | **DSR haklı çıktı; araştırmadaki güçlü sinyal rejim şansıydı** |

**Sentez:** dördü de aynı ham malzeme. Kaldıraç sırası (güçten zayıfa):
**#1 ham malzeme (yeni bilgi)** › #2 arama kalitesi › #3 deneme sayısı.
*Not: "500 deney koşsak bulur muyuz?" — hayır; DSR deneme sayısını (N) formülüne
katar, daha çok deneme eşiği YÜKSELTİR. Kripto bunu kanıtladı: DSR 0.98'i geçti,
holdout yine eledi.*

> ### ⚠ DÜZELTME (29.07.2026) — aşağıdaki holdout sayıları HATALI bir değerlendiriciden çıktı
>
> Kilitli dönem, tek başına değerlendiriliyordu. İki sonucu vardı:
> 1. Rolling pencereler holdout'un başında NaN kalıyordu (kapsama **%83**).
> 2. Daha kritiği: walk-forward ML modeli **holdout'un İÇİNDE yeniden eğitiliyordu.**
>    Yani sınav, araştırmada kabul edilen modeli değil **başka bir modeli** ölçüyordu.
>
> Düzeltme (`v2-warmup`): araştırma dilimi geçmiş olarak verilir; bilgi akışı tek
> yönlü (geçmiş→gelecek) olduğu için sızıntı değildir. Aynı hata `forward_test.py`'da
> da vardı; iki bağımsız yol artık aynı sayıyı veriyor.
>
> **Düzeltilmiş sonuç (aktif kampanya, `v2-warmup`):**
>
> | hipotez | araştırma | holdout (v1, hatalı) | holdout (v2, doğru) |
> |---|---|---|---|
> | hyp_0010 | +0.97 | −1.06 | **−0.32** (kaldı) |
> | hyp_0033 | +0.66 | −0.36 | **+0.72 → GEÇTİ** |
>
> Yani **"5. null" hükmü artık geçerli değil:** sistemin kendi bulduğu bir hipotez
> (RF + funding kalabalıklığı + momentum + likidite + gün-içi baskı) kilitli dönemden
> sağ çıktı. hyp_0010 ise gerçekten çöküyor (−1.29 fark = aşırı uyum) — sınav çalışıyor.
>
> Aşağıdaki eski koşulara ait sayılar **silinmedi**: o koşular gerçekten öyle
> raporlanmıştı ve kayıt dürüstlüğü için duruyorlar. Ama hepsi `v1` değerlendiriciden
> geldiği için **güvenilir değildir**; ML modu kullanan her holdout sonucu bu
> hatadan etkilenmiştir. `holdout_audit.sqlite`'ta eski kayıtlar `invalidated`
> işaretli (gerekçe + tarih + değerlendirici sürümüyle), silinmedi.
>
> Uyarı payı: düzeltilmiş sonuç **tek bir kilitli dönemdir**. İleri-test (2025→bugün)
> aynı stratejide **rejim-bağımlılık** gösteriyor — "alpha bulduk" demek için erken.

**Faz 5 — TAMAMLANDI. Sonuç (DÜZELTİLDİ): funding hipotezlerinden BİRİ holdout'tan
sağ çıktı (hyp_0033, +0.72). Eskiden "5. null" yazıyordu; o hüküm hatalı bir
değerlendiriciye dayanıyordu — yukarıdaki düzeltme notuna bak.**
1. ✅ `data/binance.py` — funding + OHLCV (Session keep-alive, retry, cache'li)
2. ✅ **Ölü coin havuzu** — LUNA (17.505→0.008), FTT dahil; 22/28 yüklendi
3. ✅ DSL'e `funding_rate` (info_tick=close_t; funding 8 saatte bir → günlük toplam
   ancak kapanışta bilinir; 2 sızıntı testi; ffill YOK = funding bir olaydır)
4. ✅ Kripto literatür corpus'u + rate-limit sabri → kampanya → holdout

| Aşama | Sonuç |
|---|---|
| Kampanya v1 (hisse corpus'u, rate-limit'li) | 0 kabul; 24 deneyin **11'i 429 ile atlandı** |
| **Sondaj** (LLM'siz, elle klasik funding) | **rank(−ort(funding,7)) = 0.92**, ROBUST, işaret kontrolü geçti (tersi −1.06); baseline −1.23 → *"sinyal var, arama zayıf"* |
| Kampanya v2 (kripto corpus + retry) | rate-limit **çözüldü** (24/24); **0 → 4 kabul**; en iyi DSR 0.31 |
| **Holdout (sistem adayları)** | dördü de **ELENDİ**: 0.78→−0.67, 0.62→0.16, 0.54→−0.47, 0.53→0.37 |
| **Holdout (sondaj, manuel)** | **0.92 → −0.50**; funding×reversal 0.86→0.33; zscore30 0.53→−1.26 |

**★ En can alıcı bulgu:** araştırmada funding **+0.92** vs baseline **−1.23** (arada
2.15 = "funding fiyat/hacmin üstüne bilgi katıyor"). Holdout'ta funding **−0.50** vs
baseline **−0.48** → **fark sıfır.** Funding'in kattığı değer kilitli dönemde
tamamen yok oldu; araştırmadaki 0.92 rejim şansıymış.

**★★ Asıl ders — DSR iki kez haklı çıktı:** sondaj sonucu çok caziptı (0.92, robust,
işaret kontrolü geçti, ham p=0.007, CI sıfırı içermiyor) ama DSR 0.31 (kontroller
havuzdan çıkarılınca 0.74) < 0.95 dedi → **holdout harcanmadı.** Holdout sonradan bu
disiplini doğruladı (−0.50). *"DSR aşırı muhafazakâr, bu sefer sinyal gerçek"*
denilseydi yanılınacaktı. Karşıtlık: eski kripto deneyinde DSR **0.98** demişti ama
holdout yine kesmişti. → **İki katman da gerekli; hiçbiri tek başına yetmiyor**
(DSR yanılabilir [kripto], holdout DSR'yi doğrulayabilir [funding]).

**Yan bulgular:** corpus'un etkisi ölçüldü (0→4 kabul) — LLM'e hisse anomalisi
fısıldamak aramayı gerçekten kör bırakıyormuş. Rate-limit retry'ı 11 kayıp slotu
kurtardı. Sistemin bulduğu hipotezler karmaşıktı (RF+funding+likidite+hacim) ve
klasik overfit imzası verdi (0.78 → −0.67).

**Fizibilite ölçüldü (dokümana güvenilmedi, API'ye soruldu):**
- ✅ Funding rate: **2019-09-10'dan**, 8 saatte bir, **654 USDT perpetual**, ücretsiz
- ❌ **Open interest: SADECE 30 gün** → backtest için kullanılamaz, plandan ÇIKARILDI
- ✅✅ **Ölü coin verisi DURUYOR**: LUNAUSDT (→2022-05-13 çöküşü), FTTUSDT, SRMUSDT,
  RAYUSDT hepsi funding+fiyat veriyor. **yfinance'in yapamadığı bu** → eski kripto
  deneyinin en büyük zayıflığı (survivorship) kapatılabilir. Pürüz: `exchangeInfo`
  yalnız yaşayanları listeler (712 TRADING + 123 SETTLING) → ölü sembol havuzu
  ayrıca derlenecek.

**Bulunan gerçek bug (Faz 5 girişinde):** anonim evren tarifi SABİT *"likit BÜYÜK
ölçekli hisse senetleri"* diyordu → **kripto kampanyasında LLM'e HİSSE tarif
edilmişti** (yanlış piyasa sezgisiyle hipotez üretti). Yani 2. domain deneyi adil
değildi. Düzeltildi: `CampaignConfig.anonymous_description` (kampanya kendi anonim
tarifini verir; ticker/tarih hâlâ gitmez) + 2 test.

**Kabul edilen dürüst sınır:** kripto/small-cap survivorship tam kapanmazsa,
bulunan her şey şüpheli kalır — çünkü bu yanlılık research+holdout'u **aynı yönde**
bozar, **holdout onu yakalayamaz.** (Small-cap'te 214/938 delist üyenin fiyatı
Yahoo'da yoktu; kripto'da Binance sayesinde durum daha iyi.)

---

## 🅿️ PARK (rafta — şimdi DEĞİL, sırası gelince)
- **SUNUM/UX cilası** (LinkedIn'deki "Hypothesa Terminal" benzeri — öz değil cila,
  Faz 4 sonrası): canlı terminal arayüzü (TUI, anlık durum), Telegram bildirimi,
  "çalıştırılabilir ajan" paketi. NOT: o örnek çoklu-test düzeltmesi yapmıyor
  gibi (75 fikirden biri OOS geçmiş = muhtemel şans) — bizim rigor avantajımız.
- ~~**Bedava LLM'e geç**~~ ✅ YAPILDI (tencent/hy3:free üretici + gemma-4-26b:free
  critic; maliyet $0). ⚠ Bedel: **rate-limit (429)**. Small-cap kampanyasında
  24 deneyin **11'i** atlandı → örneklem yarıya düştü. Arama kalitesi (kaldıraç #2)
  gerekirse buradan başlanır.
- ~~**Model-ağırlıklı büyük kampanya**~~ ✅ YAPILDI (Faz 4 kanonik: sabit random_forest)
- **Araştırma verimliliği deneyi** (LLM vs random/GP) — büyük bütçeyle; benchmark
  düzeltildi ama sonuç belirsiz kaldı. *(Faz 5 sonrası.)*
- ~~**Fundamentals**~~ ✅ YAPILDI (EDGAR PIT value/quality — domain #3, holdout eledi).
  **Haber/sentiment** verisi hâlâ PARK'ta (PIT açıklanma tarihiyle).
- **On-chain veri** (aktif adres, borsa giriş/çıkış) — funding'den sonra, yeni-bilgi
  kaldıracının devamı. *(Open interest ÇIKARILDI: Binance yalnız 30 gün veriyor.)*
- **Reusable skill kütüphanesi** (NVIDIA "Agent Skills" videosu deseni)
- **AgentQuant motor incelemesi** (market-impact + warmup deseni ödünç)
- **Zaman bütçesi modu** ("8 saat koş" gibi uzun otonom koşu)
- İnsan-onay checkpoint'leri / RSI (recursive self-improvement) — çok ileri

---

## 🧭 Çalışma kuralı
1. Aynı anda **tek faz, tek adım.**
2. Yeni fikir → doğrudan yapma, **PARK**'a yaz.
3. Her adım bitince: kısa özet + "sırada ne var" + bu dosyayı güncelle.
4. Kafan karışırsa: bu dosyaya bak. Neredeysek oradan devam.
