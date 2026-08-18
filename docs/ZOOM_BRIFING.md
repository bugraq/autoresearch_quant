# Zoom brifingi — "nereye geldik?"

> Toplantıdan 5 dakika önce oku, açık tut. Sağ sütun: o an ekranda ne olsun.

---

## 30 saniyede: tek cümle

> **Sistem çalışıyor ve kendini kandırmayı reddediyor.** Üç aday kilitli
> dönemi geçti, sistem "buldum" demedi — üçüncü, bağımsız bir dönemde
> çürüttü. Tek aday hâlâ ayakta ama "alpha bulduk" demiyorum.

Hoca "ne yaptın" derse **önce bunu** söyle. Detay sonra.

---

## Anlatım planı (~20 dk)

### 1) Boru hattı — 5 dk · *ekran: `agent.bat` → [4] teknik mod*

Sizin 5 adımınız birebir kurulu:

| adım | sistemde karşılığı |
|---|---|
| 1. Hipotez üretme | LLM, onaylı bir DSL'de **tipli JSON** üretir (serbest kod yazmaz) |
| 2. Modele çevirme | derleyici → StrategyGraph; statik denetleyici **sızıntıya** bakar |
| 3. Model eğitme | X = feature panelleri, y = **ileriki 10 günün getirisi**, random forest, **walk-forward + embargo** |
| 4. Backtest | tahmin → kesitsel sıralama → %20 long / %20 short → maliyetli PnL |
| 5. Metrikler | Sharpe, win rate, P&L, MaxDD, turnover + **IC/RankIC** |

**Kilit cümle:** *"Model, önümüzdeki 10 günün getirisini tahmin ediyor;
mutlak değeri değil **sıralaması** kullanılıyor, çünkü kesitsel long-short."*

Sızıntı sorusu gelirse: `y` sadece **eğitimde**; tahmin günü *t* için model
*t*'den önce fit edilmiş, araya **10 barlık embargo** konmuş. Ayrıca sinyal
kapanışta üretilip **ertesi açılışta** işleniyor (`shift(2)`).

---

### 2) Sayılar doğru mu? — 2 dk · *ekran: `runs/sharpe_verification.xlsx`*

Sizin özellikle istediğiniz kontrol:

- Motor **==** saf NumPy **==** Excel → fark **4.4e-16**
- PnL zinciri tek bir gün üzerinde açık: sinyal → ağırlık → getiri → brüt → maliyet → net
- Bir tuzak: `pandas.std()` ddof=1, `numpy.std()` ddof=0. Karıştırılırsa Sharpe kayar; motor ddof=1.

---

### 3) Başarı ölçütünüz — 5 dk · *ekran: `agent.bat` → [2] Kıyas*

Sizin tanımınız: *"alpha değil; rastgele/duygusal davranandan iyi olmak."*
Ölçtüm. Herkes **aynı evrende, aynı tarihlerde, aynı 10 bps maliyetle**:

| dönem | al-tut | rastgele | duygusal |
|---|---|---|---|
| araştırma *(kanıt değil)* | ✗ | ✓ | ✓ |
| **HOLDOUT** *(kilitli)* | ✗ | ✓ | ✓ |
| **İLERİ-TEST** *(taze)* | ✓ | ✓ | ✓ |

**Okuma:**
- Rastgele al-satçıyı ve duygusal trader'ı **her dönemde** geçiyoruz — ama bu
  **düşük bir eşik**: ikisi de işlem masrafından batıyor (ortanca maymun −%94).
- Al-tut'u **boğa piyasasında geçemiyoruz, ayı piyasasında geçiyoruz.**
  2025→bugün: al-tut **−%70**, biz **artıda**.
- Düşüşümüz al-tut'un **üçte biri** (%20 vs %76).

**Kilit cümle:** *"Piyasayı yenmiyoruz; piyasadan bağımsız duruyoruz.
Long-short olduğu için piyasa riski taşımıyor — çöküşte ayakta kalıyor."*

---

### 4) ⭐ Asıl bulgu — 5 dk · *ekran: `agent.bat` → [1] Karne*

Bu, anlatacağın **en değerli şey**. Bir alpha iddiası değil, **metodolojik bir sonuç.**

Kabul edilen 3 adayın **3'ü de** kilitli dönemi geçti. Yalnız ona baksaydık
"alpha bulduk" derdik. Üçüncü, bağımsız bir dönemde (2025→bugün, sistemin
**hiç görmediği** taze veri) sınadım:

| hipotez | araştırma | HOLDOUT | İLERİ-TEST | ileri toplam |
|---|---|---|---|---|
| **hyp_0033** | +0.66 | **+0.72** | **+0.31** | **+%6.6** |
| hyp_0002 | +0.61 | +1.29 | −0.05 | −%29 |
| hyp_0021 | **+1.14** | +0.93 | −0.36 | **−%44** |
| hyp_0025 | +0.86 | +0.59 | −1.46 | −%80 |

**Üç sonuç:**

1. **Tek kilitli dönem yeterli kanıt değil.** Holdout tek bir *rejim çekilişidir*;
   onu geçmek "genelliyor" demek değil.
2. **Çoklu-test katmanı haklıydı, holdout yanılttı.** DSR 0.20 ve 0.09 —
   istatistik "anlamlı değil" dedi, holdout "geçti" dedi, ileri-test
   istatistiği doğruladı. **Hiçbir katman tek başına yetmiyor.**
3. **En yüksek in-sample Sharpe, en sert çöktü** (+1.14 → −%44). *"En iyi
   Sharpe'ı seç"* sezgisi bu veride **ters** çalışıyor.

Bunu kalıcı bir kapı yaptım: holdout'u geçen aday **otomatik** ileri-teste
giriyor, hüküm sistemden çıkıyor (`DOĞRULANDI` / `REJİM-BAĞIMLI` / `ÇÖKTÜ` / `EKSİK`).

---

### 4b) ⭐ Aday hâlâ izleniyor — 2 dk · *ekran: `agent.bat` → [8] İleri-test*

**Staj bitti ama ölçüm bitmedi.** Holdout tek-atıştır ve tükenir; ileri-test
dönemi ise her gün büyür. Sicil append-only, yani ölçümler **birikiyor**:

| ölçüm tarihi | Sharpe | biriken getiri |
|---|---|---|
| 29.07.2026 | +0.37 | +%8.0 |
| 30.07.2026 | +0.45 | +%10.5 |
| **18.08.2026** | **+0.31** | **+%6.6** |

**Dürüst okuma — bunu sen söyle:** aday hâlâ pozitif ve hâlâ `DOĞRULANDI`,
**ama zayıflıyor.** Son 19 günün taze verisi net **negatif** geldi; biriken
getiri %10.5'ten %6.6'ya geriledi.

**Kilit cümle:** *"Bu bir zafer turu değil, bir izleme. Aday ölmedi ama
güçlenmiyor da. Zaten ileri-testin amacı bu — tükenmeyen, sürekli sınav."*

Aynı dönemde al-tut **−0.76**; fark hâlâ bizim lehimize (**+1.08**).

---

### 5) LLM'in katkısı — 2 dk · *ekran: `runs/comparison.md`*

*"LLM gerçekten arıyor mu, rastgele denemekle aynı mı?"* — ölçtüm.
Aynı veri, aynı bütçe, tek fark hipotez üreticisi:

| üretici | doğru yapıyı deneme oranı | ilk isabet |
|---|---|---|
| **LLM** | **%86** | **1. hipotez** |
| Bayes optimizasyonu | %50 | 2. |
| rastgele arama | %25 | 4. |
| genetik programlama | %22 | 2. |

**Kilit cümle:** *"LLM doğru fikir ailesini ilk denemede buluyor; LLM'siz
arayıcılar 2-4 deneme sonra ve daha düşük oranda."*

---

## Dürüst sınırlar — sen söyle, hoca sormadan

Bunları önden söylemek güven verir:

- **"Alpha bulduk" demiyorum.** Ayakta kalan aday çoklu-test düzeltmesini
  geçmiyor (DSR 0.20). Doğru ifade: **"tek aday henüz ölmedi."**
- **Tek evren, tek varlık sınıfı** (kripto perpetual). Hisse tarafı kurulu ama
  bu sonuçlar kriptodan.
- **Kâğıt üstünde** — gerçek emir, slipaj modeli, borsa mikroyapısı yok.
- **Kilitli dönem tükeniyor** — her kullanımda aşınıyor. İleri-test tükenmiyor,
  gücü bu.

---

## Ne yaptım — tek paragraf (sorarsa)

Sistem zaten kuruluydu; benim işim **denetimdi**. Her modülü okuyup gerçek
veriyle koşturdum ve **~30 gerçek hata** buldum. Çoğu "kod çöküyor" değil,
**sessizce yanlış sonuç üreten** türden:

- kilitli dönem sınavı **yanlış modeli ölçüyordu** (ML modeli holdout'un içinde
  yeniden eğitiliyordu) — bir aday −0.36'dan +0.72'ye çıktı
- sızıntı denetiminde **sömürülebilir bir açık** (saldırıyı kurup gösterdim)
- LLM **kendi geçme notunu yazabiliyordu** (`or` yüzünden kampanya eşiği
  devre dışı kalıyordu)
- EDGAR temel veride **yanlış dönem** seçimi (3.2 kat yanlış, 3 yıl bayat)
- kıyas aracı **sadece araştırma döneminde** koşuyordu — yani kendi sınavını
  kendi yazıyordu

Hepsi teste bağlı: **48 test** geçiyor.

---

## Ekranda ne açık olsun

| # | ne | nasıl |
|---|---|---|
| 1 | Kontrol paneli | `agent.bat` |
| 2 | Karne (üç dönem) | menü **[1]** |
| 3 | Kıyas matrisi | menü **[2]** |
| 4 | Görsel rapor | menü **[3]** → `dashboard.html` |
| 5 | Tek fikrin anatomisi | menü **[4]** → teknik |
| 6 | Sharpe doğrulama | menü **[9]** → `runs/sharpe_verification.xlsx` |
| 7 | 6 hipotez | `docs/HIPOTEZLER.md` |

Kod: `github.com/bugraq/autoresearch_quant`

---

## Sırada ne var (hoca sorarsa)

1. **İleri-testi düzenli koşmak** — sicil zaman serisi olarak birikiyor; her
   ölçüm adayın hâlâ yaşayıp yaşamadığını gösteriyor. En ucuz ve en dürüst
   ilerleme yolu.
2. **Hisse senedi evreninde tekrar** — PIT S&P 500 + EDGAR temel veri hazır;
   kripto sonucu tek varlık sınıfına bağlı kalmasın.
3. **Ücretli modellerle karşılaştırma** — şu an sadece bedava modeller ölçüldü.
