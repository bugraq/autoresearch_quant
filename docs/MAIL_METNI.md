# Hocaya gönderilecek mail

**Konu:** Agentic Quant — proje teslimi (kod + belgeler + sonuçlar)

**Ek:** `agentic_quant_teslim_2026-08-18.zip` (2.9 MB)

---

Hocam merhaba,

Projenin son hâlini ekte gönderiyorum. Kod, belgeler ve **üretilmiş sonuçlar**
paketin içinde — kampanyayı yeniden koşmanıza gerek kalmadan bütün çıktıları
inceleyebilirsiniz.

## Çalıştırma

Kurulum iki komut:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Doğrulama (internet/API gerektirmez, ~1 dakika):

```
.venv\Scripts\python.exe -m tests.run_all
```

**48/48 test geçmeli.** Geçiyorsa kurulum tamamdır.

Sonra **`agent.bat`'a çift tıklamanız yeterli.** Kontrol paneli açılıyor;
açılışta önce mevcut durumu (en güçlü aday + üç dönemlik karnesi + hüküm)
gösteriyor, komut listesi sonra geliyor. Komut ezberlemek gerekmiyor.

**Önerdiğim sıra:**

- **[1] Karne** — her adayın araştırma / kilitli holdout / taze ileri-test notu
- **[2] Kıyas** — rastgele al-satçıyı, al-tut'u ve duygusal trader'ı geçiyor
  muyuz (her dönemde ayrı ölçülüyor)
- **[3] Dashboard** — tarayıcıda görsel rapor
- **[4] Tek fikrin anatomisi** — bir stratejinin doğuşundan kararına kadar her
  adımı (teknik modda: LLM'e giden prompt, cümlenin sayıya dönüşü, model
  eğitimi, PnL'in açık hesabı)
- **[9] Sharpe doğrulama** — motoru saf NumPy ve Excel'le karşılaştırıyor

**Not:** Bunların hiçbiri API anahtarı istemiyor. Anahtar yalnızca **yeni
hipotez üretmek** için gerekli (menü [5]/[6]). `.env` dosyasını pakete koydum;
içindeki anahtar zaten sizin.

**Uyarı:** menü **[8] İleri-test** ilk koşuda Binance'den ~665 sembol indiriyor,
**bir saati aşabilir** — takılmış değil. Sonraki koşular saniyeler sürüyor.
Sonuçları zaten pakette olduğu için koşmanız şart değil.

## Paketin içinde üretilmiş sonuçlar da var

Sadece kod değil, **araştırma hafızasının tamamı** pakette:

- **56 hipotez**, hepsi tam tanımıyla (3 kabul, 40 red, 12 kopya, 1 revizyon) —
  her birinin kararı ve **gerekçesi** kayıtlı (`research_memory.sqlite`)
- **15 kilitli dönem kaydı**, 4 aday sicili, **6 ileri-test ölçümü**
  (`holdout_audit.sqlite`)
- **48 test dosyası** (`tests/`) — paketin içinde koşturup doğruladım
- Üretilmiş `dashboard.html` ve `runs/` altındaki bütün çıktılar
  (kıyas, ileri-test, anatomi logları, Sharpe doğrulama Excel'i)

Yani hiçbir şeyi yeniden koşmanıza gerek yok; her hipotezi ve neden
kabul/red edildiğini doğrudan inceleyebilirsiniz. Menü **[1]** ve **[3]**
bunları okuyup gösteriyor.

## Nerede ne var

- **`README.md`** — sonuçlar, kurulum, çalıştırma, boru hattının beş adımı,
  sayıların nasıl doğrulanacağı, bilinen sınırlar
- **`docs/OZET.md`** — projenin özeti
- **`docs/HIPOTEZLER.md`** — sistemin ürettiği 6 gerçek hipotez, her biri üç
  katmanda: sade anlatım / teknik / LLM'in ürettiği ham makine tanımı
- **`docs/DENETIM_KAYDI.md`** — bulunan ve düzeltilen ~30 hata
- **`docs/YOL_HARITASI.md`** — ne yapıldı, sırada ne var

## Kısaca sonuç

Sistem uçtan uca çalışıyor: LLM hipotez üretiyor (serbest kod değil, onaylı bir
DSL'de tipli yapı) → sızıntı denetimi → random forest (walk-forward + embargo)
→ backtest → metrikler → kabul/red.

**En önemli bulgu metodolojik:** kabul edilen üç adayın üçü de kilitli dönemi
geçti. Yalnız ona baksaydık "alpha bulduk" derdik. Sistemin hiç görmediği taze
veride (2025→bugün) üçü de çöktü (−%44 / −%80 / −%29). Yani **tek bir kilitli
holdout yeterli kanıt değil**; birden fazla bağımsız örneklem-dışı rejim
gerekiyor. Bunu kalıcı bir kapı hâline getirdim: holdout'u geçen aday otomatik
ileri-teste giriyor ve hüküm sistemin kendisinden çıkıyor.

Bir aday üç dönemde de pozitif kaldı (+0.66 / +0.72 / +0.31). Ancak çoklu-test
düzeltmesini geçmiyor (DSR 0.20), o yüzden **"alpha bulduk" demiyorum** —
doğru ifadesi *"tek aday henüz ölmedi"*. Üstelik ileri-testi düzenli
koşturduğum için zayıfladığı da görülüyor: 29.07'de +0.37, 30.07'de +0.45,
18.08'de +0.31. Dashboard'daki "İleri-Test Sicili" bölümünde bu seri duruyor.

Başarı ölçütünüz açısından: rastgele al-satçıyı ve duygusal trader'ı her
dönemde geçiyoruz (ama bu düşük bir eşik — ikisi de işlem masrafından batıyor).
Al-tut'u boğa piyasasında geçemiyoruz, ayı piyasasında geçiyoruz; en derin
düşüşümüz al-tut'un üçte biri. Kısacası piyasayı yenmiyoruz, piyasadan bağımsız
duruyoruz.

Ayrıca LLM'in katkısını da ölçtüm: aynı veri ve bütçeyle LLM'siz arayıcılarla
yarıştırdım. LLM doğru fikir ailesini hipotezlerin %86'sında deniyor ve ilk
hipotezde buluyor; Bayes optimizasyonu %50, rastgele arama %25, genetik
programlama %22.

Sharpe hesabını istediğiniz gibi elle doğruladım: motor = saf NumPy = Excel,
fark 4.4e-16. Menü [9] ayrıca `runs/sharpe_verification.xlsx` üretiyor, kendi
formülünüzle kontrol edebilirsiniz.

Kod ayrıca burada: https://github.com/bugraq/autoresearch_quant

Eksik veya sormak istediğiniz bir şey olursa yazın hocam.

İyi çalışmalar,
Buğra
