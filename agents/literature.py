"""
Literatür ajanı (Doküman 4.3 — 'literatürdeki mekanizmayı yeni alana uygula').

Hipotez üreticiye, akademik olarak BELGELENMİŞ, YALNIZCA fiyat/hacim ile
hesaplanabilen kesitsel anomalileri BAĞLAM (tohum) olarak verir. Böylece
fikirler rastgele değil, gerçek literatüre dayalı ve çeşitli olur.

TASARIM KARARI — neden STATİK corpus, neden canlı web araması DEĞİL:
  Proje point-in-time doğruluğu ve LLM'in gömülü GELECEK bilgisinden kaynaklanan
  sızıntının kontrolünü akademik katkı olarak savunuyor (bkz. literatür: Look-
  Ahead-Bench 2026, The Memorization Problem 2025). Canlı web araması sisteme
  güncel (bugünkü) faktör bilgisini sokar; bu hem look-ahead/memorization
  sızıntısı yaratır hem de her koşuda farklı sonuç dönerek REPRODUCIBILITY'yi
  (MVP kriter 5) bozar. Bu yüzden VARSAYILAN kaynak, repoda sabit ve
  sürüm-kontrollü bir corpus'tur: hızlı, deterministik, sızıntısız.

  Corpus'taki mekanizmalar 2016 ÖNCESİ klasik literatürden (araştırma dönemi
  başından önce yayımlanmış), böylece 'o tarihte zaten bilinen faktör' ilkesine
  uyar. Yeni bir mekanizma eklerken bunu koru.
"""
from __future__ import annotations

from llm.openai_client import OpenAICompatibleClient

_SYSTEM = ("Sen bir kantitatif finans literatürü araştırmacısısın. Web'de arama "
           "yaparak akademik olarak BELGELENMİŞ, YALNIZCA fiyat ve hacim verisinden "
           "hesaplanabilen kesitsel hisse senedi anomalilerini bulursun.")

# Klasik, fiyat/hacim-yalnızca kesitsel anomaliler — 6 mekanizma kategorisinden
# BİRER temsilci (çeşitlilik zorlaması: hepsi hacim/reversal OLMASIN). Her satır:
# 'Anomali — kısa mekanizma (alanlar: ...)'. Kaynaklar 2016 öncesi (araştırma
# dönemi öncesi) klasik literatürdür; look-ahead açısından güvenli tohumlardır.
_CORPUS: list[str] = [
    # 1) MOMENTUM (orta vade trend devamı) — Jegadeesh & Titman 1993
    "Kesitsel momentum — son 6-12 ay (son ayı atlayarak) getirisi yüksek hisseler "
    "kısa vadede üstün performansı sürdürür; yatırımcı az-tepkisi (alanlar: close)",
    # 2) REVERSAL (kısa vade aşırı-tepki geri dönüşü) — Jegadeesh 1990, Lehmann 1990
    "Kısa vadeli reversal — son 1-5 günde aşırı düşen hisseler geri döner; likidite "
    "baskısı ve aşırı-tepki düzeltmesi (alanlar: close)",
    # 3) VOLATİLİTE (düşük-vol primi) — Ang, Hodrick, Xing, Zhang 2006
    "Düşük-volatilite anomalisi — düşük idiyosinkratik/toplam oynaklıklı hisseler "
    "riske göre daha yüksek getiri sağlar (alanlar: close)",
    # 4) HACİM/LİKİDİTE (anormal hacim + Amihud illikidite) — Amihud 2002
    "İllikidite primi — düşük ortalama dolar-hacimli / yüksek |getiri|/dolar-hacim "
    "(Amihud) hisseleri illikidite primi taşır (alanlar: close, volume, dollar_volume)",
    # 5) FİYAT-SEVİYE (52-hafta yükseğe yakınlık) — George & Hwang 2004
    "52-hafta yüksek yakınlığı — fiyatı 52-hafta zirvesine yakın hisseler devam "
    "eğilimi gösterir; çıpalama yanlılığı (alanlar: close, high)",
    # 6) MEVSİMSELLİK/TAKVİM (ay-sonu / kısa-vade mevsimselliği) — Heston & Sadka 2008
    "Getiri mevsimselliği — geçmiş yıllarda aynı takvim ayında güçlü olan hisseler "
    "o ayı tekrar üstün geçme eğilimindedir (alanlar: close)",
    # Ek çeşitlilik tohumları (aynı kategorilerin farklı biçimleri):
    # HACİM — yüksek-hacim + düşük getiri etkileşimi
    "Hacim-teyitli reversal — anormal yüksek hacimle biten sert düşüşlerde geri "
    "dönüş olasılığı artar; zorunlu satışın bitişi (alanlar: close, volume)",
    # MOMENTUM — momentum + oynaklık etkileşimi (momentum crash frenleme)
    "Oynaklık-ölçekli momentum — momentum sinyali düşük oynaklıkta daha kararlı; "
    "yüksek oynaklıkta momentum çöküşü riski (alanlar: close)",
]


# KRİPTO corpus — evren hisse DEĞİLSE bu kullanılır.
#
# NEDEN AYRI: yukarıdaki corpus HİSSE anomalileri (52-hafta, ay-sonu
# mevsimselliği, Amihud). Kripto evrenine funding verip LLM'e hisse anomalisi
# fısıldamak aramayı kör bırakır — perpetual piyasanın kendi mekanizmaları var.
#
# SIZINTI SINIRI (önemli): buraya YALNIZCA literatürde yazan MEKANİZMA girer.
# Bizim kendi sondajımızda bulduğumuz spesifik parametreler (kaç günlük ortalama,
# hangi z-score penceresi, hangi tutuş süresi) BİLEREK VERİLMEZ — onları vermek
# LLM'e cevabı söylemek olurdu; aramanın işi onları bulmak.
_CRYPTO_CORPUS: list[str] = [
    # 1) FUNDING / KALABALIKLIK — perpetual piyasanın kendine özgü mekanizması.
    #    (Perpetual funding = kaldıraçlı pozisyonlanma göstergesi; kalabalık ve
    #     kaldıraçlı taraf tasfiyeye açıktır.)
    "Funding kalabalıklığı — daimi vadeli sözleşmede funding oranı POZİTİFken uzun "
    "taraf kalabalık ve kaldıraçlıdır, tasfiye riski taşır ve sonraki getirisi düşme "
    "eğilimindedir; negatif funding'de tersi geçerlidir (alanlar: funding_rate)",
    # 2) TASFİYE (likidasyon) KASKADI — kaldıraç + sert hareket etkileşimi
    "Tasfiye kaskadı — aşırı funding ile birlikte gelen sert fiyat hareketi zorunlu "
    "pozisyon kapanışlarını tetikler; kaskadın bitişinde aşırı-tepki geri döner "
    "(alanlar: funding_rate, close)",
    # 3) MOMENTUM — kripto'da belgelenmiş (Liu & Tsyvinski, kripto risk faktörleri)
    "Kripto momentum — kısa/orta vadeli (haftalar ölçeğinde) getiri devamı; "
    "yatırımcı dikkat akışı ve az-tepki (alanlar: close)",
    # 4) REVERSAL — bireysel-ağırlıklı piyasada aşırı tepki
    "Kripto kısa vadeli reversal — bireysel yatırımcı ağırlıklı, kaldıraçlı piyasada "
    "sert hareketler aşırı tepkidir ve kısmen geri döner (alanlar: close)",
    # 5) OYNAKLIK — kripto'da oynaklık ölçeklemesi
    "Oynaklık-ölçekli sinyal — yüksek oynaklık rejiminde sinyaller kararsızlaşır; "
    "oynaklığa göre ölçekleme kesitsel sıralamayı stabilize eder (alanlar: close)",
    # 6) İLLİKİDİTE — kripto'da da geçerli, ama işlem maliyeti yiyebilir
    "İllikidite primi — düşük dolar-hacimli enstrümanlar likidite primi taşır; ancak "
    "işlem maliyeti bu primi yiyebilir (alanlar: close, volume, dollar_volume)",
    # 7) FUNDING x FİYAT ETKİLEŞİMİ — kategori olarak etkileşimi teşvik eder
    "Kalabalıklık ve fiyat etkileşimi — pozisyonlanma göstergesi tek başına değil, "
    "yakın dönem fiyat hareketiyle BİRLİKTE değerlendirildiğinde tasfiye adaylarını "
    "daha iyi ayırt eder (alanlar: funding_rate, close)",
    # 8) TAKVİM — kripto 7/24 (hisse kapanışı yok)
    "Kripto takvim etkisi — piyasa 7/24 açıktır; hafta sonu ve tatil dönemlerinde "
    "likidite ve katılım düşer, fiyat hareketleri abartılı olabilir (alanlar: close, volume)",
]

_DOMAIN_CORPUS = {"equity": _CORPUS, "crypto": _CRYPTO_CORPUS}


def load_literature_mechanisms(n: int = 6, domain: str = "equity") -> list[str]:
    """Statik corpus'tan ilk n mekanizmayı döndürür (deterministik, reproducible).

    `domain`: 'equity' (hisse; varsayılan) | 'crypto' (perpetual + funding).
    Girişler mekanizma kategorilerinden birer temsilci olacak şekilde sıralıdır;
    n<=kategori sayısı ise her kategoriden en fazla bir tohum gelir.
    Bilinmeyen domain -> hisse corpus'u (güvenli varsayılan).
    """
    return _DOMAIN_CORPUS.get(domain, _CORPUS)[:n]


def fetch_literature_mechanisms(client: OpenAICompatibleClient, model: str,
                                universe_description: str, n: int = 6) -> list[str]:
    """OPSİYONEL, DEPRECATED: canlı web araması ile mekanizma çeker.

    Reproducibility ve look-ahead sızıntısı nedeniyle VARSAYILAN DEĞİLDİR
    (bkz. modül başlığı). Yalnızca models.yaml -> web_search: true iken çağrılır;
    arama başarısız/timeout olursa statik corpus'a düşer.
    """
    user = (f"Evren: {universe_description}\n\n"
            f"Web'de ara ve kesitsel hisse getirisini öngördüğü akademik olarak "
            f"belgelenmiş, YALNIZCA fiyat ve hacim (close, open, high, low, volume, "
            f"dollar_volume) ile hesaplanabilen anomaliler bul.\n"
            f"ÖNEMLİ: Aşağıdaki {n} FARKLI kategoriden HER BİRİNDEN tam olarak BİR "
            f"anomali ver (çeşitlilik şart, hepsi hacim/reversal OLMASIN):\n"
            f"  1) Zaman-serisi/kesitsel MOMENTUM (orta vade trend devamı)\n"
            f"  2) Kısa vadeli REVERSAL (aşırı tepki geri dönüşü)\n"
            f"  3) VOLATİLİTE tabanlı (düşük-vol primi / vol değişimi)\n"
            f"  4) HACİM/LİKİDİTE (anormal hacim, likidite primi)\n"
            f"  5) FİYAT-SEVİYE (52-hafta yüksek/dip, gün-içi aralık/high-low)\n"
            f"  6) MEVSİMSELLİK/TAKVİM (haftanın günü, ay-sonu, momentum-sezonu)\n"
            f"YASAK (temel veri gerektirir, LİSTELEME): Value, Book-to-Market, "
            f"Profitability, Investment, kazanç/bilanço temelli faktörler.\n"
            f"Her biri için TEK satır: 'Anomali adı — kısa mekanizma "
            f"(alanlar: ...)'. Sadece liste, başka açıklama yok.")
    try:
        # Kısa timeout: web_search yavaş/takılı kalabilir; başarısızsa statik
        # corpus'a düşer (bloke etmez).
        resp = client.chat(model, _SYSTEM, user, temperature=0.3,
                           force_json=False, max_tokens=800, web_search=True,
                           timeout=90.0)
    except Exception as e:  # noqa: BLE001 — arama başarısızsa statik corpus'a düş
        print(f"[literatür] web araması başarısız ({type(e).__name__}); statik corpus kullanılıyor.")
        return load_literature_mechanisms(n)
    lines = []
    for raw in resp.text.splitlines():
        s = raw.strip(" -*•\t")
        # baştaki "1." "2)" gibi numaralandırmayı temizle
        while s[:1].isdigit() or s[:1] in ".)":
            s = s[1:].strip()
        if len(s) > 15:
            lines.append(s)
    return lines[:n] or load_literature_mechanisms(n)
