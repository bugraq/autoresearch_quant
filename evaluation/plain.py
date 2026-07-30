"""
SADE ANLATIM KATMANI — "sadece trade bilen adam" da çıktıyı anlasın.

Bu modül HİÇBİR hesap yapmaz. Zaten hesaplanmış sayıları alır ve
teknik-olmayan Türkçeye çevirir. Kural: bir sayı basılıyorsa yanında
"bu ne demek" olmalı, ve iyi/kötü olduğu söylenmeli.

Neden ayrı modül: aynı çeviriler kampanya özeti (main.py), kıyas
(scripts/benchmark.py) ve çoklu-test tablosunda (evaluation) kullanılıyor.
Tek yerde durursa hepsi AYNI dili konuşur; üç ayrı yerde üç farklı
"Sharpe nedir" açıklaması olmaz.

DÜRÜSTLÜK KURALI (bu modülün asıl işi): sade dil, iyimser dil demek DEĞİL.
Para kaybeden bir strateji sade dilde de "para kaybediyor" der. Süslemek,
teknik jargonla gizlemekten daha kötüdür — çünkü okuyan anladığını sanır.
"""
from __future__ import annotations

#: İKİ FARKLI ARAŞTIRMA SHARPE'I VAR ve ikisi de doğru — ama yan yana
#: görülünce çelişki gibi durur. Ölçüldü (hyp_0033): karne +0.66, kıyas +0.74.
#:
#:   fold ortalaması  : araştırma dönemi 5 walk-forward dilime bölünür, her
#:                      dilimin Sharpe'ı ayrı hesaplanıp ORTALANIR. Kabul
#:                      kapısının ve hafızanın kullandığı sayı budur; daha
#:                      muhafazakârdır (kötü bir dilim ortalamayı aşağı çeker).
#:   tüm dönem        : araştırma serisi TEK PARÇA olarak ölçülür. Kıyas bunu
#:                      kullanmak ZORUNDA: al-tut ve rastgele al-satçı da tek
#:                      parça ölçülüyor; fold'lanmış bir sayıyla onlara karşı
#:                      koymak elma-armut olurdu.
#:
#: HOLDOUT ve İLERİ-TEST'te böyle bir ikilik YOKTUR (tek dilim olarak
#: değerlendirilirler) — bu yüzden o sayılar her araçta birebir aynıdır.
#: Metin tek yerde durur ki iki araç aynı cümleyi göstersin.
IKI_SHARPE_NOTU = (
    "Not: araştırma Sharpe'ı iki şekilde ölçülebilir — walk-forward fold "
    "ORTALAMASI (karne/dashboard; kabul kapısının kullandığı, daha "
    "muhafazakâr) ve TÜM DÖNEM tek parça (kıyas; al-tut/rastgele de böyle "
    "ölçüldüğü için adil karşılaştırma bunu gerektirir). İkisi de doğrudur, "
    "aynı şeyin iki ölçüsüdür. HOLDOUT ve İLERİ-TEST'te bu ikilik yoktur: "
    "o sayılar her araçta birebir aynıdır."
)


# ── Terim sözlüğü: teknik ad -> (trader dili, tek cümle açıklama) ──────
TERIMLER: "dict[str, tuple[str, str]]" = {
    "sharpe": (
        "risk başına kazanç",
        "Kazancı, iniş-çıkışın büyüklüğüne bölen not. Aynı parayı daha az "
        "heyecanla kazanmak daha iyidir. 0'ın altı = zarar."),
    "max_drawdown": (
        "en dip anındaki kayıp",
        "Zirveden en dibe paranın yüzde kaçı eridi. %40 = hesabın bir ara "
        "%40 küçüldü; çoğu insan orada dayanamaz ve çıkar."),
    "turnover": (
        "yıllık al-sat hacmi",
        "Sermayenin kaç katı kadar işlem yapılıyor. Yüksekse komisyon ve "
        "makas kârı yer — 'çok işlem = çok para' değildir."),
    "hit_rate": (
        "isabet oranı",
        "İşlem yapılan barların yüzde kaçı artıda kapandı. %50'nin altı da "
        "kâr edebilir (az sayıda büyük kazanç); tek başına yeterli değildir."),
    "total_return": (
        "toplam getiri",
        "Bu stratejiye baştan para koysaydın elindeki para % kaç değişirdi."),
    "ic": (
        "öngörü gücü",
        "Modelin tahmini ile gerçekte olan ne kadar örtüşüyor. ~0 ise "
        "yüksek kazanç bile ŞANS olabilir. Finansta 0.03 bile anlamlıdır."),
    "dsr": (
        "şans elemesi notu",
        "Yüzlerce fikir denenince biri sırf şansla parlar. Bu not, deneme "
        "sayısı düzeltildikten sonra fikrin hâlâ ayakta olma olasılığı. "
        "0.95 üstü = 'şans değil' denebilir."),
    "fdr": (
        "yanlış keşif elemesi",
        "Aynı anda çok sayıda fikri test ederken kaçının tesadüfen parlak "
        "göründüğünü hesaba katan istatistiksel süzgeç."),
    "ci": (
        "güven aralığı",
        "Gerçek değerin büyük ihtimalle içinde olduğu aralık. Alt sınır "
        "0'ın altındaysa 'aslında zarar ediyor olabilir' demektir."),
    "holdout": (
        "kilitli dönem sınavı",
        "Fikir geliştirilirken HİÇ görülmemiş bir dönem. Öğrenciye sınav "
        "sorusunu önceden vermemek gibi — gerçek not buradan çıkar."),
}


def kampanya_cost_bps(varsayilan: float = 5.0) -> float:
    """AKTİF kampanyanın işlem maliyeti (configs/campaign.yaml).

    Neden burada: şeffaflık script'leri (anatomy / benchmark / forward_test /
    verify_sharpe) maliyeti SABİT 5.0 yazıyordu; aktif kripto kampanyası ise
    10.0 kullanıyor. Sonuç: AYNI hipotez, kampanyada bir Sharpe, "her şeyi
    açıklayan" script'te BAŞKA bir Sharpe gösteriyordu — üstelik script'teki
    daha iyimser (yarı maliyet). Kıyas ve ileri-test de yarım maliyetle
    koşuyordu, yani kendimizi kayırıyorduk. Tek kaynak: kampanya config'i.
    """
    import os

    import yaml
    yol = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "configs", "campaign.yaml")
    try:
        with open(yol, encoding="utf-8") as f:
            return float(yaml.safe_load(f)["campaign"]["budget"]["cost_bps"])
    except Exception:  # noqa: BLE001 — config yoksa/bozuksa sessizce varsayılan
        return varsayilan


def esik_yorumu(ad: str, deger: "float | None") -> str:
    """Bir metriğin değerini kaba iyi/kötü ölçeğine oturt (tek satır)."""
    if deger is None:
        return "ölçülemedi"
    if ad == "sharpe":
        if deger < 0:
            return "PARA KAYBEDİYOR"
        if deger < 0.5:
            return "zayıf — masrafları zor karşılar"
        if deger < 1.0:
            return "idare eder"
        if deger < 2.0:
            return "iyi"
        return "çok iyi (bu kadar iyisi genelde bir hatanın işaretidir)"
    if ad == "max_drawdown":
        if deger < 0.10:
            return "rahat taşınır"
        if deger < 0.25:
            return "katlanılabilir"
        if deger < 0.40:
            return "sert — çoğu kişi burada çıkar"
        return "çok sert"
    if ad == "turnover":
        if deger < 20:
            return "sakin"
        if deger < 100:
            return "hareketli"
        return "çok yüksek — masraf kârı yiyor olabilir"
    if ad == "ic":
        if abs(deger) < 0.01:
            return "öngörü YOK (kazanç şans olabilir)"
        if abs(deger) < 0.03:
            return "zayıf"
        return "anlamlı öngörü"
    if ad == "dsr":
        if deger > 0.95:
            return "şansla açıklanamaz"
        if deger > 0.5:
            return "kararsız"
        return "şans olabilir"
    return ""


def para_dili(toplam_getiri: "float | None", anapara: float = 100_000) -> str:
    """Yüzdeyi somut paraya çevir — soyut yüzde, somut TL kadar konuşmaz."""
    if toplam_getiri is None:
        return "hesaplanamadı"
    son = anapara * (1.0 + toplam_getiri)
    yon = "kazanç" if toplam_getiri >= 0 else "KAYIP"
    return (f"{anapara:,.0f} birim para koysaydın {son:,.0f} olurdu "
            f"(%{toplam_getiri*100:+.1f} {yon})".replace(",", "."))


def strateji_karnesi(sharpe: "float | None", max_dd: "float | None",
                     turnover: "float | None", hit: "float | None" = None,
                     ic: "float | None" = None, toplam: "float | None" = None,
                     girinti: str = "    ") -> str:
    """Bir stratejinin metriklerini satır satır, yorumuyla birlikte yaz."""
    g = girinti
    satirlar: list[str] = []

    def satir(etiket: str, gosterim: str, yorum: str = "") -> None:
        satirlar.append(f"{g}{etiket:<24s} {gosterim:>10s}   {yorum}".rstrip())

    if sharpe is not None:
        satir(TERIMLER["sharpe"][0], f"{sharpe:+.2f}", esik_yorumu("sharpe", sharpe))
    if toplam is not None:
        satir(TERIMLER["total_return"][0], f"%{toplam*100:+.1f}")
        satirlar.append(f"{g}  -> {para_dili(toplam)}")
    if max_dd is not None:
        satir(TERIMLER["max_drawdown"][0], f"%{max_dd*100:.0f}",
              esik_yorumu("max_drawdown", max_dd))
    if turnover is not None:
        satir(TERIMLER["turnover"][0], f"{turnover:.0f}x",
              esik_yorumu("turnover", turnover))
    if hit is not None:
        satir(TERIMLER["hit_rate"][0], f"%{hit*100:.0f}")
    if ic is not None:
        satir(TERIMLER["ic"][0], f"{ic:+.3f}", esik_yorumu("ic", ic))
    return "\n".join(satirlar)


def buyuk(metin: str) -> str:
    """Türkçe-doğru büyük harf: i->İ, ı->I (Python'ın .upper()'ı i'yi I yapar)."""
    return metin.replace("i", "İ").replace("ı", "I").upper()


def sozluk_blogu(anahtarlar: "list[str]", girinti: str = "  ") -> str:
    """Kullanılan terimlerin sözlüğü — çıktının altına eklenir."""
    g = girinti
    out = [f"{g}TERİMLER — hangi kelime ne demek:"]
    for k in anahtarlar:
        if k not in TERIMLER:
            continue
        ad, aciklama = TERIMLER[k]
        out.append(f"{g}  • {buyuk(ad)} ({k})")
        for parca in _sar(aciklama, 66):
            out.append(f"{g}      {parca}")
    return "\n".join(out)


def _sar(metin: str, genislik: int) -> "list[str]":
    """Basit kelime kaydırma (textwrap'e bağımlı olmadan, hizalı çıktı için)."""
    kelimeler, satir, out = metin.split(), "", []
    for k in kelimeler:
        if len(satir) + len(k) + 1 > genislik:
            out.append(satir)
            satir = k
        else:
            satir = f"{satir} {k}".strip()
    if satir:
        out.append(satir)
    return out


def durust_hukum(toplam_getiri: "float | None", sharpe: "float | None",
                 al_tut_getiri: "float | None" = None,
                 fdr_gecti: "bool | None" = None) -> "tuple[str, list[str]]":
    """DÜRÜST hüküm: (başlık, gerekçe satırları).

    Kural sırası ÖNEMLİ — para kaybı her şeyin önüne geçer. "Rastgele
    al-satçıyı geçtik" demek, o rastgele al-satçı %95 batarken bizim %20
    batmamızı BAŞARI saymaz. Trader'ın ilk sorusu "param ne oldu"dur.

    fdr_gecti=False verilirse (çoklu-test süzgecini hiçbir fikir geçmediyse)
    hüküm ZORUNLU olarak aşağı çekilir. Aksi halde aynı çıktının bir yeri
    "hiçbiri geçemedi", öteki yeri "umut verici" der ve okuyan hangisine
    inanacağını bilemez.
    """
    gerekce: list[str] = []
    if toplam_getiri is not None and toplam_getiri < 0:
        gerekce.append(
            f"Bu strateji test döneminde PARA KAYBETTİ (%{toplam_getiri*100:+.1f}). "
            f"Başka bir yaklaşımdan 'daha az kaybetmek' başarı değildir.")
        if al_tut_getiri is not None and al_tut_getiri > 0:
            gerekce.append(
                f"Aynı dönemde hiçbir şey yapmayıp elde tutmak %{al_tut_getiri*100:+.1f} "
                f"getiriyordu — yani işlem yapmamak daha kârlıydı.")
        gerekce.append(
            "Doğru okuma: altyapı çalışıyor, ama bu evrende HENÜZ para "
            "kazandıran bir sinyal bulunamadı. Bu bir başarısızlık değil, "
            "dürüst bir NULL sonuçtur — ve raporlanması gerekir.")
        return "PARA KAZANDIRMADI", gerekce

    if sharpe is not None and sharpe < 0.5:
        gerekce.append(
            f"Risk başına kazanç ({sharpe:+.2f}) işlem masraflarını güvenle "
            f"karşılayacak seviyenin altında.")
        gerekce.append("Kâğıt üstünde artı, pratikte kaygan.")
        return "ZAYIF — HENÜZ GÜVENİLİR DEĞİL", gerekce

    if al_tut_getiri is not None and toplam_getiri is not None \
            and toplam_getiri < al_tut_getiri:
        gerekce.append(
            f"Para kazandı (%{toplam_getiri*100:+.1f}) ama pasif al-tut "
            f"(%{al_tut_getiri*100:+.1f}) daha çok kazandırdı.")
        gerekce.append(
            "Uğraşmanın karşılığı alınmamış: aynı parayı endekste bırakmak "
            "daha iyiydi. (Not: long-short strateji piyasa riski taşımaz; "
            "bu yüzden düşük getiri tek başına eleme sebebi değildir — "
            "ama 'piyasayı yendik' de denemez.)")
        return "KAZANDIRDI AMA AL-TUT'U GEÇEMEDİ", gerekce

    gerekce.append(
        "Araştırma verisinde para kazandırdı ve risk/kazanç dengesi makul.")

    if fdr_gecti is False:
        gerekce.append(
            "AMA: bu kadar çok fikir denendiği için, elde kalan en iyi sonuç "
            "bile tesadüf olarak açıklanabiliyor (çoklu-test süzgecini hiçbir "
            "fikir geçemedi). 100 kez yazı-tura atıp 8 tura üst üste gelmesi "
            "gibi — kazanç gerçek, ama tekrarlanacağının kanıtı yok.")
        gerekce.append(
            "Doğru okuma: buna 'buldum' demek için ya daha az fikirle daha "
            "güçlü bir sonuç, ya da kilitli dönemde teyit gerekir.")
        return "KAZANDIRDI AMA İSTATİSTİK ONAYLAMADI", gerekce

    gerekce.append(
        "SON SÖZ DEĞİL: bu sayı, fikrin geliştirildiği veriden çıktı. "
        "Gerçek not, hiç görülmemiş kilitli dönemden gelir "
        "(python main.py --holdout).")
    return "UMUT VERİCİ — AMA KİLİTLİ DÖNEM SINAVI BEKLİYOR", gerekce
