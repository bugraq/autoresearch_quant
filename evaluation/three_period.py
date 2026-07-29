"""
ÜÇ-DÖNEM HÜKMÜ — kilitli holdout tek başına yetmez.

Bu modülün varlık sebebi ölçülmüş bir olaydır (29.07.2026): kampanya v4'ün
kabul edilen üç adayı da kilitli dönemi GEÇTİ (holdout Sharpe +0.93 / +0.59 /
+1.29). Yalnız ona bakılsaydı "alpha bulduk" denirdi. Üçüncü, bağımsız bir
örneklem-dışı dönemde (2025→bugün, sistemin hiç görmediği taze veri) üçü de
ÇÖKTÜ (−0.36 / −1.46 / −0.05; toplam −%44 / −%80 / −%29).

Ders: holdout (2023-24) tek bir REJİM ÇEKİLİŞİDİR. Onu geçmek "genelliyor"
demek değil, "o rejimde de tuttu" demektir. Bir stratejiye güvenmek için
BİRDEN FAZLA bağımsız OOS rejiminde ayakta kalması gerekir.

Bu yüzden hüküm artık insana bırakılmıyor; kural burada, tek yerde:

    araştırma  -> fikrin geliştirildiği dönem (kanıt DEĞİL, yalnızca bağlam)
    holdout    -> kilitli, tek-atış: kampanya eşiğini geçmeli
    ileri-test -> holdout'tan SONRAKİ taze zaman: en azından POZİTİF kalmalı

İleri-test eşiği bilerek daha gevşektir (0): dönem kısa ve gürültülüdür;
ondan "iyi Sharpe" beklemek fazla ceza olur. Aradığımız şey ÇÖKMEMESİDİR.
"""
from __future__ import annotations

from dataclasses import dataclass

#: İleri-test için asgari eşik. Sıfır = "en azından para kaybettirmesin".
#: Yükseltmek daha katı olurdu ama kısa dönemde gürültü cezalandırır.
FORWARD_MIN_SHARPE = 0.0


@dataclass
class ThreePeriodVerdict:
    verdict: str            # DOĞRULANDI | REJİM-BAĞIMLI | ÇÖKTÜ | EKSİK
    passed: bool            # sistemin "güvenilir" dediği tek durum
    reasons: "list[str]"

    def line(self) -> str:
        return f"{self.verdict}: {self.reasons[0] if self.reasons else ''}"


def final_verdict(research_sharpe: "float | None",
                  holdout_sharpe: "float | None",
                  forward_sharpe: "float | None",
                  min_acceptance_sharpe: float = 0.5,
                  forward_min: float = FORWARD_MIN_SHARPE) -> ThreePeriodVerdict:
    """Üç dönemin sonucundan TEK hüküm üret.

    Kural sırası, en sert kanıttan en zayıfına:
      1. İleri-test (en taze, hiç görülmemiş) çöktüyse -> güvenilmez.
      2. Holdout kampanya eşiğinin altındaysa -> güvenilmez.
      3. İkisi de geçtiyse -> DOĞRULANDI (ama "alpha bulundu" DEĞİL; bkz. not).

    `None` = o dönem henüz ölçülmedi -> EKSİK (hüküm verilemez; sessizce
    "geçti" saymak bu modülün önlemek için var olduğu hatanın ta kendisidir).
    """
    if holdout_sharpe is None or forward_sharpe is None:
        eksik = []
        if holdout_sharpe is None:
            eksik.append("kilitli dönem sınavı yapılmadı")
        if forward_sharpe is None:
            eksik.append("ileri-test yapılmadı")
        return ThreePeriodVerdict(
            "EKSİK", False,
            [f"Hüküm verilemez — {', '.join(eksik)}.",
             "Eksik dönemi 'geçti' saymak, bu kontrolün önlemek için var "
             "olduğu hatanın kendisidir."])

    holdout_ok = holdout_sharpe >= min_acceptance_sharpe
    forward_ok = forward_sharpe > forward_min

    if holdout_ok and forward_ok:
        return ThreePeriodVerdict(
            "DOĞRULANDI", True,
            [f"İKİ bağımsız örneklem-dışı dönemde de ayakta kaldı "
             f"(holdout {holdout_sharpe:+.2f}, ileri-test {forward_sharpe:+.2f}).",
             "Bu 'alpha bulundu' DEMEK DEĞİLDİR: çok sayıda deneme içinden "
             "çıktı, çoklu-test düzeltmesi ayrıca kontrol edilmeli.",
             "Doğru okuma: bu aday HENÜZ ÖLMEDİ; izlenmeye devam edilmeli."])

    if holdout_ok and not forward_ok:
        return ThreePeriodVerdict(
            "REJİM-BAĞIMLI", False,
            [f"Kilitli dönemi geçti ({holdout_sharpe:+.2f}) ama taze veride "
             f"çöktü ({forward_sharpe:+.2f}).",
             "Holdout tek bir REJİM çekilişidir; onu geçmek genelleme kanıtı "
             "değildir. Performans döneme bağlı."])

    if not holdout_ok and forward_ok:
        return ThreePeriodVerdict(
            "REJİM-BAĞIMLI", False,
            [f"Kilitli dönemde eşiğin altında kaldı ({holdout_sharpe:+.2f} < "
             f"{min_acceptance_sharpe:.2f}) ama taze veride pozitif "
             f"({forward_sharpe:+.2f}).",
             "Çelişen OOS dönemleri = rejim bağımlılığı. Tek parlak döneme "
             "bakıp karar vermek, kendini kandırmanın en kolay yoludur."])

    return ThreePeriodVerdict(
        "ÇÖKTÜ", False,
        [f"HER İKİ örneklem-dışı dönemde de başarısız "
         f"(holdout {holdout_sharpe:+.2f}, ileri-test {forward_sharpe:+.2f}).",
         "Araştırma dönemindeki performans geçmişe uydurulmuş bir desendi."])


def verdict_table(rows: "list[tuple]", min_acceptance_sharpe: float = 0.5) -> str:
    """(hid, research, holdout, forward) satırlarından okunur tablo + hüküm.

    Tek yerde basılır ki kampanya çıktısı, dashboard ve rapor AYNI hükmü
    göstersin (daha önce üç yerde üç farklı sayı gösterme hatası yaşandı).
    """
    if not rows:
        return "  (değerlendirilecek aday yok)"
    out = [f"  {'hipotez':12s} {'araştırma':>10s} {'HOLDOUT':>9s} "
           f"{'İLERİ-TEST':>11s}   hüküm",
           "  " + "-" * 68]
    for hid, r, h, f in rows:
        v = final_verdict(r, h, f, min_acceptance_sharpe)
        rs = f"{r:+.2f}" if r is not None else "  –"
        hs = f"{h:+.2f}" if h is not None else "  –"
        fs = f"{f:+.2f}" if f is not None else "  –"
        out.append(f"  {hid:12s} {rs:>10s} {hs:>9s} {fs:>11s}   {v.verdict}")
    return "\n".join(out)
