"""
ADAY SEÇİMİ ve ÜÇ-DÖNEM KARNESİ — tek kaynak.

Neden ayrı bir modül: "hangi stratejiyi gösteriyoruz?" sorusu projede DÖRT
yerde soruluyordu (benchmark, forward_test, dashboard, anatomy) ve her biri
kendi kuralını yazmıştı. Gerçek koşuda bu şu hataya yol açtı:

    memory.accepted_hypotheses() ARAŞTIRMA Sharpe'ına göre sıralar.
    En üstteki, fikrin geliştirildiği dönemde en parlak görünen adaydır —
    yani AŞIRI UYDURULMUŞ olma ihtimali en yüksek olan. Ölçüldü: hyp_0021
    araştırmada +1.14 ile birinciydi, taze veride −%44 yaptı.

Doğru kural, kanıtın gücüne göredir:

    1. ÜÇ dönemi de geçen (DOĞRULANDI)          <- en güçlü kanıt
    2. Kilitli dönemi (holdout) geçen
    3. Yalnızca araştırmada parlayan            <- kanıt değil, yalnızca aday

Bu dosya SALT OKUNURDUR: veritabanlarını `mode=ro` açar, hiçbir şey yazmaz.
Böylece kontrol paneli/rapor gibi "sadece bakan" yerler yanlışlıkla sicili
değiştiremez.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DB = os.path.join(HERE, "research_memory.sqlite")
AUDIT_DB = os.path.join(HERE, "holdout_audit.sqlite")


@dataclass
class Aday:
    """Bir adayın üç dönemlik karnesi + neden seçildiği."""
    hypothesis_id: str
    hypothesis_json: str
    title: str
    research_sharpe: "float | None"
    holdout_sharpe: "float | None"
    forward_sharpe: "float | None"
    secim_nedeni: str
    kanit_gucu: int          # 3 = üç dönem, 2 = holdout, 1 = yalnız araştırma

    def spec(self):
        from contracts.hypothesis_spec import HypothesisSpec
        return HypothesisSpec.model_validate(json.loads(self.hypothesis_json))

    def hukum(self, min_acceptance_sharpe: float = 0.5):
        from evaluation.three_period import final_verdict
        return final_verdict(self.research_sharpe, self.holdout_sharpe,
                             self.forward_sharpe, min_acceptance_sharpe)


def _ro(path: str) -> "sqlite3.Connection | None":
    """Salt-okunur bağlantı. Dosya yoksa None (çağıran boş duruma düşer)."""
    if not os.path.exists(path):
        return None
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def holdout_sonuclari() -> "dict[str, tuple[float, bool]]":
    """{hipotez_id: (sharpe, gecti)} — yalnız AKTİF kayıtlar.

    Geçersiz kılınmış (status!='active') kayıtlar sayılmaz ama silinmez;
    audit'te gerekçesiyle durur.
    """
    c = _ro(AUDIT_DB)
    if c is None:
        return {}
    try:
        kolonlar = {r[1] for r in c.execute("PRAGMA table_info(holdout_access)")}
        sql = "SELECT hypothesis_id, sharpe, passed FROM holdout_access"
        if "status" in kolonlar:
            sql += " WHERE status='active'"
        return {h: (float(s), bool(p)) for h, s, p in c.execute(sql)}
    except sqlite3.Error:
        return {}
    finally:
        c.close()


def ileri_test_sonuclari() -> "dict[str, float]":
    """{hipotez_id: EN SON ileri-test Sharpe'ı}. Sicil yoksa boş."""
    c = _ro(AUDIT_DB)
    if c is None:
        return {}
    try:
        rows = c.execute(
            "SELECT hypothesis_id, sharpe FROM forward_test "
            "WHERE id IN (SELECT MAX(id) FROM forward_test GROUP BY hypothesis_id)"
        ).fetchall()
        return {h: float(s) for h, s in rows}
    except sqlite3.Error:      # tablo henüz yok
        return {}
    finally:
        c.close()


def _kabul_edilenler(limit: int = 50) -> "list[tuple]":
    """(hid, json, research_sharpe) — kampanya hafızasından kabul edilenler."""
    c = _ro(MEMORY_DB)
    if c is None:
        return []
    try:
        return c.execute(
            "SELECT hypothesis_id, hypothesis_json, sharpe FROM experiment "
            "WHERE decision='accept' AND hypothesis_json IS NOT NULL "
            "ORDER BY sharpe DESC LIMIT ?", (limit,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        c.close()


def _sicilden(limit: int = 50) -> "list[tuple]":
    """Aday sicilinden (kampanyalar arası) — hafıza sıfırlansa da durur.

    `--fresh` kampanya hafızasını siler; üç dönemden geçmiş aday orada
    kaybolur. Sicil parmak izi anahtarlıdır ve hayatta kalır.
    """
    c = _ro(AUDIT_DB)
    if c is None:
        return []
    try:
        return c.execute(
            "SELECT hypothesis_id, hypothesis_json, research_sharpe "
            "FROM candidate_registry ORDER BY first_seen").fetchall()
    except sqlite3.Error:
        return []
    finally:
        c.close()


def tum_adaylar(min_acceptance_sharpe: float = 0.5) -> "list[Aday]":
    """Bilinen TÜM adaylar, kanıt gücüne göre sıralı (en güçlü önce).

    Kaynak birleşimi: kampanya hafızası (kabul edilenler) + kampanyalar arası
    aday sicili. Aynı hipotez ikisinde de varsa hafızadaki kazanır (JSON'u
    kampanyanın kendi kaydıdır).
    """
    holdout = holdout_sonuclari()
    ileri = ileri_test_sonuclari()

    ham: "dict[str, tuple]" = {}
    for hid, hj, rs in _sicilden():
        if hj:
            ham[hid] = (hj, rs)
    for hid, hj, rs in _kabul_edilenler():
        ham[hid] = (hj, rs)          # hafıza sicili EZER

    adaylar = []
    for hid, (hj, rs) in ham.items():
        h_sh = holdout.get(hid, (None, False))[0]
        f_sh = ileri.get(hid)
        try:
            baslik = json.loads(hj).get("title", hid)
        except (ValueError, TypeError):
            baslik = hid
        if h_sh is not None and f_sh is not None:
            guc, neden = 3, "üç dönemde de ölçüldü"
        elif h_sh is not None:
            guc, neden = 2, "kilitli dönemde ölçüldü (ileri-test yok)"
        else:
            guc, neden = 1, "yalnız araştırma dönemi — kanıt DEĞİL"
        adaylar.append(Aday(hid, hj, baslik, rs, h_sh, f_sh, neden, guc))

    def anahtar(a: "Aday"):
        # 1) sistemin hükmü (DOĞRULANDI önce), 2) kanıt gücü, 3) en zayıf OOS
        gecti = a.hukum(min_acceptance_sharpe).passed
        oos = [s for s in (a.holdout_sharpe, a.forward_sharpe) if s is not None]
        return (not gecti, -a.kanit_gucu, -(min(oos) if oos else -99))

    return sorted(adaylar, key=anahtar)


def en_iyi_aday(min_acceptance_sharpe: float = 0.5) -> "Aday | None":
    """GÖSTERİLECEK aday: en güçlü KANITA sahip olan (en yüksek Sharpe'lı DEĞİL).

    Bu ayrım projenin ölçülmüş bir dersidir; bkz. modül başlığı.
    """
    hepsi = tum_adaylar(min_acceptance_sharpe)
    if not hepsi:
        return None
    a = hepsi[0]
    if a.hukum(min_acceptance_sharpe).passed:
        a.secim_nedeni = "ÜÇ DÖNEMİ de geçen tek aday"
    elif a.kanit_gucu >= 2 and (a.holdout_sharpe or -9) >= min_acceptance_sharpe:
        a.secim_nedeni = "kilitli dönemi geçti (ileri-testte doğrulanmadı)"
    elif a.kanit_gucu >= 2:
        a.secim_nedeni = "kilitli dönemde ölçüldü ama eşiği geçemedi"
    else:
        a.secim_nedeni = ("yalnız araştırma döneminde ölçüldü — KANIT DEĞİL, "
                          "holdout koşulmalı")
    return a


def karne_satirlari(min_acceptance_sharpe: float = 0.5) -> "list[tuple]":
    """three_period.verdict_table() için (hid, araştırma, holdout, ileri) satırları."""
    return [(a.hypothesis_id, a.research_sharpe, a.holdout_sharpe, a.forward_sharpe)
            for a in tum_adaylar(min_acceptance_sharpe)]
