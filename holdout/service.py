"""
Holdout Servisi (Doküman 10.3) — araştırmadan TAMAMEN ayrı son değerlendirme.

İlkeler:
  - LLM'den ve araştırma orchestrator'ından bağımsızdır (bu modül LLM import ETMEZ).
  - Holdout tarihlerini/serisini dışarı açıklamaz; yalnızca özet metrik döndürür.
  - Önceden belirlenmiş sayıda aday kabul eder (maximum_candidates).
  - Her aday için sonucu BİR KEZ üretir (one-shot); tekrar değerlendirme yasak.
  - Holdout sonucuyla strateji revizyonuna izin vermez (çağıran taraf uygular).
  - Bütün erişimleri audit log'a kaydeder (ayrı veritabanı).

Bu servis deterministik altyapıdır (derleme + backtest); yaratıcı hiçbir
bileşen içermez.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import MarketData
from dsl import compile_hypothesis
from backtest import run_backtest

#: Değerlendirici sürümü. Kilitli dönem sonucu, onu üreten değerlendiriciye
#: bağlıdır; sürüm kaydedilmezse "bu sayı hangi kodla çıktı" sorusu
#: cevaplanamaz. v1 -> v2 farkı ISINMA düzeltmesidir (bkz. HoldoutService).
EVALUATOR_VERSION = "v2-warmup"

_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdout_access (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    sharpe        REAL,
    passed        INTEGER,
    accessed_at   TEXT,
    -- ONE-SHOT artık UNIQUE kısıtla değil, 'aktif kayıt var mı' kuralıyla
    -- uygulanır (aşağı bak). Böylece hatalı bir değerlendiriciyle üretilmiş
    -- sonuçlar SİLİNMEDEN geçersiz kılınıp yeniden koşulabilir.
    status        TEXT DEFAULT 'active',    -- active | invalidated
    evaluator_version TEXT,
    invalidated_at    TEXT,
    invalidation_reason TEXT
);
"""

# İndeks AYRI: yeni sütunlara dayandığı için ancak taşımadan SONRA kurulabilir
# (eski audit dosyasında 'status' henüz yokken çalışırsa "no such column" verir).
_AUDIT_INDEX = ("CREATE INDEX IF NOT EXISTS ix_holdout_active "
                "ON holdout_access (hypothesis_id, status)")

# İLERİ-TEST SİCİLİ — holdout'tan FARKLI olarak tek-atış DEĞİLDİR.
# Kilitli dönem sonlu ve tükenir; ileri-test dönemi ise her gün büyür. Aynı
# aday zaman içinde tekrar tekrar ölçülür ve her ölçüm ayrı satır olur:
# stratejinin canlı performans zaman serisi. "as_of" o ölçümün yapıldığı gün.
_FORWARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS forward_test (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    as_of         TEXT NOT NULL,     -- ölçümün yapıldığı gün
    period_start  TEXT,
    period_end    TEXT,
    sharpe        REAL,
    total_return  REAL,
    verdict       TEXT,              -- üç-dönem hükmü (evaluation/three_period)
    recorded_at   TEXT
);
CREATE INDEX IF NOT EXISTS ix_forward_hyp
    ON forward_test (hypothesis_id, as_of);
"""

# Eski şemadan (hypothesis_id UNIQUE, ek sütunlar yok) yeni şemaya taşıma.
_MIGRATE_COLUMNS = {
    "status": "TEXT DEFAULT 'active'",
    "evaluator_version": "TEXT",
    "invalidated_at": "TEXT",
    "invalidation_reason": "TEXT",
}


class HoldoutError(Exception):
    """One-shot ihlali veya aday kotası aşımı."""


@dataclass
class HoldoutResult:
    hypothesis_id: str
    sharpe: float
    passed: bool
    #: Kilitli dönemin yüzde kaçında sinyal ÜRETİLEBİLDİ (ısınma sonrası).
    #: 1.0 = tam kapsama. Düşükse Sharpe daha kısa bir dilimden hesaplanmıştır.
    coverage: float = 1.0


class HoldoutService:
    """Kilitli dönem sınavı.

    `history` (araştırma dilimi) verilirse sinyal, geçmiş+holdout birleşimi
    üzerinde hesaplanıp yalnız holdout dilimine kesilir. Bu ISINMA düzeltmesidir:
      - rolling pencereler kilitli dönemin başında NaN kalmaz,
      - ML modeli holdout'un İÇİNDE yeniden eğitilmez; araştırma dönemiyle
        eğitilip kilitli döneme donmuş halde uygulanır (sınavın anlamı budur).
    Bilgi akışı tek yönlüdür (geçmiş -> gelecek), dolayısıyla sızıntı değildir.
    """

    def __init__(self, holdout_data: MarketData, audit_path: str = "holdout_audit.sqlite",
                 max_candidates: int = 20, min_sharpe: float = 0.5,
                 cost_bps: float = 5.0, history: "MarketData | None" = None) -> None:
        self._data = holdout_data          # KİLİTLİ — dışarı verilmez
        self._history = history            # araştırma dilimi (yalnız ısınma için)
        self._max = max_candidates
        self._min_sharpe = min_sharpe
        self._cost_bps = cost_bps
        self._audit = sqlite3.connect(audit_path)
        self._audit.executescript(_AUDIT_SCHEMA)   # yoksa kur
        self._migrate()                            # varsa yeni şemaya taşı
        self._audit.execute(_AUDIT_INDEX)          # sütunlar hazır: indeks
        self._audit.executescript(_FORWARD_SCHEMA)  # ileri-test sicili
        self._audit.commit()

    # ---------------- ileri-test sicili (tek-atış DEĞİL) ------------------
    def record_forward(self, hypothesis_id: str, as_of, sharpe: float,
                       total_return: float, verdict: str,
                       period_start=None, period_end=None) -> None:
        """Bir ileri-test ölçümünü sicile EKLE (üzerine yazmaz).

        Holdout tek-atıştır çünkü kilitli dönem sonludur ve tükenir.
        İleri-test dönemi ise her gün büyür: aynı aday tekrar tekrar
        ölçülebilir ve ölçülmelidir. Her satır bir ölçüm anıdır; birikince
        stratejinin canlı performans zaman serisi olur.
        """
        self._audit.execute(
            "INSERT INTO forward_test (hypothesis_id, as_of, period_start, "
            "period_end, sharpe, total_return, verdict, recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (hypothesis_id, str(as_of), str(period_start) if period_start else None,
             str(period_end) if period_end else None, float(sharpe),
             float(total_return), verdict,
             datetime.now(timezone.utc).isoformat()))
        self._audit.commit()

    def forward_log(self, hypothesis_id: "str | None" = None) -> list[tuple]:
        """İleri-test ölçüm geçmişi (en yeni sonda)."""
        sql = ("SELECT hypothesis_id, as_of, sharpe, total_return, verdict "
               "FROM forward_test")
        params: tuple = ()
        if hypothesis_id:
            sql += " WHERE hypothesis_id=?"
            params = (hypothesis_id,)
        return self._audit.execute(sql + " ORDER BY id", params).fetchall()

    def latest_forward(self) -> "dict[str, float]":
        """Her aday için EN SON ileri-test Sharpe'ı (üç-dönem hükmü için)."""
        rows = self._audit.execute(
            "SELECT hypothesis_id, sharpe FROM forward_test "
            "WHERE id IN (SELECT MAX(id) FROM forward_test GROUP BY hypothesis_id)"
        ).fetchall()
        return {h: s for h, s in rows}

    # ---------------- şema taşıma (eski audit dosyaları) ------------------
    def _migrate(self) -> None:
        """Eski audit dosyasını yeni şemaya taşı — KAYIT KAYBETMEDEN.

        Eski şemada `hypothesis_id UNIQUE` vardı; one-shot'ı veritabanı kısıtı
        uyguluyordu. Bu, hatalı bir değerlendiriciyle üretilmiş bir sonucu
        DÜZELTMEYİ de imkânsız kılıyordu (tek çare kaydı silmekti — bilimsel
        kaydı silmek en kötü seçenek). Yeni şemada one-shot 'aktif kayıt var mı'
        kuralıyla uygulanır; geçersiz kılma append-only bir olaydır.
        """
        cols = {r[1] for r in self._audit.execute(
            "PRAGMA table_info(holdout_access)")}
        for ad, tip in _MIGRATE_COLUMNS.items():
            if ad not in cols:
                self._audit.execute(
                    f"ALTER TABLE holdout_access ADD COLUMN {ad} {tip}")
        self._audit.execute(
            "UPDATE holdout_access SET status='active' WHERE status IS NULL")

        # UNIQUE kısıtı hâlâ duruyorsa tabloyu yeniden kur (SQLite'ta kısıt
        # düşürmenin tek yolu). Veri birebir kopyalanır.
        sql = self._audit.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='holdout_access'").fetchone()
        if sql and "UNIQUE" in sql[0].upper():
            self._audit.executescript("""
                ALTER TABLE holdout_access RENAME TO _holdout_old;
                CREATE TABLE holdout_access (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT NOT NULL, sharpe REAL, passed INTEGER,
                    accessed_at TEXT, status TEXT DEFAULT 'active',
                    evaluator_version TEXT, invalidated_at TEXT,
                    invalidation_reason TEXT);
                INSERT INTO holdout_access
                    (id, hypothesis_id, sharpe, passed, accessed_at, status,
                     evaluator_version, invalidated_at, invalidation_reason)
                SELECT id, hypothesis_id, sharpe, passed, accessed_at,
                       COALESCE(status,'active'), evaluator_version,
                       invalidated_at, invalidation_reason FROM _holdout_old;
                DROP TABLE _holdout_old;
                CREATE INDEX IF NOT EXISTS ix_holdout_active
                    ON holdout_access (hypothesis_id, status);
            """)

    # ---------------- geçersiz kılma (gerekçeli, silmeden) ----------------
    def invalidate(self, reason: str,
                   hypothesis_ids: "list[str] | None" = None) -> int:
        """Aktif kilitli-dönem sonuçlarını GEÇERSİZ kıl. Kayıt SİLİNMEZ.

        Ne zaman meşrudur: sonucu üreten DEĞERLENDİRİCİ hatalıysa (ör. ısınma
        hatası yüzünden sınav, araştırmada kabul edilen modeli değil başka bir
        modeli ölçüyorsa). Yani "sayı hoşuma gitmedi" değil, "bu sayı yanlış
        bir ölçüm aletinden çıktı" durumu.

        Ne zaman meşru DEĞİLDİR: holdout sonucunu görüp stratejiyi değiştirmek
        ve yeniden denemek. Bu, kilitli dönemi araştırma verisine çevirir —
        sistemin engellemek için var olduğu şeyin ta kendisi. Gerekçe zorunlu
        ve kalıcı olarak saklanır; denetlenebilir olması caydırıcılığıdır.

        Returns: geçersiz kılınan kayıt sayısı.
        """
        if not reason or not reason.strip():
            raise HoldoutError(
                "Geçersiz kılma GEREKÇE ister. Kilitli dönem kaydını gerekçesiz "
                "sıfırlamak, sınavı fiilen ortadan kaldırır.")
        now = datetime.now(timezone.utc).isoformat()
        sql = ("UPDATE holdout_access SET status='invalidated', "
               "invalidated_at=?, invalidation_reason=? WHERE status='active'")
        params: list = [now, reason.strip()]
        if hypothesis_ids:
            sql += f" AND hypothesis_id IN ({','.join('?' * len(hypothesis_ids))})"
            params += list(hypothesis_ids)
        cur = self._audit.execute(sql, params)
        self._audit.commit()
        return cur.rowcount

    def _count(self) -> int:
        """AKTİF kayıt sayısı — kota bunun üzerinden işler.

        Geçersiz kılınmış kayıtlar kotayı doldurmaz: aksi halde bir
        değerlendirici hatasını düzeltmek, düzeltmenin kendisi yüzünden
        imkânsızlaşırdı. Kayıt yine durur (dürüstlük), sadece sayılmaz.
        """
        return self._audit.execute(
            "SELECT COUNT(*) FROM holdout_access WHERE status='active'").fetchone()[0]

    def evaluate(self, hyp: HypothesisSpec) -> HoldoutResult:
        """Bir adayı kilitli dönemde BİR KEZ değerlendir. Yalnızca özet döner."""
        # ONE-SHOT: AKTİF bir kayıt varsa ikinci değerlendirme yasak. Geçersiz
        # kılınmış (invalidated) kayıt engellemez — ama o geçersiz kılma
        # gerekçesiyle birlikte audit'te kalıcıdır, silinmez.
        seen = self._audit.execute(
            "SELECT 1 FROM holdout_access WHERE hypothesis_id=? AND status='active'",
            (hyp.hypothesis_id,)).fetchone()
        if seen:
            raise HoldoutError(
                f"{hyp.hypothesis_id} zaten holdout'ta değerlendirildi (one-shot).")
        # Kota kontrolü
        if self._count() >= self._max:
            raise HoldoutError(f"Holdout aday kotası doldu ({self._max}).")

        graph = compile_hypothesis(hyp)
        signal, coverage = self._warm_signal(graph, hyp)
        result = run_backtest(graph, hyp, self._data, cost_bps=self._cost_bps,
                              signal=signal)
        sharpe = result.aggregate_sharpe() or 0.0
        passed = sharpe >= self._min_sharpe

        self._audit.execute(
            "INSERT INTO holdout_access (hypothesis_id, sharpe, passed, "
            "accessed_at, status, evaluator_version) VALUES (?,?,?,?,'active',?)",
            (hyp.hypothesis_id, sharpe, int(passed),
             datetime.now(timezone.utc).isoformat(), EVALUATOR_VERSION))
        self._audit.commit()
        # NOT: holdout tarihleri/serisi ASLA döndürülmez; sadece özet.
        return HoldoutResult(hyp.hypothesis_id, sharpe, passed, coverage)

    def _warm_signal(self, graph, hyp: HypothesisSpec):
        """Isınmalı sinyal: geçmiş+holdout üzerinde hesapla, holdout'a kes.

        history yoksa (geriye uyum) None döner ve motor eskisi gibi yalnız
        kilitli dilimden hesaplar — o zaman kapsama düşük olabilir.
        """
        from backtest.model_signal import compute_signal
        from data.synthetic import concat_market

        cols = self._data.get("close").columns
        idx = self._data.dates
        if self._history is None:
            sig = compute_signal(graph, hyp, self._data)      # ısınmasız (geriye uyum)
        else:
            sig = compute_signal(graph, hyp, concat_market(self._history, self._data))
        sig = sig.reindex(index=idx, columns=cols)
        coverage = float(sig.notna().any(axis=1).mean()) if len(sig) else 0.0
        return sig, coverage

    def audit_log(self, only_active: bool = False) -> list[tuple]:
        """Tam erişim kaydı. Geçersiz kılınanlar DA döner (silinmez, gizlenmez)."""
        sql = ("SELECT hypothesis_id, sharpe, passed, accessed_at, status, "
               "evaluator_version, invalidated_at, invalidation_reason "
               "FROM holdout_access")
        if only_active:
            sql += " WHERE status='active'"
        return self._audit.execute(sql + " ORDER BY id").fetchall()

    def close(self) -> None:
        self._audit.close()
