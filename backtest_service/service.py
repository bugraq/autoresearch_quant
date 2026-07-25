"""
BACKTEST SERVİSİ — bağımsız, model-agnostik servis sınırı.

Hoca tarifi (15.07.2026): "Bir modeli alıp (bu formül de olabilir, istatistiksel
veya ML modeli de olabilir), bununla backtest yapabileceğimiz servisi geliştir."

Bu modül o servistir. TEK çağrı:

    from backtest_service import run
    rapor = run(spec, data)          # spec: model + features + portföy + execution
    print(rapor.summary())

Servis, çağıranın compiler/graph/validator/walk-forward bilmesini GEREKTİRMEZ.
LLM'i, orchestrator'ı, hafızayı import ETMEZ → tek başına kullanılabilir
(başka bir projeden de çağrılabilir; "servis servis git" ilkesi).

DESTEKLENEN MODELLER (hepsi aynı arayüzden):
    formül        : dsl_formula        (sinyal = DSL ifadesi)
    istatistiksel : linear_regression, ridge, naive_bayes
    ML            : random_forest, gradient_boosting

BU SERVİSİ PİYASADAKİ MOTORLARDAN AYIRAN ŞEY (tarama raporundaki boşluk):
  1) SIZINTILI STRATEJİYİ ÇALIŞTIRMAYI REDDEDER (LeakageError). Hazır motorların
     hiçbiri bunu yapmıyor; backtest'i çalıştırıp yanlış sonucu döndürüyorlar.
  2) Model modunda eğitim walk-forward + EMBARGO ile yapılır (purged) → modelin
     geleceği görmesi yapısal olarak imkânsız.
  3) Tahmin KALİTESİ (IC/RankIC/ICIR/directional accuracy) ile strateji
     PERFORMANSI (Sharpe/DD/turnover) AYRI raporlanır — yüksek Sharpe + sıfır IC
     = şans işareti.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from contracts.backtest_result import BacktestResult
from contracts.decision import DecisionType
from contracts.hypothesis_spec import HypothesisSpec
from data.synthetic import MarketData
from dsl import CompileError, compile_hypothesis, validate
from backtest.walk_forward import run_walk_forward


class BacktestServiceError(Exception):
    """Servis stratejiyi çalıştıramadı (tanım hatalı ya da güvensiz)."""


class LeakageError(BacktestServiceError):
    """Strateji geleceği görüyor (look-ahead). Sonuç ÜRETİLMEZ — bilinçli."""


@dataclass
class BacktestReport:
    """Servisin çıktısı: tahmin kalitesi + strateji performansı + köken bilgisi."""

    hypothesis_id: str
    model_type: str

    # --- Tahmin kalitesi (modelin öngörü gücü; Sharpe'tan BAĞIMSIZ) ---
    ic: float
    rank_ic: float
    icir: float
    directional_accuracy: float

    # --- Strateji performansı (işlem maliyeti SONRASI) ---
    sharpe: Optional[float]
    annualized_return: Optional[float]
    max_drawdown: Optional[float]
    turnover: Optional[float]
    hit_rate: Optional[float]        # win rate — pozitif bar oranı
    total_return: Optional[float]    # birikimli P&L (fold'lar bileşik)
    positive_fold_fraction: float
    n_folds: int

    # --- Köken (tekrar üretilebilirlik) ---
    engine_version: str
    data_version: str
    seed: int
    raw: BacktestResult   # tam sonuç (getiri serisi, fold detayları, maliyet dökümü)

    def summary(self) -> str:
        sh = f"{self.sharpe:+.2f}" if self.sharpe is not None else "-"
        dd = f"%{self.max_drawdown*100:.0f}" if self.max_drawdown is not None else "-"
        to = f"{self.turnover:.0f}" if self.turnover is not None else "-"
        wr = f"%{self.hit_rate*100:.1f}" if self.hit_rate is not None else "-"
        pl = f"%{self.total_return*100:+.1f}" if self.total_return is not None else "-"
        return (
            f"[{self.hypothesis_id}] model={self.model_type}\n"
            f"  Tahmin kalitesi : IC={self.ic:+.3f}  RankIC={self.rank_ic:+.3f}  "
            f"ICIR={self.icir:+.2f}  yön isabeti=%{self.directional_accuracy*100:.1f}\n"
            f"  Performans      : Sharpe={sh}  MaxDD={dd}  turnover={to}  "
            f"pozitif fold={self.positive_fold_fraction:.0%} ({self.n_folds} fold)\n"
            f"  Kazanc          : win rate={wr}  P&L (birikimli)={pl}\n"
            f"  Köken           : {self.engine_version} / {self.data_version} / seed={self.seed}"
        )


def run(spec: HypothesisSpec, data: MarketData, *, cost_bps: float = 5.0,
        n_folds: int = 5, seed: int = 42,
        enforce_leakage_check: bool = True) -> BacktestReport:
    """Model + veri -> backtest raporu. Servisin TEK giriş noktası.

    spec.model.type ne olursa olsun (formül / istatistiksel / ML) aynı çağrı.

    Raises:
        LeakageError: strateji geleceği görüyor (enforce_leakage_check=True iken).
        BacktestServiceError: strateji derlenemedi / tanımı geçersiz.
    """
    # 1) Derle (deterministik: aynı spec -> aynı graph)
    try:
        graph = compile_hypothesis(spec)
    except CompileError as e:
        raise BacktestServiceError(f"Strateji derlenemedi: {e}") from e

    # 2) GÜVENLİK KAPISI — servisin sözleşmesi: YALNIZCA geçerli stratejiyi koştur.
    # Denetimden geçmeyene sayı ÜRETMEYİZ; yanlış bir sayı, sayı yokluğundan kötüdür.
    # NOT: validator sızıntıyı 'revise' der (execution kaydırılarak düzeltilebilir),
    # 'reject' demez — bu yüzden karar seviyesine değil, SORUN TİPİNE bakılır.
    if enforce_leakage_check:
        decision = validate(graph, spec)
        if decision.decision != DecisionType.accept:
            reasons = "; ".join(f"{i.type}: {i.description}" for i in decision.issues)
            if any(i.type == "temporal_leakage" for i in decision.issues):
                raise LeakageError(
                    f"SIZINTI — strateji geleceği görüyor; backtest ÇALIŞTIRILMADI. {reasons}")
            raise BacktestServiceError(
                f"Strateji denetimden geçemedi — backtest ÇALIŞTIRILMADI. {reasons}")

    # 3) Walk-forward backtest (model modunda: geçmişe fit, geleceğe tahmin)
    result = run_walk_forward(graph, spec, data, n_folds=n_folds,
                              cost_bps=cost_bps, seed=seed)
    return _to_report(spec, result)


def _avg(vals) -> Optional[float]:
    """None'ları atlayarak ortalama; hiç değer yoksa None."""
    xs = [v for v in vals if v is not None]
    return float(sum(xs) / len(xs)) if xs else None


def _compound(vals) -> Optional[float]:
    """Ardışık dönem getirilerini bileşikle: prod(1+r) - 1."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    acc = 1.0
    for v in xs:
        acc *= (1.0 + v)
    return float(acc - 1.0)


def _to_report(spec: HypothesisSpec, r: BacktestResult) -> BacktestReport:
    e = r.exposures
    folds = r.per_fold_metrics
    return BacktestReport(
        hypothesis_id=r.hypothesis_id,
        model_type=e.get("model_type", spec.model.type),
        ic=float(e.get("ic", 0.0)),
        rank_ic=float(e.get("rank_ic", 0.0)),
        icir=float(e.get("icir", 0.0)),
        directional_accuracy=float(e.get("dir_acc", 0.5)),
        sharpe=r.aggregate_sharpe(),
        annualized_return=(sum(m.annualized_return for m in folds) / len(folds)
                           if folds else None),
        max_drawdown=(max((m.max_drawdown for m in folds), default=None)),
        turnover=(max((m.turnover for m in folds), default=None)),
        hit_rate=_avg(m.hit_rate for m in folds),
        # Fold'lar ardışık dönemler -> birikimli P&L bileşiktir, ortalama değil.
        total_return=_compound(m.total_return for m in folds),
        positive_fold_fraction=float(e.get("positive_fold_fraction", 0.0)),
        n_folds=int(e.get("n_folds", len(folds))),
        engine_version=r.engine_version,
        data_version=r.data_version,
        seed=r.seed,
        raw=r,
    )
