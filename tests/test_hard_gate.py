"""
Hard gate testleri — ÖDÜL HACKLEME (reward hacking) kapıları.

Temel ilke: değerlendirilen taraf kendi geçme notunu yazamaz. LLM hipotezin
`falsification` alanlarını doldurur; bunlar ÖN KAYIT (pre-registration)
belgesidir, kabul kapısı DEĞİL. Kampanya eşiği TABANDIR — hipotezin kendi
taahhüdü eşiği yalnızca SIKILAŞTIRABİLİR.

Kapatılan açık: `minimum_positive_walk_forward_folds` kampanya eşiğinin
YERİNE geçiyordu (`hyp.… or min_positive_folds`). LLM buraya 0.1 yazınca,
5 dönemin 4'ünde para kaybedip birinde patlayan bir strateji KABUL alıyordu —
üstelik modülün kendi docstring'i "kabul kapısını LLM'e vermeyiz" diyordu.
"""
from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.decision import DecisionType
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from evaluation.hard_gate import evaluate

KAMPANYA_SHARPE = 0.5
KAMPANYA_FOLD = 0.5


def _hyp(fold_esigi=None, oos_sharpe: float = 0.5) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="h_gate", title="t", claim="c",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="t", description="d"),
        universe=Universe(source="sp500_point_in_time"), features=[],
        signal=Expression(op="cross_sectional_rank",
                          inputs=[Expression(op="field", field="close")]),
        portfolio=Portfolio(type="cross_sectional_long_short"),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification(
            minimum_oos_sharpe=oos_sharpe,
            minimum_positive_walk_forward_folds=fold_esigi))


def _res(pozitif_oran: float, fold_sharpes: list, dd: float = 0.05,
         turnover: float = 20.0) -> BacktestResult:
    return BacktestResult(
        hypothesis_id="h_gate",
        per_fold_metrics=[
            FoldMetrics(fold_id=f"f{i}", split="validation", sharpe=s,
                        annualized_return=0.1, volatility=0.1,
                        max_drawdown=dd, turnover=turnover)
            for i, s in enumerate(fold_sharpes)],
        exposures={"positive_fold_fraction": pozitif_oran})


# Tek dönemde patlayan, dördünde kaybeden strateji: ORTALAMA Sharpe yüksek
# (0.96, eşiği geçer) ama 5 fold'un yalnız 1'i pozitif -> tutarlılık eşiğine takılmalı.
_TUTARSIZ = (0.2, [5.0, -0.05, -0.05, -0.05, -0.05])
_TUTARLI = (0.8, [1.2, 1.0, 0.9, 1.1, -0.1])


def test_llm_gevsek_esik_yazarak_kapiyi_acamaz():
    """ASIL TEST: LLM'in gevşek beyanı kabul kapısını etkilemez."""
    res = _res(*_TUTARSIZ)
    assert res.aggregate_sharpe() > KAMPANYA_SHARPE, \
        "test kurgusu bozuk: Sharpe eşiği geçmiyor, fold kontrolü izole değil"
    dec = evaluate(res, _hyp(fold_esigi=0.1), KAMPANYA_SHARPE, KAMPANYA_FOLD)
    assert dec.decision == DecisionType.reject, (
        "LLM 'minimum_positive_walk_forward_folds: 0.1' yazarak kabul kapısını "
        "gevşetti — ödül hackleme açığı geri geldi.")
    assert any(i.type == "fold_inconsistency" for i in dec.issues)
    print("  [ok] gevşek LLM eşiği yok sayıldı, kampanya tabanı uygulandı")


def test_llm_daha_siki_esik_yazabilir():
    """Kendi kendini daha sıkı bağlamak SERBEST (ön kaydın anlamı budur)."""
    dec = evaluate(_res(*_TUTARLI), _hyp(fold_esigi=0.95),
                   KAMPANYA_SHARPE, KAMPANYA_FOLD)
    assert dec.decision == DecisionType.reject, \
        "hipotez %95 taahhüt etti, %80 tutturdu — kendi sözüne göre elenmeliydi"
    assert any("kendi" in (i.description or "") for i in dec.issues)
    print("  [ok] daha sıkı beyan UYGULANIYOR (sıkılaştırma serbest)")


def test_esik_beyan_edilmezse_kampanya_tabani():
    dec_kotu = evaluate(_res(*_TUTARSIZ), _hyp(), KAMPANYA_SHARPE, KAMPANYA_FOLD)
    dec_iyi = evaluate(_res(*_TUTARLI), _hyp(), KAMPANYA_SHARPE, KAMPANYA_FOLD)
    assert dec_kotu.decision == DecisionType.reject
    assert dec_iyi.decision == DecisionType.accept, \
        f"tutarlı strateji reddedildi: {[i.type for i in dec_iyi.issues]}"
    print("  [ok] beyan yoksa kampanya tabanı (%50) işliyor")


def test_gevsetme_girisimi_kayda_gecer():
    """Sessizce yok saymak denetlenemez olurdu — girişim not düşülür."""
    dec = evaluate(_res(*_TUTARLI), _hyp(fold_esigi=0.1),
                   KAMPANYA_SHARPE, KAMPANYA_FOLD)
    assert dec.decision == DecisionType.accept, "yanlış pozitif: tutarlı strateji elendi"
    assert any(i.type == "weaker_own_threshold" for i in dec.issues), \
        "gevşetme girişimi kayda geçmedi (iz sürülemez)"
    print("  [ok] gevşetme girişimi kabul kaydına not düşüldü")


def test_diger_kampanya_esikleri_llm_den_bagimsiz():
    """Sharpe / drawdown / turnover kapıları da yalnız kampanyadan gelir."""
    # Hipotez kendi Sharpe iddiasını 0.01'e çekiyor: kabul eşiği yine 0.5 olmalı.
    dusuk = _res(0.8, [0.3, 0.2, 0.25, 0.3, -0.1])
    dec = evaluate(dusuk, _hyp(oos_sharpe=0.01), KAMPANYA_SHARPE, KAMPANYA_FOLD)
    assert dec.decision == DecisionType.reject, "LLM Sharpe eşiğini gevşetebildi"
    assert any(i.type == "below_acceptance_sharpe" for i in dec.issues)

    # Aşırı drawdown ve turnover: kampanya sabitleri her hâlükârda uygular.
    dd = evaluate(_res(0.8, [1.2, 1.0, 0.9, 1.1, -0.1], dd=0.60), _hyp(),
                  KAMPANYA_SHARPE, KAMPANYA_FOLD, max_drawdown=0.40)
    assert any(i.type == "excessive_drawdown" for i in dd.issues)
    tn = evaluate(_res(0.8, [1.2, 1.0, 0.9, 1.1, -0.1], turnover=999.0), _hyp(),
                  KAMPANYA_SHARPE, KAMPANYA_FOLD, max_turnover=300.0)
    assert any(i.type == "excessive_turnover" for i in tn.issues)
    print("  [ok] Sharpe/drawdown/turnover kapıları LLM'den bağımsız")


def main() -> None:
    test_llm_gevsek_esik_yazarak_kapiyi_acamaz()
    test_llm_daha_siki_esik_yazabilir()
    test_esik_beyan_edilmezse_kampanya_tabani()
    test_gevsetme_girisimi_kayda_gecer()
    test_diger_kampanya_esikleri_llm_den_bagimsiz()
    print("OK — hard gate testleri geçti (değerlendirilen taraf notunu yazamaz).")


if __name__ == "__main__":
    main()
