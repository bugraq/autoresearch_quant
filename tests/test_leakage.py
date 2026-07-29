"""
Sızıntı (leakage) testleri — Doküman 23.4 'leakage mutation testleri'.

Kasıtlı hatalı stratejileri validator'ın YAKALADIĞINI, geçerli stratejiyi
ise KABUL ettiğini doğrular. Bu, projenin en kritik güvencesi:
sızıntıyı 'test ederek' değil, 'ifade edilemez kılarak + eşitsizlikle' önlüyoruz.
"""
from contracts.decision import DecisionType
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism,
    Execution,
    Falsification,
    HypothesisFamily,
    HypothesisSpec,
    Portfolio,
    Universe,
)
from dsl import CompileError, compile_hypothesis, validate


def _hyp(signal: Expression, trade_time: str = "open_t_plus_1", **feat) -> HypothesisSpec:
    """Test için minimal ama geçerli bir HypothesisSpec kabuğu."""
    return HypothesisSpec(
        hypothesis_id="hyp_test",
        title="test",
        claim="test",
        family=HypothesisFamily.reversal,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"),
        features=feat.get("features", []),
        signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.1, short_quantile=0.1),
        execution=Execution(signal_time="close_t", trade_time=trade_time,
                            holding_period_days=5),
        falsification=Falsification(),
    )


def _cs_rank_of(inner: Expression) -> Expression:
    return Expression(op="cross_sectional_rank", inputs=[inner])


def test_valid_strategy_accepted():
    # negate(return(close,5)) @ close_t, işlem open_t+1 -> GEÇERLİ
    sig = _cs_rank_of(Expression(op="negate", inputs=[
        Expression(op="return", window=5, inputs=[Expression(op="field", field="close")])]))
    graph = compile_hypothesis(_hyp(sig, trade_time="open_t_plus_1"))
    dec = validate(graph, _hyp(sig, trade_time="open_t_plus_1"))
    assert dec.decision == DecisionType.accept, dec.issues
    print("  [ok] geçerli strateji kabul edildi")


def test_same_bar_execution_leak():
    # close_t sinyal + close_t execution -> SIZINTI (revise)
    sig = _cs_rank_of(Expression(op="return", window=5,
                                 inputs=[Expression(op="field", field="close")]))
    h = _hyp(sig, trade_time="close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.revise
    assert any(i.type == "temporal_leakage" for i in dec.issues)
    print("  [ok] close_t sinyal + close_t execution sızıntısı yakalandı")


def test_funding_rate_same_bar_execution_leak():
    """funding_rate 8 saatte bir (00/08/16 UTC) ödenir; günlük alan o günün
    ödemeler TOPLAMIDIR → ancak gün KAPANIŞINDA bilinir (FIELD_BASE_TICK=1).
    Aynı barda (close_t) işlem yapmak 16:00 ödemesini sabahtan bilmek olurdu."""
    sig = _cs_rank_of(Expression(op="field", field="funding_rate"))
    h = _hyp(sig, trade_time="close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.revise
    assert any(i.type == "temporal_leakage" for i in dec.issues), dec.issues
    print("  [ok] funding_rate close_t sızıntısı yakalandı")


def test_funding_rate_next_open_is_safe():
    """Aynı funding sinyali bir sonraki açılışta işlenirse GEÇERLİ."""
    sig = _cs_rank_of(Expression(op="field", field="funding_rate"))
    h = _hyp(sig, trade_time="open_t_plus_1")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.accept, dec.issues
    print("  [ok] funding_rate open_t+1 işlemi güvenli")


def test_execution_before_signal_leak():
    # sinyal close_t (tick 1), işlem open_t (tick 0) -> işlem sinyalden ÖNCE
    sig = _cs_rank_of(Expression(op="field", field="close"))
    h = _hyp(sig, trade_time="open_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.revise
    assert any(i.type == "temporal_leakage" for i in dec.issues)
    print("  [ok] işlemin sinyalden önce olması yakalandı")


def test_negative_window_rejected():
    # lag(close, -3) = 3 gün İLERİ bak -> negatif pencere reddi
    sig = _cs_rank_of(Expression(op="lag", window=-3,
                                 inputs=[Expression(op="field", field="close")]))
    h = _hyp(sig)
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.reject
    assert any(i.type == "invalid_parameter" for i in dec.issues)
    print("  [ok] negatif pencere (ileri bakış) reddedildi")


def test_lag_is_safe():
    # lag(close, 1) tick'i geriye çeker (close_{t-1}); close_t işlemde bile güvenli
    sig = _cs_rank_of(Expression(op="lag", window=1,
                                 inputs=[Expression(op="field", field="close")]))
    h = _hyp(sig, trade_time="close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.accept, dec.issues
    print("  [ok] lag geriye kaydırıyor, güvenli")


def test_unknown_operator_rejected():
    sig = Expression(op="magic_alpha", inputs=[Expression(op="field", field="close")])
    try:
        compile_hypothesis(_hyp(sig))
        raise AssertionError("bilinmeyen operatör derlendi")
    except CompileError:
        print("  [ok] bilinmeyen operatör derlemede reddedildi")


def test_unknown_field_rejected():
    # 'insider_tip' diye bir alan yok -> alternatif/izinsiz veri
    sig = _cs_rank_of(Expression(op="field", field="insider_tip"))
    try:
        compile_hypothesis(_hyp(sig))
        raise AssertionError("bilinmeyen alan derlendi")
    except CompileError:
        print("  [ok] izinsiz veri alanı derlemede reddedildi")


def test_disallowed_field_rejected():
    # Kampanya sadece 'close'a izin veriyor ama sinyal 'volume' kullanıyor -> red
    sig = _cs_rank_of(Expression(op="rolling_mean", window=20,
                                 inputs=[Expression(op="field", field="volume")]))
    h = _hyp(sig)
    dec = validate(compile_hypothesis(h), h, allowed_fields={"close"})
    assert dec.decision == DecisionType.reject
    assert any(i.type == "disallowed_field" for i in dec.issues)
    print("  [ok] izin verilmeyen veri alanı (campaign kısıtı) reddedildi")


def test_degenerate_conditional_rejected():
    # conditional'ın iki dalı aynı -> sahte koşullama (reward hacking) -> reddedilmeli
    same = Expression(op="field", field="close")
    cond = Expression(op="greater_than", inputs=[
        Expression(op="volatility", window=20, inputs=[Expression(op="field", field="close")]),
        Expression(op="const", value=0.02)])
    sig = _cs_rank_of(Expression(op="conditional", inputs=[cond, same, same]))
    h = _hyp(sig)
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.reject
    assert any(i.type == "degenerate_conditional" for i in dec.issues)
    print("  [ok] dejenere conditional (iki dalı aynı) reddedildi")


def test_excessive_complexity_rejected():
    # 45 iç içe negate -> karmaşıklık sınırı aşımı
    e: Expression = Expression(op="field", field="close")
    for _ in range(45):
        e = Expression(op="negate", inputs=[e])
    h = _hyp(_cs_rank_of(e))
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.reject
    assert any(i.type == "excessive_complexity" for i in dec.issues)
    print("  [ok] aşırı karmaşık strateji reddedildi")


# ---------------------------------------------------------------------------
# MODEL MODU — feature'lar da denetlenmeli (kapatilan acik)
# ---------------------------------------------------------------------------
# Acik neydi: model modunda strateji sinyali, sinyal IFADESINDEN degil modelin
# BUTUN feature'lardan urettigi tahminden gelir. Validator yalnizca signal
# dugumunun tick'ine bakiyordu; sinyal ifadesinin ATIF YAPMADIGI bir feature
# gec bilgi (close_t) tasiyip modele X olarak girebiliyor, islem de close_t'de
# yapilabiliyordu = klasik ayni-bar sizintisi, "TEMIZ" damgasiyla.


def _iki_featurelu(model_type: str, trade_time: str) -> HypothesisSpec:
    """features=[open_t (guvenli), close_t (gec)]; sinyal SADECE guvenliye atif yapar."""
    from contracts.dsl import NamedFeature
    from contracts.hypothesis_spec import ModelSpec
    guvenli = NamedFeature(name="guvenli", expression=Expression(
        op="return", window=5, inputs=[Expression(op="field", field="open")]))
    gec = NamedFeature(name="gec_close", expression=Expression(
        op="return", window=1, inputs=[Expression(op="field", field="close")]))
    h = _hyp(_cs_rank_of(Expression(op="feature_ref", name="guvenli")),
             trade_time=trade_time, features=[guvenli, gec])
    h.model = ModelSpec(type=model_type)
    return h


def test_model_modunda_gizli_gec_feature_yakalanir():
    h = _iki_featurelu("random_forest", "close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision != DecisionType.accept,         "Model modunda close_t feature'i + close_t islemi KABUL edildi (sizinti acigi geri geldi)"
    sizinti = [i for i in dec.issues if i.type == "temporal_leakage"]
    assert sizinti, f"temporal_leakage bulunamadi: {[i.type for i in dec.issues]}"
    assert "gec_close" in " ".join(i.description for i in sizinti),         "Hangi feature'in sizdirdigi soylenmiyor"
    print("  [ok] model modu: sinyalde GORUNMEYEN close_t feature'i yakalandi")


def test_model_modunda_bir_bar_sonra_islem_guvenli():
    """Yanlis pozitif olmamali: islem bir bar sonraysa close_t feature serbest."""
    h = _iki_featurelu("random_forest", "open_t_plus_1")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.accept,         f"gecerli model stratejisi reddedildi: {[i.type for i in dec.issues]}"
    print("  [ok] model modu: open_t+1 islemi ile close_t feature GUVENLI")


def test_dsl_formul_modunda_kullanilmayan_feature_sorun_degil():
    """dsl_formula'da model yok; sinyal agacinda olmayan feature hicbir seye girmez."""
    h = _iki_featurelu("dsl_formula", "close_t")
    dec = validate(compile_hypothesis(h), h)
    assert dec.decision == DecisionType.accept,         f"dsl_formula modunda yanlis pozitif: {[i.type for i in dec.issues]}"
    print("  [ok] dsl_formula: kullanilmayan feature yanlis alarm uretmiyor")


def main():
    test_valid_strategy_accepted()
    test_funding_rate_same_bar_execution_leak()
    test_funding_rate_next_open_is_safe()
    test_same_bar_execution_leak()
    test_execution_before_signal_leak()
    test_negative_window_rejected()
    test_lag_is_safe()
    test_unknown_operator_rejected()
    test_unknown_field_rejected()
    test_disallowed_field_rejected()
    test_degenerate_conditional_rejected()
    test_excessive_complexity_rejected()
    test_model_modunda_gizli_gec_feature_yakalanir()
    test_model_modunda_bir_bar_sonra_islem_guvenli()
    test_dsl_formul_modunda_kullanilmayan_feature_sorun_degil()
    print("OK — tüm sızıntı/geçerlilik testleri geçti.")


if __name__ == "__main__":
    main()
