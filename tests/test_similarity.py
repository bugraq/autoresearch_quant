"""
Benzerlik/yenilik kontrolü testleri (Doküman 14).
Aynı yapı -> yapısal duplicate; farklı yapı -> değil.
"""
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from memory.similarity import NoveltyIndex


def _hyp(hid: str, signal: Expression) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id=hid, title=hid, claim="t", family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="x", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=signal,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def _mom(w: int) -> Expression:
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=w, inputs=[Expression(op="field", field="close")])])


def test_identical_structure_is_duplicate():
    idx = NoveltyIndex()
    idx.add(_hyp("hyp_a", _mom(60)))
    assert idx.check_structural(_hyp("hyp_b", _mom(60))) == "hyp_a"
    print("  [ok] aynı yapı yapısal duplicate olarak yakalandı")


def test_different_window_not_duplicate():
    # 60g vs 5g: token 'return:60' vs 'return:5' farklı -> Jaccard < 0.90
    idx = NoveltyIndex()
    idx.add(_hyp("hyp_a", _mom(60)))
    assert idx.check_structural(_hyp("hyp_b", _mom(5))) is None
    print("  [ok] farklı pencere duplicate sayılmadı")


def test_first_is_never_duplicate():
    idx = NoveltyIndex()
    assert idx.check_structural(_hyp("hyp_a", _mom(60))) is None
    print("  [ok] ilk hipotez duplicate değil")


def test_structural_check_is_sign_blind():
    """NEDEN inversion modunda yapısal kontrol atlanıyor (loop 3a):
    büyük bir ağaca negate eklemek operatör çok-kümesini ~aynı bırakır;
    yapısal imza bir sinyalle TERSİNİ ayırt EDEMEZ. Bu test o körlüğü
    belgeler — davranış değişirse (imza işaret-duyarlı olursa) loop'taki
    atlama kaldırılabilir."""
    # Gerçekçi boyutta (LLM'in ürettiği gibi) çok-faktörlü ağaç: ~11 token.
    # Jaccard = n/(n+1) olduğundan körlük ancak n >= 10'da 0.90 eşiğini aşar —
    # gerçek koşudaki duplicate'ler de böyle büyük ağaçlardı.
    big = Expression(op="multiply", inputs=[
        Expression(op="subtract", inputs=[
            Expression(op="return", window=5, inputs=[
                Expression(op="field", field="close")]),
            Expression(op="rolling_mean", window=20, inputs=[
                Expression(op="field", field="close")])]),
        Expression(op="multiply", inputs=[
            Expression(op="zscore", window=60, inputs=[
                Expression(op="field", field="volume")]),
            Expression(op="volatility", window=20, inputs=[
                Expression(op="field", field="close")])])])
    sig = Expression(op="cross_sectional_rank", inputs=[big])
    inverted = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="negate", inputs=[big])])
    idx = NoveltyIndex()
    idx.add(_hyp("hyp_a", sig))
    assert idx.check_structural(_hyp("hyp_b", inverted)) == "hyp_a", \
        "yapısal imza işaret-duyarlı olmuş; loop'taki inversion atlaması gözden geçirilebilir"
    print("  [ok] yapısal imza işaret-körü (inversion'da atlanmasının gerekçesi)")


def test_behavioral_signed_correlation():
    """İşaretli davranışsal kontrol: tembel kopya (+corr) duplicate,
    gerçek inversion (-corr) YENİ bahis — inversion modunun asıl hakemi."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.normal(size=(200, 15)))
    idx = NoveltyIndex()
    idx.add(_hyp("hyp_a", _mom(60)), signal=sig)
    assert idx.check_behavioral(sig * 1.0001) == "hyp_a", "tembel kopya kaçtı"
    assert idx.check_behavioral(-sig) is None, "gerçek inversion duplicate sayıldı"
    print("  [ok] davranışsal kontrol işaretli: kopya yakalanır, inversion serbest")


def _funding_sig():
    """Momentum'dan yapısal olarak TAMAMEN farklı sinyal (funding tabanlı)."""
    return Expression(op="cross_sectional_rank", inputs=[
        Expression(op="negate", inputs=[Expression(op="field", field="funding_rate")])])


def test_originality_score_ranges():
    """Özgünlük skoru [0,1]: boş kütüphane=1, birebir=0, benzer<farklı.
    AlphaAgent-hizalı novelty-regularization'ın çekirdeği — binary duplicate
    eşiğinin ALTINDAKİ tekrarları da sürekli skorla görünür kılar."""
    idx = NoveltyIndex()
    # Boş kütüphane: her şey tam özgün
    assert idx.originality_score(_hyp("a", _mom(30))) == 1.0
    idx.add(_hyp("a", _mom(30)))
    # Birebir aynı yapı: özgünlük 0
    assert idx.originality_score(_hyp("b", _mom(30))) == 0.0
    # Benzer (momentum, farklı pencere) < Farklı (funding)
    sim = idx.originality_score(_hyp("c", _mom(20)))
    dif = idx.originality_score(_hyp("d", _funding_sig()))
    assert 0.0 < sim < dif <= 1.0, f"benzer {sim} < farklı {dif} olmalı"
    print(f"  [ok] özgünlük skoru: boş=1.0, birebir=0.0, benzer={sim:.2f} < farklı={dif:.2f}")


def test_nearest_identifies_closest():
    """nearest() en benzer faktörün hid'ini + benzerliğini döndürür (LLM'e
    'şuna benzeme' geri bildirimi bunun üstüne kurulacak)."""
    idx = NoveltyIndex()
    assert idx.nearest(_hyp("a", _mom(30))) == (None, 0.0), "boş kütüphane"
    idx.add(_hyp("mom", _mom(30)))
    idx.add(_hyp("fund", _funding_sig()))
    hid, sim = idx.nearest(_hyp("q", _mom(25)))   # momentum sorgusu -> mom'a yakın
    assert hid == "mom" and sim > 0, f"en yakın mom olmalı, geldi: {hid} {sim}"
    print(f"  [ok] nearest: momentum sorgusu -> '{hid}' (benzerlik {sim:.2f})")


def main():
    test_identical_structure_is_duplicate()
    test_different_window_not_duplicate()
    test_first_is_never_duplicate()
    test_structural_check_is_sign_blind()
    test_behavioral_signed_correlation()
    test_originality_score_ranges()
    test_nearest_identifies_closest()
    print("OK — benzerlik/yenilik kontrolü çalışıyor.")


if __name__ == "__main__":
    main()
