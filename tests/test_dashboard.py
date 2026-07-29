"""
Dashboard smoke testi — mini bir hafızadan HTML üretiliyor mu.
"""
import os
import tempfile

from contracts.backtest_result import BacktestResult, FoldMetrics
from contracts.decision import Decision, DecisionSource, DecisionType
from contracts.dsl import Expression
from contracts.hypothesis_spec import (
    EconomicMechanism, Execution, Falsification, HypothesisFamily,
    HypothesisSpec, Portfolio, Universe,
)
from dashboard import generate_dashboard
from memory import MemoryStore


def _hyp() -> HypothesisSpec:
    sig = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60, inputs=[Expression(op="field", field="close")])])
    return HypothesisSpec(
        hypothesis_id="hyp_d1", title="60g momentum", claim="t",
        family=HypothesisFamily.momentum,
        economic_mechanism=EconomicMechanism(type="momentum", description="y"),
        universe=Universe(source="sp500_point_in_time"), features=[], signal=sig,
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.3, short_quantile=0.3),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def test_generate_dashboard():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "mem.sqlite")
        store = MemoryStore(db)
        result = BacktestResult(
            hypothesis_id="hyp_d1",
            per_fold_metrics=[FoldMetrics(fold_id="f0", split="research", sharpe=0.8,
                                          annualized_return=0.1, volatility=0.12,
                                          max_drawdown=0.09, turnover=5.0)],
            net_returns=[0.001, -0.002, 0.003] * 40)
        dec = Decision(hypothesis_id="hyp_d1", decision=DecisionType.accept,
                       source=DecisionSource.gate)
        store.record(_hyp(), dec, "accepted", result=result)
        store.close()

        out = os.path.join(d, "dash.html")
        generate_dashboard(db, os.path.join(d, "yok.sqlite"), out, campaign_name="test")
        assert os.path.exists(out)
        content = open(out, encoding="utf-8").read()
        for token in ["Araştırma Paneli", "Kabul Edilen Stratejiler",
                      "Araştırma Hunisi", "Çoklu Test", "Holdout", "hyp_d1"]:
            assert token in content, f"eksik bölüm: {token}"
        # ARASTIRMA SHARPE'I KANIT DEGIL: kabul tablosu uc-donem hukmunu de
        # gostermeli. Olculdu ki en yuksek arastirma Sharpe'li aday taze
        # veride EN SERT coken oldu; yalniz Sharpe siralamasi yaniltiyordu.
        assert "Üç-dönem hükmü" in content,             "kabul tablosunda hüküm sütunu yok — yalnız Sharpe sıralaması yanıltır"
        assert "KANIT DEĞİLDİR" in content,             "araştırma Sharpe'ının kanıt olmadığı uyarısı kayboldu"
        print("  [ok] dashboard tüm bölümlerle üretildi")


def test_elle_sonda_kampanya_adayindan_ayrilir():
    """Holdout audit'ine elle yazilmis sonda girerse BASLIK yaniltmamali.

    Gercek olay: kilitli donem audit'ine kampanya disi 4 sonda girmisti ve
    biri 'gecti'. Ayrim yapilmayinca dashboard "kilitli donemi 1/6 gecti"
    diyordu — oysa gecen sey sistemin bulgusu degil, insanin elle denedigi
    bir varyantti. Ayrim KANITA dayanir: hafizada kaydi olmayan kimlik
    kampanya urunu degildir.
    """
    import sqlite3

    from dashboard.report import _holdout_counts

    with tempfile.TemporaryDirectory() as d:
        mem = os.path.join(d, "mem.sqlite")
        store = MemoryStore(mem)
        res = BacktestResult(
            hypothesis_id="hyp_d1",
            per_fold_metrics=[FoldMetrics(fold_id="f0", split="research", sharpe=0.9,
                                          annualized_return=0.11, volatility=0.12,
                                          max_drawdown=0.15, turnover=40.0)],
            net_returns=[0.001, -0.0005, 0.0012] * 90)
        store.record(_hyp(), Decision(hypothesis_id="hyp_d1",
                                      decision=DecisionType.accept,
                                      source=DecisionSource.gate),
                     "accepted", result=res)
        store.close()

        hold = os.path.join(d, "hold.sqlite")
        c = sqlite3.connect(hold)
        c.execute("CREATE TABLE holdout_access (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                  " hypothesis_id TEXT UNIQUE, sharpe REAL, passed INTEGER,"
                  " accessed_at TEXT)")
        c.executemany("INSERT INTO holdout_access (hypothesis_id, sharpe, passed) "
                      "VALUES (?,?,?)",
                      [("hyp_d1", -0.4, 0),          # kampanya adayi: KALDI
                       ("probe_elle_1", 0.7, 1),     # elle sonda: gecti
                       ("probe_elle_2", -0.2, 0)])
        c.commit(); c.close()

        gecen, aday, s_gecen, s_aday = _holdout_counts(hold, mem)
        assert (gecen, aday) == (0, 1), f"kampanya sayimi yanlis: {gecen}/{aday}"
        assert (s_gecen, s_aday) == (1, 2), f"sonda sayimi yanlis: {s_gecen}/{s_aday}"

        out = generate_dashboard(mem, hold, os.path.join(d, "o.html"),
                                 campaign_name="t")
        html = open(out, encoding="utf-8").read()
        assert "elle sonda" in html, "sonda etiketi HTML'de yok"
        assert "sistemin bulgusu DEĞİLDİR" in html, "banner uyarisi yok"
        assert "KİLİTLİ DÖNEMDE ÇÖKTÜ" in html,             "kampanya adaylarinin hepsi kaldi ama baslik bunu soylemiyor"
        print("  [ok] elle sonda ayrildi: kampanya 0/1, sonda 1/2 — baslik dogru")


def test_hafiza_yoksa_ayrim_yapilmaz_ama_patlamaz():
    """Hafiza verilmezse (eski cagri) hepsi kampanya sayilir; cokme olmaz."""
    import sqlite3

    from dashboard.report import _holdout_counts
    with tempfile.TemporaryDirectory() as d:
        hold = os.path.join(d, "h.sqlite")
        c = sqlite3.connect(hold)
        c.execute("CREATE TABLE holdout_access (hypothesis_id TEXT, passed INTEGER)")
        c.execute("INSERT INTO holdout_access VALUES ('x', 1)")
        c.commit(); c.close()
        assert _holdout_counts(hold, None) == (1, 1, 0, 0)
        assert _holdout_counts(os.path.join(d, "yok.sqlite")) == (0, 0, 0, 0)
        print("  [ok] hafizasiz/dosyasiz cagri guvenli")


def test_metriksiz_kabul_dashboardu_cokertmez():
    """sharpe=None olan bir kabul kaydi TUM dashboard'i cokertmemeli.

    Gercek hata: _details, 'Sharpe {sharpe:.2f}' biciminde yaziyordu; metrigi
    olmayan tek bir kabul kaydi (geriye-donuk kayit / backfill sirasi / elle
    ekleme) TypeError firlatip dashboard.html'in TAMAMINI uretilemez yapiyordu.
    """
    with tempfile.TemporaryDirectory() as d:
        mem = os.path.join(d, "m.sqlite")
        store = MemoryStore(mem)
        store.record(_hyp(), Decision(hypothesis_id="hyp_d1",
                                      decision=DecisionType.accept,
                                      source=DecisionSource.gate),
                     "accepted")                      # result YOK -> sharpe None
        store.close()
        out = generate_dashboard(mem, os.path.join(d, "yok.sqlite"),
                                 os.path.join(d, "o.html"), campaign_name="t")
        html = open(out, encoding="utf-8").read()
        assert "Sharpe —" in html, "eksik metrik icin nazik gosterim yok"
        assert "hyp_d1" in html
        print("  [ok] metriksiz kabul: dashboard cokmedi, 'Sharpe —' basildi")


def main():
    test_generate_dashboard()
    test_metriksiz_kabul_dashboardu_cokertmez()
    test_elle_sonda_kampanya_adayindan_ayrilir()
    test_hafiza_yoksa_ayrim_yapilmaz_ama_patlamaz()
    print("OK — dashboard testi geçti.")


if __name__ == "__main__":
    main()
