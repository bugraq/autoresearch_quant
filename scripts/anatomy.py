"""
TEK HİPOTEZ ANATOMİSİ — her adım açık, hiçbir şey gizli değil (hoca talebi, 22.07.2026).

Bir (1) hipotezi doğuşundan kararına kadar takip eder ve HER ADIMI yazdırır:

  0. Kampanya bağlamı (veri, evren, kısıtlar)
  1. LLM'e giden TAM PROMPT (system + user) — LLM tam olarak ne gördü?
  2. LLM'in HAM CEVABI (JSON) + token/süre
  3. Hipotez CÜMLE olarak: iddia, ekonomik mekanizma, yanlışlama koşulu
  4. Metin -> YAPI: HypothesisSpec alan alan
  5. Yapı -> GRAF: derlenmiş StrategyGraph, düğüm düğüm
  6. SIZINTI DENETİMİ: info_tick eşitsizliği açık açık
  7. Graf -> SAYI: her düğümün ürettiği panel (boyut + örnek satırlar + istatistik)
  8. MODEL EĞİTİMİ: X/y nedir, nerede, hangi tarihlerde, embargo nasıl
  9. Sinyal -> AĞIRLIK -> PnL: tek gün üzerinde açık hesap
 10. METRİKLER: fold fold + toplam (Sharpe, win rate, P&L...)
 11. KARAR: hard gate kriterleri tek tek

Kullanım:
    .venv/Scripts/python.exe scripts/anatomy.py              # gerçek LLM ile
    .venv/Scripts/python.exe scripts/anatomy.py --canned     # LLM çağırmadan (bedava)
    .venv/Scripts/python.exe scripts/anatomy.py --log        # runs/anatomy.log'a da yaz

Not: --canned modunda 1-2. adımlarda prompt YİNE gösterilir (gerçek prompt'tur),
yalnızca LLM çağrısı yapılmaz; hazır bir örnek hipotez kullanılır.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)

COST_BPS = 5.0
_LOG_LINES: list[str] = []
_WRITE_LOG = False


# ==========================================================================
# Yazdırma yardımcıları
# ==========================================================================
def P(s: str = "") -> None:
    print(s)
    if _WRITE_LOG:
        _LOG_LINES.append(s)


def step(no: str, title: str) -> None:
    P("")
    P("╔" + "═" * 76 + "╗")
    P(f"║ ADIM {no}: {title}".ljust(77) + "║")
    P("╚" + "═" * 76 + "╝")


def sub(title: str) -> None:
    P("")
    P(f"--- {title} " + "-" * max(0, 72 - len(title)))


def box(text: str, width: int = 76) -> None:
    """Uzun metni çerçeve içinde, satır kaydırarak yazdır."""
    import textwrap
    for para in text.split("\n"):
        for line in (textwrap.wrap(para, width - 4) or [""]):
            P(f"  | {line}")


def panel_stats(name: str, df: pd.DataFrame, n_rows: int = 4) -> None:
    """Bir sayısal panelin (tarih x varlık) kimliğini bas: boyut, örnek, istatistik."""
    vals = df.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    dolu = len(finite) / vals.size * 100 if vals.size else 0.0
    P(f"  {name}")
    P(f"    boyut     : {df.shape[0]} tarih x {df.shape[1]} varlik "
      f"({vals.size:,} hucre, %{dolu:.1f} dolu)")
    if len(finite):
        P(f"    istatistik: min={finite.min():+.6f}  ort={finite.mean():+.6f}  "
          f"max={finite.max():+.6f}  std={finite.std():.6f}")
    else:
        P("    istatistik: TAMAMEN BOS (NaN) - bu feature hicbir sey uretmiyor!")
    ilk_dolu = df.dropna(how="all")
    if len(ilk_dolu):
        P(f"    ilk dolu satir: {ilk_dolu.index[0].date()} "
          f"(ilk {len(df) - len(ilk_dolu)} satir NaN = pencere isiniyor)")
        # Boş sütun göstermenin anlamı yok: en dolu 6 varlığı seç.
        son = ilk_dolu.iloc[-n_rows:]
        kolon = son.notna().sum().sort_values(ascending=False).head(6).index
        P(f"    ornek (son {n_rows} bar, en dolu 6 varlik):")
        for line in son[kolon].round(6).to_string().split("\n"):
            P(f"      {line}")


# ==========================================================================
# ADIM 0 — Bağlam
# ==========================================================================
def setup(use_llm: bool):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, ".env"))

    import main as M
    campaign = M.load_yaml("campaign.yaml")["campaign"]
    models = M.load_yaml("models.yaml")["models"]
    data_cfg = M.load_yaml("data.yaml")["data"]
    cfg = M.build_config(campaign)

    step("0", "KAMPANYA BAGLAMI — sistem neyle calisiyor?")
    P(f"\n  Kampanya       : {campaign['name']}")
    P(f"  Amac           : {cfg.goal}")
    P(f"  Veri kaynagi   : {data_cfg.get('source')}")
    P(f"  Tarih araligi  : {campaign.get('start_date')} -> {campaign.get('end_date')}")
    P(f"  Sabit model    : {campaign.get('model', 'dsl_formula')}")
    P(f"  Uretici LLM    : {models['hypothesis_generator'].get('model', '?')}")
    P(f"  Butce          : {cfg.max_experiments} deney")

    P("\n  LLM'in gorebilecegi ALANLAR (allowed_fields):")
    P(f"    {', '.join(cfg.allowed_fields)}")
    P("\n  LLM'in kullanabilecegi OPERATORLER (allowed_operators):")
    P(f"    {', '.join(cfg.allowed_operators)}")
    P("\n  >> LLM bunlarin DISINA cikamaz. Serbest Python YAZAMAZ; yalnizca bu")
    P("     operatorlerden olusan tipli bir agac (DSL) tanimlar.")

    P("\n  Veri yukleniyor...")
    t0 = time.time()
    data, holdout = M.load_data(campaign, data_cfg, cfg.research_fraction)
    P(f"  [ok] {time.time() - t0:.1f} sn")
    P(f"    arastirma dilimi: {len(data.dates)} bar x "
      f"{len(data.get('close').columns)} varlik  ({data.dates[0].date()} -> {data.dates[-1].date()})")
    P(f"    KILITLI holdout : {len(holdout.dates)} bar  "
      f"({holdout.dates[0].date()} -> {holdout.dates[-1].date()})  <- bu kosuda ASLA kullanilmaz")
    P(f"    yilliklastirma  : {data.bars_per_year} bar/yil")
    P(f"    mevcut alanlar  : {', '.join(sorted(data.fields.keys()))}")
    return campaign, models, cfg, data


# ==========================================================================
# ADIM 1-3 — LLM
# ==========================================================================
def generate(campaign, models, cfg, use_llm: bool):
    from agents.hypothesis_generator import _build_system_prompt, _build_user_prompt
    from agents.literature import load_literature_mechanisms
    from memory.store import MemoryStore
    from orchestrator.loop import GenerationMode, _build_context

    literature = load_literature_mechanisms(
        domain=str(campaign.get("literature_domain", "equity")))
    memory = MemoryStore(os.path.join(HERE, "research_memory.sqlite"))
    ctx = _build_context(cfg, memory, cfg.max_experiments,
                         GenerationMode.new, None, literature=literature)

    step("1", "LLM'E GIDEN TAM PROMPT — model tam olarak ne gordu?")
    system = _build_system_prompt(ctx)
    user = _build_user_prompt(ctx)
    P(f"\n  [SYSTEM PROMPT]  ({len(system):,} karakter)")
    box(system)
    P(f"\n  [USER PROMPT]  ({len(user):,} karakter)")
    box(user)
    P(f"\n  >> Toplam ~{(len(system) + len(user)) // 4:,} token gidiyor.")
    if cfg.anonymize_universe:
        P("  >> DIKKAT: anonymize_universe=true -> prompt'ta HIC ticker/tarih YOK.")
        P("     Sebep: LLM egitim verisinden 'hangi hisse ne zaman yukseldi' hatirlayip")
        P("     parametre icine look-ahead sizdirabilir. Anonimlestirme bunu keser.")

    step("2", "LLM'IN HAM CEVABI")
    if use_llm:
        from llm.providers import make_provider
        provider = make_provider(models["hypothesis_generator"])
        P(f"\n  Model cagriliyor: {provider.model} (temperature={provider.temperature})")
        t0 = time.time()
        resp = provider.client.chat(provider.model, system, user,
                                    temperature=provider.temperature,
                                    max_tokens=provider.max_tokens)
        dt = time.time() - t0
        P(f"  [ok] {dt:.1f} sn | prompt={resp.prompt_tokens} tok, "
          f"cevap={resp.completion_tokens} tok | model={resp.model}")
        P("\n  [HAM CEVAP]")
        box(resp.text)
        # ONARIM YOLU (gercek dongude de var): LLM bazen bozuk/eksik JSON dondurur.
        # Sistem hatayi LLM'e GERI besleyip bir kez duzeltme ister. Bu adim
        # gorunur olmali - "LLM her zaman dogru uretiyor" izlenimi yanlis olur.
        try:
            hyp = provider._parse(resp.text, "hyp_anatomi")
        except Exception as e:  # noqa: BLE001
            P("\n  !! LLM'IN CIKTISI GECERSIZ — sema dogrulamasi basarisiz:")
            box(str(e)[:800])
            P("\n  >> ONARIM: hata LLM'e geri besleniyor, bir kez duzeltme isteniyor")
            P("     (dusuk sicaklik = 0.2). Gercek kampanyada da aynen boyle olur.")
            repair = (f"{user}\n\nÖnceki çıktın geçersizdi. Hata: {e}\n"
                      f"Şemaya ve operatör aritelerine birebir uyan, "
                      f"SADECE geçerli JSON döndür.")
            t0 = time.time()
            resp = provider.client.chat(provider.model, system, repair,
                                        temperature=0.2, max_tokens=provider.max_tokens)
            P(f"\n  [ok] onarim cevabi {time.time() - t0:.1f} sn | "
              f"{resp.completion_tokens} tok")
            P("\n  [ONARILMIS CEVAP]")
            box(resp.text)
            hyp = provider._parse(resp.text, "hyp_anatomi")
            P("\n  >> Onarim basarili. (Bu da basarisiz olsaydi deney "
              "'compile_error' olarak KAYDEDILIR, sessizce yutulmazdi.)")
    else:
        P("\n  (--canned) LLM cagrilmadi; hazir ornek hipotez kullaniliyor.")
        P("  Yukaridaki prompt GERCEKTIR - kosarken LLM'e giden tam metin budur.")
        hyp = _canned_hypothesis(campaign)
        P("\n  [ORNEK CEVAP - LLM boyle bir JSON dondurur]")
        box(json.dumps(json.loads(hyp.model_dump_json()), ensure_ascii=False, indent=2)[:2500])

    step("3", "HIPOTEZ — CUMLE OLARAK (insan ne okuyor?)")
    P(f"\n  Kimlik    : {hyp.hypothesis_id}")
    P(f"  Baslik    : {hyp.title}")
    P(f"  Aile      : {hyp.family.value if hasattr(hyp.family, 'value') else hyp.family}")
    P("\n  IDDIA:")
    box(hyp.claim)
    P("\n  EKONOMIK MEKANIZMA (neden calismasi bekleniyor?):")
    P(f"    tip: {hyp.economic_mechanism.type}")
    box(hyp.economic_mechanism.description)
    P("\n  YANLISLAMA KOSULU (hangi durumda 'yanlis' diyecegiz?):")
    f = hyp.falsification
    P(f"    minimum Sharpe            : {getattr(f, 'minimum_sharpe', '-')}")
    P(f"    minimum pozitif fold orani: {getattr(f, 'minimum_positive_walk_forward_folds', '-')}")
    P("\n  >> Ekonomik mekanizma ZORUNLU alan. Gerekce yazamayan hipotez sema")
    P("     dogrulamasindan gecemez — 'veri madenciligi' hipotezleri boylece elenir.")
    return hyp


def _canned_hypothesis(campaign):
    """LLM'siz mod icin gercekci bir ornek (LLM'in urettiginin aynisi yapida).

    Kampanyanin SABIT modelini kullanir: model dsl_formula degilse feature'lar
    tanimlanir ve ADIM 8'de gercek ML egitim yolu gosterilir.
    """
    from contracts.dsl import Expression, NamedFeature
    from contracts.hypothesis_spec import (
        EconomicMechanism, Execution, Falsification, HypothesisFamily,
        HypothesisSpec, ModelSpec, Portfolio, Universe,
    )
    model_type = str(campaign.get("model", "dsl_formula"))
    ret5 = Expression(op="return", window=5,
                      inputs=[Expression(op="field", field="close")])
    feats = [] if model_type == "dsl_formula" else [
        NamedFeature(name="getiri_5bar", expression=ret5),
        NamedFeature(name="oynaklik_20bar",
                     expression=Expression(op="volatility", window=20,
                                           inputs=[Expression(op="field", field="close")])),
        NamedFeature(name="hacim_z",
                     expression=Expression(op="zscore", window=20,
                                           inputs=[Expression(op="field",
                                                              field="dollar_volume")])),
    ]
    return HypothesisSpec(
        hypothesis_id="hyp_anatomi",
        title="Kisa vadeli asiri tepki geri doner",
        claim=("Son 5 gunde kesitsel olarak en cok yukselen varliklar, takip eden "
               "gunlerde diger varliklara gore geride kalir; cunku kisa vadeli "
               "hareket likidite talebinden kaynaklanir ve geri doner."),
        family=HypothesisFamily.reversal,
        economic_mechanism=EconomicMechanism(
            type="liquidity_provision",
            description=("Aceleci alicilar likidite talep eder; piyasa yapicilar bu "
                         "talebi karsilamak icin envanter riski alir ve karsiliginda "
                         "prim ister. Bu prim, fiyatin geri donmesiyle odenir.")),
        universe=Universe(source=campaign.get("universe", "sp500_point_in_time")),
        features=feats,
        model=ModelSpec(type=model_type),
        signal=Expression(op="cross_sectional_rank", inputs=[
            Expression(op="negate", inputs=[ret5])]),
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=0.2, short_quantile=0.2),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=5),
        falsification=Falsification())


# ==========================================================================
# ADIM 4-6 — Metin -> yapı -> graf -> sızıntı denetimi
# ==========================================================================
def to_graph(hyp):
    from dsl import compile_hypothesis, validate

    step("4", "METIN -> YAPI: HypothesisSpec alan alan")
    P("\n  LLM'in cumlesi burada TIPLI bir nesneye donustu. Artik belirsizlik yok:")
    P(f"\n  universe.source         : {hyp.universe.source}")
    P(f"  portfolio.type          : {hyp.portfolio.type}")
    P(f"  portfolio.weighting     : {hyp.portfolio.weighting}")
    P(f"  portfolio long/short    : %{(hyp.portfolio.long_quantile or 0)*100:.0f} / "
      f"%{(hyp.portfolio.short_quantile or 0)*100:.0f}")
    P(f"  execution.signal_time   : {hyp.execution.signal_time}")
    P(f"  execution.trade_time    : {hyp.execution.trade_time}")
    P(f"  execution.rebalance     : {hyp.execution.rebalance}")
    P(f"  execution.holding_period: {hyp.execution.holding_period_days} bar")
    P(f"  model.type              : {hyp.model.type}")
    P("\n  SINYAL IFADESI (ic ice operatorler):")
    box(json.dumps(json.loads(hyp.signal.model_dump_json(exclude_none=True)),
                   ensure_ascii=False, indent=2))

    step("5", "YAPI -> GRAF: derleme (compile_hypothesis)")
    t0 = time.time()
    graph = compile_hypothesis(hyp)
    P(f"\n  [ok] derlendi ({time.time() - t0:.3f} sn) — {len(graph.nodes)} dugum")
    P("\n  Her dugum bir islem. 'max_info_time' = bu degerin EN ERKEN ne zaman")
    P("  bilinebilecegi. Sizinti kontrolu tamamen bu etikete dayanir.\n")
    P(f"  {'dugum':<26} {'operator':<30} {'girdi':<26} {'bilinme ani':<14}")
    P(f"  {'-'*26} {'-'*30} {'-'*26} {'-'*14}")
    for n in graph.nodes:
        prm = {k: v for k, v in n.params.items() if not k.startswith("_")}
        op = n.op + (f" {prm}" if prm else "")
        P(f"  {n.node_id[:26]:<26} {op[:30]:<30} {','.join(n.input_ids)[:26]:<26} "
          f"{n.max_info_time:<14}")
    P(f"\n  sinyal dugumu : {graph.signal_node_id}")
    if graph.feature_node_ids:
        P(f"  feature dugumleri: {graph.feature_node_ids}")

    step("6", "SIZINTI DENETIMI — strateji gelecegi goruyor mu?")
    nodes = {n.node_id: n for n in graph.nodes}
    sig = nodes[graph.signal_node_id]
    sig_tick = int(sig.params.get("_info_tick", 0))
    from dsl.operators import parse_time_token
    trade_tick = parse_time_token(hyp.execution.trade_time)
    P("\n  Kural:  sinyalin bilinme ani  <  islemin yapildigi an")
    P(f"\n    sinyal '{sig.max_info_time}'  -> info_tick = {sig_tick}")
    P(f"    islem  '{hyp.execution.trade_time}' -> info_tick = {trade_tick}")
    P(f"\n    {sig_tick} < {trade_tick} ?  -> {'EVET, temiz' if sig_tick < trade_tick else 'HAYIR — SIZINTI!'}")
    decision = validate(graph, hyp)
    P(f"\n  Denetim karari: {decision.decision.value.upper()}")
    for i in decision.issues:
        P(f"    - [{i.type}] {i.description}")
    if not decision.issues:
        P("    (sorun yok)")
    P("\n  >> Bu kapi GECILMEZSE backtest HIC CALISTIRILMAZ. Yanlis bir sayi,")
    P("     sayi yoklugundan daha kotudur.")
    return graph


# ==========================================================================
# ADIM 7-8 — Sayısallaşma + model eğitimi
# ==========================================================================
def to_numbers(graph, hyp, data):
    from backtest.evaluator import _eval_node
    from backtest.model_signal import compute_signal, forward_return

    step("7", "GRAF -> SAYI: cumle burada tabloya donusuyor")
    P("\n  Her dugum sirayla calisir; ciktisi bir PANEL'dir (tarih x varlik).")
    P("  Asagida her dugumun urettigi tablonun kimligi var.\n")
    vals: dict = {}
    for n in graph.nodes:
        vals[n.node_id] = _eval_node(n, vals, data)
        v = vals[n.node_id]
        prm = {k: v2 for k, v2 in n.params.items() if not k.startswith("_")}
        etiket = f"[{n.node_id}] {n.op}" + (f" {prm}" if prm else "")
        if isinstance(v, pd.DataFrame):
            panel_stats(etiket, v)
        else:
            P(f"  {etiket}\n    (skaler/sabit) = {v}")
        P("")

    P("  >> ISTE 'sayisala donusme' burada oldu: LLM'in cumlesi, yukaridaki")
    P("     dugumler zinciriyle bir sayi tablosuna indi. Son dugumun tablosu = SINYAL.")

    step("8", "MODEL EGITIMI — feature/target ne, nerede egitiliyor?")
    if hyp.model.type == "dsl_formula":
        P("\n  model.type = dsl_formula  -> EGITILEN MODEL YOK.")
        P("  Sinyal dogrudan yukaridaki DSL ifadesinin ciktisidir (formul).")
        P("  Ogrenilen parametre olmadigi icin asiri-uyum (overfit) riski dusuktur;")
        P("  buna karsilik model kutusu (RF/GB) kullanilmaz.")
        signal = compute_signal(graph, hyp, data)
    else:
        h = max(1, int(hyp.execution.holding_period_days or 1))
        P(f"\n  model.type = {hyp.model.type}  -> ML/istatistiksel model EGITILIYOR.")
        P("\n  FEATURE'LAR (X)  : yukaridaki 'feature' dugumlerinin panelleri")
        P(f"    {list(graph.feature_node_ids.keys())}")
        P(f"\n  TARGET (y)       : ileriki getiri, close(t+{h})/close(t) - 1")
        P(f"    holding_period_days = {h}  ->  model {h} bar sonrasini tahmin ediyor")
        P("\n  >> Model 'bu varlik yukselir mi' DEGIL, 'hangi varliklar digerlerine")
        P("     gore daha iyi olacak' sorusunu cevaplar (kesitsel). Cikti surekli")
        P("     bir sayidir; entry/take-profit/stop-loss DEGILDIR.")

        fwd = forward_return(data, h)
        P("\n  TARGET paneli:")
        panel_stats("  y = ileriki getiri", fwd)

        P("\n  EGITIM NEREDE? -> backtest/model_signal.py :: evaluate_model_signal()")
        P("  Zaman 6 bloga bolunur. Blok i, YALNIZCA kendinden ONCEKI veriyle")
        P("  egitilmis modelle tahmin edilir (purged walk-forward + embargo):\n")
        dates = data.get("close").index
        blocks = np.array_split(np.arange(len(dates)), 6)
        P(f"  {'blok':<6} {'test basi':<12} {'test sonu':<12} {'egitim bitis':<14} {'embargo':<10}")
        P(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*14} {'-'*10}")
        for i, blk in enumerate(blocks):
            if not len(blk):
                continue
            emb_i = max(0, blk[0] - h)
            P(f"  {i:<6} {str(dates[blk[0]].date()):<12} {str(dates[blk[-1]].date()):<12} "
              f"{str(dates[emb_i].date()):<14} {h} bar")
        P(f"\n  >> EMBARGO ({h} bar): hedefi test donemine SARKAN son ornekler")
        P("     egitimden DUSULUR. Yoksa model, tahmin edecegi donemin getirisini")
        P("     dolayli olarak egitimde gormus olurdu — klasik sizinti.")
        P("  >> Ilk blok(lar) egitim verisi yetersiz oldugu icin NaN kalir = ISLEM YOK.")

        P("\n  Model egitiliyor...")
        t0 = time.time()
        signal = compute_signal(graph, hyp, data)
        P(f"  [ok] {time.time() - t0:.1f} sn")

    P("\n  URETILEN SINYAL (= tum zincirin ciktisi):")
    panel_stats("  sinyal", signal)
    return signal


# ==========================================================================
# ADIM 9-11 — PnL, metrikler, karar
# ==========================================================================
def to_pnl(hyp, data, signal, cfg):
    from backtest.engine import (
        _build_weights, _execution_prices, _universe_mask, compute_pnl, fold_metrics)
    from dsl.operators import parse_time_token

    step("9", "SINYAL -> AGIRLIK -> PnL: tek gun uzerinde acik hesap")
    masked = _universe_mask(hyp, data, signal)
    weights = _build_weights(masked, hyp.portfolio)
    trade_tick = parse_time_token(hyp.execution.trade_time)
    bar_offset, phase = trade_tick // 2, ("open" if trade_tick % 2 == 0 else "close")
    exec_px = _execution_prices(data, phase)
    exec_ret = exec_px.pct_change(fill_method=None)

    P(f"\n  1) Evren filtresi  : sinyal, uygun olmayan varliklarda NaN yapildi")
    P(f"  2) Agirliklandirma : {hyp.portfolio.type}, {hyp.portfolio.weighting}, "
      f"gross={hyp.portfolio.gross_exposure}")
    P(f"  3) Islem hizasi    : {hyp.execution.trade_time} -> weights.shift({bar_offset+1})")

    valid = weights[(weights != 0).any(axis=1)].index
    if not len(valid):
        P("\n  (Hicbir gunde pozisyon yok — sinyal bos.)")
        return None, None
    t = valid[len(valid) // 2]
    ti = data.dates.get_loc(t)
    if ti + bar_offset + 1 >= len(data.dates):
        t = valid[len(valid) // 3]
        ti = data.dates.get_loc(t)
    t1, t2 = data.dates[ti + bar_offset], data.dates[ti + bar_offset + 1]

    # 8h barda t+1 ve t+2 AYNI takvim gunune dusebilir; sutun adi cakisip biri
    # kaybolmasin diye tam zaman damgasi kullanilir.
    def _et(ts) -> str:
        return ts.strftime("%Y-%m-%d") if len(data.dates) and \
            getattr(data, "bars_per_year", 252) <= 366 else ts.strftime("%m-%d %H:%M")

    sub(f"Ornek bar: t = {t}")
    tbl = pd.DataFrame({
        "sinyal_t": masked.loc[t],
        "agirlik w_t": weights.loc[t],
        f"{phase} t+{bar_offset} [{_et(t1)}]": exec_px.loc[t1],
        f"{phase} t+{bar_offset+1} [{_et(t2)}]": exec_px.loc[t2],
        "getiri r": exec_ret.loc[t2],
    })
    tbl["katki = w*r"] = tbl["agirlik w_t"] * tbl["getiri r"]
    tbl = tbl[tbl["agirlik w_t"] != 0]
    for line in tbl.head(12).round(6).to_string().split("\n"):
        P(f"  {line}")
    if len(tbl) > 12:
        P(f"  ... ({len(tbl)} pozisyondan ilk 12'si)")
    brut = float(tbl["katki = w*r"].sum())
    turn = float((weights - weights.shift(1)).abs().sum(axis=1).loc[t])
    P(f"\n  Brut PnL (sutun toplami) = {brut:+.8f}")
    P(f"  Devir (turnover)         = {turn:.6f}")
    P(f"  Maliyet = devir*{COST_BPS:.0f}bps    = {turn * COST_BPS / 1e4:.8f}")
    P(f"  NET PnL                  = {brut - turn * COST_BPS / 1e4:+.8f}")

    step("10", "METRIKLER — sayilar nereden geliyor?")
    net_pnl, turnover_t = compute_pnl(signal, hyp, data, COST_BPS)
    bpy = data.bars_per_year
    P(f"\n  Net getiri serisi: {len(net_pnl)} bar")
    P(f"  Yilliklastirma   : sqrt({bpy}) = {np.sqrt(bpy):.4f}\n")

    from backtest.walk_forward import run_walk_forward
    from dsl import compile_hypothesis
    res = run_walk_forward(compile_hypothesis(hyp), hyp, data,
                           n_folds=5, cost_bps=COST_BPS, signal=signal)
    P(f"  {'fold':<8} {'Sharpe':>9} {'yillik':>9} {'MaxDD':>8} {'winrate':>9} {'P&L':>9}")
    P(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*9}")
    for m in res.per_fold_metrics:
        wr = f"%{m.hit_rate*100:.1f}" if m.hit_rate is not None else "-"
        pl = f"%{m.total_return*100:+.1f}" if m.total_return is not None else "-"
        P(f"  {m.fold_id:<8} {m.sharpe:>+9.3f} {m.annualized_return:>+9.3f} "
          f"{m.max_drawdown:>8.3f} {wr:>9} {pl:>9}")
    P(f"\n  ORTALAMA Sharpe = {res.aggregate_sharpe():+.4f}")

    sub("Sharpe ELLE dogrulama (motor gercekten dogru mu?)")
    x = net_pnl.to_numpy(dtype=float)
    mean = float(x.sum() / len(x))
    std = float(np.sqrt(sum((xi - mean) ** 2 for xi in x) / (len(x) - 1)))
    P(f"  ortalama = {mean:+.10f}")
    P(f"  std(ddof=1) = {std:.10f}")
    P(f"  Sharpe = ortalama/std*sqrt({bpy}) = {mean/std*np.sqrt(bpy):+.6f}")
    fm = fold_metrics(net_pnl, turnover_t, "tum", "verify", bars_per_year=bpy)
    P(f"  motor (tum donem)                = {fm.sharpe:+.6f}")
    P(f"  FARK = {abs(fm.sharpe - mean/std*np.sqrt(bpy)):.2e}   [OK]")

    sub("Tahmin kalitesi (Sharpe'tan BAGIMSIZ olcu)")
    e = res.exposures
    P(f"  IC     = {e.get('ic', 0):+.4f}   (sinyal ile gerceklesen getirinin kesitsel korelasyonu)")
    P(f"  RankIC = {e.get('rank_ic', 0):+.4f}   (siralama korelasyonu, aykirilara dayanikli)")
    P(f"  ICIR   = {e.get('icir', 0):+.4f}   (IC'nin kararliligi)")
    P(f"  yon isabeti = %{e.get('dir_acc', 0.5)*100:.1f}   (baseline %50)")
    P("\n  >> Yuksek Sharpe + SIFIR IC = sans isareti. Ikisi birlikte okunmali.")

    step("11", "KARAR — hard gate kriterleri tek tek")
    from evaluation.hard_gate import HARD_MAX_DRAWDOWN, HARD_MAX_TURNOVER, evaluate
    dec = evaluate(res, hyp, min_acceptance_sharpe=cfg.min_acceptance_sharpe,
                   min_positive_folds=cfg.min_positive_folds)
    worst_dd = max((m.max_drawdown for m in res.per_fold_metrics), default=0.0)
    max_turn = max((m.turnover for m in res.per_fold_metrics), default=0.0)
    pos = e.get("positive_fold_fraction", 0.0)
    rows = [
        ("Sharpe", res.aggregate_sharpe(), ">=", cfg.min_acceptance_sharpe),
        ("pozitif fold orani", pos, ">=", cfg.min_positive_folds),
        ("en kotu MaxDD", worst_dd, "<=", HARD_MAX_DRAWDOWN),
        ("en yuksek turnover", max_turn, "<=", HARD_MAX_TURNOVER),
    ]
    P(f"\n  {'kriter':<22} {'deger':>10} {'':>3} {'esik':>8}   sonuc")
    P(f"  {'-'*22} {'-'*10} {'-'*3} {'-'*8}   -----")
    for ad, deg, op, esik in rows:
        ok = (deg >= esik) if op == ">=" else (deg <= esik)
        P(f"  {ad:<22} {deg:>10.3f} {op:>3} {esik:>8.3f}   {'GECTI' if ok else 'KALDI'}")
    P(f"\n  KARAR: {dec.decision.value.upper()}")
    for i in dec.issues:
        P(f"    - [{i.type}] {i.description}")
    P("\n  >> 'accept' burada BITMEZ. Kabul edilen aday sonra sirasiyla:")
    P("     robustness (permutasyon, maliyet 2x) -> coklu-test (DSR + BH-FDR)")
    P("     -> ve ancak kampanya bitince KILITLI HOLDOUT'ta TEK ATIS denenir.")
    return res, net_pnl


# ==========================================================================
# SADE MOD — "bu konuyu hic bilmeyen anlasin" (hoca talebi)
# Teknik detay/JSON/boyut YOK. Her adim duz Turkce, her terim aciklanir.
# ==========================================================================
_FAMILY_TR = {
    "momentum": "kazanan kazanmaya devam eder (momentum)",
    "reversal": "asiri hareket geri doner (ortalamaya donus)",
    "volume": "islem hacmi yon gosterir",
    "volatility": "oynaklik bilgi tasir",
    "liquidity": "likidite/alim-satim kolayligi onemli",
    "cross_sectional_interaction": "iki etkenin birlesimi (etkilesim)",
    "regime_conditioned": "piyasa rejimine gore degisen kural",
    "composite": "birden fazla sinyalin birlesimi",
}


def _n(title: str) -> None:
    P("")
    P("─" * 78)
    P(f"  {title}")
    P("─" * 78)


def _sade_strateji_cumlesi(hyp) -> str:
    """Hipotezi tek cumlelik duz Turkce stratejiye cevir (dashboard ile ayni dil)."""
    try:
        import re
        from dashboard.report import _plain_strategy
        txt = _plain_strategy(json.loads(hyp.model_dump_json()))
        return re.sub(r"</?b>", "", txt)   # HTML vurgu etiketlerini temizle
    except Exception:  # noqa: BLE001
        return "(sade ceviri uretilemedi)"


def narrate(campaign, models, cfg, use_llm: bool):
    """Tek hipotezi SADE dille, adim adim anlat. Teknik ciktinin yalinlastirilmisi."""
    import main as M
    from dsl import compile_hypothesis, validate
    from backtest.model_signal import compute_signal
    from backtest.walk_forward import run_walk_forward
    from evaluation.hard_gate import evaluate

    P("\n" + "═" * 78)
    P("  BIR YATIRIM FIKRI NASIL SINANIR? — bastan sona, sade anlatim")
    P("═" * 78)
    P("""
  Asagida yapay zekanin urettigi TEK bir yatirim fikrini, dogdugu andan
  'ise yarar mi' kararina kadar adim adim izleyecegiz. Amac: borsa/istatistik
  bilmeyen birinin bile her adimi anlamasi. Teknik terimler geçtikçe hemen
  yaninda ne demek oldugunu yazacagiz.""")

    # --- Hazirlik (sessiz) ---
    _n("HAZIRLIK: elimizde ne var?")
    campaign_ = M.load_yaml("campaign.yaml")["campaign"]
    data, holdout = M.load_data(campaign_, M.load_yaml("data.yaml")["data"],
                                cfg.research_fraction)
    yil = f"{campaign_.get('start_date')} — {campaign_.get('end_date')}"
    n_varlik = len(data.get("close").columns)
    P(f"""
  Piyasa      : {campaign_.get('universe', '?')} (kripto vadeli islemler)
  Kac varlik  : {n_varlik} coin
  Tarih       : {yil}
  Veri        : fiyat (acilis/kapanis), islem hacmi, funding (asagida acikliyoruz)

  ONEMLI: veriyi IKIYE boluyoruz.
    • Bir kismiyla fikri gelistirip test edecegiz (arastirma verisi).
    • Diger kismina HIC DOKUNMUYORUZ ({len(holdout.dates)} bar kilitli kasada).
      Neden? Bir ogrenciye sinav sorularini onceden verirseniz notu sahte olur.
      O kilitli kismi yalnizca en SONDA, tek sefer aciyoruz — gercek sinav.""")

    # --- ADIM 1: fikir ---
    _n("ADIM 1 / 6  —  Yapay zeka bir fikir uretiyor")
    hyp = generate_quiet(campaign, models, cfg, use_llm, data)
    aile = hyp.family.value if hasattr(hyp.family, "value") else str(hyp.family)
    P(f"""
  Yapay zekaya soruyoruz: "Bu piyasada fiyatlarin yonunu tahmin edebilecek
  bir oruntu bul, ve NEDEN ise yarayacagini ekonomik olarak acikla."

  Onemli kisit: yapay zeka cani ne isterse yapamaz. Ona bir 'lego seti'
  veriyoruz — sadece izin verdigimiz yapi taslariyla (ortalama, siralama,
  degisim orani gibi) fikir kurabiliyor. Boylece urettigi her fikri bilgisayar
  birebir, hatasiz calistirabiliyor.

  URETTIGI FIKIR:
""")
    P(f"    ┌{'─'*70}┐")
    for ln in _wrap(hyp.title, 66):
        P(f"    │ {ln:<68} │")
    P(f"    ├{'─'*70}┤")
    P(f"    │ {'Ne turden bir fikir:':<68} │")
    for ln in _wrap(_FAMILY_TR.get(aile, aile), 66):
        P(f"    │   {ln:<66} │")
    P(f"    │ {' ':<68} │")
    P(f"    │ {'Ne iddia ediyor:':<68} │")
    for ln in _wrap(hyp.claim, 66):
        P(f"    │   {ln:<66} │")
    P(f"    │ {' ':<68} │")
    P(f"    │ {'Neden calismali (ekonomik gerekce):':<68} │")
    for ln in _wrap(hyp.economic_mechanism.description, 66):
        P(f"    │   {ln:<66} │")
    P(f"    └{'─'*70}┘")
    P(f"""
  DUZ TURKCE OZET:
    {_sade_strateji_cumlesi(hyp)}

  >> Dikkat: yapay zeka bir GEREKCE yazmak zorunda. "Su yuzden calisir"
     diyemeyen fikir daha en bastan eleniyor. Boylece 'rastgele denedim,
     tuttu' turu bos fikirleri en basta atiyoruz.""")

    # --- ADIM 2: modele cevirme + sizinti ---
    _n("ADIM 2 / 6  —  Fikri bilgisayarin diline ceviriyoruz")
    graph = compile_hypothesis(hyp)
    dec = validate(graph, hyp)
    temiz = dec.decision.value == "accept"
    P(f"""
  Fikir bir cumleydi. Simdi onu bilgisayarin adim adim isleyecegi bir
  islem zincirine ceviriyoruz ({len(graph.nodes)} kucuk islem).

  EN KRITIK KONTROL — 'gelecege bakma' (sizinti):
    Bir stratejinin en sinsi hatasi, farkinda olmadan GELECEGI kullanmasidir.
    Ornek: "bugun al" derken yanlislikla yarinin fiyatini hesaba katmak.
    Gercekte imkansiz olan bu bilgi, testte muhtesem ama sahte kar uretir.

    Biz her islemin 'ne zaman bilinebilecegini' etiketliyoruz ve sunu
    matematiksel olarak kontrol ediyoruz:
        sinyalin bilindigi an  <  islemin yapildigi an  ?

    Bu fikir icin sonuc: {'✓ TEMIZ — gelecege bakmiyor.' if temiz else '✗ SIZINTI VAR!'}
  {'' if temiz else '  (Sizintili fikir burada DURDURULUR, hic test edilmez.)'}
  >> Bu kapiyi gecemeyen fikir backtest'e ALINMAZ. Yanlis bir sonuc,
     sonucsuzluktan daha tehlikelidir.""")

    # --- ADIM 3: sayiya dokme + egitim ---
    _n("ADIM 3 / 6  —  Fikir rakamlara donusuyor (ve model ogreniyor)")
    signal = compute_signal(graph, hyp, data)
    if hyp.model.type == "dsl_formula":
        P(f"""
  Fikrin her varlik icin bir 'puan' uretmesini istiyoruz. Yukaridaki islem
  zinciri, her gun her coin icin bir sayi hesapliyor — yuksek puan 'yukselir',
  dusuk puan 'duser' demek.

  Bu fikirde OGRENEN bir model yok; puan dogrudan formulden geliyor.
  (Avantaji: ezberleme/asiri-uyum riski dusuk.)""")
    else:
        h = max(1, int(hyp.execution.holding_period_days or 1))
        P(f"""
  Bu fikirde bir MODEL var ({_model_tr(hyp.model.type)}) — yani gecmisten
  ders cikaran bir algoritma. Nasil ogreniyor?

    • GORDUGU (girdi)    : {len(graph.feature_node_ids)} gosterge — {', '.join(list(graph.feature_node_ids.keys())[:4])}
    • TAHMIN ETTIGI (hedef): {h} bar sonraki getiri (fiyat yukseldi mi, ne kadar?)

  KRITIK: model SADECE gecmisle egitilir, gelecegi asla gormez. Zamani
  dilimlere boluyoruz; her dilimde model YALNIZCA kendinden ONCEKI gunlerle
  egitilir, sonra hic gormedigi gunlerde sinanir. Aradaki 'bosluk' (embargo),
  egitimin test donemine sizmasini engeller.

  >> Model 'su coin yukselir mi' demiyor; 'bugun HANGI coinler digerlerinden
     daha iyi olacak' diye siraliyor. Cikti bir puan — al/sat seviyesi degil.""")
    P("""
  Kritik ayrim: model bir PUAN/SIRALAMA uretiyor; "su fiyattan al, sunda
  kar-al, sunda zarar-kes" gibi hazir emir DEGIL. Kimi al, kimi sat kararini
  bir sonraki adimdaki portfoy kurallari veriyor.""")

    # --- ADIM 4: para ile test ---
    _n("ADIM 4 / 6  —  Gecmis veride PARA ile deniyoruz (backtest)")
    res = run_walk_forward(graph, hyp, data, n_folds=5, cost_bps=COST_BPS, signal=signal)
    lq = int((hyp.portfolio.long_quantile or 0.2) * 100)
    P(f"""
  'Backtest' = fikri gecmise goturup, sanki o gunlerde gercekten para
  koymusuz gibi gun gun ne olurdu hesaplamak.

  Kurallar:
    • Her gun en iyi %{lq} coini AL (long), en kotu %{lq}'i ac-sat (short).
    • Al-sat MASRAFI dusulur (komisyon/spread) — bedava islem yok.
    • Bugunun bilgisiyle karar verilir, YARIN islem yapilir (gerceklik boyle).

  Ayrica tek bir donemde degil, {len(res.per_fold_metrics)} AYRI donemde ayri ayri
  test ediyoruz. Neden? Bir fikir sadece 2021'de tuttuysa sanstir; asil iyi
  fikir farkli donemlerin cogunda calisir.""")

    # --- ADIM 5: karne ---
    _n("ADIM 5 / 6  —  Karne: fikir ne kadar iyi?")
    sharpe = res.aggregate_sharpe() or 0.0
    folds = res.per_fold_metrics
    wr = _avg_sade(m.hit_rate for m in folds)
    pl = _compound_sade(m.total_return for m in folds)
    pos = res.exposures.get("positive_fold_fraction", 0.0)
    ic = res.exposures.get("ic", 0.0)
    P(f"""
  SHARPE ORANI = {sharpe:+.2f}
     Ne demek: "aldigin risk basina ne kadar kazandin". Kazanci, inis-cikisin
     (riskin) buyuklugune bolen bir not. Yuksek = ayni riske daha cok kazanc.
     Kaba olcek: 1'in ustu iyi, 2'nin ustu cok iyi, 0'in alti para kaybi.

  KAZANMA ORANI (win rate) = %{wr*100:.0f}
     Ne demek: islem yapilan gunlerin yuzde kaci artida kapandi.
     Not: %50'nin alti da kar edebilir (az kazandiran cok gun + cok kazandiran
     az gun). O yuzden tek basina yeterli degil, Sharpe ile birlikte okunur.

  TOPLAM GETIRI (P&L) = %{pl*100:+.0f}
     Ne demek: bu fikre bu donemde para koysaydin, elindeki para % kac degisirdi.

  TUTARLILIK = {len(folds)} donemin %{pos*100:.0f}'i artida
     Ne demek: fikir donemler arasi ne kadar istikrarli. Hepsi arti = saglam.

  ONGORU GUCU (IC) = {ic:+.3f}
     Ne demek: modelin tahmini ile gercekte olan ne kadar ortusuyor.
     ~0 ise: yuksek Sharpe bile olsa SANS olabilir. Sifirdan belirgin
     farkli olmasi, isin sansla degil gercek ongoruyle oldugunun isareti.""")

    # --- Sharpe elle dogrulama (hoca: gercekten dogru mu?) ---
    from backtest.engine import compute_pnl, fold_metrics
    net_pnl, turnover_t = compute_pnl(signal, hyp, data, COST_BPS)
    bpy = data.bars_per_year
    x = net_pnl.to_numpy(dtype=float)
    mean = float(x.sum() / len(x))
    std = float(np.sqrt(sum((xi - mean) ** 2 for xi in x) / (len(x) - 1)))
    elle = mean / std * np.sqrt(bpy)
    motor = fold_metrics(net_pnl, turnover_t, "t", "v", bars_per_year=bpy).sharpe
    P(f"""
  "Bu Sharpe rakami dogru mu, uydurma mi?" — elle kontrol:
     Gunluk ortalama kazanc  = {mean:+.6f}
     Gunluk inis-cikis (std) = {std:.6f}
     Sharpe = ortalama / inis-cikis × √{bpy} = {elle:+.3f}
     Programin buldugu        = {motor:+.3f}   → {'AYNI ✓' if abs(elle-motor) < 1e-6 else 'FARKLI!'}
     (Ayrica Excel'de de dogrulanabilir: scripts/verify_sharpe.py bir tablo uretir.)""")

    # --- ADIM 6: karar ---
    _n("ADIM 6 / 6  —  Karar")
    gate = evaluate(res, hyp, min_acceptance_sharpe=cfg.min_acceptance_sharpe,
                    min_positive_folds=cfg.min_positive_folds)
    kabul = gate.decision.value == "accept"
    P(f"""
  Fikir, onceden belirlenmis (ve yapay zekanin degistiremedigi) esikleri
  gecti mi? Sharpe yeterli mi, farkli donemlerde tutarli mi, kayiplar
  makul mu, cok mu sik islem yapiyor?

  BU FIKRIN KARARI:  {'✓ KABUL' if kabul else '✗ RED'}""")
    if not kabul:
        for i in gate.issues:
            P(f"     • {_issue_tr(i.type)}")
    P(f"""
  {'AMA is burada BITMEZ:' if kabul else 'Peki kabul edilseydi ne olurdu?'}
  Kabul edilen bir fikir daha bitmis sayilmaz. Sirasiyla:
     1) Saglamlik testleri (masraf 2 kati olsa? veriyi karistirsak? hala tutar mi)
     2) Sans elemesi (yuzlerce fikir denenince biri sirf sansla parlak gorunur;
        istatistik bunu ayiklar — 'coklu test duzeltmesi')
     3) VE en son: kilitli kasadaki o hic dokunulmamis donem — tek sefer, gercek sinav.

  Iste projenin farki bu: cogu sistem Adim 5'teki parlak Sharpe'i gorup
  "buldum!" der. Biz onun sansmi gercek mi oldugunu elemeden inanmiyoruz.""")

    P("\n" + "═" * 78)
    P("  BITTI. Bir fikir dogdu, gerekce buldu, rakama dondu, parayla sinandi,")
    P("  ve dogrulukları elle teyit edilerek bir karara baglandi.")
    P("═" * 78 + "\n")
    return hyp, res


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width) or [""]


def _model_tr(t: str) -> str:
    return {
        "linear_regression": "dogrusal regresyon — en basit ogrenen model",
        "ridge": "ridge regresyon — asiri-uyuma karsi frenli dogrusal model",
        "naive_bayes": "naive Bayes — olasilik tabanli siniflandirici",
        "random_forest": "rastgele orman — cok sayida karar agacinin ortalamasi",
        "gradient_boosting": "gradyan artirma — hatayi adim adim duzelten agaclar",
    }.get(t, t)


def _issue_tr(t: str) -> str:
    return {
        "below_acceptance_sharpe": "Getiri/risk (Sharpe) esigin altinda kaldi",
        "excessive_drawdown": "Kayiplar (dususler) fazla derin",
        "excessive_turnover": "Cok sik islem yapiyor (masraf yer)",
        "fold_inconsistency": "Donemler arasi tutarsiz — bir donem tutmus, otekiler tutmamis",
    }.get(t, t)


def _avg_sade(vals):
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _compound_sade(vals):
    acc = 1.0
    for v in vals:
        if v is not None:
            acc *= (1.0 + v)
    return acc - 1.0


def generate_quiet(campaign, models, cfg, use_llm: bool, data):
    """LLM'den hipotez al ama SESSIZ (sade mod icin). Onarim yolu dahil."""
    from agents.hypothesis_generator import _build_system_prompt, _build_user_prompt
    from agents.literature import load_literature_mechanisms
    from memory.store import MemoryStore
    from orchestrator.loop import GenerationMode, _build_context

    if not use_llm:
        return _canned_hypothesis(campaign)

    lit = load_literature_mechanisms(domain=str(campaign.get("literature_domain", "equity")))
    memory = MemoryStore(os.path.join(HERE, "research_memory.sqlite"))
    ctx = _build_context(cfg, memory, cfg.max_experiments,
                         GenerationMode.new, None, literature=lit)
    from llm.providers import make_provider
    provider = make_provider(models["hypothesis_generator"])
    system = _build_system_prompt(ctx)
    user = _build_user_prompt(ctx)
    P("\n  (Yapay zeka dusunuyor, ~1 dakika surebilir...)")
    resp = provider.client.chat(provider.model, system, user,
                                temperature=provider.temperature,
                                max_tokens=provider.max_tokens)
    try:
        return provider._parse(resp.text, "hyp_anatomi")
    except Exception as e:  # noqa: BLE001
        repair = (f"{user}\n\nÖnceki çıktın geçersizdi. Hata: {e}\n"
                  f"SADECE şemaya uyan geçerli JSON döndür.")
        resp = provider.client.chat(provider.model, system, repair,
                                    temperature=0.2, max_tokens=provider.max_tokens)
        return provider._parse(resp.text, "hyp_anatomi")


def main() -> None:
    global _WRITE_LOG
    ap = argparse.ArgumentParser(description="Tek hipotezin bastan sona anatomisi")
    ap.add_argument("--canned", action="store_true",
                    help="LLM cagirma (bedava); hazir ornek hipotez kullan. "
                         "Prompt yine gercek prompttur.")
    ap.add_argument("--sade", action="store_true",
                    help="SADE anlatim: teknik detay/JSON yok; konuyu bilmeyenin "
                         "anlayacagi duz Turkce, adim adim.")
    ap.add_argument("--log", action="store_true",
                    help="Ciktiyi runs/anatomy.log dosyasina da yaz.")
    args = ap.parse_args()
    _WRITE_LOG = args.log

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if args.sade:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, ".env"))
        import main as M
        campaign = M.load_yaml("campaign.yaml")["campaign"]
        models = M.load_yaml("models.yaml")["models"]
        cfg = M.build_config(campaign)
        narrate(campaign, models, cfg, use_llm=not args.canned)
    else:
        P("\n" + "#" * 78)
        P("#  TEK HIPOTEZ ANATOMISI — dogusundan kararina, her adim acik")
        P("#" * 78)
        campaign, models, cfg, data = setup(not args.canned)
        hyp = generate(campaign, models, cfg, use_llm=not args.canned)
        graph = to_graph(hyp)
        signal = to_numbers(graph, hyp, data)
        to_pnl(hyp, data, signal, cfg)
        P("\n" + "#" * 78)
        P("#  BITTI — hipotez dogdu, sayiya dondu, test edildi, karara baglandi.")
        P("#" * 78 + "\n")

    if _WRITE_LOG:
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        fname = "anatomy_sade.log" if args.sade else "anatomy.log"
        path = os.path.join(HERE, "runs", fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG_LINES))
        print(f"Log yazildi: {path}")


if __name__ == "__main__":
    main()
