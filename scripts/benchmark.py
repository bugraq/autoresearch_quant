"""
KIYAS: bizim yaklasim RASTGELE al-satciyi ve PASIF al-tutcuyu geciyor mu?
(hoca talebi, 22.07.2026)

Hocanin basari tanimi:
  "Su an icin bir basari alpha bulmak DEGIL; parayi rastgele al-sat yapan
   birine veya psikolojik davranan birine gore YUKARIDA bulmak da bir basari."

Ayni evren, ayni tarih, ayni islem maliyeti ve ayni portfoy kurallariyla dort
seyi yaristirir:

  1) AL-TUT (pasif/piyasa)    : tum varliklari esit agirlikla al ve TUT.
  2) RASTGELE AL-SATCI (maymun): her bar rastgele long/short. Tek maymun sansli
                                 olabilir; N maymun kosup DAGILIMINI cikaririz.
  3) PSIKOLOJIK (duygusal)     : dunku kazanani kovalayan naif trend-takipcisi.
  4) BIZIM STRATEJI            : en guclu KANITA sahip aday (evaluation/aday.py).

═══════════════════════════════════════════════════════════════════════════
NEDEN HER DONEMDE AYRI OLCULUYOR — bu betikte duzeltilen ASIL HATA
═══════════════════════════════════════════════════════════════════════════
Eskiden kiyas YALNIZCA ARASTIRMA doneminde kosuyordu (load_data(...)[0]).
Ama arastirma donemi, stratejinin SECILDIGI donemdir: aday zaten "orada iyi
gorundugu icin" kabul edilmistir. Orada maymunu gecmesi kacinilmazdir ve
hicbir sey kanitlamaz — kendi sinavini kendi yazmak gibidir.

Hocanin sorusu ("rastgele al-sat yapandan iyi miyiz?") GERCEK DUNYA sorusudur;
cevabi yalnizca ORNEKLEM-DISI donemden gelir. Bu yuzden yaris her donemde
ayri kosuluyor ve sonuc bir MATRIS olarak basiliyor:

    donem x rakip  ->  geciyor muyuz?

Olculdu (29.07.2026): arastirmada 3/3 rakibi geciyorduk; ayni aday kilitli
donemde al-tut'un ALTINA dustu. Tek tablo bu farki gizliyordu.

ISINMA: arastirma disindaki her donem, KENDINDEN ONCEKI veriyle isitilir.
Aksi halde rolling pencereler donemin basinda NaN kalir ve ML modeli donemin
ICINDE yeniden egitilir — yani kabul edilen model degil BASKA bir model
olculur (holdout/service.py'da ayni hata duzeltilmisti).

Kullanim:
    .venv/Scripts/python.exe scripts/benchmark.py
    .venv/Scripts/python.exe scripts/benchmark.py --monkeys 500 --log
    .venv/Scripts/python.exe scripts/benchmark.py --ileri     # taze donemi de kat

NOT: Kripto evreninde "SPY" yoktur; piyasa proxy'si = evrenin esit-agirlikli
sepeti (ayni evren -> en durust karsilastirma).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# MALIYET KAMPANYADAN OKUNUR (configs/campaign.yaml -> budget.cost_bps).
# Sabit 5.0 yazmak, aktif kripto kampanyasi 10.0 kullanirken bu betigi
# YARIM maliyetle kosturuyordu: ayni hipotez kampanyada baska, burada
# baska (daha iyimser) Sharpe gosteriyordu. Config yoksa 5.0 varsayilir.
from evaluation.plain import kampanya_cost_bps
COST_BPS = kampanya_cost_bps(5.0)
_LOG: list[str] = []
_WRITE = False


def P(s: str = "") -> None:
    print(s)
    if _WRITE:
        _LOG.append(s)


def _sharpe(net_pnl: pd.Series, bpy: int) -> float:
    x = net_pnl.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(bpy))


def _total_return(net_pnl: pd.Series) -> float:
    x = net_pnl.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.prod(1.0 + x) - 1.0) if len(x) else 0.0


def _max_dd(net_pnl: pd.Series) -> float:
    x = net_pnl.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.0
    eq = np.cumprod(1.0 + x)
    return float(-(eq / np.maximum.accumulate(eq) - 1.0).min())


def _stats(net_pnl: pd.Series, bpy: int) -> dict:
    return {"sharpe": _sharpe(net_pnl, bpy),
            "yillik": float(net_pnl.mean() * bpy),
            "toplam": _total_return(net_pnl),
            "maxdd": _max_dd(net_pnl)}


# --------------------------------------------------------------------------
# Yarismacilar — hepsi AYNI motordan (compute_pnl) gecer: adil
# --------------------------------------------------------------------------
def _template_hyp(universe_source: str, long_q: float = 0.2):
    """Rastgele/psikolojik traderlar icin ORTAK portfoy+execution sablonu.
    Yalnizca SINYAL degisir; kisitlar ayni -> adil karsilastirma."""
    from contracts.dsl import Expression
    from contracts.hypothesis_spec import (
        EconomicMechanism, Execution, Falsification, HypothesisFamily,
        HypothesisSpec, Portfolio, Universe,
    )
    return HypothesisSpec(
        hypothesis_id="benchmark", title="benchmark", claim="benchmark",
        family=HypothesisFamily.composite,
        economic_mechanism=EconomicMechanism(type="none", description="benchmark"),
        universe=Universe(source=universe_source),
        features=[], signal=Expression(op="field", field="close"),
        portfolio=Portfolio(type="cross_sectional_long_short",
                            long_quantile=long_q, short_quantile=long_q),
        execution=Execution(signal_time="close_t", trade_time="open_t_plus_1",
                            holding_period_days=1),
        falsification=Falsification())


def buy_and_hold(data) -> pd.Series:
    """Esit-agirlikli AL-TUT: her bar mevcut varliklarin ortalama getirisi.
    Al-tut oldugu icin devir ~0; tek seferlik giris maliyeti ihmal edilir."""
    close = data.get("close")
    try:
        adj = data.get("adjusted_close")
    except KeyError:
        adj = close
    ret = adj.pct_change(fill_method=None)
    # index_membership varsa (survivorship): varlik yalnizca uyeyken sepette
    if "index_membership" in data.fields:
        ret = ret.where(data.get("index_membership") > 0)
    return ret.mean(axis=1).dropna()   # esit agirlik = kesitsel ortalama


def random_trader(data, template, seed: int, cost_bps: float = COST_BPS) -> pd.Series:
    """Bir 'maymun': her bar rastgele skor -> ayni long/short kurallari + maliyet."""
    from backtest.engine import compute_pnl
    close = data.get("close")
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(rng.standard_normal(close.shape),
                       index=close.index, columns=close.columns)
    sig = sig.where(close.notna())          # sadece verisi olan varlikta pozisyon
    net, _ = compute_pnl(sig, template, data, cost_bps)
    return net


def psychological_trader(data, template) -> pd.Series:
    """'Duygusal' trader: dunku kazanani kovalar (asiri tepki / momentum-chasing).
    Tipik retail davranisi — yukselen neyse ona atlar."""
    from backtest.engine import compute_pnl
    close = data.get("close")
    yday_ret = close.pct_change(fill_method=None)     # dunku getiri = sinyal
    net, _ = compute_pnl(yday_ret, template, data, COST_BPS)
    return net


def our_strategy(data, hyp, graph, history=None, cost_bps: float = COST_BPS):
    """Bizim aday, verilen donemde. history verilirse ISINMALI hesaplanir.

    Isinma zorunlu (arastirma disindaki donemlerde): sinyal gecmis+donem
    birlesiminde hesaplanip yalniz doneme kesilir. Aksi halde rolling
    pencereler bosta kalir ve ML modeli donemin ICINDE yeniden egitilir.
    Bilgi akisi tek yonlu (gecmis -> donem): sizinti degildir.
    """
    from backtest.engine import compute_pnl
    from backtest.model_signal import compute_signal
    from data.synthetic import concat_market

    if history is None:
        signal = compute_signal(graph, hyp, data)
    else:
        signal = compute_signal(graph, hyp, concat_market(history, data)).reindex(
            index=data.dates, columns=data.get("close").columns)
    net, _ = compute_pnl(signal, hyp, data, cost_bps)
    return net


# --------------------------------------------------------------------------
def _yaris(data, template, hyp, graph, monkeys: int, history=None) -> dict:
    """Bir DONEMDE dort yarismaciyi kostur -> tum istatistikler."""
    bpy = data.bars_per_year
    bh = _stats(buy_and_hold(data), bpy)

    m_sh, m_tot, m_sh_free = [], [], []
    for i in range(monkeys):
        net = random_trader(data, template, seed=1000 + i)
        m_sh.append(_sharpe(net, bpy))
        m_tot.append(_total_return(net))
        m_sh_free.append(_sharpe(random_trader(data, template, seed=1000 + i,
                                               cost_bps=0.0), bpy))
    m_sh, m_sh_free = np.array(m_sh), np.array(m_sh_free)

    psy = _stats(psychological_trader(data, template), bpy)
    our = _stats(our_strategy(data, hyp, graph, history), bpy)
    our_free = _sharpe(our_strategy(data, hyp, graph, history, cost_bps=0.0), bpy)

    return {
        "bpy": bpy, "bh": bh, "psy": psy, "our": our, "our_free": our_free,
        "m_sh_med": float(np.median(m_sh)), "m_sh_best": float(m_sh.max()),
        "m_tot_med": float(np.median(m_tot)),
        "m_sh_med_free": float(np.median(m_sh_free)),
        "m_sh_best_free": float(m_sh_free.max()),
        "pctile": float((m_sh < our["sharpe"]).mean() * 100),
        "pctile_free": float((m_sh_free < our_free).mean() * 100),
        "bars": len(data.dates),
        "aralik": f"{data.dates[0].date()}→{data.dates[-1].date()}",
    }


def _gecti(y: dict) -> "dict[str, bool]":
    """Bu donemde hangi rakibi geciyoruz? (Sharpe karsilastirmasi)"""
    return {"al-tut": y["our"]["sharpe"] > y["bh"]["sharpe"],
            "rastgele": y["our"]["sharpe"] > y["m_sh_med"],
            "duygusal": y["our"]["sharpe"] > y["psy"]["sharpe"]}


def _donem_tablosu(ad: str, y: dict, monkeys: int) -> None:
    P(f"\n  ── {ad}  ({y['aralik']}, {y['bars']} bar) " + "─" * max(0, 50 - len(ad)))
    P(f"\n  {'Yaklasim':<34} {'Sharpe':>8} {'Yillik':>9} {'Toplam':>9} {'MaxDD':>7}")
    P(f"  {'-'*34} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")

    def row(a, s):
        P(f"  {a:<34} {s['sharpe']:>+8.2f} {s['yillik']*100:>+8.1f}% "
          f"{s['toplam']*100:>+8.1f}% {s['maxdd']*100:>6.0f}%")

    row("Al-tut (pasif piyasa)", y["bh"])
    P(f"  {'Rastgele al-satci — ORTANCA maymun':<34} {y['m_sh_med']:>+8.2f} "
      f"{'':>9} {y['m_tot_med']*100:>+8.1f}%")
    P(f"  {'Rastgele al-satci — EN SANSLI':<34} {y['m_sh_best']:>+8.2f} "
      f"{'(' + str(monkeys) + ' maymunun en iyisi)':>26}")
    row("Psikolojik (duygusal) trader", y["psy"])
    P(f"  {'-'*34} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")
    row("BIZIM STRATEJI", y["our"])
    P(f"\n  Masrafsiz kontrol (masraf=0, 'gercek sinyal mi?' testi):"
      f"  maymun ortanca {y['m_sh_med_free']:+.2f} | bizim {y['our_free']:+.2f}"
      f"  → maymunlarin %{y['pctile_free']:.0f}'inden iyi")


# --------------------------------------------------------------------------
def _load():
    import main as M
    return {"campaign": M.load_yaml("campaign.yaml")["campaign"],
            "data_cfg": M.load_yaml("data.yaml")["data"]}


def main() -> None:
    global _WRITE
    ap = argparse.ArgumentParser(description="Bizim yaklasim random/al-tut'u geciyor mu?")
    ap.add_argument("--monkeys", type=int, default=200,
                    help="Kac rastgele al-satci (maymun) kosulsun (varsayilan 200).")
    ap.add_argument("--ileri", action="store_true",
                    help="TAZE donemi de kat (Binance'den indirir, ILK kosuda 1 saat+).")
    ap.add_argument("--log", action="store_true", help="runs/benchmark.log'a yaz.")
    args = ap.parse_args()
    _WRITE = args.log
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import main as M
    from dsl import compile_hypothesis
    from evaluation.aday import en_iyi_aday

    cfgs = _load()
    campaign, data_cfg = cfgs["campaign"], cfgs["data_cfg"]
    cfg = M.build_config(campaign)

    P("\n" + "═" * 78)
    P("  KIYAS: rastgele al-satciyi, al-tut'u ve duygusal trader'i geciyor muyuz?")
    P("═" * 78)
    P(f"  Kosum tarihi: {datetime.now():%Y-%m-%d %H:%M}   |   "
      f"Islem masrafi: {COST_BPS:.0f} bps (kampanyadan)")
    P("""
  Hocanin olcutu: su an amac 'alpha bulmak' degil. Amac, parayi rastgele
  al-sat yapan birinden ve hic ugrasmayan (al-tut) birinden DAHA IYI
  yonetebildigimizi gostermek. Herkes AYNI piyasada, AYNI tarihlerde,
  AYNI islem masrafiyla yarisiyor.

  DIKKAT — kiyas HER DONEMDE ayri kosuluyor. Arastirma doneminde kazanmak
  bir sey kanitlamaz: aday zaten "orada iyi gorundugu icin" secilmistir.
  Asil cevap ORNEKLEM-DISI (holdout / ileri-test) satirlarindadir.""")

    # --- ADAY: en guclu KANITA sahip olan (en yuksek Sharpe'li DEGIL) ---
    aday = en_iyi_aday(cfg.min_acceptance_sharpe
                       if hasattr(cfg, "min_acceptance_sharpe") else 0.5)
    if aday is None:
        P("\n  Kabul edilmis strateji YOK — kiyas edilecek aday yok.")
        P("  Once bir kampanya kos:  python main.py    (veya agent.bat -> Kampanya)")
        return
    hyp = aday.spec()
    graph = compile_hypothesis(hyp)
    P(f"""
  Aday strateji : {aday.hypothesis_id}  "{aday.title[:58]}"
  Neden bu aday : {aday.secim_nedeni}
                  (aday secimi ARASTIRMA Sharpe'ina gore YAPILMAZ — en parlak
                   arastirma skoru genelde en asiri-uydurulmus adaydir.)""")

    P("\n  Veri yukleniyor (arastirma + kilitli holdout)...")
    research, holdout = M.load_data(campaign, data_cfg, cfg.research_fraction)
    template = _template_hyp(campaign.get("universe", "sp500_point_in_time"))

    donemler: "list[tuple[str, dict]]" = []
    P(f"  [1/2] ARASTIRMA doneminde yaris ({args.monkeys} maymun)...")
    donemler.append(("ARASTIRMA (in-sample — kanit DEGIL)",
                     _yaris(research, template, hyp, graph, args.monkeys)))
    P(f"  [2/2] KILITLI HOLDOUT doneminde yaris (isinmali)...")
    donemler.append(("HOLDOUT (kilitli, *OOS)",
                     _yaris(holdout, template, hyp, graph, args.monkeys,
                            history=research)))

    if args.ileri:
        P("  [+] TAZE donem indiriliyor (ILK kosuda 1 saat+ surebilir)...")
        try:
            from data.synthetic import concat_market
            from scripts.forward_test import _load_forward_data
            f_bas = pd.Timestamp(str(campaign["end_date"])).date() + timedelta(days=1)
            fdata = _load_forward_data(campaign, data_cfg, f_bas, date.today())
            if len(fdata.dates) >= 20:
                donemler.append(("ILERI-TEST (taze, *OOS)",
                                 _yaris(fdata, template, hyp, graph, args.monkeys,
                                        history=concat_market(research, holdout))))
            else:
                P("  (taze donem cok kisa — atlandi)")
        except Exception as e:  # noqa: BLE001
            P(f"  Taze veri cekilemedi: {type(e).__name__}: {str(e)[:150]}")

    # ---------------- DONEM DONEM TABLOLAR ----------------
    P("\n" + "─" * 78)
    P("  SONUCLAR")
    P("─" * 78)
    for ad, y in donemler:
        _donem_tablosu(ad, y, args.monkeys)

    # ---------------- ASIL CIKTI: GECME MATRISI ----------------
    P("\n" + "═" * 78)
    P("  HANGI RAKIBI, HANGI DONEMDE GECIYORUZ?   (✓ geciyoruz / ✗ gecemiyoruz)")
    P("═" * 78)
    P(f"\n  {'Donem':<36}{'al-tut':>9}{'rastgele':>10}{'duygusal':>10}{'  toplam':>9}")
    P(f"  {'-'*36}{'-'*9}{'-'*10}{'-'*10}{'-'*9}")
    for ad, y in donemler:
        g = _gecti(y)
        isaret = lambda b: "  ✓" if b else "  ✗"   # noqa: E731
        P(f"  {ad:<36}{isaret(g['al-tut']):>9}{isaret(g['rastgele']):>10}"
          f"{isaret(g['duygusal']):>10}{f'{sum(g.values())}/3':>9}")

    oos = [(ad, y) for ad, y in donemler if "*OOS" in ad]
    P("""
  Okuma: ilk satir (ARASTIRMA) KANIT DEGILDIR — aday orada secildigi icin
  kazanmasi beklenir. Anlamli olan *OOS isaretli satirlardir.""")

    # RISK TARAFI — Sharpe tek basina long-short bir stratejiye haksizlik eder:
    # al-tut piyasa riskini TASIR (borsa cakilirsa o da cakilir), long-short
    # tasimaz. Bunu bir "biz kazandik" puanina cevirmiyoruz (o, olcutu sonradan
    # kendine gore secmek olurdu) — sadece OLGUYU koyuyoruz, trader kendi karar
    # versin: ayni parayi ne kadar dususe maruz birakiyoruz?
    P(f"\n  Risk tarafi (olgu, hukum degil) — en derin dusus (MaxDD):")
    P(f"  {'Donem':<36}{'BIZIM':>10}{'al-tut':>10}")
    P(f"  {'-'*36}{'-'*10}{'-'*10}")
    for ad, y in donemler:
        P(f"  {ad:<36}{y['our']['maxdd']*100:>9.0f}%{y['bh']['maxdd']*100:>9.0f}%")
    P("""  Al-tut piyasa riski TASIR; long-short tasimaz. Dusuk getiri tek basina
  eleme sebebi degildir — ama 'piyasayi yendik' de denemez.""")

    # ---------------- DURUST HUKUM ----------------
    # Kiyas sayisi ("3'te 2 geciyoruz") TEK BASINA hukum olamaz: ortanca maymun
    # -%95 batarken bizim -%20 batmamiz "gectik" sayilirsa cikti, para kaybini
    # basari gibi gosterir. Once PARAYA bakilir.
    from evaluation.plain import _sar, durust_hukum

    P("\n" + "─" * 78)
    P("  HUKUM")
    P("─" * 78)
    if not oos:
        P("\n  Ornek-disi donem olculemedi — hukum verilemez.")
    for ad, y in oos:
        g = _gecti(y)
        baslik, gerekce = durust_hukum(y["our"]["toplam"], y["our"]["sharpe"],
                                       y["bh"]["toplam"])
        P(f"\n  [{ad}]  → {baslik}")
        for satir in gerekce:
            for parca in _sar(satir, 70):
                P(f"     {parca}")
        gecemedik = [k for k, v in g.items() if not v]
        if gecemedik:
            P(f"     Bu donemde GECEMEDIGIMIZ rakip(ler): {', '.join(gecemedik)}.")
        else:
            P("     Bu donemde uc rakibi de geciyoruz.")

    # Uc-donem hukmu (sistemin kalici kurali) — tek kaynak
    v = aday.hukum()
    P(f"\n  Sistemin uc-donem hukmu ({aday.hypothesis_id}): {v.verdict}")
    for r in v.reasons:
        for parca in _sar(r, 70):
            P(f"     {parca}")

    P("""
  NOT: "maymunu gecmek" DUSUK bir esiktir — rastgele maymun cilginca al-sat
  yapip masraftan batar. Onu gecmek "iyi strateji" demek degil, "delice islem
  yapmiyoruz" demektir. Asil olcu al-tut ve masrafsiz kontroldur.""")
    P("═" * 78 + "\n")

    # SONUCU KALICI YAP. Kiyas, hocanin BASARI OLCUTUdur ama sonucu yalnizca
    # bir terminal logunda duruyordu: dashboard'da (hocaya gosterilen gorsel
    # rapor) bu bolum HIC YOKTU. JSON'a yazilinca dashboard okuyup gosterebilir
    # ve "ne zaman olculdu" damgasi da beraberinde gider (bayat sayi tehlikesi).
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    import json as _json
    ozet = {
        "kosum_tarihi": datetime.now().isoformat(timespec="seconds"),
        "cost_bps": COST_BPS, "maymun_sayisi": args.monkeys,
        "aday": {"hypothesis_id": aday.hypothesis_id, "title": aday.title,
                 "secim_nedeni": aday.secim_nedeni, "hukum": v.verdict},
        "donemler": [
            {"ad": ad, "oos": "*OOS" in ad, "aralik": y["aralik"],
             "bizim": y["our"], "al_tut": y["bh"], "duygusal": y["psy"],
             "maymun_ortanca_sharpe": y["m_sh_med"],
             "bizim_masrafsiz_sharpe": y["our_free"],
             "maymun_yuzdeligi": y["pctile"], "gecti": _gecti(y)}
            for ad, y in donemler],
    }
    with open(os.path.join(HERE, "runs", "benchmark.json"), "w",
              encoding="utf-8") as f:
        _json.dump(ozet, f, ensure_ascii=False, indent=2)

    if _WRITE:
        p = os.path.join(HERE, "runs", "benchmark.log")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG))
        print(f"Log yazildi: {p}")


if __name__ == "__main__":
    main()
