"""
Walking Skeleton — uçtan uca araştırma döngüsü.

İki mod:
  python main.py            Kampanya koşusu (hipotez üret -> test -> hafıza).
                            HOLDOUT'A DOKUNMAZ — araştırma istediği kadar
                            tekrarlanabilir, kilitli dönem tüketilmez.
  python main.py --holdout  YALNIZCA holdout değerlendirmesi (LLM yok):
                            hafızadaki kabul edilmiş adaylar kilitli dönemde
                            BİR KEZ sınanır (one-shot, audit log). Kampanya
                            bittiğinde, bilinçli bir kararla çağrılır.

Bu ayrım Doküman 10.3'ün gereğidir: holdout her koşunun sonunda otomatik
tüketilirse araştırma-değerlendirme ayrımı fiilen erir (insan-döngüsü sızıntısı).

Modeli değiştirmek için: configs/models.yaml. Kod DEĞİŞMEZ.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv

from contracts.hypothesis_spec import HypothesisSpec
from dashboard import generate_dashboard
from data import make_adapter, split_by_fraction
from evaluation import build_report, print_report
from holdout import HoldoutError, HoldoutService
from llm import make_critic, make_provider
from memory import MemoryStore
from orchestrator import CampaignConfig, run_campaign

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "research_memory.sqlite")
HOLDOUT_DB = os.path.join(HERE, "holdout_audit.sqlite")


def load_yaml(name: str) -> dict:
    with open(os.path.join(HERE, "configs", name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(campaign: dict) -> CampaignConfig:
    budget = campaign.get("budget", {})
    risk = campaign.get("risk_constraints", {})
    hpol = campaign.get("holdout_policy", {})
    return CampaignConfig(
        goal=campaign["goal"],
        universe_description=campaign["universe_description"],
        allowed_fields=campaign.get("allowed_fields", []),
        allowed_operators=campaign.get("allowed_operators", []),
        allowed_horizons=campaign.get("allowed_horizons", []),
        allowed_rebalance=campaign.get("allowed_rebalance", []),
        portfolio_types=campaign.get("portfolio_types", []),
        model=str(campaign.get("model", "dsl_formula")),   # kampanya başına sabit model
        max_experiments=int(budget.get("maximum_experiments", 8)),
        max_llm_tokens=int(budget.get("maximum_llm_tokens", 300000)),
        cost_bps=float(budget.get("cost_bps", 5.0)),
        min_acceptance_sharpe=float(risk.get("min_acceptance_sharpe", 0.5)),
        max_drawdown=float(risk.get("max_drawdown", 0.40)),
        max_turnover=float(risk.get("max_turnover", 300.0)),
        min_positive_folds=float(risk.get("min_positive_folds", 0.5)),
        min_originality=float(risk.get("min_originality", 0.0)),
        research_fraction=float(hpol.get("research_fraction", 0.7)),
        parameter_optimization=bool(budget.get("parameter_optimization", False)),
        anonymize_universe=bool(campaign.get("anonymize_universe", True)),
        anonymous_description=(str(campaign["anonymous_description"])
                               if campaign.get("anonymous_description") else None),
    )


def load_data(campaign: dict, data_cfg: dict, research_fraction: float):
    """Veri yükle ve araştırma / KİLİTLİ holdout olarak böl (tek yerden)."""
    src = data_cfg.get("source")
    if src in ("yfinance", "sp500_pit", "sp600_pit", "binance"):
        data_cfg.setdefault(src, {})
        data_cfg[src]["start"] = str(campaign["start_date"])
        data_cfg[src]["end"] = str(campaign["end_date"])
    full = make_adapter(data_cfg).load()
    # FUNDAMENTALS (value/quality): config'te fundamentals:true ise EDGAR'dan PIT
    # book_to_market + roe alanları eklenir (bölmeden ÖNCE → hem araştırma hem
    # holdout aynı alanları görür). İlk çağrı EDGAR'dan çeker (cache'lenir).
    if data_cfg.get(src, {}).get("fundamentals"):
        from data.edgar import add_fundamentals
        print("  [edgar] PIT value/quality (book_to_market, roe) ekleniyor...")
        add_fundamentals(full)
    return split_by_fraction(full, research_fraction)


def _backfill_audits(memory: MemoryStore, data, cfg: CampaignConfig) -> None:
    """Reviewer raporu olmayan kabullere Backtest Auditor raporu üret (geriye-uyum).

    Eski kabuller auditor özelliğinden önce kaydedildiği için reviews_json boştur;
    veri yüklüyken deterministik denetimi çalıştırıp kaydı güncelleriz.
    """
    pending = memory.accepted_without_reviews()
    if not pending:
        return
    from agents.backtest_auditor import BacktestAuditor
    from backtest import run_backtest
    from dsl import compile_hypothesis
    auditor = BacktestAuditor()
    done = 0
    for row_id, _hid, hj in pending:
        try:
            hyp = HypothesisSpec.model_validate_json(hj)
            graph = compile_hypothesis(hyp)
            res = run_backtest(graph, hyp, data, cost_bps=cfg.cost_bps)
            memory.set_reviews(row_id, [auditor.audit(hyp, res, data, cfg.cost_bps)])
            done += 1
        except Exception:  # noqa: BLE001 — bir kayıt patlarsa diğerleri sürsün
            pass
    if done:
        print(f"[reviewer] {done} eski kabule Backtest Auditor raporu geriye-dolduruldu.")


# Eleme aşamalarının trader diline çevirisi (teknik stage adı -> ne oldu).
_ELEME_TR = {
    "compile_error":         "kural dışı yazılmıştı (bilgisayar çeviremedi)",
    "static_rejected":       "GELECEĞE BAKIYORDU (sızıntı) — testten önce eledik",
    "critic_rejected":       "ekonomik gerekçesi tutmadı (bağımsız denetçi eledi)",
    "duplicate":             "aynı fikrin tekrarıydı (daha önce denenmişti)",
    "low_originality":       "mevcut bir fikre çok benziyordu (yeniden istendi)",
    "degenerate_conditional": "koşulu boşa çalışıyordu (kural fiilen ölüydü)",
    "gate_rejected":         "geçmişte para kazandırmadı (eşiği geçemedi)",
    "robustness_rejected":   "kırılgandı (masraf 2x / parametre oynayınca çöktü)",
    "parameter_search":      "parametre arama denemesi (ayrı fikir değil)",
    "accepted":              "GEÇTİ",
}


def print_trader_summary(memory: MemoryStore, rows, cfg: CampaignConfig) -> None:
    """TEKNİK TERİM İÇERMEYEN kampanya özeti.

    Hedef okur: ML/ajan bilmeyen, sadece alım-satım bilen biri. Kural — basılan
    her sayının yanında "bu ne demek" ve "iyi mi kötü mü" olmalı; para kaybı
    asla süslenmemeli (bkz. evaluation/plain.py dürüstlük kuralı).
    """
    from evaluation.plain import (
        durust_hukum, sozluk_blogu, strateji_karnesi,
    )

    stages = memory.stage_counts()
    kabul = memory.accepted_full()          # (hid, title, sharpe, dd, turnover, returns)
    toplam = memory.total_experiments()
    # Parametre aramaları ayrı FİKİR değil; "kaç fikir denendi" sayısına girmez.
    fikir_sayisi = toplam - stages.get("parameter_search", 0)

    print("\n" + "=" * 74)
    print("  TRADER ÖZETİ — teknik terim yok")
    print("=" * 74)

    print(f"\n  NE YAPTIK?")
    print(f"    Bilgisayar {fikir_sayisi} tane alım-satım fikri üretti ve her birini")
    print(f"    geçmiş veride, işlem masrafı düşülerek parayla denedi.")
    print(f"    {len(kabul)} tanesi bütün elemeleri geçti.")

    print(f"\n  FİKİRLER NEREDE ELENDİ?")
    # Elenenler önce (çoktan aza), GEÇTİ en sonda — huni yukarıdan aşağı okunur.
    for stage, n in sorted(stages.items(),
                           key=lambda kv: (kv[0] == "accepted", -kv[1])):
        if stage == "parameter_search":
            continue
        print(f"    {n:4d}  {_ELEME_TR.get(stage, stage)}")
    if stages.get("parameter_search"):
        print(f"    ({stages['parameter_search']} ek deneme: kabul edilen fikirlerin "
              f"ayar araması —\n     ayrı fikir sayılmaz ama şans hesabına katılır.)")

    if not kabul:
        print("\n  SONUÇ: Hiçbir fikir elemeleri geçemedi.")
        print("    Bu bir arıza DEĞİL. Sistem, para kazandırmayan fikirleri")
        print("    kabul etmemek için kurulmuştur. 'Bulamadık' demek,")
        print("    olmayan bir şeyi 'bulduk' demekten çok daha değerlidir.")
        print("=" * 74 + "\n")
        return

    print(f"\n  ELEMELERİ GEÇENLER")
    dsr_ile = {r.hypothesis_id: r for r in rows}
    for hid, title, sharpe, dd, turn, rets in kabul:
        toplam_getiri = None
        if rets:
            birikim = 1.0
            for x in rets:
                birikim *= (1.0 + x)
            toplam_getiri = birikim - 1.0
        print(f"\n    [{hid}] {title[:66]}")
        print(strateji_karnesi(sharpe=sharpe, max_dd=dd, turnover=turn,
                               toplam=toplam_getiri, girinti="      "))
        r = dsr_ile.get(hid)
        if r is not None:
            from evaluation.plain import esik_yorumu
            print(f"      {'şans elemesi notu':<24s} {r.dsr:>10.2f}   "
                  f"{esik_yorumu('dsr', r.dsr)}")
            if not r.survives_fdr:
                print("        -> UYARI: bu kadar çok fikir denendiği için bu "
                      "sonuç tesadüf\n           olabilir; istatistik onaylamadı.")

    # DÜRÜST HÜKÜM — en iyi kabulün parasına bakarak
    en_iyi = max(kabul, key=lambda k: (k[2] or -99))
    rets = en_iyi[5] or []
    tg = None
    if rets:
        b = 1.0
        for x in rets:
            b *= (1.0 + x)
        tg = b - 1.0
    # Çoklu-test süzgecini HİÇ kimse geçmediyse hüküm aşağı çekilir — aynı
    # çıktının iki yeri farklı şey söylemesin.
    fdr_gecti = any(r.survives_fdr for r in rows) if rows else None
    baslik, gerekce = durust_hukum(tg, en_iyi[2], fdr_gecti=fdr_gecti)
    print(f"\n  HÜKÜM (en iyi fikir için): {baslik}")
    for satir in gerekce:
        from evaluation.plain import _sar
        for parca in _sar(satir, 68):
            print(f"    {parca}")

    print()
    print(sozluk_blogu(["sharpe", "max_drawdown", "turnover", "dsr", "holdout"],
                       girinti="  "))
    print("=" * 74 + "\n")


def _run_forward_gate(campaign: dict, cfg: CampaignConfig, sonuclar: list,
                      research_data, holdout_data) -> "int | None":
    """Holdout'u geçen adayları TAZE veride sına ve üç-dönem hükmünü bas.

    Ağ/veri hatası araştırmayı bloklamaz: ileri-test yapılamazsa hüküm
    'EKSİK' kalır — sessizce 'geçti' SAYILMAZ (bkz. three_period).
    """
    from datetime import date, timedelta

    import pandas as pd

    from contracts.hypothesis_spec import HypothesisSpec
    from data.synthetic import concat_market
    from evaluation.three_period import final_verdict, verdict_table

    gecenler = [(h, r, s) for h, r, s, p in sonuclar if p]
    print(f"\n=== ÜÇ-DÖNEM KAPISI ({len(gecenler)} aday kilitli dönemi geçti) ===")
    print("  Kilitli dönem TEK bir rejim çekilişidir. Ölçüldü: holdout'u geçen")
    print("  3 adayın 3'ü de taze veride çöktü. Bu yüzden ikinci, bağımsız bir")
    print("  örneklem-dışı dönemde (sistemin hiç görmediği zaman) sınanıyorlar.\n")

    baslangic = pd.Timestamp(str(campaign["end_date"])).date() + timedelta(days=1)
    bitis = date.today()
    try:
        from scripts.forward_test import _load_forward_data, _sharpe
        data_cfg = load_yaml("data.yaml")["data"]
        print(f"  Taze dönem yükleniyor: {baslangic} → {bitis} "
              f"(ilk koşuda uzun sürebilir; sonrası cache'ten)...")
        taze = _load_forward_data(campaign, data_cfg, baslangic, bitis)
    except Exception as e:  # noqa: BLE001 — ağ/veri sorunu hükmü bloklamasın
        print(f"  Taze veri alınamadı ({type(e).__name__}: {str(e)[:120]}).")
        print("  Hüküm EKSİK kalıyor — 'geçti' SAYILMIYOR.")
        print(verdict_table([(h, r, s, None) for h, r, s in gecenler],
                            cfg.min_acceptance_sharpe))
        return None      # ölçülemedi -> hüküm EKSİK (0 ile karıştırılmamalı)

    gecmis = concat_market(research_data, holdout_data) if research_data is not None \
        else holdout_data
    holdout_svc = HoldoutService(holdout_data, audit_path=HOLDOUT_DB,
                                 cost_bps=cfg.cost_bps)
    memory = MemoryStore(DB_PATH)
    hyp_by_id = {h: j for h, j, _s in memory.accepted_hypotheses(limit=50)}
    memory.close()

    satirlar = []
    for hid, r_sh, h_sh in gecenler:
        try:
            f_sh, f_tot = _forward_metrics(hyp_by_id[hid], taze, gecmis,
                                           cfg.cost_bps, _sharpe)
        except Exception as e:  # noqa: BLE001
            print(f"  {hid}: ileri-test hesaplanamadı ({type(e).__name__})")
            satirlar.append((hid, r_sh, h_sh, None))
            continue
        v = final_verdict(r_sh, h_sh, f_sh, cfg.min_acceptance_sharpe)
        holdout_svc.record_forward(hid, bitis, f_sh, f_tot, v.verdict,
                                   baslangic, bitis)
        satirlar.append((hid, r_sh, h_sh, f_sh))
        print(f"  {hid}: ileri-test Sharpe={f_sh:+.2f}  toplam={f_tot*100:+.0f}%")
    holdout_svc.close()

    print()
    print(verdict_table(satirlar, cfg.min_acceptance_sharpe))
    dogrulanan = [s for s in satirlar
                  if final_verdict(*s[1:], cfg.min_acceptance_sharpe).passed]
    print()
    if dogrulanan:
        print(f"  {len(dogrulanan)}/{len(satirlar)} aday İKİ bağımsız OOS döneminde de")
        print("  ayakta kaldı. 'Alpha bulundu' DEĞİL — 'henüz ölmedi'. İzlemeye devam.")
    else:
        print("  Hiçbir aday iki OOS döneminde birden ayakta kalamadı.")
        print("  Kilitli dönemi geçmeleri REJİM ŞANSIYDI — sistem bunu yakaladı.")
    return len(dogrulanan)


def _forward_metrics(hyp_json: str, taze, gecmis, cost_bps: float, sharpe_fn):
    """Bir hipotezin taze dönemdeki (Sharpe, toplam getiri) değeri.

    Sinyal ISITILARAK hesaplanır (geçmiş = araştırma+holdout): aksi hâlde model
    taze dönemin İÇİNDE yeniden eğitilir ve ölçtüğümüz şey kabul edilen model
    olmaz (bkz. holdout ısınma düzeltmesi).
    """
    import json

    from backtest.engine import compute_pnl
    from backtest.model_signal import compute_signal
    from contracts.hypothesis_spec import HypothesisSpec
    from data.synthetic import concat_market
    from dsl import compile_hypothesis

    hyp = HypothesisSpec.model_validate(json.loads(hyp_json))
    graph = compile_hypothesis(hyp)
    sig = compute_signal(graph, hyp, concat_market(gecmis, taze)).reindex(
        index=taze.dates, columns=taze.get("close").columns)
    net, _ = compute_pnl(sig, hyp, taze, cost_bps)
    equity = (1.0 + net).cumprod()
    return (sharpe_fn(net, getattr(taze, "bars_per_year", 365)),
            float(equity.iloc[-1] - 1.0) if len(equity) else 0.0)


def run_holdout_mode(campaign: dict, cfg: CampaignConfig, holdout_data,
                     research_data=None, invalidate_reason: "str | None" = None,
                     skip_forward: bool = False) -> None:
    """--holdout: hafızadaki kabul edilmiş adayları kilitli dönemde sına.

    LLM'siz, deterministik son sınav. One-shot: aynı aday ikinci kez
    değerlendirilMEZ (audit DB koşular arası korunur).
    """
    if not os.path.exists(DB_PATH):
        print("Hafıza yok (research_memory.sqlite). Önce kampanya koş: python main.py")
        return
    memory = MemoryStore(DB_PATH)
    policy = campaign.get("holdout_policy", {}) or {}
    max_cand = int(policy.get("maximum_candidates", 20))
    candidates = memory.accepted_hypotheses(limit=max_cand)
    if not candidates:
        print("Holdout adayı yok: hafızada kabul edilmiş hipotez bulunmuyor.")
        memory.close()
        return

    # ISINMA: araştırma dilimi GEÇMİŞ olarak verilir — rolling pencereler ve
    # walk-forward ML modeli kilitli dönemin başında sıfırdan başlamaz, model
    # holdout'un içinde YENİDEN EĞİTİLMEZ. Bilgi akışı tek yönlü (geçmiş->gelecek).
    holdout = HoldoutService(holdout_data, audit_path=HOLDOUT_DB,
                             max_candidates=max_cand,
                             min_sharpe=cfg.min_acceptance_sharpe,
                             cost_bps=cfg.cost_bps,
                             history=research_data)
    # GEÇERSİZ KILMA (istenmişse) — değerlendirmeden ÖNCE, gerekçeyle, silmeden.
    if invalidate_reason:
        n = holdout.invalidate(invalidate_reason)
        print(f"\n[geçersiz kılma] {n} kilitli-dönem kaydı GEÇERSİZ işaretlendi "
              f"(silinmedi; gerekçe audit'te kalıcı).")
        print(f"  Gerekçe: {invalidate_reason}")
        print("  Bu kayıtlar artık kotayı doldurmuyor ve yeniden "
              "değerlendirilebilir.\n")

    print(f"=== HOLDOUT (kilitli dönem, one-shot, {len(candidates)} aday) ===")
    print("  Bu, fikirlerin geliştirilirken HİÇ görmediği bir dönem. Öğrenciye")
    print("  sınav sorusunu önceden vermemek gibi — gerçek not buradan çıkar.")
    print("  Her aday YALNIZCA BİR KEZ sınanır; sonuca bakıp fikri düzeltmek")
    print("  yasaktır (yoksa kilitli dönem de araştırma verisine dönüşür).\n")
    if research_data is None:
        print("  UYARI: geçmiş verilmedi — kilitli dönemin başı ısınmayla harcanır.")
    sonuclar = []
    for hid, hjson, research_sharpe in candidates:
        hyp = HypothesisSpec.model_validate_json(hjson)
        try:
            res = holdout.evaluate(hyp, campaign=campaign.get('name'),
                                   research_sharpe=research_sharpe)
        except HoldoutError as e:
            print(f"  {hid}  atlandı: {e}")
            continue
        flag = "GEÇTİ" if res.passed else "KALDI"
        cov = (f"  kapsama=%{res.coverage*100:.0f}" if res.coverage < 0.995 else "")
        print(f"  {hid}  araştırma Sharpe={research_sharpe:.2f} -> "
              f"holdout Sharpe={res.sharpe:.2f}  [{flag}]{cov}")
        sonuclar.append((hid, research_sharpe, res.sharpe, res.passed))

    # ---- ÜÇ-DÖNEM KAPISI: holdout'u geçen aday OTOMATİK ileri-teste girer ----
    # Neden otomatik: ölçüldü ki holdout'u geçen 3 adayın 3'ü de taze veride
    # çöktü. Tek kilitli dönem bir REJİM çekilişidir. İleri-testi ayrı bir
    # script'te insan kararına bırakmak, "3/3 geçti" diye erken sevinmeyi
    # mümkün kılıyordu; artık hüküm sistemin kendisinden çıkıyor.
    # NİHAİ HÜKÜM üç-dönem kapısından gelir; SADE OKUMA onu yansıtmalı.
    # (Aksi hâlde çıktının bir yeri "hiçbiri ayakta kalamadı", öteki yeri
    # "3/3 ayakta kaldı" der — bu oturumda bir kez yaşandı; okuyan hangisine
    # inanacağını bilemez.)
    dogrulanan = None
    if sonuclar and any(p for *_x, p in sonuclar) and not skip_forward:
        dogrulanan = _run_forward_gate(campaign, cfg, sonuclar,
                                       research_data, holdout_data)

    if sonuclar:
        gecen = sum(1 for *_x, p in sonuclar if p)
        print("\n  SADE OKUMA:")
        if gecen == 0:
            print("    Hiçbir fikir kilitli dönemde ayakta kalmadı. Araştırma")
            print("    döneminde iyi görünen sonuçlar, yeni veride tekrarlanmadı —")
            print("    yani o kazançlar gerçek bir kural değil, geçmişe uydurulmuş")
            print("    desenlerdi. Sistemin işi tam olarak bunu yakalamaktı.")
        elif dogrulanan == 0:
            print(f"    {gecen}/{len(sonuclar)} fikir kilitli dönemi geçti AMA")
            print("    hiçbiri ikinci, bağımsız dönemde ayakta kalamadı.")
            print("    Kilitli dönemi geçmeleri REJİM ŞANSIYDI — tek bir kilitli")
            print("    dönem yeterli kanıt değildir. Sistem bunu yakaladı.")
        elif dogrulanan:
            print(f"    {gecen}/{len(sonuclar)} fikir kilitli dönemi, {dogrulanan} tanesi")
            print("    ikinci bağımsız dönemi de geçti. Ciddiye alınacak bir işaret —")
            print("    ama 'alpha bulundu' DEĞİL: çok sayıda deneme içinden çıktı,")
            print("    çoklu-test düzeltmesi ayrıca kontrol edilmeli.")
        else:
            print(f"    {gecen}/{len(sonuclar)} fikir kilitli dönemde ayakta kaldı.")
            print("    UYARI: ikinci dönem (ileri-test) ÖLÇÜLMEDİ — hüküm EKSİK.")
            print("    Tek bir kilitli dönem yeterli kanıt değildir.")
        dusenler = [(h, a, b) for h, a, b, p in sonuclar if a > 0 and b < a - 0.3]
        if dusenler:
            print("\n    Araştırmada iyi görünüp kilitli dönemde düşenler:")
            for h, a, b in dusenler:
                print(f"      {h}: {a:+.2f} -> {b:+.2f}  "
                      f"(fark {b-a:+.2f} = geçmişe aşırı uyum işareti)")
    holdout.close()
    memory.close()

    out = generate_dashboard(DB_PATH, HOLDOUT_DB, os.path.join(HERE, "dashboard.html"),
                             campaign_name=campaign["name"],
                             bars_per_year=holdout_data.bars_per_year,
                             min_acceptance_sharpe=cfg.min_acceptance_sharpe)
    print(f"\nDashboard: {out}")


def main() -> None:
    # Windows konsolu (cp1254) LLM'den gelen özel karakterlerde (em-dash, emoji)
    # patlayabilir; --detay çıktısı ham hipotez metnini basar. errors='replace'
    # ile güvene al (kampanya bir print yüzünden düşmesin).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="Otonom quant araştırma kampanyası")
    parser.add_argument("--fresh", action="store_true",
                        help="Yeni kampanya: hafızayı SIFIRLA. Varsayılan: mevcut kampanyaya DEVAM et.")
    parser.add_argument("--holdout", action="store_true",
                        help="Kampanya KOŞMA; hafızadaki kabul edilmiş adayları kilitli "
                             "holdout döneminde değerlendir (one-shot, LLM'siz).")
    parser.add_argument("--live", action="store_true",
                        help="Canlı ajan terminali: kampanyayı gerçek zamanlı panelde "
                             "izle (rich TUI). Mantık değişmez, yalnızca görünüm.")
    parser.add_argument("--holdout-invalidate", metavar="GEREKÇE",
                        help="Mevcut kilitli-dönem sonuçlarını GEÇERSİZ kıl "
                             "(silmez; gerekçe+tarihle audit'e yazar) ve yeniden "
                             "değerlendirmeye izin ver. YALNIZCA değerlendirici "
                             "hatalıysa meşrudur — sonucu beğenmediğin için DEĞİL.")
    parser.add_argument("--ileri-test-atla", action="store_true",
                        dest="skip_forward",
                        help="--holdout sonrası otomatik ileri-testi ATLA "
                             "(taze veri indirme uzun sürüyorsa). Hüküm o zaman "
                             "EKSİK kalır — 'geçti' SAYILMAZ.")
    parser.add_argument("--detay", action="store_true",
                        help="DETAYLI çıktı: her deneyin her adımı (üretim, derleme, "
                             "sızıntı, sinyal, backtest fold'ları, gate) tek tek basılır.")
    args = parser.parse_args()
    if args.fresh and args.holdout:
        parser.error("--fresh ile --holdout birlikte kullanılamaz "
                     "(sıfırlanmış hafızada holdout adayı olmaz).")

    load_dotenv(os.path.join(HERE, ".env"))   # API key'i ortama yükle (koda girmez)
    campaign = load_yaml("campaign.yaml")["campaign"]
    models = load_yaml("models.yaml")["models"]
    data_cfg = load_yaml("data.yaml")["data"]
    cfg = build_config(campaign)

    # Veri her iki modda da gerekli (holdout modu kilitli dilimi kullanır).
    data, holdout_data = load_data(campaign, data_cfg, cfg.research_fraction)

    if args.holdout:
        run_holdout_mode(campaign, cfg, holdout_data, research_data=data,
                         invalidate_reason=args.holdout_invalidate,
                         skip_forward=args.skip_forward)
        return

    # ---- Kampanya modu (holdout'a DOKUNULMAZ) ----
    # Model TAK-ÇALIŞTIR: üretici + bağımsız eleştirmen config'ten kurulur
    gen_cfg = models["hypothesis_generator"]
    provider = make_provider(gen_cfg)
    critic = make_critic(models.get("quant_critic", {"provider": "dummy"}))

    # Literatür grounding (Doküman 4.3): hipotez üreticiye 'bilinen anomali'
    # tohumları ver. VARSAYILAN = statik corpus (reproducible + point-in-time
    # güvenli; bkz. agents/literature.py başlığı). models.yaml -> web_search: true
    # ile isteğe bağlı canlı arama denenir (reproducibility/look-ahead riskli).
    from agents.literature import load_literature_mechanisms
    if gen_cfg.get("web_search") and hasattr(provider, "client"):
        from agents.literature import fetch_literature_mechanisms
        from orchestrator.loop import ANONYMOUS_UNIVERSE
        print("Literatür aranıyor (web_search, en fazla ~90 sn; olmazsa statik corpus)...")
        # Anonimleştirme açıkken literatür ajanı da ticker/tarih GÖRMEZ.
        lit_universe = ((cfg.anonymous_description or ANONYMOUS_UNIVERSE)
                        if cfg.anonymize_universe else campaign["universe_description"])
        literature = fetch_literature_mechanisms(
            provider.client, provider.model, lit_universe)
    else:
        # Corpus evrene göre seçilir: kripto evrenine HİSSE anomalisi (52-hafta,
        # ay-sonu, Amihud) fısıldamak aramayı kör bırakır — perpetual piyasanın
        # kendi mekanizmaları var (funding/kalabalıklık, tasfiye kaskadı).
        domain = str(campaign.get("literature_domain", "equity"))
        literature = load_literature_mechanisms(domain=domain)
        print(f"Literatür (statik corpus, reproducible, alan={domain}):")
    for m in literature:
        print(f"  • {m[:100]}")
    print()

    # DEVAM (varsayılan) veya SIFIRLA (--fresh). Devam: novelty/çoklu-test/öğrenme
    # koşular arası birikir; aynı hipotez tekrar üretilmez (Doküman: campaign = çok deney).
    if args.fresh:
        # ARAŞTIRMA hafızası sıfırlanır (yeni kampanya = yeni N, doğru).
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        # HOLDOUT AUDIT'İ ASLA SİLİNMEZ. Eskiden burada os.remove(HOLDOUT_DB)
        # vardı ve bu, projenin en güçlü bilimsel iddiasını rutin bir bayrakla
        # yok ediyordu: kilitli dönem kaydı one-shot, append-only ve geçersiz
        # kılma için GEREKÇE zorunlu — ama `--fresh` hepsini gerekçesiz,
        # izsiz siliyordu. Üstelik bu, "yeni kampanya" için VARSAYILAN yol
        # olduğundan kazara oluyordu (bu oturumda iki kez oldu).
        #
        # Kilitli dönem KAMPANYA BAŞINA değil, PROJE ÇAPINDA sonlu bir
        # kaynaktır: her kullanım onu bir miktar aşındırır. Bu yüzden audit
        # kampanyalar arası KORUNUR ve aday kotası ortak sayılır — yeni bir
        # kampanya açmak, kilitli dönemi sıfırlamaz.
        if os.path.exists(HOLDOUT_DB):
            import shutil
            from datetime import datetime as _dt
            yedek_dir = os.path.join(HERE, "arsiv")
            os.makedirs(yedek_dir, exist_ok=True)
            damga = _dt.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(HOLDOUT_DB,
                        os.path.join(yedek_dir, f"holdout_audit_{damga}.sqlite"))
            print(f"(--fresh) Araştırma hafızası sıfırlandı. HOLDOUT AUDIT "
                  f"KORUNDU (yedeği: arsiv/holdout_audit_{damga}.sqlite).")
            print("          Kilitli dönem proje çapında sonlu bir kaynaktır; "
                  "yeni kampanya onu sıfırlamaz, aday kotası ortaktır.")
        else:
            print("(--fresh) Yeni kampanya: araştırma hafızası sıfırlandı.")
    memory = MemoryStore(DB_PATH)

    print(f"=== Kampanya: {campaign['name']} ===")
    print(f"Evren: {cfg.universe_description}")
    print(f"Sağlayıcı: {models['hypothesis_generator']['provider']} | "
          f"Bütçe: {cfg.max_experiments} deney\n")

    if args.live:
        # Canlı panel: loop'un DÜZ metin print'leri yutulur (redirect), her karar
        # on_event kancasıyla panele akar. Panel GERÇEK stdout'a basar (reporter
        # file=real_out) → redirect'ten etkilenmez. TUI süs; kanca güvenli (kırılsa
        # kampanya sürer). Kampanya bitince özet normal basılır.
        import io as _io
        import sys as _sys
        from contextlib import redirect_stdout

        from ui.live import LiveReporter
        real_out = _sys.stdout
        try:
            real_out.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        with LiveReporter(file=real_out) as reporter:
            with redirect_stdout(_io.StringIO()):
                run_campaign(provider, data, memory, cfg, critic=critic,
                             literature=literature, on_event=reporter.on_event)
    else:
        run_campaign(provider, data, memory, cfg, critic=critic, literature=literature,
                     verbose=args.detay)

    # GERİYE-DOLDURMA: reviewer özelliğinden önce kabul edilmiş hipotezlerin
    # Backtest Auditor raporu yok. Veri hâlâ yüklüyken üret (dashboard tam olsun).
    _backfill_audits(memory, data, cfg)

    print("\n--- ÖZET ---")
    print(f"Toplam deney (multiple-testing muhasebesi): {memory.total_experiments()}")
    print(f"Karar dağılımı: {memory.summary_by_decision()}")
    print("\nLeaderboard (kabul edilenler, Sharpe'a göre):")
    for hid, title, sharpe, dd in memory.leaderboard():
        print(f"  {hid}  {title:32s}  Sharpe={sharpe:.2f}  MaxDD=%{(dd or 0)*100:.0f}")

    # Multiple testing raporu — "kabul" != "istatistiksel geçerli"
    backtested = memory.backtested_experiments()
    # YILLIKLAŞTIRMA ÖLÇEĞİ VERİDEN: aksi halde bu tablo 252 varsayar ve aynı
    # stratejinin Sharpe'ı leaderboard'dakinden (8h kripto: 1095) ~2x farklı
    # görünür. DSR/FDR bundan etkilenmez (per-period), ann_sharpe ve CI etkilenir.
    rows = build_report(backtested, bars_per_year=data.bars_per_year)
    print_report(rows, n_trials=len(backtested))

    # SADE ÖZET EN SONA: teknik tablolardan sonra, herkesin okuyabileceği
    # dilde toparlama. Teknik çıktı kaldırılmadı — üstüne bir katman eklendi.
    print_trader_summary(memory, rows, cfg)

    # Holdout BİLEREK burada koşulmaz (Doküman 10.3): her koşuda otomatik
    # tüketilseydi kilitli dönem fiilen araştırma verisine dönerdi.
    n_accepted = len(memory.accepted_hypotheses())
    if n_accepted:
        print(f"\nHoldout adayı bekliyor: {n_accepted} kabul edilmiş hipotez. "
              f"Kampanya bitti diyorsan: python main.py --holdout")

    # Token/maliyet görünürlüğü (Doküman 17.3) — üretici + eleştirmen
    pt = getattr(provider, "total_prompt_tokens", 0) + getattr(critic, "total_prompt_tokens", 0)
    ct = getattr(provider, "total_completion_tokens", 0) + getattr(critic, "total_completion_tokens", 0)
    if pt or ct:
        print(f"\nToken kullanımı (üretici+critic): prompt={pt}, completion={ct}, toplam={pt+ct}")

    memory.close()

    # Research dashboard (tek dosya, offline) — hocaya göstermek için
    out = generate_dashboard(DB_PATH, HOLDOUT_DB, os.path.join(HERE, "dashboard.html"),
                             campaign_name=campaign["name"],
                             bars_per_year=data.bars_per_year,
                             min_acceptance_sharpe=cfg.min_acceptance_sharpe)
    print(f"\nDashboard: {out}")


if __name__ == "__main__":
    main()
