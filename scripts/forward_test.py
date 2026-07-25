"""
İLERİ-TEST (forward-test): holdout'un CANLI, bitmeyen versiyonu.

Kilitli holdout tek-atıştır ve tükenir (geçmiş veri sınırlı). Ama GELECEK
sınırsızdır: sistemin eğitildiği/değerlendirildiği dönem bittikten SONRAKİ her
gün, hiç kimsenin görmediği taze veridir. İleri-test bunu kullanır.

  Sistemin gördüğü son tarih : campaign.end_date (araştırma + holdout dahil)
  İleri-test dönemi          : o tarihten + 1 gün  →  BUGÜN   (el değmemiş)

Kabul edilen en iyi stratejiyi bu taze dönemde gün gün kâğıt üzerinde koşar,
biriken getiri eğrisini çıkarır ve pasif al-tut ile karşılaştırır. Bu, tanım
gereği out-of-sample'dır — holdout'un tekrarlanabilir, süregelen hâli.

Kullanım:
    .venv/Scripts/python.exe scripts/forward_test.py
    .venv/Scripts/python.exe scripts/forward_test.py --gunluk-cek   # (ileride: canlı bar)

NOT: İlk koşu taze dönemi (2025→bugün) Binance'den indirir (birkaç dakika);
sonraki koşular cache'ten hızlıdır. Strateji funding kullanıyorsa funding de çekilir.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

COST_BPS = 5.0
_LOG: list[str] = []
_WRITE = False


def P(s: str = "") -> None:
    print(s)
    if _WRITE:
        _LOG.append(s)


def _sharpe(net: pd.Series, bpy: int) -> float:
    x = net.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(bpy))


def _sparkline(equity: pd.Series, width: int = 60) -> str:
    """Biriken sermaye eğrisini ASCII sparkline olarak çiz."""
    blocks = "▁▂▃▄▅▆▇█"
    x = equity.to_numpy(dtype=float)
    if len(x) < 2:
        return "(veri yok)"
    # width noktaya indir
    idx = np.linspace(0, len(x) - 1, min(width, len(x))).astype(int)
    xs = x[idx]
    lo, hi = xs.min(), xs.max()
    if hi - lo < 1e-12:
        return blocks[0] * len(xs)
    norm = (xs - lo) / (hi - lo)
    return "".join(blocks[min(7, int(v * 7.999))] for v in norm)


def _load_forward_data(campaign, data_cfg, forward_start, forward_end):
    """Sistemin görmediği TAZE dönemi çek (SPLIT YOK — tüm dönem tek parça)."""
    from data.adapter import make_adapter
    cfg = copy.deepcopy(data_cfg)
    src = cfg.get("source")
    if src in ("yfinance", "sp500_pit", "sp600_pit", "binance"):
        cfg.setdefault(src, {})
        cfg[src]["start"] = str(forward_start)
        cfg[src]["end"] = str(forward_end)
    data = make_adapter(cfg).load()
    if cfg.get(src, {}).get("fundamentals"):
        from data.edgar import add_fundamentals
        add_fundamentals(data)
    return data


def main() -> None:
    global _WRITE
    ap = argparse.ArgumentParser(description="İleri-test: holdout sonrası taze veride kâğıt takip")
    ap.add_argument("--log", action="store_true", help="runs/forward_test.log'a yaz.")
    args = ap.parse_args()
    _WRITE = args.log
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import main as M
    from backtest.engine import compute_pnl
    from backtest.model_signal import compute_signal
    from contracts.hypothesis_spec import HypothesisSpec
    from dsl import compile_hypothesis
    from memory.store import MemoryStore
    from scripts.benchmark import buy_and_hold

    campaign = M.load_yaml("campaign.yaml")["campaign"]
    data_cfg = M.load_yaml("data.yaml")["data"]
    cfg = M.build_config(campaign)

    P("\n" + "═" * 78)
    P("  İLERİ-TEST — strateji zaman içinde tutuyor mu? (tüm OOS dönemleri)")
    P("═" * 78)
    P("""
  DİKKAT: tek bir parlak döneme kanmak, kendini kandırmanın en kolay yoludur.
  Bu araç stratejiyi TÜM örneklem-dışı (OOS) dönemlerde yan yana koyar:
    • Holdout   — sistemin gördüğü verinin kilitli son dilimi (2023-24)
    • İleri-test — sistemin gördüğü tarihten SONRAKİ taze zaman (2025→bugün)
  Gerçek alpha HER OOS döneminde tutar. Bir dönem uçup diğerinde çökmek =
  rejim-bağımlılık = güvenilmez sinyal.""")

    # --- Kabul edilen en iyi strateji ---
    db = os.path.join(HERE, "research_memory.sqlite")
    if not os.path.exists(db):
        P("\n  research_memory.sqlite yok — önce bir kampanya koş (agent.bat -> 1).")
        return
    acc = MemoryStore(db).accepted_hypotheses(limit=1)
    if not acc or not acc[0][1]:
        P("\n  Kabul edilen strateji yok — ileri-test edilecek aday yok.")
        P("  (Sistem henüz hiçbir hipotezi kabul etmedi; bu da dürüst bir durum.)")
        return
    hid, hj, sh_research = acc[0]
    hyp = HypothesisSpec.model_validate(json.loads(hj))

    # --- İleri-test dönemi: sistemin gördüğü son tarih + 1 gün → bugün ---
    seen_end = pd.Timestamp(str(campaign["end_date"])).date()
    forward_start = seen_end + timedelta(days=1)
    forward_end = date.today()
    gun = (forward_end - forward_start).days
    P(f"""
  Aday strateji     : {hid}  "{hyp.title}"
                      (araştırma Sharpe ~{sh_research:+.2f})
  Sistemin gördüğü  : ... {seen_end}  (araştırma + holdout buraya kadar)
  İLERİ-TEST dönemi : {forward_start}  →  {forward_end}   ({gun} gün, EL DEĞMEMİŞ)

  Bu dönem sistemin hiç görmediği taze veridir — ne eğitimde, ne holdout'ta.
  Kabul edilen stratejiyi burada kâğıt üzerinde gün gün koşuyoruz.""")

    if gun < 30:
        P(f"\n  ⚠ İleri-test dönemi çok kısa ({gun} gün). Anlamlı sonuç için "
          f"campaign.end_date ile bugün arası daha uzun olmalı.")

    graph = compile_hypothesis(hyp)

    def _run(d):
        """Bir dönemde stratejiyi koştur → (Sharpe, toplam, MaxDD, al-tut Sharpe, equity)."""
        sig = compute_signal(graph, hyp, d)
        net, _ = compute_pnl(sig, hyp, d, COST_BPS)
        eq = (1.0 + net).cumprod()
        b = buy_and_hold(d)
        bpy_ = getattr(d, "bars_per_year", 365)
        dd = float(-(eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
        tot = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
        return _sharpe(net, bpy_), tot, dd, _sharpe(b, bpy_), eq

    # --- Araştırma + holdout (sistemin gördüğü dönem; cache'ten hızlı) ---
    P("\n  Sistemin gördüğü dönem yükleniyor (araştırma + kilitli holdout)...")
    research, holdout = M.load_data(campaign, data_cfg, cfg.research_fraction)
    r_sh, r_tot, r_dd, _, _ = _run(research)
    h_sh, h_tot, h_dd, _, _ = _run(holdout)

    # --- İleri-test (taze dönem) ---
    P("  Taze/el değmemiş dönem indiriliyor (ilk koşu birkaç dakika sürebilir)...")
    try:
        fdata = _load_forward_data(campaign, data_cfg, forward_start, forward_end)
    except Exception as e:  # noqa: BLE001
        P(f"\n  Taze veri çekilemedi: {type(e).__name__}: {str(e)[:200]}")
        P("  (Holdout sonuçları yine de aşağıda; ağ/kaynak sorunu olabilir.)")
        fdata = None

    if fdata is not None and len(fdata.dates) >= 20:
        f_sh, f_tot, f_dd, fb_sh, f_eq = _run(fdata)
    else:
        f_sh = f_tot = f_dd = fb_sh = None
        f_eq = None

    # --- Tablo: TÜM DÖNEMLER YAN YANA ---
    P("\n" + "─" * 78)
    P("  STRATEJİ ZAMAN İÇİNDE  (%s bps masraf; * = örneklem-DIŞI)" % int(COST_BPS))
    P("─" * 78)
    P(f"\n  {'Dönem':<30}{'Sharpe':>9}{'Toplam':>10}{'MaxDD':>8}")
    P(f"  {'-'*30}{'-'*9}{'-'*10}{'-'*8}")
    P(f"  {'Araştırma (in-sample)':<30}{r_sh:>+9.2f}{r_tot*100:>+9.0f}%{r_dd*100:>7.0f}%")
    P(f"  {'Holdout (2023-24) *OOS':<30}{h_sh:>+9.2f}{h_tot*100:>+9.0f}%{h_dd*100:>7.0f}%")
    if f_sh is not None:
        P(f"  {'İleri-test (2025→bugün) *OOS':<30}{f_sh:>+9.2f}{f_tot*100:>+9.0f}%{f_dd*100:>7.0f}%")
        P(f"\n  İleri-test biriken getiri eğrisi:\n    {_sparkline(f_eq)}")

    # --- Yorum: OOS TUTARLILIĞI (asıl soru) ---
    P("\n" + "─" * 78)
    P("  NE ANLAMA GELİYOR? — asıl soru: OOS dönemleri TUTARLI mı?")
    P("─" * 78)
    oos = [("holdout", h_sh)] + ([("ileri-test", f_sh)] if f_sh is not None else [])
    pozitif = [ad for ad, s in oos if s > 0.3]
    negatif = [ad for ad, s in oos if s < -0.1]
    if len(oos) >= 2 and pozitif and negatif:
        P(f"""
  ⚠ REJİM-BAĞIMLI — güvenilmez sinyal.
  Strateji in-sample'da parlıyor ({r_sh:+.2f}) ama örneklem-dışı dönemler
  ÇELİŞİYOR: {', '.join(pozitif)} pozitif, {', '.join(negatif)} negatif.
  Bir OOS döneminde uçup diğerinde çökmek, performansın piyasa REJİMİNE
  bağlı olduğunu (dönem şansı) gösterir — HER koşulda tutan bir alpha değil.

  Bu tam da tek bir parlak dönemin (örn. yalnızca ileri-test) neden YANILTICI
  olduğunun kanıtı: holdout'u gizleyip yalnız ileri-testi gösterseydik
  "alpha bulduk" derdik. Sistem (ve çok-dönem OOS) bunu yakalıyor.""")
    elif len(oos) >= 2 and not negatif and len(pozitif) == len(oos):
        P(f"""
  ✓ TUTARLI — nadir ve güçlü. Strateji in-sample ({r_sh:+.2f}) VE her iki
  örneklem-dışı dönemde ({', '.join(f'{ad}:{s:+.2f}' for ad,s in oos)}) pozitif.
  Bu, rejimden bağımsız gerçek bir sinyalin işaretidir — yine de dönem
  uzadıkça izlenmeli (ileri-test tükenmez; bu onun gücü).""")
    else:
        P(f"""
  Örneklem-dışı dönemler: {', '.join(f'{ad}:{s:+.2f}' for ad,s in oos)}.
  Kesin yargı için daha fazla OOS dönemi gerekli — ileri-test her yeni günle
  büyür (holdout'un bitmeyen hâli).""")
    P("═" * 78 + "\n")

    if _WRITE:
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        p = os.path.join(HERE, "runs", "forward_test.log")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG))
        print(f"Log yazildi: {p}")


if __name__ == "__main__":
    main()
