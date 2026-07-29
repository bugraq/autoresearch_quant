"""
Research Dashboard — kampanya sonuçlarından tek dosyalık statik HTML üretir.

Sunucu/derleme yok: research_memory.sqlite + holdout_audit.sqlite okunur,
kendi kendine yeten (inline CSS/SVG) bir dashboard.html yazılır. Her bölümün
başında Türkçe başlık + kısa açıklama vardır; üstte anlatısal bir özet cümle.
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
from datetime import datetime

from evaluation import build_report, evaluate_strategies
from memory import MemoryStore

# Pipeline aşamaları — funnel sırası (üstten alta)
_STAGE_ORDER = [
    ("compile_error", "Derleme hatası"),
    ("static_rejected", "Sızıntı / statik red"),
    ("critic_rejected", "Critic reddi (ekonomik)"),
    ("duplicate", "Tekrar (novelty)"),
    ("gate_rejected", "Performans kapısı reddi"),
    ("robustness_rejected", "Sağlamlık testi reddi"),
    ("accepted", "KABUL"),
]
_REJECT_STAGES = {"compile_error", "static_rejected", "critic_rejected",
                  "gate_rejected", "robustness_rejected"}

_CSS = """
:root { --bg:#0f1117; --card:#1a1d27; --border:#2a2f3d; --fg:#e6e8ee;
  --muted:#8a90a2; --accent:#5b8def; --good:#3fb950; --bad:#f85149; --warn:#d29922; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f6f7f9; --card:#fff; --border:#e2e5ea; --fg:#1a1d27;
    --muted:#5c6472; --accent:#2f6bd8; --good:#1a7f37; --bad:#cf222e; --warn:#9a6700; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.55; }
.wrap { max-width:1080px; margin:0 auto; padding:36px 22px 72px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }
.lead { color:var(--muted); font-size:13px; margin:0 0 22px; }
.banner { background:linear-gradient(90deg,rgba(91,141,239,.14),transparent);
  border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:10px; padding:14px 18px; font-size:15px; margin-bottom:26px; }
.banner b { color:var(--fg); } .banner .hl { color:var(--accent); font-weight:700; }
section { margin-top:34px; }
h2 { font-size:15px; margin:0 0 3px; letter-spacing:.02em; }
.desc { color:var(--muted); font-size:12.5px; margin:0 0 12px; max-width:760px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:16px; border-top:3px solid var(--border); }
.tile.g { border-top-color:var(--good); } .tile.r { border-top-color:var(--bad); }
.tile.b { border-top-color:var(--accent); } .tile.w { border-top-color:var(--warn); }
.tile .n { font-size:30px; font-weight:700; line-height:1; }
.tile .l { color:var(--muted); font-size:12px; margin-top:6px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:16px 18px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
tr:last-child td { border-bottom:none; }
th { color:var(--muted); font-weight:600; font-size:12px; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
td.num.good { color:var(--good); font-weight:600; } td.num.bad { color:var(--bad); font-weight:600; }
.bar { height:22px; border-radius:5px; background:var(--accent); min-width:3px; }
.bar.good { background:var(--good); } .bar.bad { background:var(--bad); }
.frow { display:flex; align-items:center; gap:12px; margin:7px 0; }
.frow .lbl { width:190px; font-size:13px; } .frow .cnt { width:34px; text-align:right;
  color:var(--muted); font-variant-numeric:tabular-nums; font-weight:600; }
.frow .track { flex:1; background:var(--border); border-radius:5px; }
.pill { padding:2px 9px; border-radius:20px; font-size:11px; font-weight:700; }
.pill.good { background:rgba(63,185,80,.16); color:var(--good); }
.pill.bad { background:rgba(248,81,73,.16); color:var(--bad); }
.pill.warn { background:rgba(210,153,34,.16); color:var(--warn); }
.pill.muted { background:var(--border); color:var(--muted); }
.detail { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:14px 16px; margin-bottom:12px; }
.detail .dh { font-weight:600; margin-bottom:8px; font-size:14px; }
.detail .did { color:var(--accent); font-family:ui-monospace,monospace; margin-right:6px; }
.detail .dsh { float:right; color:var(--good); font-weight:700; font-size:13px; }
.detail .drow { font-size:13px; margin:5px 0; }
.detail .drow b { color:var(--muted); font-weight:600; }
.detail code { background:var(--bg); border:1px solid var(--border); border-radius:5px;
  padding:3px 7px; font-size:12.5px; display:inline-block; margin-top:3px;
  font-family:ui-monospace,monospace; word-break:break-all; }
.foot { color:var(--muted); font-size:12px; margin-top:44px;
  border-top:1px solid var(--border); padding-top:16px; }
"""


def _q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()


def _esc(x) -> str:
    return html.escape(str(x))


def _section(title: str, desc: str, body: str) -> str:
    return f'<section><h2>{_esc(title)}</h2><div class="desc">{_esc(desc)}</div>{body}</section>'


def _aktif_kosul(conn) -> str:
    """status sütunu varsa 'yalnız aktif' kosulu; yoksa (eski audit) bos.

    Geçersiz kılınmış kayıtlar audit'te DURUR (silinmez) ama sonuç
    tablolarında aktif sonuçla aynı kefeye konmaz — yoksa aynı hipotez iki
    kez, iki farklı Sharpe'la listelenir ve hangisi geçerli belli olmaz.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(holdout_access)")}
    return " WHERE status='active'" if "status" in cols else ""


def _kampanya_adaylari(memory_db: "str | None") -> "set[str] | None":
    """Hafızadaki hipotez kimlikleri. None = hafıza yok (ayrım yapılamaz)."""
    if not memory_db or not os.path.exists(memory_db):
        return None
    conn = sqlite3.connect(memory_db)
    try:
        return {h for (h,) in _q(conn, "SELECT DISTINCT hypothesis_id FROM experiment")}
    finally:
        conn.close()


def _holdout_counts(holdout_db: str, memory_db: "str | None" = None) -> tuple:
    """(gecen, aday, sondaj_gecen, sondaj_aday) — KAMPANYA adayı vs ELLE SONDAJ.

    Neden ayrım şart: holdout audit'ine kampanya dışı, elle yazılmış sondalar
    da girebiliyor (gerçekten girdi). Bunları kampanya sonucuyla aynı kefeye
    koymak başlığı yanıltır: "kilitli dönemi 1/6 geçti" cümlesindeki 1, sistemin
    BULDUĞU bir strateji değil, insanın elle denediği bir varyant olabilir.
    Ayrım kanıta dayanır: hafızada kaydı olmayan kimlik = kampanya ürünü değil.
    """
    if not os.path.exists(holdout_db):
        return (0, 0, 0, 0)
    conn = sqlite3.connect(holdout_db)
    rows = _q(conn, "SELECT hypothesis_id, passed FROM holdout_access"
                    + _aktif_kosul(conn))
    conn.close()
    bilinen = _kampanya_adaylari(memory_db)
    if bilinen is None:                       # ayrım yapılamıyor: hepsi kampanya
        return (sum(1 for _h, p in rows if p), len(rows), 0, 0)
    kamp = [(h, p) for h, p in rows if h in bilinen]
    sondaj = [(h, p) for h, p in rows if h not in bilinen]
    return (sum(1 for _h, p in kamp if p), len(kamp),
            sum(1 for _h, p in sondaj if p), len(sondaj))


def _banner(conn, holdout_db: str, memory_db: "str | None" = None) -> str:
    total = _q(conn, "SELECT COUNT(*) FROM experiment")[0][0]
    acc = _q(conn, "SELECT COUNT(*) FROM experiment WHERE decision='accept'")[0][0]
    passed, cand, s_pass, s_cand = _holdout_counts(holdout_db, memory_db)
    hold_txt = (f"bunlardan <span class='hl'>{passed}</span> tanesi kilitli holdout "
                f"dönemini geçti" if cand else "holdout değerlendirmesi henüz yapılmadı")
    if s_cand:
        hold_txt += (f" (ayrıca kampanya dışı, elle denenmiş {s_cand} sonda kilitli "
                     f"dönemde çalıştırılmış; {s_pass} tanesi geçmiş — bunlar sistemin "
                     f"bulgusu DEĞİLDİR)")
    return (f'<div class="banner">Bu kampanyada LLM otonom olarak '
            f'<b>{total}</b> hipotez üretip test etti; '
            f'<span class="hl">{acc}</span> tanesi tüm süzgeçlerden geçip kabul edildi, '
            f'{hold_txt}. Aşağıdaki her bölüm sürecin bir yönünü gösterir.</div>')


def _trader_ozeti(memory_db: str, holdout_db: str, bars_per_year: int) -> str:
    """EN ÜSTTEKİ SADE PANEL — teknik terim bilmeyen için tek bakışta hüküm.

    Aşağıdaki teknik bölümler duruyor; bu onların yerine değil, ÖNÜNE geçer.
    Panelin işi tek soruyu cevaplamak: "sonuç ne, paraya ne oldu?"
    """
    from evaluation.plain import TERIMLER, durust_hukum, esik_yorumu, para_dili

    store = MemoryStore(memory_db)
    kabul = store.accepted_full()      # (hid, title, sharpe, dd, turnover, returns)
    rows = build_report(store.backtested_experiments(), bars_per_year=bars_per_year)
    stages = store.stage_counts()
    toplam = store.total_experiments()
    store.close()
    fikir = toplam - stages.get("parameter_search", 0)
    _passed, cand, _s_pass, _s_cand = _holdout_counts(holdout_db, memory_db)

    if not kabul:
        return ('<div class="card"><p><b>SONUÇ: hiçbir fikir elemeleri geçemedi.</b></p>'
                f'<p class="desc">Bilgisayar {fikir} alım-satım fikri üretip her birini '
                'geçmiş veride, işlem masrafı düşülerek denedi; hiçbiri eşikleri '
                'aşamadı. Bu bir arıza değildir — sistem, para kazandırmayan fikirleri '
                'kabul etmemek için kurulmuştur. Olmayan bir şeyi “bulduk” demektense '
                '“bulamadık” demek çok daha değerlidir.</p></div>')

    def _toplam_getiri(rets):
        if not rets:
            return None
        b = 1.0
        for x in rets:
            b *= (1.0 + x)
        return b - 1.0

    en_iyi = max(kabul, key=lambda k: (k[2] or -99))
    tg = _toplam_getiri(en_iyi[5])
    fdr_gecti = any(r.survives_fdr for r in rows) if rows else None
    baslik, gerekce = durust_hukum(tg, en_iyi[2], fdr_gecti=fdr_gecti)
    renk = "bad" if ("KAZANDIRMADI" in baslik or "ZAYIF" in baslik) else "warn"
    if baslik.startswith("UMUT"):
        renk = "good"

    satirlar = []
    for hid, title, sharpe, dd, turn, rets in kabul[:5]:
        t = _toplam_getiri(rets)
        satirlar.append(
            f"<tr><td>{_esc(hid)}</td><td>{_esc(title[:54])}</td>"
            f'<td class="num">{sharpe:+.2f}</td>'
            f'<td>{_esc(esik_yorumu("sharpe", sharpe))}</td>'
            f'<td class="num">{("%%%+.0f" % (t*100)) if t is not None else "–"}</td>'
            f'<td class="num">%{(dd or 0)*100:.0f}</td>'
            f'<td>{_esc(esik_yorumu("max_drawdown", dd or 0))}</td></tr>')

    sozluk = "".join(
        f"<li><b>{_esc(ad)}</b> — {_esc(acik)}</li>"
        for k, (ad, acik) in TERIMLER.items()
        if k in ("sharpe", "max_drawdown", "turnover", "dsr", "holdout"))

    # KİLİTLİ DÖNEM SONUCU VARSA O ESASTIR — araştırma dönemindeki kazanç,
    # fikrin geliştirildiği veriden çıkar; asıl not hiç görülmemiş dönemden.
    if not cand:
        holdout_not = ('<p class="desc"><b>Kilitli dönem sınavı henüz yapılmadı.</b> '
                       'Yukarıdaki sayılar, fikirlerin GELİŞTİRİLDİĞİ dönemden çıktı — '
                       'yani öğrencinin kendi çalıştığı sorulardan aldığı not. Gerçek '
                       'not için: <code>python main.py --holdout</code></p>')
    elif _passed == 0:
        renk = "bad"
        baslik = "KİLİTLİ DÖNEMDE ÇÖKTÜ"
        holdout_not = (f'<p class="desc"><b>ASIL SONUÇ BU:</b> {cand} aday hiç '
                       'görmediği kilitli dönemde sınandı ve <b>hiçbiri ayakta '
                       'kalmadı</b>. Yukarıdaki kazançlar, fikirlerin geliştirildiği '
                       'dönemde geçerliydi; yeni veride tekrarlanmadı. Yani bunlar '
                       'gerçek bir piyasa kuralı değil, geçmişe uydurulmuş '
                       'desenlerdi — sistemin işi tam olarak bunu yakalamaktı.</p>')
    else:
        # ÜÇ-DÖNEM HÜKMÜ ESASTIR. Eskiden burada koşulsuz "KİLİTLİ DÖNEMİ n/m
        # GEÇTİ" (yeşil) yazılıyordu. Ölçüldü ki bu YANILTICI: holdout'u geçen
        # 3 adayın 3'ü de taze veride çöktü. Başlık artık ikinci, bağımsız OOS
        # dönemini de hesaba katar — yoksa panel "geçti", alttaki üç-dönem
        # tablosu "rejim-bağımlı" der ve okuyan hangisine inanacağını bilemez.
        from evaluation.three_period import final_verdict
        uc = _uc_donem_satirlari(memory_db, holdout_db)
        hukumler = [final_verdict(r, h, f, 0.5) for _h, _t, r, h, f in uc]
        dogrulanan = sum(1 for v in hukumler if v.passed)
        olculdu = [v for v in hukumler if v.verdict != "EKSİK"]
        if not olculdu:
            renk = "warn"
            baslik = f"KİLİTLİ DÖNEMİ {_passed}/{cand} GEÇTİ — İLERİ-TEST EKSİK"
            holdout_not = (f'<p class="desc"><b>{_passed}/{cand}</b> aday kilitli '
                           'dönemi geçti. AMA hüküm <b>EKSİK</b>: ikinci, bağımsız '
                           'dönem (ileri-test) henüz ölçülmedi. Tek bir kilitli '
                           'dönem yeterli kanıt değildir — ölçülmemiş dönemi '
                           '“geçti” saymak tam da bu kontrolün önlemek için var '
                           'olduğu hatadır. <code>python main.py --holdout</code>'
                           '</p>')
        elif dogrulanan:
            renk = "good"
            baslik = f"İKİ OOS DÖNEMİNİ DE {dogrulanan}/{len(uc)} GEÇTİ"
            holdout_not = (f'<p class="desc"><b>ASIL SONUÇ BU:</b> {dogrulanan} aday '
                           'İKİ bağımsız örneklem-dışı dönemde de ayakta kaldı. '
                           'Ciddiye alınacak bir işaret — ama <b>“alpha bulundu” '
                           'değil</b>: çok sayıda deneme içinden çıktı. Doğru okuma: '
                           '<b>henüz ölmedi</b>.</p>')
        else:
            renk = "bad"
            baslik = "REJİM-BAĞIMLI — TAZE VERİDE ÇÖKTÜ"
            holdout_not = (f'<p class="desc"><b>ASIL SONUÇ BU:</b> {_passed} aday '
                           'kilitli dönemi geçti AMA hiçbiri sistemin hiç görmediği '
                           'taze veride ayakta kalamadı. Kilitli dönem TEK bir rejim '
                           'çekilişidir; onu geçmek genelleme kanıtı değildir. '
                           'Yalnızca holdout’a bakılsaydı “alpha bulduk” denecekti — '
                           'sistem bunu yakaladı.</p>')

    return (
        f'<div class="card">'
        f'<p><span class="pill {renk}">{_esc(baslik)}</span></p>'
        f'<p class="desc">Bilgisayar <b>{fikir}</b> alım-satım fikri üretti, her birini '
        f'geçmiş veride işlem masrafı düşerek denedi; <b>{len(kabul)}</b> tanesi bütün '
        f'elemeleri geçti.</p>'
        + "".join(f'<p class="desc">{_esc(g)}</p>' for g in gerekce)
        + (f'<p class="desc">Somut olarak: {_esc(para_dili(tg))}.</p>' if tg is not None else "")
        + '<table><tr><th>Kimlik</th><th>Strateji</th><th>Risk başına kazanç</th>'
          '<th></th><th>Toplam getiri</th><th>En dip kayıp</th><th></th></tr>'
        + "".join(satirlar) + "</table>"
        + holdout_not
        + f'<p class="desc">Terimler:</p><ul class="desc">{sozluk}</ul>'
        + "</div>")


def _tiles(conn, memory_db: str) -> str:
    total = _q(conn, "SELECT COUNT(*) FROM experiment")[0][0]
    by_dec = dict(_q(conn, "SELECT decision, COUNT(*) FROM experiment GROUP BY decision"))
    fams = _q(conn, "SELECT COUNT(DISTINCT family) FROM experiment WHERE sharpe IS NOT NULL")[0][0]
    store = MemoryStore(memory_db)
    structures = store.distinct_structure_count()   # farklı YAPI (pencereden bağımsız)
    store.close()
    items = [("Toplam hipotez", total, "b"), ("Kabul edilen", by_dec.get("accept", 0), "g"),
             ("Reddedilen", by_dec.get("reject", 0), "r"),
             ("Tekrar (elendi)", by_dec.get("duplicate", 0), "w"),
             ("Farklı yapı", structures, "b"), ("Denenen aile", fams, "b")]
    return '<div class="tiles">' + "".join(
        f'<div class="tile {c}"><div class="n">{v}</div><div class="l">{_esc(l)}</div></div>'
        for l, v, c in items) + "</div>"


def _funnel(conn) -> str:
    counts = dict(_q(conn, "SELECT stage, COUNT(*) FROM experiment GROUP BY stage"))
    mx = max(list(counts.values()) + [1])
    rows = []
    for stage, label in _STAGE_ORDER:
        c = counts.get(stage, 0)
        w = int(100 * c / mx)
        cls = "good" if stage == "accepted" else ("bad" if c and stage in _REJECT_STAGES else "")
        rows.append(
            f'<div class="frow"><div class="lbl">{_esc(label)}</div>'
            f'<div class="track"><div class="bar {cls}" style="width:{w}%"></div></div>'
            f'<div class="cnt">{c}</div></div>')
    return '<div class="card">' + "".join(rows) + "</div>"


def _total_return_pct(returns_json: "str | None") -> "float | None":
    """Net getiri serisinden birikimli getiriyi (%) hesaplar: prod(1+r)-1.
    "Bu stratejiye para koysaydın % kaç kazanırdın" — trader'ın anladığı sayı."""
    if not returns_json:
        return None
    try:
        acc = 1.0
        for x in json.loads(returns_json):
            acc *= (1.0 + float(x))
        return (acc - 1.0) * 100.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _ret_cell(returns_json: "str | None") -> str:
    """Getiri hücresi — yeşil kazanç / kırmızı kayıp."""
    tot = _total_return_pct(returns_json)
    if tot is None:
        return '<td class="num">—</td>'
    cls = "good" if tot >= 0 else "bad"
    return f'<td class="num {cls}">{tot:+.0f}%</td>'


def _leaderboard(conn) -> str:
    rows = _q(conn, """SELECT hypothesis_id, title, sharpe, max_drawdown, returns_json
                       FROM experiment
                       WHERE decision='accept' AND sharpe IS NOT NULL
                       ORDER BY sharpe DESC LIMIT 20""")
    if not rows:
        return '<div class="card desc">Bu kampanyada kabul edilen strateji olmadı.</div>'
    body = "".join(
        f"<tr><td>{_esc(h)}</td><td>{_esc(t)}</td>"
        f'<td class="num">{s:.2f}</td>{_ret_cell(rj)}'
        f'<td class="num">%{(d or 0)*100:.0f}</td></tr>'
        for h, t, s, d, rj in rows)
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Strateji</th>'
            '<th>Sharpe</th><th>Getiri</th><th>Maks. düşüş</th></tr>'
            + body + "</table></div>"
            + '<div class="desc" style="margin-top:6px">Getiri = araştırma '
            'döneminde birikimli kazanç (para % kaç değişti). Sharpe = risk başına '
            'getiri (yüksek iyi). Bunlar araştırma dönemi; kesin yargı holdout.</div>')


def _multiple_testing(memory_db: str, bars_per_year: int = 252) -> str:
    store = MemoryStore(memory_db)
    rows = build_report(store.backtested_experiments(), bars_per_year=bars_per_year)
    store.close()
    if not rows:
        return '<div class="card desc">Backtest edilen deney yok.</div>'
    body = ""
    for r in rows:
        fdr = ('<span class="pill good">GEÇTİ</span>' if r.survives_fdr
               else '<span class="pill muted">geçmedi</span>')
        dsr = f"{r.dsr:.2f}" + (" ★" if r.dsr > 0.95 else "")
        copy_tag = f" ×{r.n_copies}" if r.n_copies > 1 else ""
        var_tag = f"+{r.n_param_variants}" if r.n_param_variants else "–"
        body += (f"<tr><td>{_esc(r.hypothesis_id)}{_esc(copy_tag)}</td>"
                 f'<td class="num">{r.ann_sharpe:.2f}</td>'
                 f'<td class="num">{r.raw_p:.3f}</td><td class="num">{dsr}</td>'
                 f'<td class="num">[{r.ci_low:.2f}, {r.ci_high:.2f}]</td>'
                 f'<td class="num">{var_tag}</td>'
                 f"<td>{fdr}</td></tr>")
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Sharpe</th>'
            '<th>ham p</th><th>DSR</th><th>%95 güven aralığı</th>'
            '<th>varyant</th><th>FDR</th></tr>'
            + body + "</table></div>"
            '<div class="desc" style="margin-top:8px">★ = DSR &gt; 0.95: deneme sayısı '
            'düzeltildikten sonra bile anlamlı. Güven aralığı sıfırı içeriyorsa sonuç '
            'kesin değildir. varyant (+N) = optimizer bu stratejinin N pencere '
            'varyantını aradı; korelasyonlu oldukları için bağımsız deneme sayılmaz '
            '(Doküman 10.1) — sadece birincil stratejiler paydaya girer. '
            '×N = N deneme birebir aynı getiriyi üretti (ölü parametre).</div>')


def _pareto(memory_db: str) -> str:
    """Çok amaçlı Pareto sıralaması (Doküman 11.2)."""
    store = MemoryStore(memory_db)
    evals = evaluate_strategies(store.accepted_full())
    store.close()
    if not evals:
        return '<div class="card desc">Değerlendirilecek kabul edilmiş strateji yok.</div>'
    body = ""
    for e in evals:
        star = '<span class="pill good">Pareto-optimal</span>' if e.pareto_optimal else '–'
        body += (f"<tr><td>{_esc(e.hypothesis_id)}</td>"
                 f'<td class="num">{e.sharpe:.2f}</td>'
                 f'<td class="num">{e.sharpe_lb:.2f}</td>'
                 f'<td class="num">%{e.max_drawdown*100:.0f}</td>'
                 f'<td class="num">{e.turnover:.0f}</td>'
                 f'<td class="num">{e.score:.2f}</td><td>{star}</td></tr>')
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Sharpe</th>'
            '<th>Sharpe alt-sınır</th><th>Maks DD</th><th>Turnover</th>'
            '<th>Skor</th><th>Pareto</th></tr>' + body + "</table></div>"
            '<div class="desc" style="margin-top:8px">Skor = Sharpe_alt-sınır '
            '− 0.5·DD − 0.002·turnover (bütçe tahsisi için yardımcı sinyal). '
            'Pareto-optimal = hiçbir stratejiye tüm boyutlarda yenik düşmeyen.</div>')


def _uc_donem_satirlari(memory_db: str, holdout_db: str) -> "list[tuple]":
    """(hid, baslik, arastirma, holdout, ileri-test) — AKTIF kayitlardan.

    Uc kaynak birlestirilir: arastirma Sharpe'i hafizadan, holdout aktif
    audit kaydindan, ileri-test ise sicilin EN SON olcumunden. Gecersiz
    kilinmis holdout kayitlari ve eski ileri-test olcumleri disarida kalir.
    """
    if not os.path.exists(holdout_db):
        return []
    hc = sqlite3.connect(holdout_db)
    try:
        kol = {r[1] for r in hc.execute("PRAGMA table_info(holdout_access)")}
        hash_var = "hypothesis_hash" in kol
        sec = "hypothesis_id, sharpe" + (", hypothesis_hash" if hash_var else "")
        sql = f"SELECT {sec} FROM holdout_access"
        if "status" in kol:
            sql += " WHERE status='active'"
        satir = _q(hc, sql)
        holdout = {r[0]: r[1] for r in satir}
        hashler = {r[0]: r[2] for r in satir} if hash_var else {}
        ileri = {}
        try:
            ileri = dict(_q(hc, "SELECT hypothesis_id, sharpe FROM forward_test "
                                "WHERE id IN (SELECT MAX(id) FROM forward_test "
                                "GROUP BY hypothesis_id)"))
        except sqlite3.Error:
            pass          # eski audit dosyasi: forward_test tablosu yok
    finally:
        hc.close()
    if not holdout:
        return []
    # ARAŞTIRMA BİLGİSİ ÖNCE SİCİLDEN. Sicil kampanyalar arası yaşar; canlı
    # hafıza `--fresh` ile sıfırlanır. Sicilde yoksa (eski kayıt) hafızaya
    # düşülür — orada da kimlik çakışması parmak iziyle denetlenir.
    sicil = {}
    hc2 = sqlite3.connect(holdout_db)
    try:
        sicil = {f: (t, r, c) for f, t, r, c in _q(
            hc2, "SELECT fingerprint, title, research_sharpe, campaign "
                 "FROM candidate_registry")}
    except sqlite3.Error:
        pass          # eski audit dosyası: sicil tablosu yok
    finally:
        hc2.close()

    mc = sqlite3.connect(memory_db)
    try:
        arastirma = {h: (t, s, hj) for h, t, s, hj in _q(
            mc, "SELECT hypothesis_id, title, sharpe, hypothesis_json FROM "
                "experiment WHERE decision='accept' AND sharpe IS NOT NULL")}
    finally:
        mc.close()
    out = []
    for hid, h_sh in holdout.items():
        t, r_sh, hj = arastirma.get(hid, ("(hafızada yok)", None, None))
        # KİMLİK DOĞRULAMASI: hypothesis_id kampanyalar arası TEKİL DEĞİL
        # (--fresh sayacı sıfırlar), audit ise kampanyalar arası yaşar.
        # Gerçek örnek: v2'nin hyp_0033'ü üç dönemi geçti, v4'ün hyp_0033'ü
        # bambaşka bir hipotez. Parmak izi tutmuyorsa araştırma Sharpe'ını
        # JOIN ETME — yanlış hipotezi raporlamak, hiç raporlamamaktan kötüdür.
        beklenen = hashler.get(hid)
        # Sicilde parmak iziyle kayıtlıysa ORASI esastır (kampanya bağımsız).
        if beklenen and beklenen in sicil:
            s_title, s_r, s_camp = sicil[beklenen]
            t = f"{s_title or ''}" + (f"  [{s_camp}]" if s_camp else "")
            r_sh = s_r
        elif beklenen and hj:
            try:
                from contracts.hypothesis_spec import HypothesisSpec
                from holdout.service import hypothesis_fingerprint
                if hypothesis_fingerprint(
                        HypothesisSpec.model_validate_json(hj)) != beklenen:
                    t, r_sh = "(BAŞKA kampanyanın aynı kimlikli hipotezi)", None
            except Exception:  # noqa: BLE001
                pass
        out.append((hid, t, r_sh, h_sh, ileri.get(hid)))
    out.sort(key=lambda r: (r[4] is None, -(r[4] or 0)))
    return out


def _uc_donem(memory_db: str, holdout_db: str, min_sharpe: float) -> str:
    """UC-DONEM HUKMU bolumu — kilitli donem tek basina yetmez."""
    from evaluation.three_period import final_verdict

    satirlar = _uc_donem_satirlari(memory_db, holdout_db)
    if not satirlar:
        return ('<div class="card desc">Kilitli dönem sınavı yapılmadı — '
                'üç-dönem hükmü için önce <code>python main.py --holdout</code>.</div>')
    _PILL = {"DOĞRULANDI": "good", "REJİM-BAĞIMLI": "warn",
             "ÇÖKTÜ": "bad", "EKSİK": "muted"}
    govde = ""
    dogrulanan = 0
    for hid, title, r_sh, h_sh, f_sh in satirlar:
        v = final_verdict(r_sh, h_sh, f_sh, min_sharpe)
        dogrulanan += int(v.passed)
        govde += (
            f"<tr><td>{_esc(hid)}</td><td>{_esc((title or '')[:44])}</td>"
            f'<td class="num">{("%+.2f" % r_sh) if r_sh is not None else "–"}</td>'
            f'<td class="num">{("%+.2f" % h_sh) if h_sh is not None else "–"}</td>'
            f'<td class="num">{("%+.2f" % f_sh) if f_sh is not None else "–"}</td>'
            f'<td><span class="pill {_PILL.get(v.verdict, "muted")}">'
            f'{_esc(v.verdict)}</span></td></tr>')
    if dogrulanan:
        son = (f'<p class="desc"><b>{dogrulanan}/{len(satirlar)}</b> aday İKİ '
               'bağımsız örneklem-dışı dönemde de ayakta kaldı. Ciddiye alınacak '
               'bir işaret — ama <b>“alpha bulundu” değil</b>: çok sayıda deneme '
               'içinden çıktı, çoklu-test düzeltmesi ayrıca kontrol edilmeli. '
               'Doğru okuma: bu aday <b>henüz ölmedi</b>.</p>')
    else:
        son = ('<p class="desc"><b>Hiçbir aday iki OOS döneminde birden ayakta '
               'kalamadı.</b> Kilitli dönemi geçmiş olmaları rejim şansıydı — '
               'tek bir kilitli dönem yeterli kanıt değildir. Sistem bunu '
               'yakaladı; “holdout geçti” diye erken sevinilmedi.</p>')
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Strateji</th>'
            '<th>Araştırma</th><th>Holdout</th><th>İleri-test</th><th>Hüküm</th>'
            '</tr>' + govde + "</table>" + son + "</div>")


def _holdout(holdout_db: str, memory_db: "str | None" = None) -> str:
    if not os.path.exists(holdout_db):
        return '<div class="card desc">Holdout değerlendirmesi yapılmadı.</div>'
    conn = sqlite3.connect(holdout_db)
    rows = _q(conn, "SELECT hypothesis_id, sharpe, passed FROM holdout_access"
                    + _aktif_kosul(conn) + " ORDER BY sharpe DESC")
    gecersiz = _q(conn, "SELECT hypothesis_id, sharpe, invalidation_reason "
                        "FROM holdout_access WHERE status='invalidated' "
                        "ORDER BY id") if _aktif_kosul(conn) else []
    conn.close()
    bilinen = _kampanya_adaylari(memory_db)
    if not rows:
        return '<div class="card desc">Holdout adayı yok.</div>'

    def _pill(passed) -> str:
        return ('<span class="pill good">GEÇTİ</span>' if passed
                else '<span class="pill bad">KALDI</span>')

    def _kaynak(h: str) -> str:
        if bilinen is None:
            return "–"
        if h in bilinen:
            return "kampanya"
        return '<span class="pill warn">elle sonda</span>'

    body = "".join(
        f'<tr><td>{_esc(h)}</td><td class="num">{s:.2f}</td><td>{_pill(p)}</td>'
        f'<td>{_kaynak(h)}</td></tr>'
        for h, s, p in rows)
    not_ = ""
    if bilinen is not None and any(h not in bilinen for h, _s, _p in rows):
        not_ = ('<p class="desc"><b>“elle sonda” ne demek:</b> bu kimlikler '
                'kampanya hafızasında yok — yani sistemin ürettiği hipotezler '
                'değil, insan tarafından elle yazılıp kilitli dönemde denenmiş '
                'varyantlar. Kilitli dönemin amacı tek-atışlık olmaktır; elle '
                'birden fazla varyant denemek bu korumayı zayıflatır ve sonuçları '
                'sistemin başarısı gibi okumak YANLIŞ olur. Dürüstlük için '
                'gizlenmiyor, ayrı işaretleniyor.</p>')
    if gecersiz:
        satir = "".join(
            f'<tr><td>{_esc(h)}</td><td class="num">{(sh or 0):.2f}</td>'
            f'<td colspan="2">{_esc(rsn or "—")}</td></tr>'
            for h, sh, rsn in gecersiz)
        not_ += ('<p class="desc" style="margin-top:14px"><b>GEÇERSİZ KILINMIŞ '
                 'eski sonuçlar</b> (silinmedi — kayıt dürüstlüğü). Bunlar hatalı '
                 'bir değerlendiriciyle üretildiği için sonuç sayılmıyor; '
                 'gerekçeleri aşağıda:</p>'
                 '<table><tr><th>Kimlik</th><th>Eski Sharpe</th>'
                 '<th>Geçersiz kılma gerekçesi</th></tr>' + satir + '</table>')
    return ('<div class="card"><table><tr><th>Kimlik</th><th>Holdout Sharpe</th>'
            '<th>Sonuç</th><th>Kaynak</th></tr>' + body + "</table>" + not_ + "</div>")


def _signal_formula(e: dict) -> str:
    """DSL sinyal ağacını okunabilir formüle çevir (görsel için)."""
    op = e["op"]
    if op == "field":
        return e.get("field") or "?"
    if op == "const":
        return str(e.get("value"))
    if op == "feature_ref":
        return e.get("name") or "?"
    inner = ", ".join(_signal_formula(i) for i in e.get("inputs", []) if isinstance(i, dict))
    w = f", pencere={e['window']}" if e.get("window") else ""
    return f"{op}({inner}{w})"


# --- Düz Türkçe çeviri (quant olmayan biri okuyunca anlasın) ---------------
_FIELD_TR = {
    "close": "kapanış fiyatı", "open": "açılış fiyatı", "high": "gün-içi en yüksek",
    "low": "gün-içi en düşük", "adjusted_close": "düzeltilmiş kapanış",
    "volume": "işlem hacmi", "dollar_volume": "dolar hacmi", "market_cap": "piyasa değeri",
    "funding_rate": "funding oranı", "book_to_market": "defter/piyasa değeri (ucuzluk)",
    "roe": "özkaynak kârlılığı (ROE)",
}


def _describe_node(e: dict, feats: dict) -> str:
    """DSL düğümünü düz Türkçe bir ifadeye çevirir (özyinelemeli)."""
    op = e.get("op")
    if op == "field":
        return _FIELD_TR.get(e.get("field"), e.get("field") or "?")
    if op == "const":
        return str(e.get("value"))
    if op == "feature_ref":
        name = e.get("name") or "?"
        return _describe_node(feats[name], feats) if name in feats else name
    ins = [i for i in e.get("inputs", []) if isinstance(i, dict)]
    a = _describe_node(ins[0], feats) if ins else "?"
    b = _describe_node(ins[1], feats) if len(ins) > 1 else "?"
    w = e.get("window")
    gun = f"{w} günlük" if w else ""
    if op == "return":
        return f"son {gun} getirisi"
    if op in ("rolling_mean", "ewma"):
        return f"{a} — {gun} ortalaması"
    if op in ("rolling_std", "volatility"):
        return f"{a} — {gun} oynaklığı"
    if op == "zscore":
        return f"{a} — {gun} z-skoru (ortalamadan kaç std sapmış)"
    if op == "delta":
        return f"{a} — {gun} değişimi"
    if op in ("rolling_min",):
        return f"{a} — {gun} dip"
    if op in ("rolling_max",):
        return f"{a} — {gun} tepe"
    if op in ("cross_sectional_rank", "quantile", "rolling_rank"):
        return f"{a} sıralaması"
    if op in ("normalize", "demean", "neutralize_market", "neutralize_sector"):
        return f"piyasadan arındırılmış {a}"
    if op == "winsorize":
        return f"aşırı uçları budanmış {a}"
    if op == "negate":
        return f"{a} (tersi — düşük olanı öne alır)"
    if op == "multiply":
        return f"{a} × {b}"
    if op in ("divide", "ratio"):
        return f"{a} / {b}"
    if op == "add":
        return f"{a} + {b}"
    if op == "subtract":
        return f"{a} eksi {b}"
    if op == "greater_than":
        return f"{a} > {b} koşulu"
    if op == "less_than":
        return f"{a} < {b} koşulu"
    if op == "conditional":
        c = _describe_node(ins[2], feats) if len(ins) > 2 else "?"
        return f"eğer {a} ise {b}, değilse {c}"
    if op == "correlation":
        return f"{a} ile {b} korelasyonu ({gun})"
    if op == "intraday_range":
        return f"gün-içi fiyat aralığı (yüksek-düşük)/kapanış{(' — ' + gun + ' ort') if gun else ''}"
    if op == "close_location":
        return f"kapanışın gün-içi aralıktaki yeri (tepeye/dibe yakınlık){(' — ' + gun + ' ort') if gun else ''}"
    return f"{op}({a})"


# Yönü koruyup yalnızca ölçekleyen/sıralayan katmanlar — çekirdek metriği bulmak
# için bunları soyarız (sinyalin AL/SAT yönünü değiştirmezler).
_RANK_LIKE_OPS = {"cross_sectional_rank", "quantile", "rolling_rank"}


def _core_and_direction(e: dict, feats: dict) -> "tuple[str, bool]":
    """Sinyal ağacından çekirdek metriği ve YÖNÜ çıkarır.

    negate/rank/feature_ref katmanlarını soyar. Döner: (çekirdek düz-Türkçe,
    ters_mi). ters_mi=True ise DÜŞÜK değer AL demektir (negate sayısı tek).
    Böylece "tersi — düşük olanı öne alır" gibi kafa karıştıran ara ifadeler
    yerine doğrudan "en düşük olanı AL" diyebiliriz.
    """
    inverted = False
    node = e
    seen = 0
    while isinstance(node, dict) and seen < 12:
        seen += 1
        op = node.get("op")
        ins = [i for i in node.get("inputs", []) if isinstance(i, dict)]
        if op == "negate" and ins:
            inverted = not inverted
            node = ins[0]
        elif op in _RANK_LIKE_OPS and ins:
            node = ins[0]                       # sıralama: yönü korur
        elif op == "feature_ref":
            name = node.get("name")
            if name in feats:
                node = feats[name]              # feature tanımına in
            else:
                return name or "?", inverted
        else:
            break
    return _describe_node(node, feats), inverted


_REBALANCE_TR = {"daily": "her bar", "weekly": "haftalık", "monthly": "aylık"}
_MODEL_TR = {
    "linear_regression": "doğrusal regresyon", "ridge": "ridge regresyon",
    "naive_bayes": "olasılık (naive Bayes)", "random_forest": "rastgele orman",
    "gradient_boosting": "gradyan artırma",
}


def _fields_used(e: dict, feats: dict, acc: set) -> None:
    """Bir ifade ağacında kullanılan ham veri alanlarını (düz Türkçe) toplar."""
    if not isinstance(e, dict):
        return
    op = e.get("op")
    if op == "field":
        acc.add(_FIELD_TR.get(e.get("field"), e.get("field") or "?"))
    elif op == "feature_ref" and e.get("name") in feats:
        _fields_used(feats[e["name"]], feats, acc)
    for i in e.get("inputs", []):
        _fields_used(i, feats, acc)


def _rebalance_tail(h: dict) -> str:
    ex = h.get("execution", {})
    reb, hold = ex.get("rebalance"), ex.get("holding_period_days")
    if reb and reb != "daily":
        return f" Pozisyonlar {_REBALANCE_TR.get(reb, reb)} yenilenir."
    if hold and int(hold) > 1:
        return f" Her pozisyon ~{int(hold)} bar tutulur."
    return ""


def _plain_strategy(h: dict) -> str:
    """Hipotezi tek cümlelik, TRADER'IN ANLAYACAĞI düz Türkçeye çevirir.

    - Model varsa (random_forest vb.): "model şu göstergelerden getiriyi tahmin
      eder; en yüksek beklenen AL, en düşük SAT."
    - Formül ise: "<metrik> en <yüksek/düşük> olanı AL, ...SAT." Yön negate'ten okunur.
    """
    feats = {f.get("name"): f.get("expression", {})
             for f in h.get("features", []) if isinstance(f, dict)}
    ptype = h.get("portfolio", {}).get("type", "")
    tail = _rebalance_tail(h)
    mtype = h.get("model", {}).get("type", "dsl_formula")

    # --- Model modu: sinyal DSL değil, model tahmini. Dürüst anlatım. ---
    if mtype != "dsl_formula":
        fset: set = set()
        for f in h.get("features", []):
            if isinstance(f, dict):
                _fields_used(f.get("expression", {}), feats, fset)
        flist = ", ".join(sorted(fset)) if fset else "fiyat/hacim"
        model_tr = _MODEL_TR.get(mtype, mtype)
        long_short = "long_short" in ptype
        sat = ", en düşük getiri beklenenleri açığa SAT" if long_short else ""
        return (f"Bir <b>{model_tr}</b> modeli, <b>{flist}</b> göstergelerinden her "
                f"varlığın yakın gelecekteki getirisini tahmin eder; en yüksek getiri "
                f"beklenen varlıkları AL{sat}." + tail)

    # --- Formül modu: yönü negate'ten oku, doğrudan AL/SAT söyle. ---
    core, inverted = _core_and_direction(h.get("signal", {}), feats)
    hi_al = not inverted
    yuksek, dusuk = "en <b>yüksek</b>", "en <b>düşük</b>"
    al_uc, sat_uc = (yuksek, dusuk) if hi_al else (dusuk, yuksek)
    if "long_short" in ptype:
        action = f"<b>{core}</b> {al_uc} olan varlıkları AL, {sat_uc} olanları açığa SAT"
    elif "long_only" in ptype:
        action = f"<b>{core}</b> {al_uc} olan varlıkları AL (sadece alım yapar)"
    else:
        action = f"<b>{core}</b> değerine göre varlık seçer"
    return action + "." + tail


# İssue tipi -> insan-dostu Türkçe başlık (reddetme nedeni)
_REASON_TR = {
    "compile_error": "Derlenmedi (geçersiz strateji yapısı)",
    "lookahead": "Geleceğe bakma (sızıntı) tespit edildi",
    "leakage": "Veri sızıntısı tespit edildi",
    "disallowed_field": "İzin verilmeyen veri alanı kullandı",
    "disallowed_operator": "İzin verilmeyen operatör kullandı",
    "yapısal_duplicate": "Daha önce denenen bir stratejiyle aynı (yapısal tekrar)",
    "davranışsal_duplicate": "Başka bir stratejiyle neredeyse aynı sinyali üretti (tekrar)",
    "metinsel_duplicate": "Açıklaması daha önceki bir hipotezle neredeyse aynı (metinsel tekrar)",
    "not_robust": "Sağlamlık testlerini geçemedi (şansa/ayara aşırı bağımlı)",
    "claim_signal_mismatch": "İddia ile sinyal uyuşmuyor (critic reddi)",
    "sharpe_below_threshold": "Getiri/risk (Sharpe) eşiğin altında",
    "drawdown_exceeded": "Maksimum düşüş sınırı aşıldı",
    "turnover_exceeded": "İşlem sıklığı (turnover) sınırı aşıldı",
    "insufficient_positive_folds": "Dönemler arası tutarsız (yeterli pozitif fold yok)",
}


def _humanize_issue(issues_json: str | None) -> str:
    """issues_json'daki ilk sorunu insan-dostu bir cümleye çevirir."""
    if not issues_json:
        return "—"
    try:
        issues = json.loads(issues_json)
    except (json.JSONDecodeError, TypeError):
        return "—"
    if not issues:
        return "—"
    it = issues[0]
    typ = it.get("type", "")
    label = _REASON_TR.get(typ)
    desc = it.get("description", "")
    if label:
        return f"{label}" + (f" — {desc}" if desc else "")
    return desc or typ or "—"


def _details(conn) -> str:
    """Hipotez detayı (Doküman 20) — kabul edilen stratejilerin TAM içeriği."""
    rows = _q(conn, """SELECT hypothesis_id, sharpe, hypothesis_json, model_name,
                              prompt_hash, seed, returns_json
                       FROM experiment
                       WHERE decision='accept' AND hypothesis_json IS NOT NULL
                       ORDER BY sharpe DESC LIMIT 6""")
    if not rows:
        return '<div class="card desc">Detay gösterilecek kabul edilmiş strateji yok.</div>'
    cards = []
    for hid, sharpe, hj, model_name, prompt_hash, seed, rj in rows:
        h = json.loads(hj)
        mech = h.get("economic_mechanism", {})
        fails = mech.get("expected_failure_conditions", []) or []
        f = h.get("falsification", {})
        tot = _total_return_pct(rj)
        ret_txt = (f' · <span style="color:var(--{"good" if tot >= 0 else "bad"})">'
                   f'getiri {tot:+.0f}%</span>') if tot is not None else ""
        # sharpe None olabilir (metriksiz kabul kaydı: geriye-dönük kayıtlar,
        # backfill sırası, elle eklenen kayıt). Biçimlendirme çökmemeli —
        # dashboard'ın tamamı tek bir eksik sayı yüzünden üretilememişti.
        sharpe_txt = f"Sharpe {sharpe:.2f}" if sharpe is not None else "Sharpe —"
        cards.append(f"""<div class="detail">
  <div class="dh"><span class="did">{_esc(hid)}</span> {_esc(h.get('title',''))}
    <span class="dsh">{sharpe_txt}{ret_txt}</span></div>
  <div class="drow"><b>Ne yapıyor (düz anlatım):</b> {_plain_strategy(h)}</div>
  <div class="drow"><b>İddia:</b> {_esc(h.get('claim',''))}</div>
  <div class="drow"><b>Ekonomik mekanizma:</b> {_esc(mech.get('type',''))} — {_esc(mech.get('description',''))}</div>
  <div class="drow"><b>Beklenen başarısızlık koşulları:</b> {_esc(', '.join(fails) or '—')}</div>
  <div class="drow"><b>Aile / portföy:</b> {_esc(h.get('family',''))} · {_esc(h.get('portfolio',{}).get('type',''))}</div>
  <div class="drow"><b>Sinyal (DSL formülü):</b><br><code>{_esc(_signal_formula(h.get('signal',{})))}</code></div>
  <div class="drow"><b>Çürütme eşiği (ön kayıt):</b> min OOS Sharpe {f.get('minimum_oos_sharpe','—')}, maks turnover {f.get('maximum_turnover','—')}, maks DD {f.get('maximum_drawdown','—')}</div>
  <div class="drow"><b>Tekrar-üretilebilirlik:</b> model {_esc(model_name or '—')} · prompt {_esc(prompt_hash or '—')} · seed {_esc(seed if seed is not None else '—')}</div>
</div>""")
    return "".join(cards)


def _all_hypotheses(conn) -> str:
    """Denenen HER hipotez (kabul+red) — düz Türkçe strateji + sonuç + neden.

    Bu bölüm kampanyanın asıl hikâyesidir: LLM ne denedi, ne oldu, NİYE. Kabul
    çıkmasa bile (gerçek veride sık olur) sistemin ne yaptığı buradan anlaşılır.
    """
    rows = _q(conn, """SELECT hypothesis_id, title, family, decision, sharpe,
                              hypothesis_json, issues_json, stage,
                              parent_hypothesis_id
                       FROM experiment ORDER BY id""")
    if not rows:
        return '<div class="card desc">Henüz hipotez üretilmedi.</div>'

    def _pill(dec: str) -> str:
        if dec == "accept":
            return '<span class="pill good">KABUL</span>'
        if dec == "duplicate":
            return '<span class="pill muted">TEKRAR</span>'
        return '<span class="pill bad">RED</span>'

    # Parametre-arama denemeleri LLM hipotezi DEĞİL, optimizer'ın pencere
    # varyantları — detay listesini boğmasınlar; parent başına TEK satıra katla.
    # (Çoklu-test sayımında yine tam olarak yer alırlar.)
    param_counts: dict = {}
    main_rows = []
    for r in rows:
        if r[7] == "parameter_search":
            param_counts[r[8] or "?"] = param_counts.get(r[8] or "?", 0) + 1
        else:
            main_rows.append(r)

    cards = []
    for hid, title, family, dec, sharpe, hj, issues, _stage, _parent in main_rows:
        h = json.loads(hj) if hj else {}
        plain = _plain_strategy(h) if h else "—"
        sh = f' · araştırma Sharpe {sharpe:.2f}' if sharpe is not None else ""
        if dec == "accept":
            reason = "Tüm süzgeçlerden geçti (sızıntı, performans, sağlamlık)."
        else:
            reason = _humanize_issue(issues)
        extra = ""
        if hid in param_counts:
            extra = (f'<div class="drow"><b>Parametre araması:</b> optimizer bu '
                     f'hipotezin pencerelerinde {param_counts[hid]} varyant denedi '
                     f'(hepsi çoklu-test sayımında).</div>')
        cards.append(f"""<div class="detail">
  <div class="dh"><span class="did">{_esc(hid)}</span> {_esc(title or '')}
    <span style="float:right">{_pill(dec)}</span></div>
  <div class="drow"><b>Aile:</b> {_esc(family or '—')}{_esc(sh)}</div>
  <div class="drow"><b>Ne yapıyor:</b> {plain}</div>
  <div class="drow"><b>Sonuç / neden:</b> {_esc(reason)}</div>
  {extra}
</div>""")
    return "".join(cards)


def _lineage(conn) -> str:
    """Hipotez soy ağacı (Doküman 13/20) — parent -> child ilişkileri."""
    rows = _q(conn, """SELECT parent_hypothesis_id, hypothesis_id, relation_type, decision
                       FROM experiment WHERE parent_hypothesis_id IS NOT NULL
                       ORDER BY id""")
    if not rows:
        return '<div class="card desc">Henüz türetilmiş (revision/inversion) hipotez yok.</div>'
    body = "".join(
        f'<tr><td>{_esc(p)}</td><td>→ {_esc(rel or "?")} →</td><td>{_esc(c)}</td>'
        f'<td>{_esc(dec)}</td></tr>'
        for p, c, rel, dec in rows)
    return ('<div class="card"><table><tr><th>Ebeveyn</th><th>İlişki</th>'
            '<th>Türev</th><th>Sonuç</th></tr>' + body + "</table></div>")


def _render_report(rep: dict) -> str:
    """Bir ReviewReport (dict) -> renkli kontrol listesi HTML."""
    pill = {"ok": "good", "warn": "warn", "fail": "bad"}
    label = {"ok": "TEMİZ", "warn": "DİKKAT", "fail": "SORUN"}
    v = rep.get("verdict", "ok")
    head = (f'<div class="drow"><b>{_esc(rep.get("reviewer",""))}:</b> '
            f'<span class="pill {pill.get(v,"muted")}">{label.get(v, v)}</span></div>')
    items = "".join(
        f'<div class="drow" style="margin-left:10px">'
        f'<span class="pill {pill.get(c.get("status"),"muted")}">'
        f'{_esc(c.get("status"))}</span> '
        f'<b>{_esc(c.get("name"))}:</b> {_esc(c.get("detail"))}</div>'
        for c in rep.get("checks", []))
    return head + items


def _reviewers(memory_db: str, bars_per_year: int = 252) -> str:
    """Bağımsız reviewer ajanları (Doküman 15): Backtest Auditor + Statistical Reviewer.

    Auditor raporu kabul sırasında saklanır (reviews_json); Statistical Reviewer
    çoklu-test satırından rapor-zamanı hesaplanır.
    """
    from agents.statistical_reviewer import StatisticalReviewer

    store = MemoryStore(memory_db)
    rows = build_report(store.backtested_experiments(), bars_per_year=bars_per_year)
    store.close()
    stat_by_hid = {r.hypothesis_id: r for r in rows}

    conn = sqlite3.connect(memory_db)
    accepted = _q(conn, """SELECT hypothesis_id, title, reviews_json FROM experiment
                           WHERE decision='accept' ORDER BY sharpe DESC LIMIT 6""")
    conn.close()
    if not accepted:
        return '<div class="card desc">Kabul edilmiş strateji yok — reviewer raporu üretilmedi.</div>'

    reviewer = StatisticalReviewer()
    cards = []
    for hid, title, reviews_json in accepted:
        blocks = []
        if reviews_json:
            try:
                for rep in json.loads(reviews_json):
                    blocks.append(_render_report(rep))
            except (json.JSONDecodeError, TypeError):
                pass
        if hid in stat_by_hid:
            blocks.append(_render_report(reviewer.review(stat_by_hid[hid]).model_dump()))
        cards.append(f'<div class="detail"><div class="dh">'
                     f'<span class="did">{_esc(hid)}</span> {_esc(title)}</div>'
                     + "".join(blocks) + "</div>")
    return "".join(cards)


def _procedural(memory_db: str) -> str:
    """Procedural memory (Doküman 12.3): hangi araştırma hamlesi işe yaradı."""
    from memory.procedural import build_procedural_lessons
    store = MemoryStore(memory_db)
    lessons = build_procedural_lessons(store)
    store.close()
    if not lessons:
        return ('<div class="card desc">Henüz süreç dersi çıkmadı (türetilmiş '
                'hipotez / yeterli deney yok).</div>')
    items = "".join(f'<div class="drow">• {_esc(l)}</div>' for l in lessons)
    return f'<div class="card">{items}</div>'


def _families(conn) -> str:
    rows = _q(conn, """SELECT family,
                         SUM(CASE WHEN decision='accept' THEN 1 ELSE 0 END),
                         COUNT(*)
                       FROM experiment WHERE sharpe IS NOT NULL GROUP BY family
                       ORDER BY 3 DESC""")
    if not rows:
        return '<div class="card desc">Backtest edilen aile yok.</div>'
    mx = max([t for _, _, t in rows] + [1])
    out = []
    for fam, acc, tot in rows:
        w = int(100 * tot / mx)
        out.append(
            f'<div class="frow"><div class="lbl">{_esc(fam)}</div>'
            f'<div class="track"><div class="bar" style="width:{w}%"></div></div>'
            f'<div class="cnt">{acc}/{tot}</div></div>')
    return '<div class="card">' + "".join(out) + "</div>"


def generate_dashboard(memory_db: str, holdout_db: str, out_path: str,
                       campaign_name: str = "", bars_per_year: int = 252,
                       min_acceptance_sharpe: float = 0.5) -> str:
    """bars_per_year: Sharpe yıllıklaştırma ölçeği (hisse 252 / kripto 365 / 8h 1095).
    Verilmezse çoklu-test tablosu 252 varsayar ve leaderboard ile ÇELİŞİR."""
    conn = sqlite3.connect(memory_db)
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    parts = [
        f'<h1>Araştırma Paneli</h1>',
        f'<p class="lead">Kampanya: <b>{_esc(campaign_name)}</b> · Oluşturma: {ts}</p>',
        _banner(conn, holdout_db, memory_db),
        _section("Sonuç — sade anlatım (teknik terim yok)",
                 "Aşağıdaki bütün bölümler sürecin nasıl işlediğini gösterir. "
                 "Bu ilk bölüm tek soruyu cevaplar: sonuç ne, paraya ne oldu? "
                 "Alım-satım bilen ama makine öğrenmesi bilmeyen biri için yazıldı.",
                 _trader_ozeti(memory_db, holdout_db, bars_per_year)),
        _section("Üç-Dönem Hükmü — kilitli holdout TEK BAŞINA yetmez",
                 "Ölçüldü: kilitli dönemi geçen 3 adayın 3'ü de, sistemin hiç "
                 "görmediği taze veride (2025→bugün) çöktü. Holdout tek bir REJİM "
                 "çekilişidir. Bu yüzden hüküm İKİ bağımsız örneklem-dışı dönemin "
                 "birlikte değerlendirilmesinden çıkar; ölçülmemiş dönem 'geçti' "
                 "sayılmaz (EKSİK).",
                 _uc_donem(memory_db, holdout_db, min_acceptance_sharpe)),
        _section("Kampanya Özeti",
                 "Bu turda üretilen hipotezlerin karar dağılımı. 'Farklı yapı' = "
                 "pencereden bağımsız kaç farklı strateji YAPISI üretildi (yalnız "
                 "pencere değişikliği aynı yapı sayılır — gerçek çeşitlilik ölçüsü).",
                 _tiles(conn, memory_db)),
        _section("Araştırma Hunisi — Hipotezler Nerede Elendi?",
                 "Her hipotez soldan sağa bu aşamalardan geçer; bir aşamada elenirse "
                 "orada durur. Kırmızı = elendi, mavi = tekrar, yeşil = kabul. "
                 "Sağdaki sayı o aşamada sonlanan hipotez adedidir.",
                 _funnel(conn)),
        _section("Tüm Denenen Hipotezler — LLM Ne Denedi, Ne Oldu, Niçin?",
                 "Kampanyanın asıl hikâyesi. LLM'in ürettiği her hipotez düz Türkçe "
                 "olarak ne yaptığıyla birlikte listelenir; yanında sonucu (KABUL / RED "
                 "/ TEKRAR) ve — reddedildiyse — insan diliyle nedeni yazar. Kabul "
                 "çıkmasa bile sistemin neyi neden elediği buradan net görülür.",
                 _all_hypotheses(conn)),
        _section("En İyi Stratejiler",
                 "Tüm süzgeçlerden geçip kabul edilen stratejiler, araştırma dönemi "
                 "Sharpe oranına göre sıralı.",
                 _leaderboard(conn)),
        _section("Hipotez Detayı — Bir Strateji Neyden İbaret?",
                 "Leaderboard'daki kısa başlık yalnızca etikettir. Her hipotez aslında "
                 "şu zengin içeriği taşır: test edilebilir iddia, ekonomik mekanizma, "
                 "beklenen başarısızlık koşulları, asıl DSL sinyal formülü ve sonuçları "
                 "görmeden taahhüt edilen çürütme eşiği (ön kayıt).",
                 _details(conn)),
        _section("Çoklu Test Düzeltmesi — 'Kabul' ≠ 'İstatistiksel Geçerli'",
                 "Çok sayıda deneme yapıldığında yüksek bir Sharpe tesadüfen çıkabilir. "
                 "Deflated Sharpe (DSR) ve FDR bunu düzeltir: FDR 'GEÇTİ' değilse sonuç "
                 "istatistiksel olarak kanıtlanmış sayılmaz.",
                 _multiple_testing(memory_db, bars_per_year)),
        _section("Bağımsız Reviewer Ajanları (Doküman 15)",
                 "Üretici LLM'den AYRI, deterministik iki denetçi. Backtest Auditor "
                 "backtest'in GEÇERLİLİĞİNİ denetler (sızıntı/survivorship/maliyet/"
                 "likidite); Statistical Reviewer 'kabul' ile 'istatistiksel doğrulandı'yı "
                 "ayırır (FDR/DSR/güven aralığı/fold). TEMİZ/DİKKAT/SORUN her kontrol için.",
                 _reviewers(memory_db, bars_per_year)),
        _section("Çok Amaçlı Sıralama (Pareto)",
                 "Kabul edilen stratejiler tek Sharpe ile değil; Sharpe alt güven "
                 "sınırı, drawdown ve turnover birlikte değerlendirilir. Pareto-optimal "
                 "olanlar hiçbir boyutta başkasına tümüyle yenik düşmez — reward "
                 "hacking'e karşı ek bir süzgeç.",
                 _pareto(memory_db)),
        _section("Kilitli Dönem Sınavı (Holdout)",
                 "Araştırma sırasında hiç görülmeyen, kilitli bir dönemde yapılan son "
                 "test. Bir stratejinin gerçekten genelleyip genellemediği buradan "
                 "anlaşılır (araştırma ajanı bu veriye asla erişemez).",
                 _holdout(holdout_db, memory_db)),
        _section("Aile Performansı — Bütçe Dağılımı",
                 "Her strateji ailesinin kabul/toplam oranı. Sistem araştırma bütçesini "
                 "başarılı ailelere Thompson sampling (bandit) ile kaydırır.",
                 _families(conn)),
        _section("Hipotez Soy Ağacı (Lineage)",
                 "Bir hipotezin başka bir hipotezden nasıl türetildiği: revision "
                 "(champion'ı geliştir), inversion (başarısızı ters çevir). Araştırmanın "
                 "kör deneme değil, yönlü bir keşif olduğunu gösterir.",
                 _lineage(conn)),
        _section("Süreç Hafızası (Procedural Memory, Doküman 12.3)",
                 "Sistem yalnızca 'hangi faktör iyi'yi değil, 'hangi ARAŞTIRMA "
                 "HAMLESİ işe yarıyor'u da öğrenir: revizyon/ters-çevirme/birleştirme "
                 "kabul oranları, doygun aileler ve en çok elemenin yapıldığı aşama. "
                 "Bu dersler bir sonraki hipotez üretimine geri beslenir.",
                 _procedural(memory_db)),
    ]
    conn.close()
    doc = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Araştırma Paneli — {_esc(campaign_name)}</title>'
           f'<style>{_CSS}</style></head><body><div class="wrap">'
           + "".join(parts)
           + '<div class="foot">LLM Tabanlı Otonom Quant Araştırmacısı · '
             'otomatik üretilmiş rapor</div></div></body></html>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def _cli_main() -> None:
    """Komut satırından dashboard üret. `python -m dashboard` bunu çağırır."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Kampanya adını aktif config'ten oku (eski sabit isim yerine güncel ad).
    name = "arastirma_kampanyasi"
    try:
        import io as _io

        import yaml as _yaml
        _c = _yaml.safe_load(_io.open(os.path.join(here, "configs", "campaign.yaml"),
                                      encoding="utf-8"))
        name = _c["campaign"].get("name", name)
    except Exception:  # noqa: BLE001 — config yoksa genel ad
        pass
    # Yıllıklaştırma ölçeğini data.yaml'dan türet (veri YÜKLEMEDEN) — aksi halde
    # bu tek başına koşan rapor 252 varsayıp kampanya çıktısıyla çelişirdi.
    bpy = 252
    try:
        from data import bars_per_year_from_config
        _d = _yaml.safe_load(_io.open(os.path.join(here, "configs", "data.yaml"),
                                      encoding="utf-8"))
        bpy = bars_per_year_from_config(_d["data"])
    except Exception:  # noqa: BLE001
        pass
    out = generate_dashboard(
        os.path.join(here, "research_memory.sqlite"),
        os.path.join(here, "holdout_audit.sqlite"),
        os.path.join(here, "dashboard.html"),
        campaign_name=name, bars_per_year=bpy)
    print(f"Dashboard yazıldı: {out}")


if __name__ == "__main__":
    _cli_main()
