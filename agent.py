"""
Ajan kontrol paneli — TEK giriş noktası. Komut ezberlemek YOK.

    python agent.py        (veya agent.bat'a çift tıkla)

TASARIM: bu bir menü değil, bir KOKPİTTİR. Açılınca önce SONUÇ görürsün —
"elimizde ne var, işe yarıyor mu, neyi geçiyoruz" — komut listesi sonra gelir.
Eski hâli düz 10 maddelik bir listeydi; kullanıcı hangi maddenin cevabı hangi
soruya verdiğini bilemiyordu ve iki temel araç (Sharpe doğrulama, testler)
menüde HİÇ YOKTU (yalnız README'de yazıyordu).

Durum paneli SALT SQLITE okur (evaluation/aday.py, mode=ro) — saniyeler sürer,
veri indirmez, hiçbir şeye yazmaz. Ağır işler yalnız sen seçince koşar.

Alt işleri (kampanya/holdout/kıyas) kendi süreçlerinde çalıştırır; main.py ve
compare.py'ye DOKUNMAZ, onların üstünde bir kabuktur.
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def _python() -> str:
    """Alt komutları çalıştıracak yorumlayıcı. Proje .venv'ini TERCİH et — bağımlılıklar
    (openai, rich, pandas...) orada kuruludur. agent.py yanlışlıkla sistem python'uyla
    başlatılsa bile (openai yok → ModuleNotFoundError), alt işler .venv ile koşar."""
    for rel in (os.path.join(".venv", "Scripts", "python.exe"),   # Windows
                os.path.join(".venv", "bin", "python")):          # Unix
        cand = os.path.join(HERE, rel)
        if os.path.exists(cand):
            return cand
    return sys.executable


PY = _python()


def _console():
    from rich.console import Console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    return Console()


# ---------------------------------------------------------------------------
# MENÜ — soruya göre gruplanmış. (tuş, etiket, "bu hangi soruyu cevaplar", eylem)
# ---------------------------------------------------------------------------
_GRUPLAR = [
    ("SONUÇLARA BAK  (hızlı — veri indirmez)", [
        ("1", "Karne: üç dönemde ne oldu?",
         "Her adayın araştırma / kilitli holdout / taze ileri-test notu + hüküm.",
         "KARNE"),
        ("2", "Kıyas: rastgeleyi ve al-tut'u geçiyor muyuz?",
         "Hocanın başarı ölçütü. Her dönemde ayrı yarış + geçme matrisi.",
         "BENCHMARK"),
        ("3", "Dashboard'u aç (tarayıcı)",
         "Görsel rapor: leaderboard, çoklu-test, holdout, huni.",
         "DASHBOARD"),
        ("4", "Tek fikri baştan sona anlat",
         "BİR stratejiyi doğuşundan kararına izle. Sade veya teknik.",
         "ANATOMY"),
    ]),
    ("YENİ ÖLÇÜM YAP  (yavaş — veri indirir / LLM çağırır)", [
        ("5", "Kampanyayı sürdür (yeni hipotezler üret)",
         "Ajan fikir üretir, test eder, kabul/ret verir. Görünüm seçilir.",
         "CAMPAIGN"),
        ("6", "Yeni kampanya (hafızayı SIFIRLA)",
         "Baştan başlar. Önceki deneyler silinir (aday sicili korunur).",
         "CAMPAIGN_FRESH"),
        ("7", "Holdout sınavı (kilitli dönem, tek-atış)",
         "Kabul edilen adayları hiç görülmemiş kilitli dilimde sınar. LLM'siz.",
         ["main.py", "--holdout"]),
        ("8", "İleri-test (taze veri, 2025→bugün)",
         "Holdout'un canlı hâli. Tüm OOS dönemlerini yan yana koyar.",
         "FORWARD"),
    ]),
    ("DENETLE  (sayılar doğru mu?)", [
        ("9", "Sharpe gerçekten doğru mu? (Python + Excel)",
         "Motoru saf-NumPy ve Excel'le karşılaştırır; PnL zincirini gün gün açar.",
         ["scripts/verify_sharpe.py"]),
        ("t", "Bütün testleri koş",
         "45 test dosyası: sızıntı, ödül-hackleme, ısınma, hizalama, PIT veri.",
         ["-m", "tests.run_all"]),
        ("k", "LLM karşılaştırması",
         "5 modeli aynı veri/bütçeyle yarıştırır: hangisi daha iyi hipotez üretir.",
         ["compare.py"]),
        ("d", "Durum / ayarlar",
         "Aktif evren, model, bütçe, tarih aralığı (configs/*.yaml).",
         "STATUS"),
    ]),
]
_MENU = [m for _, grup in _GRUPLAR for m in grup] + [("0", "Çıkış", "", "QUIT")]


# ---------------------------------------------------------------------------
# DURUM PANELİ — açılışta görünen "elimizde ne var"
# ---------------------------------------------------------------------------
def _durum_ozeti() -> dict:
    """Salt-okunur durum. Hata olursa panel çökmez, 'bilinmiyor' der.

    Kontrol paneli projenin ana giriş noktasıdır: burada bir istisna, kullanıcı
    için projenin tamamen bozuk görünmesi demektir.
    """
    d = {"aday_sayisi": 0, "gecen": 0, "en_iyi": None, "hata": None}
    try:
        from evaluation.aday import tum_adaylar
        adaylar = tum_adaylar()
        d["aday_sayisi"] = len(adaylar)
        d["gecen"] = sum(1 for a in adaylar if a.hukum().passed)
        d["en_iyi"] = adaylar[0] if adaylar else None
    except Exception as e:  # noqa: BLE001
        d["hata"] = f"{type(e).__name__}: {e}"
    return d


def _log_yasi(ad: str) -> "str | None":
    """runs/<ad> ne kadar eski? Bayat log, silinmiş bir iddiayı yaşatabilir."""
    p = os.path.join(HERE, "runs", ad)
    if not os.path.exists(p):
        return None
    gun = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).days
    return "bugün" if gun == 0 else f"{gun} gün önce"


def _render_durum(console) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    d = _durum_ozeti()

    baslik = Text()
    baslik.append("AGENTIC QUANT", style="bold cyan")
    baslik.append("  ·  otonom quant araştırmacısı\n", style="cyan")
    baslik.append("hipotez üret → modele çevir → sızıntı-güvenli backtest → "
                  "çoklu-test → kilitli holdout → ileri-test", style="dim")
    console.print(Panel(baslik, border_style="cyan"))

    if d["hata"]:
        console.print(f"[yellow]Durum okunamadı ({d['hata']}). "
                      f"Menü yine de çalışır.[/yellow]")
        return
    if not d["aday_sayisi"]:
        console.print("[yellow]  Henüz kabul edilmiş aday yok.[/yellow]  "
                      "[dim]Başlamak için: [5] Kampanyayı sürdür[/dim]")
        return

    a = d["en_iyi"]
    v = a.hukum()
    renk = {"DOĞRULANDI": "green", "REJİM-BAĞIMLI": "yellow",
            "ÇÖKTÜ": "red", "EKSİK": "dim"}.get(v.verdict, "white")

    t = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    t.add_column("dönem", style="white", width=26)
    t.add_column("Sharpe", justify="right", width=8)
    t.add_column("", style="dim")
    for etiket, deger, not_ in [
            ("araştırma (in-sample)", a.research_sharpe, "kanıt DEĞİL — fikir burada seçildi"),
            ("HOLDOUT (kilitli) *OOS", a.holdout_sharpe, "tek-atış, hiç görülmemiş dilim"),
            ("İLERİ-TEST (taze) *OOS", a.forward_sharpe, "2025→bugün, el değmemiş")]:
        t.add_row(etiket, "  –" if deger is None else f"{deger:+.2f}", not_)

    ic = Text()
    ic.append(f"En güçlü aday: {a.hypothesis_id}\n", style="bold white")
    ic.append(f"{a.title[:64]}\n", style="dim")
    console.print(Panel(ic, border_style=renk, padding=(0, 1),
                        title=f"[{renk}]HÜKÜM: {v.verdict}[/{renk}]",
                        subtitle=f"[dim]{d['aday_sayisi']} aday · "
                                 f"üç dönemi geçen: {d['gecen']}[/dim]"))
    console.print(t)
    if v.reasons:
        console.print(f"  [dim]→ {v.reasons[-1]}[/dim]")

    # Bayat log uyarısı: kullanıcı runs/*.log dosyalarını "güncel gerçek"
    # sanabilir. Gerçek koşuda benchmark.log iki gün eskiydi ve koddan
    # SİLİNMİŞ bir iddiayı ("3/3 geçtik, eşik sağlandı") hâlâ gösteriyordu.
    yas = [(ad, _log_yasi(ad)) for ad in ("benchmark.log", "forward_test.log")]
    eski = [f"{ad} ({y})" for ad, y in yas if y and y != "bugün"]
    if eski:
        console.print(f"  [dim yellow]Not: {', '.join(eski)} — "
                      f"yeniden koşmadan güncel sayma.[/dim yellow]")


def _render_menu(console) -> None:
    from rich.markup import escape
    from rich.table import Table

    # KÖŞELİ PARANTEZ KAÇIRILMALI: rich "[t]" gibi bir dizgeyi BİÇİM ETİKETİ
    # sanar ve tanımadığı için sessizce yutar. Harf tuşları (t/k/d) menüde
    # görünmez oluyordu — madde duruyor ama hangi tuşa basılacağı kayboluyordu.
    # Sayılar ("[1]") tesadüfen etkilenmiyordu; hata yalnız harflerde görünürdü.
    for baslik, maddeler in _GRUPLAR:
        console.print(f"\n[bold]{baslik}[/bold]")
        t = Table(box=None, expand=True, pad_edge=False, show_header=False)
        t.add_column("", style="bold cyan", width=4, justify="right")
        t.add_column("", style="white", width=42)
        t.add_column("", style="dim", ratio=1)
        for key, label, desc, _ in maddeler:
            t.add_row(escape(f"[{key}]"), label, desc)
        console.print(t)
    console.print("\n  [bold cyan]" + escape("[0]") + "[/bold cyan] Çıkış")


# ---------------------------------------------------------------------------
def _run(console, args: list[str]) -> None:
    """Alt komutu KENDİ sürecinde çalıştır (canlı panel/çıktı doğrudan terminale)."""
    console.print(f"\n[dim]→ {' '.join(args)} çalıştırılıyor (.venv python)…[/dim]\n")
    try:
        subprocess.run([PY, *args], cwd=HERE, check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Kullanıcı durdurdu.[/yellow]")
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]Hata: {type(e).__name__}: {e}[/red]")


def _open_dashboard(console) -> None:
    path = os.path.join(HERE, "dashboard.html")
    if not os.path.exists(path):
        console.print("[yellow]dashboard.html yok — önce bir kampanya koş ([5] veya [6]).[/yellow]")
        return
    webbrowser.open(f"file://{path}")
    console.print(f"[green]Dashboard tarayıcıda açıldı:[/green] {path}")


def _show_karne(console) -> None:
    """Üç-dönem karnesi — hükmü tek kaynaktan (evaluation/three_period)."""
    from evaluation.aday import karne_satirlari, tum_adaylar
    from evaluation.three_period import verdict_table

    satirlar = karne_satirlari()
    if not satirlar:
        console.print("\n[yellow]Henüz aday yok — önce kampanya koş ([5]).[/yellow]")
        return
    console.print("\n[bold]ÜÇ-DÖNEM KARNESİ[/bold]  "
                  "[dim](araştırma kanıt DEĞİL; hüküm iki OOS dönemine bakar)[/dim]\n")
    console.print(verdict_table(satirlar))
    console.print("\n[dim]DOĞRULANDI = iki bağımsız OOS döneminde de ayakta kaldı "
                  "('alpha bulundu' demek DEĞİL).\n"
                  "REJİM-BAĞIMLI = bir dönemde tuttu, diğerinde çöktü.[/dim]\n")
    for a in tum_adaylar():
        v = a.hukum()
        console.print(f"  [bold]{a.hypothesis_id}[/bold] — {a.title[:60]}")
        console.print(f"    [dim]{v.verdict}: {v.reasons[0] if v.reasons else ''}[/dim]")


def _show_status(console) -> None:
    import io

    import yaml
    from rich.table import Table

    def _y(name):
        with io.open(os.path.join(HERE, "configs", name), encoding="utf-8") as f:
            return yaml.safe_load(f)

    try:
        camp = _y("campaign.yaml")["campaign"]
        data = _y("data.yaml")["data"]
        models = _y("models.yaml")["models"]
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Config okunamadı: {e}[/red]")
        return
    t = Table(title="Aktif yapılandırma", box=None)
    t.add_column("", style="cyan"); t.add_column("", style="white")
    t.add_row("Kampanya", str(camp.get("name", "?")))
    t.add_row("Veri kaynağı", str(data.get("source", "?")))
    t.add_row("Evren", str(camp.get("universe", "?")))
    t.add_row("Model (sabit)", str(camp.get("model", "dsl_formula")))
    t.add_row("Üretici LLM", str(models.get("hypothesis_generator", {}).get("model", "?")))
    t.add_row("Bütçe", f"{camp.get('budget', {}).get('maximum_experiments', '?')} deney")
    t.add_row("İşlem maliyeti",
              f"{camp.get('budget', {}).get('cost_bps', '?')} bps "
              f"(tüm araçlar bunu kullanır)")
    t.add_row("Özgünlük eşiği",
              str(camp.get("risk_constraints", {}).get("min_originality", 0.0)))
    t.add_row("Tarih aralığı",
              f"{camp.get('start_date', '?')} → {camp.get('end_date', '?')}")
    t.add_row("", "[dim](bu tarihten SONRASI ileri-test dönemi)[/dim]")
    console.print(t)


def main() -> None:
    from rich.prompt import Prompt
    console = _console()
    actions = {k: a for k, _, _, a in _MENU}
    while True:
        console.print()
        _render_durum(console)
        _render_menu(console)
        # EOF/Ctrl+C'de CIRKIN TRACEBACK basma. "Enter'a bas" adiminda zaten
        # yakalaniyordu ama SECIM sorusunda yakalanmiyordu: girdi borudan
        # gelirse ya da kullanici Ctrl+D/Ctrl+C yaparsa panel stack trace ile
        # cokuyordu (kontrol paneli, projenin ana giris noktasi).
        try:
            choice = Prompt.ask("\nSeçim", choices=[k for k, *_ in _MENU],
                                default="1", show_choices=False)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Görüşürüz.[/dim]")
            return
        action = actions[choice]
        if action == "QUIT":
            console.print("[dim]Görüşürüz.[/dim]")
            return
        if action == "DASHBOARD":
            _open_dashboard(console)
        elif action == "KARNE":
            _show_karne(console)
        elif action in ("CAMPAIGN", "CAMPAIGN_FRESH"):
            if action == "CAMPAIGN_FRESH":
                ok = Prompt.ask("[yellow]Kampanya hafızası SİLİNECEK "
                                "(aday sicili ve holdout kaydı korunur). Emin misin?[/yellow]",
                                choices=["e", "h"], default="h")
                if ok != "e":
                    continue
            gorunum = Prompt.ask(
                "Görünüm? (p = canlı panel, d = detaylı akış [her adım], o = sade özet)",
                choices=["p", "d", "o"], default="d")
            args = ["main.py"]
            if action == "CAMPAIGN_FRESH":
                args.append("--fresh")
            if gorunum == "p":
                args.append("--live")
            elif gorunum == "d":
                args.append("--detay")
            _run(console, args)
        elif action == "ANATOMY":
            sade = Prompt.ask(
                "Anlatım nasıl olsun? (s = sade/herkes anlar, t = teknik/detaylı)",
                choices=["s", "t"], default="s")
            kaynak = Prompt.ask(
                "Hangi fikir? (b = BULUNAN aday [sicilden, üç-dönem karneli], "
                "y = yeni fikir üret)", choices=["b", "y"], default="b")
            args = ["scripts/anatomy.py", "--log"]
            if kaynak == "b":
                # Sicildeki gerçek adayı anlat: hocaya gösterilecek olan bu.
                args.append("--aday")
                if sade == "s":
                    args.append("--sade")   # sade mod aday desteklemezse teknik akar
            else:
                canli = Prompt.ask(
                    "Yapay zeka çağrılsın mı? (h = çağırma, hazır örnek — bedava/hızlı)",
                    choices=["e", "h"], default="h")
                if sade == "s":
                    args.append("--sade")
                if canli == "h":
                    args.append("--canned")
            _run(console, args)
        elif action == "BENCHMARK":
            # Taze dönem Binance'den indirilir; ilk koşuda uzun sürer. Kullanıcı
            # bilerek seçsin — habersiz 1 saatlik indirme "takıldı" sanılır.
            ileri = Prompt.ask(
                "Taze dönem (2025→bugün) de katılsın mı? "
                "(h = hayır/hızlı, e = evet — cache yoksa uzun sürer)",
                choices=["e", "h"], default="h")
            args = ["scripts/benchmark.py", "--log"]
            if ileri == "e":
                args.append("--ileri")
            _run(console, args)
        elif action == "FORWARD":
            _run(console, ["scripts/forward_test.py", "--log"])
        elif action == "STATUS":
            _show_status(console)
        elif isinstance(action, list):
            _run(console, action)
        console.print("\n[dim]— Menüye dönmek için Enter —[/dim]")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            return


if __name__ == "__main__":
    # agent.py sistem python'uyla başlatılmış olabilir (rich/pandas/openai orada
    # kurulu DEĞİL → import çökerdi). .venv varsa kendini onunla yeniden başlat.
    # Sonsuz döngü yok: PY zaten aktif yorumlayıcıysa (agent.bat .venv ile açtıysa)
    # bu blok atlanır.
    if os.path.exists(PY) and os.path.realpath(PY) != os.path.realpath(sys.executable):
        sys.exit(subprocess.run([PY, os.path.abspath(__file__), *sys.argv[1:]]).returncode)
    try:
        main()
    except ImportError as e:
        # .venv yok + sistem python'da bağımlılık yok: açık yönlendirme.
        print(f"\nEksik bağımlılık: {e}\nÇözüm: proje .venv'ini kur/aktive et "
              f"veya `pip install -r requirements.txt`.")
        sys.exit(1)
