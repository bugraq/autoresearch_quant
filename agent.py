"""
Ajan kontrol paneli — TEK giriş noktası. Komut ezberlemek YOK.

    python agent.py        (veya agent.bat'a çift tıkla)

Açılır, menüden seçersin, iş biter, menüye döner. Alt işleri (kampanya/holdout/
karşılaştırma) kendi süreçlerinde çalıştırır — canlı terminal, dashboard, Deney A
hepsi buradan. main.py/compare.py'ye DOKUNMAZ; onların üstünde bir kabuktur.
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser

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


# (tuş, etiket, açıklama, eylem)  — eylem: ["dosya", "bayrak"...] | özel anahtar
_MENU = [
    ("1", "Kampanyayı izle (mevcut kampanyaya devam)",
     "Ajan hipotez üretir/test eder. Görünüm seçilir: canlı panel / DETAYLI akış "
     "(her adım tek tek) / sade özet.",
     "CAMPAIGN"),
    ("2", "Yeni kampanya (sıfırdan)",
     "Hafızayı SIFIRLAR ve baştan başlar. (Önceki deneyler silinir.) Görünüm seçilir.",
     "CAMPAIGN_FRESH"),
    ("3", "LLM karşılaştırması",
     "5 LLM'i birbiriyle yarıştırır (2 bedava + 2 ucuz + 1 orta): aynı veri/bütçeyle "
     "hangi model daha iyi hipotez üretiyor.",
     ["compare.py"]),
    ("4", "Holdout değerlendirmesi",
     "Kabul edilen adayları kilitli dönemde tek-atış sınar (LLM'siz, one-shot).",
     ["main.py", "--holdout"]),
    ("5", "Dashboard'u aç (tarayıcı)",
     "Son kampanyanın görsel raporu: leaderboard, çoklu-test, holdout, funnel.",
     "DASHBOARD"),
    ("6", "Tek fikri baştan sona anlat (sade veya teknik)",
     "BİR yatırım fikrini doğuşundan kararına izler. Sade mod: konuyu bilmeyen "
     "anlar. Teknik mod: prompt, sayısallaşma, eğitim, PnL, metrikler tam açık.",
     "ANATOMY"),
    ("7", "Kıyas: random/al-tut'u geçiyor muyuz?",
     "Bizim stratejiyi rastgele al-satçı (maymun), pasif al-tut ve duygusal "
     "trader ile aynı koşullarda yarıştırır (hocanın 'başarı' ölçütü).",
     "BENCHMARK"),
    ("8", "İleri-test (holdout'un canlı versiyonu)",
     "Kabul edilen stratejiyi sistemin gördüğü tarihten SONRAKİ taze veride koşar; "
     "tüm OOS dönemlerini yan yana koyup rejim-bağımlılığı yakalar. (İlk koşu birkaç dk.)",
     "FORWARD"),
    ("9", "Durum / ayarlar",
     "Aktif evren, model, bütçe ve veri kaynağını göster (configs/*.yaml).",
     "STATUS"),
    ("0", "Çıkış", "", "QUIT"),
]


def _render_menu(console) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    header = Text()
    header.append("Otonom quant araştırma ajanı\n", style="bold cyan")
    header.append("LLM hipotez → model → sızıntı-güvenli backtest → çoklu-test → holdout",
                  style="dim")
    console.print(Panel(header, title="🔬 AGENTIC QUANT", border_style="cyan"))

    t = Table(box=None, expand=True, pad_edge=False)
    t.add_column("", style="bold cyan", width=3, justify="right")
    t.add_column("", style="bold white")
    t.add_column("", style="dim", ratio=1)
    for key, label, desc, _ in _MENU:
        t.add_row(f"[{key}]", label, desc)
    console.print(t)


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
        console.print("[yellow]dashboard.html yok — önce bir kampanya koş (1 veya 2).[/yellow]")
        return
    webbrowser.open(f"file://{path}")
    console.print(f"[green]Dashboard tarayıcıda açıldı:[/green] {path}")


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
    t.add_row("Özgünlük eşiği",
              str(camp.get("risk_constraints", {}).get("min_originality", 0.0)))
    t.add_row("Tarih aralığı",
              f"{camp.get('start_date', '?')} → {camp.get('end_date', '?')}")
    console.print(t)


def main() -> None:
    from rich.prompt import Prompt
    console = _console()
    actions = {k: a for k, _, _, a in _MENU}
    while True:
        console.print()
        _render_menu(console)
        # EOF/Ctrl+C'de CIRKIN TRACEBACK basma. "Enter'a bas" adiminda zaten
        # yakalaniyordu ama SECIM sorusunda yakalanmiyordu: girdi borudan
        # gelirse ya da kullanici Ctrl+D/Ctrl+C yaparsa panel stack trace ile
        # cokuyordu (kontrol paneli, projenin ana giris noktasi).
        try:
            choice = Prompt.ask("\nSeçim", choices=[k for k, *_ in _MENU],
                                default="1")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Görüşürüz.[/dim]")
            return
        action = actions[choice]
        if action == "QUIT":
            console.print("[dim]Görüşürüz.[/dim]")
            return
        if action == "DASHBOARD":
            _open_dashboard(console)
        elif action in ("CAMPAIGN", "CAMPAIGN_FRESH"):
            if action == "CAMPAIGN_FRESH":
                ok = Prompt.ask("[yellow]Hafıza SİLİNECEK, emin misin?[/yellow]",
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
            _run(console, ["scripts/benchmark.py", "--log"])
        elif action == "FORWARD":
            _run(console, ["scripts/forward_test.py", "--log"])
        elif action == "STATUS":
            _show_status(console)
        elif isinstance(action, list):
            if action == ["main.py", "--fresh", "--live"]:
                ok = Prompt.ask("[yellow]Hafıza SİLİNECEK, emin misin?[/yellow]",
                                choices=["e", "h"], default="h")
                if ok != "e":
                    continue
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
