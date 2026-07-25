"""
Canlı ajan terminali — kampanyayı GERÇEK ZAMANLI izleme (rich TUI).

`python main.py --live` ile ajan çalışırken terminalde:
  - üst: kampanya kimliği (evren/model/bütçe) + anlık durum (deney i/N, kabul/red/
    dup, en iyi Sharpe, token)
  - orta: pipeline hunisi — her hipotez nerede eleniyor (üretim→sızıntı→özgünlük→
    critic→gate→sağlamlık→KABUL)
  - alt: son olayların akan, renkli listesi (kabul=yeşil, red=kırmızı, duplicate/
    düşük-özgünlük=sarı)

Tasarım ilkesi: bu katman SÜSTÜR, mantığı DEĞİŞTİRMEZ. run_campaign'e opsiyonel
`on_event(kind, info)` kancasıyla bağlanır; kanca verilmezse sistem eskisi gibi
düz metin basar (bkz. orchestrator/loop.py). Böylece TUI kırılsa bile araştırma
çekirdeği etkilenmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Olay -> pipeline hunisindeki aşama etiketi (kayıt STAGE'leriyle hizalı)
_STAGE_LABEL = {
    "generated": "Üretildi",
    "compile_error": "Derleme hatası",
    "static_rejected": "Sızıntı (statik)",
    "duplicate": "Tekrar (duplicate)",
    "low_originality": "Düşük özgünlük",
    "critic_rejected": "Critic reddi",
    "degenerate_conditional": "Dejenere koşul",
    "gate_rejected": "Gate (Sharpe/risk)",
    "robustness_rejected": "Sağlamlık",
    "accepted": "✓ KABUL",
}
# Huni sırası (yukarıdan aşağıya elenme yolu)
_FUNNEL_ORDER = ["generated", "static_rejected", "duplicate", "low_originality",
                 "critic_rejected", "degenerate_conditional", "gate_rejected",
                 "robustness_rejected", "accepted"]

_RESULT_STYLE = {
    "accepted": "bold green",
    "gate_rejected": "red",
    "robustness_rejected": "red",
    "static_rejected": "red",
    "compile_error": "red",
    "duplicate": "yellow",
    "low_originality": "yellow",
    "critic_rejected": "magenta",
    "degenerate_conditional": "magenta",
}


@dataclass
class _State:
    title: str = "Kampanya"
    universe: str = ""
    model: str = ""
    budget: int = 0
    done: int = 0
    accepts: int = 0
    rejects: int = 0
    dups: int = 0
    best_sharpe: float = float("-inf")
    best_title: str = "—"
    tokens: int = 0
    stage_counts: dict = field(default_factory=dict)
    feed: list = field(default_factory=list)   # (sıra, hid, başlık, stage, sharpe)


class LiveReporter:
    """run_campaign'in on_event kancasına bağlanan canlı görüntüleyici."""

    def __init__(self, max_feed: int = 14, file=None) -> None:
        self._s = _State()
        self._max_feed = max_feed
        self._file = file          # panel BU akışa basar (genelde gerçek stdout);
        self._live: Live | None = None   # loop'un print'lerini yutan redirect'ten
                                         # etkilenmemesi için ayrı tutulur

    # --- yaşam döngüsü -----------------------------------------------------
    def __enter__(self) -> "LiveReporter":
        # Windows: konsol cp1254'e düşerse emoji/blok karakteri encode EDEMEZ
        # (çökme). UTF-8'e geç ve rich'i ANSI moduna al (legacy Windows render'ı
        # atlanır) — modern terminalde tam görünür, eski terminalde çökmez.
        import sys
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — reconfigure yoksa (eski Python) yut
            pass
        from rich.console import Console
        self._console = Console(file=self._file, force_terminal=True,
                                legacy_windows=False)
        self._live = Live(self._render(), console=self._console,
                          refresh_per_second=8, screen=False)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live:
            self._live.update(self._render())
            self._live.__exit__(*exc)

    # --- olay girişi (on_event kancası buraya bağlanır) --------------------
    def on_event(self, kind: str, info: dict) -> None:
        s = self._s
        if kind == "start":
            s.title = info.get("title", "Kampanya")
            s.universe = info.get("universe", "")
            s.model = info.get("model", "")
            s.budget = int(info.get("budget", 0))
        elif kind == "decision":
            stage = info.get("stage", "generated")
            s.stage_counts[stage] = s.stage_counts.get(stage, 0) + 1
            s.stage_counts["generated"] = s.stage_counts.get("generated", 0) + 1
            sharpe = info.get("sharpe")
            if stage == "accepted":
                s.accepts += 1
                if sharpe is not None and sharpe > s.best_sharpe:
                    s.best_sharpe, s.best_title = sharpe, info.get("title", "—")
            elif stage in ("duplicate", "low_originality"):
                s.dups += 1
            else:
                s.rejects += 1
            s.done = s.accepts + s.rejects + s.dups
            s.tokens = int(info.get("tokens", s.tokens))
            s.feed.append((s.done, info.get("hid", "?"),
                           info.get("title", ""), stage, sharpe))
            s.feed = s.feed[-self._max_feed:]
        if self._live:
            self._live.update(self._render())

    # --- render ------------------------------------------------------------
    def _header(self) -> Panel:
        s = self._s
        sub = Text()
        sub.append(f"{s.universe}\n", style="dim")
        sub.append(f"model: {s.model}   bütçe: {s.budget} deney", style="dim cyan")
        return Panel(sub, title=f"🔬 AGENTIC QUANT — {s.title}",
                     border_style="cyan", padding=(0, 1))

    def _stats(self) -> Table:
        s = self._s
        t = Table.grid(expand=True)
        for _ in range(6):
            t.add_column(justify="center", ratio=1)
        best = "—" if s.best_sharpe == float("-inf") else f"{s.best_sharpe:.2f}"
        t.add_row(
            Text(f"Deney\n{s.done}/{s.budget}", justify="center"),
            Text(f"✓ Kabul\n{s.accepts}", style="bold green", justify="center"),
            Text(f"✗ Red\n{s.rejects}", style="red", justify="center"),
            Text(f"⊘ Tekrar\n{s.dups}", style="yellow", justify="center"),
            Text(f"En iyi Sharpe\n{best}", style="bold cyan", justify="center"),
            Text(f"Token\n{s.tokens:,}", style="dim", justify="center"))
        return t

    def _funnel(self) -> Table:
        s = self._s
        t = Table(title="Pipeline hunisi (nerede eleniyor)", title_style="dim",
                  box=None, expand=True, pad_edge=False)
        t.add_column("aşama", style="white")
        t.add_column("sayı", justify="right")
        t.add_column("", ratio=1)
        gen = max(s.stage_counts.get("generated", 0), 1)
        for stage in _FUNNEL_ORDER:
            c = s.stage_counts.get(stage, 0)
            if c == 0 and stage not in ("generated", "accepted"):
                continue
            bar = "█" * int(20 * c / gen)
            style = _RESULT_STYLE.get(stage, "white")
            t.add_row(_STAGE_LABEL.get(stage, stage), str(c),
                      Text(bar, style=style))
        return t

    def _feed(self) -> Table:
        t = Table(title="Canlı akış", title_style="dim", box=None,
                  expand=True, pad_edge=False)
        t.add_column("#", justify="right", style="dim", width=3)
        t.add_column("hipotez", style="white", ratio=3)
        t.add_column("sonuç", ratio=2)
        for seq, hid, title, stage, sharpe in self._s.feed:
            style = _RESULT_STYLE.get(stage, "white")
            label = _STAGE_LABEL.get(stage, stage)
            extra = f"  (Sharpe {sharpe:.2f})" if sharpe is not None else ""
            short = (title[:46] + "…") if len(title) > 47 else title
            t.add_row(str(seq), f"{hid}  {short}",
                      Text(label + extra, style=style))
        return t

    def _render(self) -> Group:
        return Group(self._header(), Text(""), self._stats(), Text(""),
                     self._funnel(), Text(""), self._feed())


def _demo() -> None:
    """Sahte olaylarla canlı paneli göster (LLM/veri gerektirmez)."""
    import random
    import time

    events = [
        ("hyp_0001", "Funding-crowding reversal in perpetuals", "gate_rejected", -0.16),
        ("hyp_0002", "Volatility-regime conditioned momentum", "duplicate", None),
        ("hyp_0003", "On-chain flow divergence with liquidity filter", "accepted", 0.71),
        ("hyp_0004", "Momentum with volume confirmation", "low_originality", None),
        ("hyp_0005", "Short-term reversal, illiquidity scaled", "robustness_rejected", 0.44),
        ("hyp_0006", "Cross-sectional funding × reversal interaction", "accepted", 0.83),
        ("hyp_0007", "52-week-high proximity trend", "static_rejected", None),
        ("hyp_0008", "Quality-tilted low volatility basket", "gate_rejected", 0.21),
        ("hyp_0009", "Liquidity-timed intraday-range breakout", "critic_rejected", None),
        ("hyp_0010", "Sentiment-shift confirmed momentum", "accepted", 0.66),
    ]
    tok = 0
    with LiveReporter() as r:
        r.on_event("start", {"title": "demo_kampanya",
                             "universe": "~660 kripto perpetual (funding + OHLCV)",
                             "model": "random_forest", "budget": len(events)})
        time.sleep(0.6)
        for hid, title, stage, sharpe in events:
            tok += random.randint(4000, 9000)
            r.on_event("decision", {"hid": hid, "title": title, "stage": stage,
                                    "sharpe": sharpe, "tokens": tok})
            time.sleep(0.6)
        time.sleep(1.2)
    print("\n[demo bitti] Gerçek koşu: python main.py --live")


if __name__ == "__main__":
    _demo()
