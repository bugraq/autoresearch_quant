"""
İyileşme (öğrenme) eğrisi — araştırma İTERASYONLAR boyunca daha iyi strateji
buluyor mu? (Doküman 26 araştırma verimliliği; Jiang/NVIDIA "line goes up" analoğu).

best-so-far = deney sırasına göre Sharpe'ların KÜMÜLATİF MAX'ı. Düz çizgi = arama
ilerlemiyor; yükselen çizgi = makine öğreniyor/iyileştiriyor. Birden çok üreticiyi
(LLM vs random/GP/Bayesian) aynı eksende kıyaslamak "beats the baseline" grafiğidir.

Çıktı: self-contained SVG (harici bağımlılık yok, light/dark uyumlu) — dashboard'a
gömülür veya tek başına tarayıcıda açılır.
"""
from __future__ import annotations


def best_so_far(sharpes: "list[float]") -> "list[float]":
    """Sharpe dizisinin kümülatif maksimumu (best-so-far eğrisi)."""
    out: list[float] = []
    cur = float("-inf")
    for s in sharpes:
        cur = max(cur, s)
        out.append(cur)
    return out


# Renk paleti (çizgiler) — koyu/açık temada da okunur.
_PALETTE = ["#4f8cff", "#ff8a3d", "#35c46b", "#c86bff", "#e5484d", "#f5c518"]


def improvement_svg(curves: "dict[str, list[float]]", title: str = "İyileşme eğrisi",
                    ylabel: str = "best-so-far Sharpe", width: int = 640,
                    height: int = 380) -> str:
    """Üretici -> best-so-far eğrisi sözlüğünü çok-çizgili SVG'ye çevirir."""
    pad_l, pad_r, pad_t, pad_b = 56, 130, 40, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    series = {lbl: best_so_far(v) for lbl, v in curves.items() if v}
    if not series:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="60">' \
               f'<text x="10" y="30">Veri yok</text></svg>'

    n_max = max(len(v) for v in series.values())
    all_y = [y for v in series.values() for y in v]
    y_min, y_max = min(all_y), max(all_y)
    if y_max - y_min < 1e-9:
        y_max, y_min = y_max + 0.5, y_min - 0.5

    def px(i: int) -> float:
        return pad_l + (plot_w * i / max(1, n_max - 1))

    def py(y: float) -> float:
        return pad_t + plot_h * (1 - (y - y_min) / (y_max - y_min))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui,sans-serif" font-size="12">',
        '<style>'
        ':root{--fg:#1a1a1a;--muted:#888;--grid:#e2e2e2;--bg:transparent}'
        '@media (prefers-color-scheme:dark){:root{--fg:#e8e8e8;--muted:#999;--grid:#333}}'
        'text{fill:var(--fg)} .muted{fill:var(--muted)} .grid{stroke:var(--grid)}'
        '</style>',
        f'<text x="{pad_l}" y="22" font-weight="600">{title}</text>',
    ]
    # Y ekseni ızgara + etiketler (5 çizgi)
    for k in range(5):
        yy = y_min + (y_max - y_min) * k / 4
        gy = py(yy)
        parts.append(f'<line class="grid" x1="{pad_l}" y1="{gy:.1f}" '
                     f'x2="{pad_l+plot_w}" y2="{gy:.1f}" stroke-width="1"/>')
        parts.append(f'<text class="muted" x="{pad_l-8}" y="{gy+4:.1f}" '
                     f'text-anchor="end">{yy:.2f}</text>')
    # Eksen başlıkları
    parts.append(f'<text class="muted" x="{pad_l+plot_w/2:.0f}" y="{height-12}" '
                 f'text-anchor="middle">deney sayısı →</text>')
    parts.append(f'<text class="muted" x="16" y="{pad_t+plot_h/2:.0f}" '
                 f'transform="rotate(-90 16 {pad_t+plot_h/2:.0f})" '
                 f'text-anchor="middle">{ylabel}</text>')
    # Çizgiler + lejant
    for idx, (lbl, v) in enumerate(series.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        pts = " ".join(f"{px(i):.1f},{py(y):.1f}" for i, y in enumerate(v))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2.5"/>')
        ly = pad_t + 6 + idx * 20
        parts.append(f'<line x1="{pad_l+plot_w+14}" y1="{ly}" x2="{pad_l+plot_w+34}" '
                     f'y2="{ly}" stroke="{color}" stroke-width="3"/>')
        final = v[-1]
        parts.append(f'<text x="{pad_l+plot_w+38}" y="{ly+4}">{lbl} '
                     f'<tspan class="muted">({final:.2f})</tspan></text>')
    parts.append('</svg>')
    return "".join(parts)
