"""
İyileşme (öğrenme) eğrisi testleri — best-so-far + SVG üretimi.
"""
from __future__ import annotations

from evaluation.improvement import best_so_far, improvement_svg


def test_best_so_far_monotonic():
    """best-so-far kümülatif MAX olmalı — asla düşmez."""
    seq = [0.2, -0.5, 0.6, 0.1, 0.9, 0.3]
    bsf = best_so_far(seq)
    assert bsf == [0.2, 0.2, 0.6, 0.6, 0.9, 0.9], f"beklenmeyen: {bsf}"
    assert all(bsf[i] <= bsf[i + 1] for i in range(len(bsf) - 1)), "best-so-far düştü"
    print("  [ok] best_so_far monotonik artan (kümülatif max)")


def test_best_so_far_empty():
    assert best_so_far([]) == []
    print("  [ok] boş dizi güvenli")


def test_improvement_svg_valid():
    """Çok-çizgili SVG üretilmeli; her seri için polyline + lejant."""
    curves = {"random": [0.1, 0.3, 0.2, 0.5], "gp": [-0.2, 0.4, 0.6, 0.9]}
    svg = improvement_svg(curves, title="test")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<polyline") == 2, "her üretici için bir çizgi olmalı"
    assert "random" in svg and "gp" in svg, "lejant etiketleri eksik"
    print(f"  [ok] SVG üretildi ({len(svg)} byte, 2 çizgi + lejant)")


def main():
    test_best_so_far_monotonic()
    test_best_so_far_empty()
    test_improvement_svg_valid()
    print("OK — iyileşme eğrisi testleri geçti.")


if __name__ == "__main__":
    main()
