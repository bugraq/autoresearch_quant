"""
Benchmark (maymun testi) doğrulama testleri.

İki iddia (ağ/dış veri GEREKTİRMEZ — sentetik veri):
  A) MASRAFSIZ rastgele al-satçı ~0 Sharpe üretir. Rastgele işlem bilgi
     taşımaz; motor bunu 0 civarı göstermeli. (Sistematik sapma varsa motorun
     kendisi sahte alpha üretiyor demektir — kritik kalibrasyon kontrolü.)
  B) Veriye GERÇEK sinyal gömülüyse, o sinyali kullanan strateji rastgele
     maymun dağılımının ÜSTÜNDE olur (aksi halde 'başarı' iddiası boş olur).
"""
import numpy as np

from data import gen_cross_sectional_momentum, gen_random
from scripts.benchmark import (
    _sharpe, _template_hyp, buy_and_hold, random_trader,
)


def _monkey_median_sharpe(data, cost_bps, n=60):
    tmpl = _template_hyp("sp500_point_in_time")
    bpy = getattr(data, "bars_per_year", 252)
    sh = [_sharpe(random_trader(data, tmpl, seed=s, cost_bps=cost_bps), bpy)
          for s in range(n)]
    return float(np.median(sh)), np.array(sh)


def test_costless_monkey_is_near_zero():
    """A: masrafsız rastgele al-satçının ortanca Sharpe'ı ~0 olmalı."""
    data = gen_random(n_sec=25, n_days=600, seed=7)
    med, _ = _monkey_median_sharpe(data, cost_bps=0.0, n=60)
    assert abs(med) < 0.5, f"masrafsız maymun ortancası ~0 değil: {med:+.2f}"
    print(f"  [ok] masrafsız rastgele al-satçı ortancası ~0: {med:+.2f} "
          f"(motor sahte alpha üretmiyor)")


def test_cost_drags_monkeys_down():
    """Masraf, rastgele (yüksek devirli) al-satçıyı aşağı çeker: net < gross."""
    data = gen_random(n_sec=25, n_days=600, seed=7)
    med_free, _ = _monkey_median_sharpe(data, cost_bps=0.0, n=40)
    med_cost, _ = _monkey_median_sharpe(data, cost_bps=5.0, n=40)
    assert med_cost < med_free, \
        f"masraf maymunu düşürmedi: masraflı {med_cost:+.2f} >= masrafsız {med_free:+.2f}"
    print(f"  [ok] masraf rastgele al-satçıyı düşürdü: "
          f"{med_free:+.2f} -> {med_cost:+.2f}")


def test_real_signal_beats_monkeys():
    """B: veriye gömülü momentum sinyalini kullanan strateji maymunları geçer."""
    from backtest.engine import compute_pnl
    from contracts.dsl import Expression
    from dsl import compile_hypothesis
    from backtest import evaluate_signal

    # Belirgin momentum (yüksek drift): gürültüye gömülü zayıf sinyalde maymun
    # dağılımı geniş olur; test'in amacı 'gerçek sinyal maymunları geçer'i
    # göstermek, sinyalin ne kadar güçlü kurulduğunu değil.
    data = gen_cross_sectional_momentum(n_sec=25, n_days=700, seed=3,
                                        drift_spread=0.0016)
    bpy = getattr(data, "bars_per_year", 252)

    # Gömülü sinyali kullanan strateji (masrafsız — saf sinyal gücü)
    tmpl = _template_hyp("sp500_point_in_time")
    mom = Expression(op="cross_sectional_rank", inputs=[
        Expression(op="return", window=60,
                   inputs=[Expression(op="field", field="close")])])
    tmpl.signal = mom
    sig = evaluate_signal(compile_hypothesis(tmpl), data)
    our_net, _ = compute_pnl(sig, tmpl, data, 0.0)
    our_sharpe = _sharpe(our_net, bpy)

    _, monkeys = _monkey_median_sharpe(data, cost_bps=0.0, n=60)
    pctile = float((monkeys < our_sharpe).mean() * 100)
    assert pctile >= 80, \
        f"gerçek sinyal maymunları geçemedi: %{pctile:.0f} yüzdelik, Sharpe={our_sharpe:+.2f}"
    print(f"  [ok] gerçek sinyal maymunların %{pctile:.0f}'ini geçti "
          f"(Sharpe={our_sharpe:+.2f})")


def test_buy_and_hold_runs():
    """Al-tut serisi üretiliyor ve sonlu (NaN/inf değil)."""
    data = gen_random(n_sec=20, n_days=400, seed=1)
    bh = buy_and_hold(data)
    assert len(bh) > 100 and np.isfinite(bh.to_numpy()).all()
    print(f"  [ok] al-tut serisi üretildi: {len(bh)} bar, "
          f"Sharpe={_sharpe(bh, 252):+.2f}")


if __name__ == "__main__":
    test_costless_monkey_is_near_zero()
    test_cost_drags_monkeys_down()
    test_real_signal_beats_monkeys()
    test_buy_and_hold_runs()
    print("OK — benchmark (maymun testi) doğrulandı: motor sahte alpha üretmiyor, "
          "gerçek sinyal maymunları geçiyor.")
