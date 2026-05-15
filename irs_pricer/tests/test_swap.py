from datetime import date
from irs_pricer.curve import DiscountCurve
from irs_pricer.swap import build_plain_vanilla_swap
from irs_pricer.pricer import price_swap

def build_synth_curve(ref=date(2025,10,15)):
    pillars = [
        (0.0, 0.015),
        (0.25, 0.017),
        (0.5, 0.018),
        (1.0, 0.020),
        (2.0, 0.022),
        (3.0, 0.0235),
        (5.0, 0.025),
        (7.0, 0.026),
        (10.0, 0.0275),
    ]
    return DiscountCurve(ref, pillars)

def test_par_swap_is_near_zero():
    curve = build_synth_curve()
    start = curve.ref_date
    end = date(2030,10,15)
    notional = 1_000_000
    # Find par rate from a guess
    guess = 0.02
    s0 = build_plain_vanilla_swap(curve.ref_date, start, end, notional, guess)
    par = price_swap(curve, s0).par_rate
    spar = build_plain_vanilla_swap(curve.ref_date, start, end, notional, par)
    pv = price_swap(curve, spar)
    assert abs(pv.pv) < 1000.0  # within $1k tolerance

def test_signs_payer_receiver():
    curve = build_synth_curve()
    start = curve.ref_date
    end = date(2030,10,15)
    notional = 1_000_000
    s0 = build_plain_vanilla_swap(curve.ref_date, start, end, notional, 0.02)
    par = price_swap(curve, s0).par_rate
    rich = par + 0.003
    payer = build_plain_vanilla_swap(curve.ref_date, start, end, notional, rich, payer="payer")
    receiver = build_plain_vanilla_swap(curve.ref_date, start, end, notional, rich, payer="receiver")
    pv_p = price_swap(curve, payer).pv
    pv_r = price_swap(curve, receiver).pv
    assert pv_p < 0.0 and pv_r > 0.0

def test_dv01_positive_for_receiver_when_rates_rise_value_drops():
    curve = build_synth_curve()
    start = curve.ref_date
    end = date(2030,10,15)
    notional = 1_000_000
    s0 = build_plain_vanilla_swap(curve.ref_date, start, end, notional, 0.02)
    par = price_swap(curve, s0).par_rate
    rec = build_plain_vanilla_swap(curve.ref_date, start, end, notional, par, payer="receiver")
    res = price_swap(curve, rec)
    # DV01 should be approximately positive number of dollars per bp for receiver (fixed receiver = long duration)
    assert res.dv01 < 0
