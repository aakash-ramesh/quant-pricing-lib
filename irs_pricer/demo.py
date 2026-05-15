from datetime import date
from irs_pricer.curve import DiscountCurve
from irs_pricer.swap import build_plain_vanilla_swap
from irs_pricer.pricer import price_swap

def build_synthetic_curve(ref=date(2025,10,15)):
    # Simple upward sloping curve (continuously-compounded zeros)
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

def main():
    curve = build_synthetic_curve()
    start = date(2025,10,15)
    end   = date(2030,10,15)
    notional = 1_000_000

    # Price a par swap by setting fixed rate ~ par_rate on first pass
    guess_rate = 0.025
    swap = build_plain_vanilla_swap(curve.ref_date, start, end, notional, guess_rate)
    pv0 = price_swap(curve, swap)
    print("Initial guess PV:", pv0)

    # Use computed par rate to build a near-par swap
    par_rate = pv0.par_rate
    par_swap = build_plain_vanilla_swap(curve.ref_date, start, end, notional, par_rate)
    pv_par = price_swap(curve, par_swap)
    print("Par swap (should be ~0 PV):", pv_par)

    # Bump fixed rate to see payer/receiver
    payer_swap = build_plain_vanilla_swap(curve.ref_date, start, end, notional, par_rate+0.002, payer="payer")
    receiver_swap = build_plain_vanilla_swap(curve.ref_date, start, end, notional, par_rate+0.002, payer="receiver")
    print("Payer PV (should be negative if fixed rate > par):", price_swap(curve, payer_swap))
    print("Receiver PV (should be positive if fixed rate > par):", price_swap(curve, receiver_swap))

if __name__ == "__main__":
    main()
