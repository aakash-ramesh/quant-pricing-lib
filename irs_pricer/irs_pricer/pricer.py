from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from datetime import date
from .daycount import DAYCOUNT_MAP
from .curve import DiscountCurve
from .swap import Swap

@dataclass
class SwapPV:
    pv_fixed: float
    pv_float: float
    pv: float
    par_rate: float
    dv01: float

def _year_fraction(dc_name: str, d1: date, d2: date) -> float:
    return DAYCOUNT_MAP[dc_name](d1, d2)

def price_swap(curve: DiscountCurve, swap: Swap) -> SwapPV:
    # fixed leg PV
    fixed_cf_pv = 0.0
    accr_sum = 0.0
    sched = swap.fixed_leg.schedule
    for i in range(1, len(sched)):
        d1, d2 = sched[i-1], sched[i]
        accr = _year_fraction(swap.fixed_leg.daycount, d1, d2)
        T = (d2 - curve.ref_date).days / 365.0
        df = curve.df(max(T, 0.0))
        fixed_cf_pv += swap.fixed_leg.notional * swap.fixed_leg.rate * accr * df
        accr_sum += accr
    # add principal exchange? (for IRS, none on fixed leg)

    # float leg PV: par construction with forward rates from the curve
    float_cf_pv = 0.0
    fsched = swap.float_leg.schedule
    for i in range(1, len(fsched)):
        d1, d2 = fsched[i-1], fsched[i]
        accr = _year_fraction(swap.float_leg.daycount, d1, d2)
        T1 = (d1 - curve.ref_date).days / 365.0
        T2 = (d2 - curve.ref_date).days / 365.0
        fwd = curve.fwd_simple(max(T1,0.0), max(T2,0.0), accr) + swap.float_leg.index_spread
        df = curve.df(max(T2, 0.0))
        float_cf_pv += swap.float_leg.notional * fwd * accr * df
    # Add PV of notional exchange at maturity for floating (DF difference construction).
    # For a standard IRS with no exchange of notional, the expected value of floating coupons equals
    # notional*(1-DF(end)) when start ~ ref_date. We'll approximate by adding the DF(N) * notional - notional
    # adjustment relative to the fixed-leg structure to mimic par behavior.
    # Alternatively, compute explicitly: PV float = N*(DF(start)-DF(end)) + sum(spread*accr*DF).
    start_T = (swap.start_date - curve.ref_date).days / 365.0
    end_T = (swap.end_date - curve.ref_date).days / 365.0
    float_cf_pv += swap.float_leg.notional * (curve.df(max(start_T,0.0)) - curve.df(max(end_T,0.0)))

    pv_fixed = fixed_cf_pv
    pv_float = float_cf_pv
    pv = (pv_float - pv_fixed) if swap.payer == "payer" else (pv_fixed - pv_float)

    # Par rate (ignoring spread): PV_float / (N * sum accruals discounted approx as DF at pay dates)
    # Better: exact par rate = PV_float / sum(N*accr*DF(pay dates))
    denom = 0.0
    for i in range(1, len(sched)):
        d1, d2 = sched[i-1], sched[i]
        accr = _year_fraction(swap.fixed_leg.daycount, d1, d2)
        T = (d2 - curve.ref_date).days / 365.0
        denom += swap.fixed_leg.notional * accr * curve.df(max(T, 0.0))
    par_rate = (pv_float) / denom if denom != 0 else 0.0

    # DV01 via bumping +1bp
    bumped = curve.bump(1.0)
    pv_bumped = price_swap_no_dv01(bumped, swap).pv
    dv01 = (pv_bumped - pv) / 0.0001  # dollars per 1bp

    return SwapPV(pv_fixed=pv_fixed, pv_float=pv_float, pv=pv, par_rate=par_rate, dv01=dv01)

def price_swap_no_dv01(curve: DiscountCurve, swap: Swap) -> SwapPV:
    # helper so DV01 bump doesn't recurse infinitely
    fixed_cf_pv = 0.0
    accr_sum = 0.0
    sched = swap.fixed_leg.schedule
    for i in range(1, len(sched)):
        d1, d2 = sched[i-1], sched[i]
        accr = DAYCOUNT_MAP[swap.fixed_leg.daycount](d1, d2)
        T = (d2 - curve.ref_date).days / 365.0
        df = curve.df(max(T, 0.0))
        fixed_cf_pv += swap.fixed_leg.notional * swap.fixed_leg.rate * accr * df
        accr_sum += accr
    float_cf_pv = 0.0
    fsched = swap.float_leg.schedule
    for i in range(1, len(fsched)):
        d1, d2 = fsched[i-1], fsched[i]
        accr = DAYCOUNT_MAP[swap.float_leg.daycount](d1, d2)
        T1 = (d1 - curve.ref_date).days / 365.0
        T2 = (d2 - curve.ref_date).days / 365.0
        fwd = curve.fwd_simple(max(T1,0.0), max(T2,0.0), accr) + swap.float_leg.index_spread
        df = curve.df(max(T2, 0.0))
        float_cf_pv += swap.float_leg.notional * fwd * accr * df
    start_T = (swap.start_date - curve.ref_date).days / 365.0
    end_T = (swap.end_date - curve.ref_date).days / 365.0
    float_cf_pv += swap.float_leg.notional * (curve.df(max(start_T,0.0)) - curve.df(max(end_T,0.0)))

    pv_fixed = fixed_cf_pv
    pv_float = float_cf_pv
    pv = (pv_float - pv_fixed) if swap.payer == "payer" else (pv_fixed - pv_float)

    denom = 0.0
    for i in range(1, len(sched)):
        d1, d2 = sched[i-1], sched[i]
        accr = DAYCOUNT_MAP[swap.fixed_leg.daycount](d1, d2)
        T = (d2 - curve.ref_date).days / 365.0
        denom += swap.fixed_leg.notional * accr * curve.df(max(T, 0.0))
    par_rate = (pv_float) / denom if denom != 0 else 0.0

    return SwapPV(pv_fixed=pv_fixed, pv_float=pv_float, pv=pv, par_rate=par_rate, dv01=float("nan"))
