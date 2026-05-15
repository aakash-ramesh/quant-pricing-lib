from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from datetime import date
import math

@dataclass
class DiscountCurve:
    """Simple log-linear zero-rate curve with continuously-compounded zeros.
    Pillars are (tenor_in_years, zero_rate_ccy).
    """
    ref_date: date
    pillars: List[Tuple[float, float]]  # (T, r_ccy)
    
    def __post_init__(self):
        self.pillars.sort(key=lambda x: x[0])
        assert self.pillars[0][0] >= 0.0

    def zero_rate(self, T: float) -> float:
        # Log-linear in zero yield: r(T) interpolates linearly between pillars in T
        if T <= self.pillars[0][0]:
            return self.pillars[0][1]
        for i in range(len(self.pillars)-1):
            t0, r0 = self.pillars[i]
            t1, r1 = self.pillars[i+1]
            if t0 <= T <= t1:
                w = (T - t0) / (t1 - t0) if t1 > t0 else 0.0
                return r0 * (1 - w) + r1 * w
        return self.pillars[-1][1]

    def df(self, T: float) -> float:
        return math.exp(-self.zero_rate(T) * T)

    def fwd_simple(self, T1: float, T2: float, daycount: float) -> float:
        # Simple forward rate implied by discount factors over [T1, T2], given accrual fraction
        if T2 <= T1:
            return 0.0
        d1, d2 = self.df(T1), self.df(T2)
        return (d1/d2 - 1.0) / daycount

    def bump(self, bp: float) -> "DiscountCurve":
        # parallel bump in basis points on zero rates
        bump_amt = bp / 10000.0
        new_pillars = [(t, r + bump_amt) for (t, r) in self.pillars]
        return DiscountCurve(self.ref_date, new_pillars)
