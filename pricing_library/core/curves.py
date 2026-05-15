from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from pricing_library.core.daycount import DayCount, year_fraction


@dataclass(frozen=True)
class DiscountCurve:
    """Continuously compounded zero-rate curve with linear interpolation in maturity."""

    ref_date: date
    pillars: tuple[tuple[float, float], ...]

    def __init__(self, ref_date: date, pillars: Iterable[tuple[float, float]]):
        sorted_pillars = tuple(sorted(pillars, key=lambda x: x[0]))
        if not sorted_pillars:
            raise ValueError("at least one curve pillar is required")
        if sorted_pillars[0][0] < 0:
            raise ValueError("curve pillars must have non-negative maturities")
        for left, right in zip(sorted_pillars, sorted_pillars[1:]):
            if right[0] <= left[0]:
                raise ValueError("curve pillar maturities must be strictly increasing")
        object.__setattr__(self, "ref_date", ref_date)
        object.__setattr__(self, "pillars", sorted_pillars)

    def time_from_reference(self, target: date, convention: str | DayCount = DayCount.ACT_365F) -> float:
        return year_fraction(self.ref_date, target, convention)

    def zero_rate(self, maturity: float) -> float:
        if maturity < 0:
            raise ValueError("maturity cannot be negative")
        if maturity <= self.pillars[0][0]:
            return self.pillars[0][1]

        for (t0, r0), (t1, r1) in zip(self.pillars, self.pillars[1:]):
            if t0 <= maturity <= t1:
                weight = (maturity - t0) / (t1 - t0)
                return r0 * (1.0 - weight) + r1 * weight
        return self.pillars[-1][1]

    def discount_factor(self, maturity: float) -> float:
        return math.exp(-self.zero_rate(maturity) * maturity)

    def discount_factor_for_date(self, target: date, convention: str | DayCount = DayCount.ACT_365F) -> float:
        return self.discount_factor(self.time_from_reference(target, convention))

    def forward_rate(self, start: float, end: float, accrual_fraction: float) -> float:
        if end <= start:
            raise ValueError("forward end must be after start")
        if accrual_fraction <= 0:
            raise ValueError("accrual fraction must be positive")
        df_start = self.discount_factor(start)
        df_end = self.discount_factor(end)
        return (df_start / df_end - 1.0) / accrual_fraction

    def bump_parallel(self, basis_points: float) -> "DiscountCurve":
        bump = basis_points / 10_000.0
        return DiscountCurve(self.ref_date, ((t, r + bump) for t, r in self.pillars))


class FlatCurve(DiscountCurve):
    """Flat continuously compounded curve."""

    def __init__(self, ref_date: date, rate: float):
        super().__init__(ref_date, ((0.0, rate), (100.0, rate)))

