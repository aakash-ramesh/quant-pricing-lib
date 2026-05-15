from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

from pricing_library.core.daycount import DayCount, year_fraction

InterpolationMode = Literal["linear_zero", "linear_discount", "log_linear_discount"]
Compounding = Literal["continuous", "simple", "annual", "semiannual", "quarterly", "monthly"]
CurveInputType = Literal["zero_rates", "discount_factors"]

_COMPOUNDING_FREQUENCIES: dict[str, int] = {
    "annual": 1,
    "semiannual": 2,
    "quarterly": 4,
    "monthly": 12,
}


@dataclass(frozen=True)
class DiscountCurve:
    """Discount curve with configurable rate compounding and interpolation."""

    ref_date: date
    pillars: tuple[tuple[float, float], ...]
    interpolation: InterpolationMode
    compounding: Compounding
    input_type: CurveInputType
    _discount_pillars: tuple[tuple[float, float], ...]

    def __init__(
        self,
        ref_date: date,
        pillars: Iterable[tuple[float, float]],
        *,
        interpolation: InterpolationMode = "linear_zero",
        compounding: Compounding = "continuous",
        input_type: CurveInputType = "zero_rates",
    ):
        if interpolation not in {"linear_zero", "linear_discount", "log_linear_discount"}:
            raise ValueError("unsupported interpolation mode")
        if compounding not in {"continuous", "simple", "annual", "semiannual", "quarterly", "monthly"}:
            raise ValueError("unsupported compounding convention")
        if input_type not in {"zero_rates", "discount_factors"}:
            raise ValueError("input_type must be 'zero_rates' or 'discount_factors'")

        sorted_pillars = tuple(sorted(pillars, key=lambda x: x[0]))
        if not sorted_pillars:
            raise ValueError("at least one curve pillar is required")
        if sorted_pillars[0][0] < 0:
            raise ValueError("curve pillars must have non-negative maturities")
        for left, right in zip(sorted_pillars, sorted_pillars[1:]):
            if right[0] <= left[0]:
                raise ValueError("curve pillar maturities must be strictly increasing")
        if input_type == "discount_factors" and any(value <= 0 for _, value in sorted_pillars):
            raise ValueError("discount factors must be positive")

        discount_pillars = tuple(
            (maturity, self._pillar_discount_factor(maturity, value, compounding, input_type))
            for maturity, value in sorted_pillars
        )
        object.__setattr__(self, "ref_date", ref_date)
        object.__setattr__(self, "pillars", sorted_pillars)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "compounding", compounding)
        object.__setattr__(self, "input_type", input_type)
        object.__setattr__(self, "_discount_pillars", discount_pillars)

    @classmethod
    def from_discount_factors(
        cls,
        ref_date: date,
        discount_factors: Iterable[tuple[float, float]],
        *,
        interpolation: InterpolationMode = "log_linear_discount",
        compounding: Compounding = "continuous",
    ) -> "DiscountCurve":
        return cls(
            ref_date,
            discount_factors,
            interpolation=interpolation,
            compounding=compounding,
            input_type="discount_factors",
        )

    @classmethod
    def from_money_market_rates(
        cls,
        ref_date: date,
        rates: Iterable[tuple[float, float]],
        *,
        interpolation: InterpolationMode = "log_linear_discount",
    ) -> "DiscountCurve":
        discount_factors = ((maturity, 1.0 / (1.0 + rate * maturity)) for maturity, rate in rates)
        return cls.from_discount_factors(ref_date, discount_factors, interpolation=interpolation)

    @staticmethod
    def _pillar_discount_factor(
        maturity: float,
        value: float,
        compounding: Compounding,
        input_type: CurveInputType,
    ) -> float:
        if input_type == "discount_factors":
            return value
        return _discount_factor_from_rate(value, maturity, compounding)

    def time_from_reference(self, target: date, convention: str | DayCount = DayCount.ACT_365F) -> float:
        return year_fraction(self.ref_date, target, convention)

    def zero_rate(self, maturity: float) -> float:
        if maturity < 0:
            raise ValueError("maturity cannot be negative")
        if maturity == 0:
            return self._pillar_zero_rate(self._discount_pillars[0][0], self._discount_pillars[0][1])
        return _zero_rate_from_discount_factor(
            self.discount_factor(maturity),
            maturity,
            self.compounding,
        )

    def discount_factor(self, maturity: float) -> float:
        if maturity < 0:
            raise ValueError("maturity cannot be negative")
        if maturity == 0:
            return 1.0
        if maturity <= self._discount_pillars[0][0]:
            return _discount_factor_from_rate(self.zero_rate_at_pillar(0), maturity, self.compounding)

        for left, right in zip(self._discount_pillars, self._discount_pillars[1:]):
            t0, df0 = left
            t1, df1 = right
            if t0 <= maturity <= t1:
                weight = (maturity - t0) / (t1 - t0)
                if self.interpolation == "linear_discount":
                    return df0 * (1.0 - weight) + df1 * weight
                if self.interpolation == "log_linear_discount":
                    return math.exp(math.log(df0) * (1.0 - weight) + math.log(df1) * weight)
                r0 = self._pillar_zero_rate(t0, df0)
                r1 = self._pillar_zero_rate(t1, df1)
                rate = r0 * (1.0 - weight) + r1 * weight
                return _discount_factor_from_rate(rate, maturity, self.compounding)

        return _discount_factor_from_rate(self.zero_rate_at_pillar(-1), maturity, self.compounding)

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
        return DiscountCurve(
            self.ref_date,
            ((maturity, self.zero_rate(maturity) + bump) for maturity, _ in self._discount_pillars),
            interpolation=self.interpolation,
            compounding=self.compounding,
            input_type="zero_rates",
        )

    def zero_rate_at_pillar(self, index: int) -> float:
        maturity, discount_factor = self._discount_pillars[index]
        return self._pillar_zero_rate(maturity, discount_factor)

    def _pillar_zero_rate(self, maturity: float, discount_factor: float) -> float:
        if maturity == 0:
            if self.input_type == "zero_rates":
                return self.pillars[0][1]
            if len(self._discount_pillars) == 1:
                return 0.0
            next_maturity, next_discount = self._discount_pillars[1]
            return _zero_rate_from_discount_factor(next_discount, next_maturity, self.compounding)
        return _zero_rate_from_discount_factor(discount_factor, maturity, self.compounding)


class FlatCurve(DiscountCurve):
    """Flat curve with configurable compounding."""

    def __init__(self, ref_date: date, rate: float, *, compounding: Compounding = "continuous"):
        super().__init__(ref_date, ((0.0, rate), (100.0, rate)), compounding=compounding)


def _discount_factor_from_rate(rate: float, maturity: float, compounding: Compounding) -> float:
    if maturity == 0:
        return 1.0
    if compounding == "continuous":
        return math.exp(-rate * maturity)
    if compounding == "simple":
        denominator = 1.0 + rate * maturity
        if denominator <= 0:
            raise ValueError("simple-compounded rate produces a non-positive discount denominator")
        return 1.0 / denominator
    frequency = _COMPOUNDING_FREQUENCIES[compounding]
    denominator = 1.0 + rate / frequency
    if denominator <= 0:
        raise ValueError("periodic-compounded rate produces a non-positive discount denominator")
    return denominator ** (-frequency * maturity)


def _zero_rate_from_discount_factor(
    discount_factor: float,
    maturity: float,
    compounding: Compounding,
) -> float:
    if maturity <= 0:
        raise ValueError("maturity must be positive when converting discount factors to zero rates")
    if discount_factor <= 0:
        raise ValueError("discount factor must be positive")
    if compounding == "continuous":
        return -math.log(discount_factor) / maturity
    if compounding == "simple":
        return (1.0 / discount_factor - 1.0) / maturity
    frequency = _COMPOUNDING_FREQUENCIES[compounding]
    return frequency * (discount_factor ** (-1.0 / (frequency * maturity)) - 1.0)
