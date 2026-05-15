from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pricing_library.core.curves import DiscountCurve
from pricing_library.core.daycount import DayCount, year_fraction
from pricing_library.core.schedule import BusinessDayAdjustment, generate_schedule

SwapDirection = Literal["payer", "receiver"]


@dataclass(frozen=True)
class FixedLeg:
    notional: float
    fixed_rate: float
    day_count: str | DayCount
    payment_dates: tuple[date, ...]


@dataclass(frozen=True)
class FloatingLeg:
    notional: float
    spread: float
    day_count: str | DayCount
    reset_dates: tuple[date, ...]


@dataclass(frozen=True)
class InterestRateSwap:
    """Plain-vanilla fixed-vs-floating interest rate swap."""

    trade_date: date
    start_date: date
    maturity_date: date
    fixed_leg: FixedLeg
    floating_leg: FloatingLeg
    direction: SwapDirection = "payer"

    def __post_init__(self) -> None:
        if self.direction not in {"payer", "receiver"}:
            raise ValueError("direction must be 'payer' or 'receiver'")
        if self.fixed_leg.notional <= 0 or self.floating_leg.notional <= 0:
            raise ValueError("swap notionals must be positive")


@dataclass(frozen=True)
class SwapValuation:
    fixed_leg_pv: float
    floating_leg_pv: float
    pv: float
    par_rate: float
    annuity: float
    dv01: float


class InterestRateSwapPricer:
    """Discount-curve pricer for vanilla swaps."""

    def price(self, swap: InterestRateSwap, discount_curve: DiscountCurve) -> SwapValuation:
        fixed_leg_pv, annuity = self._fixed_leg_pv_and_annuity(swap, discount_curve)
        floating_leg_pv = self._floating_leg_pv(swap, discount_curve)
        pv = self._net_pv(swap.direction, fixed_leg_pv, floating_leg_pv)
        par_rate = floating_leg_pv / annuity if annuity else 0.0

        bumped_curve = discount_curve.bump_parallel(1.0)
        bumped_fixed_pv, _ = self._fixed_leg_pv_and_annuity(swap, bumped_curve)
        bumped_float_pv = self._floating_leg_pv(swap, bumped_curve)
        dv01 = self._net_pv(swap.direction, bumped_fixed_pv, bumped_float_pv) - pv

        return SwapValuation(
            fixed_leg_pv=fixed_leg_pv,
            floating_leg_pv=floating_leg_pv,
            pv=pv,
            par_rate=par_rate,
            annuity=annuity,
            dv01=dv01,
        )

    @staticmethod
    def _net_pv(direction: SwapDirection, fixed_leg_pv: float, floating_leg_pv: float) -> float:
        if direction == "payer":
            return floating_leg_pv - fixed_leg_pv
        return fixed_leg_pv - floating_leg_pv

    @staticmethod
    def _fixed_leg_pv_and_annuity(
        swap: InterestRateSwap,
        discount_curve: DiscountCurve,
    ) -> tuple[float, float]:
        dates = swap.fixed_leg.payment_dates
        annuity = 0.0
        for start, end in zip(dates, dates[1:]):
            accrual = year_fraction(start, end, swap.fixed_leg.day_count)
            maturity = discount_curve.time_from_reference(end)
            annuity += swap.fixed_leg.notional * accrual * discount_curve.discount_factor(maturity)
        return swap.fixed_leg.fixed_rate * annuity, annuity

    @staticmethod
    def _floating_leg_pv(swap: InterestRateSwap, discount_curve: DiscountCurve) -> float:
        dates = swap.floating_leg.reset_dates
        pv = 0.0
        for start, end in zip(dates, dates[1:]):
            accrual = year_fraction(start, end, swap.floating_leg.day_count)
            t_start = discount_curve.time_from_reference(start)
            t_end = discount_curve.time_from_reference(end)
            forward = discount_curve.forward_rate(max(t_start, 0.0), max(t_end, 0.0), accrual)
            discount = discount_curve.discount_factor(max(t_end, 0.0))
            pv += swap.floating_leg.notional * (forward + swap.floating_leg.spread) * accrual * discount
        return pv


def build_vanilla_interest_rate_swap(
    trade_date: date,
    start_date: date,
    maturity_date: date,
    *,
    notional: float,
    fixed_rate: float,
    direction: SwapDirection = "payer",
    fixed_frequency_months: int = 6,
    floating_frequency_months: int = 3,
    fixed_day_count: str | DayCount = DayCount.THIRTY_360,
    floating_day_count: str | DayCount = DayCount.ACT_360,
    floating_spread: float = 0.0,
    business_day_adjustment: BusinessDayAdjustment = "modified_following",
) -> InterestRateSwap:
    fixed_dates = tuple(
        generate_schedule(
            start_date,
            maturity_date,
            fixed_frequency_months,
            adjustment=business_day_adjustment,
        )
    )
    floating_dates = tuple(
        generate_schedule(
            start_date,
            maturity_date,
            floating_frequency_months,
            adjustment=business_day_adjustment,
        )
    )
    return InterestRateSwap(
        trade_date=trade_date,
        start_date=start_date,
        maturity_date=maturity_date,
        direction=direction,
        fixed_leg=FixedLeg(notional, fixed_rate, fixed_day_count, fixed_dates),
        floating_leg=FloatingLeg(notional, floating_spread, floating_day_count, floating_dates),
    )

