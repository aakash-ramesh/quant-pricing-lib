from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date
from statistics import fmean, pstdev

from pricing_library.core.daycount import DayCount, year_fraction
from pricing_library.core.schedule import business_days


@dataclass(frozen=True)
class Dividend:
    ex_date: date
    amount: float


@dataclass(frozen=True)
class ASRContract:
    """Accelerated share repurchase contract from the corporate client's perspective."""

    cash_notional: float
    initial_spot: float
    discount: float
    trade_date: date
    maturity_date: date
    averaging_dates: tuple[date, ...]
    upfront_fraction: float = 0.8
    realized_average: float | None = None
    realized_observations: int = 0
    average_cap: float | None = None
    average_floor: float | None = None
    low_level: float | None = None
    high_level: float | None = None
    low_notional_multiplier: float = 1.0
    high_notional_multiplier: float = 1.0
    dividends: tuple[Dividend, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.cash_notional <= 0:
            raise ValueError("cash_notional must be positive")
        if self.initial_spot <= 0:
            raise ValueError("initial_spot must be positive")
        if not 0.0 <= self.upfront_fraction <= 1.0:
            raise ValueError("upfront_fraction must be between 0 and 1")
        if self.maturity_date < self.trade_date:
            raise ValueError("maturity_date cannot be before trade_date")
        if any(d <= self.trade_date for d in self.averaging_dates):
            raise ValueError("averaging_dates must be after trade_date")

    @classmethod
    def with_business_day_averaging(
        cls,
        *,
        cash_notional: float,
        initial_spot: float,
        discount: float,
        trade_date: date,
        maturity_date: date,
        upfront_fraction: float = 0.8,
        **kwargs: object,
    ) -> "ASRContract":
        return cls(
            cash_notional=cash_notional,
            initial_spot=initial_spot,
            discount=discount,
            trade_date=trade_date,
            maturity_date=maturity_date,
            averaging_dates=tuple(business_days(trade_date, maturity_date)),
            upfront_fraction=upfront_fraction,
            **kwargs,
        )


@dataclass(frozen=True)
class ASRValuation:
    pv: float
    standard_error: float
    expected_final_average: float
    expected_total_shares: float
    upfront_shares: float
    expected_settlement_shares: float
    paths: int
    seed: int | None


class ASRMonteCarloPricer:
    """Risk-neutral ASR pricer for averaging, caps/floors, discounts, and variable notional."""

    def __init__(
        self,
        *,
        spot: float,
        rate: float,
        volatility: float,
        borrow_or_dividend_yield: float = 0.0,
        paths: int = 10_000,
        seed: int | None = 11,
        antithetic: bool = True,
    ):
        if spot <= 0:
            raise ValueError("spot must be positive")
        if volatility < 0:
            raise ValueError("volatility cannot be negative")
        if paths <= 0:
            raise ValueError("paths must be positive")
        self.spot = spot
        self.rate = rate
        self.volatility = volatility
        self.borrow_or_dividend_yield = borrow_or_dividend_yield
        self.paths = paths
        self.seed = seed
        self.antithetic = antithetic

    def price(self, contract: ASRContract) -> ASRValuation:
        rng = random.Random(self.seed)
        discounted_values: list[float] = []
        final_averages: list[float] = []
        total_shares: list[float] = []
        settlement_shares: list[float] = []
        upfront_shares = contract.cash_notional * contract.upfront_fraction / contract.initial_spot

        time_points = self._event_dates(contract)
        while len(discounted_values) < self.paths:
            shocks = [rng.gauss(0.0, 1.0) for _ in time_points]
            path = self._simulate_path(contract, time_points, shocks)
            self._append_path_value(contract, path, upfront_shares, discounted_values, final_averages, total_shares, settlement_shares)
            if self.antithetic and len(discounted_values) < self.paths:
                anti_path = self._simulate_path(contract, time_points, [-z for z in shocks])
                self._append_path_value(contract, anti_path, upfront_shares, discounted_values, final_averages, total_shares, settlement_shares)

        return ASRValuation(
            pv=fmean(discounted_values),
            standard_error=pstdev(discounted_values) / math.sqrt(len(discounted_values))
            if len(discounted_values) > 1
            else 0.0,
            expected_final_average=fmean(final_averages),
            expected_total_shares=fmean(total_shares),
            upfront_shares=upfront_shares,
            expected_settlement_shares=fmean(settlement_shares),
            paths=len(discounted_values),
            seed=self.seed,
        )

    def _append_path_value(
        self,
        contract: ASRContract,
        path: dict[date, float],
        upfront_shares: float,
        discounted_values: list[float],
        final_averages: list[float],
        total_shares: list[float],
        settlement_shares: list[float],
    ) -> None:
        average = self._final_average(contract, path)
        net_average = self._bounded_average(contract, average)
        multiplier = self._notional_multiplier(contract, average)
        shares = contract.cash_notional * multiplier / net_average
        remaining = shares - upfront_shares
        terminal_spot = path[contract.maturity_date]
        maturity = year_fraction(contract.trade_date, contract.maturity_date, DayCount.ACT_365F)

        final_averages.append(average)
        total_shares.append(shares)
        settlement_shares.append(remaining)
        discounted_values.append(remaining * terminal_spot * math.exp(-self.rate * maturity))

    def _simulate_path(self, contract: ASRContract, event_dates: list[date], shocks: list[float]) -> dict[date, float]:
        current_date = contract.trade_date
        spot = self.spot
        path: dict[date, float] = {}
        dividends_by_date = {div.ex_date: div.amount for div in contract.dividends}

        for event_date, shock in zip(event_dates, shocks):
            dt = year_fraction(current_date, event_date, DayCount.ACT_365F)
            if dt > 0:
                drift = (
                    self.rate
                    - self.borrow_or_dividend_yield
                    - 0.5 * self.volatility * self.volatility
                ) * dt
                diffusion = self.volatility * math.sqrt(dt) * shock
                spot *= math.exp(drift + diffusion)
            if event_date in dividends_by_date:
                spot = max(spot - dividends_by_date[event_date], 0.01)
            path[event_date] = spot
            current_date = event_date

        return path

    @staticmethod
    def _event_dates(contract: ASRContract) -> list[date]:
        events = set(contract.averaging_dates)
        events.add(contract.maturity_date)
        events.update(div.ex_date for div in contract.dividends if contract.trade_date < div.ex_date <= contract.maturity_date)
        return sorted(events)

    @staticmethod
    def _final_average(contract: ASRContract, path: dict[date, float]) -> float:
        future_observations = [path[d] for d in contract.averaging_dates]
        if contract.realized_average is None or contract.realized_observations == 0:
            return fmean(future_observations)
        total = contract.realized_average * contract.realized_observations + sum(future_observations)
        return total / (contract.realized_observations + len(future_observations))

    @staticmethod
    def _bounded_average(contract: ASRContract, average: float) -> float:
        net_average = average - contract.discount
        if contract.average_floor is not None:
            net_average = max(net_average, contract.average_floor)
        if contract.average_cap is not None:
            net_average = min(net_average, contract.average_cap)
        return max(net_average, 0.01)

    @staticmethod
    def _notional_multiplier(contract: ASRContract, average: float) -> float:
        if contract.low_level is None or contract.high_level is None:
            return contract.low_notional_multiplier
        if abs(contract.high_level - contract.low_level) < 1e-12:
            return contract.low_notional_multiplier
        raw = contract.low_notional_multiplier + (
            (average - contract.low_level)
            / (contract.high_level - contract.low_level)
            * (contract.high_notional_multiplier - contract.low_notional_multiplier)
        )
        lower = min(contract.low_notional_multiplier, contract.high_notional_multiplier)
        upper = max(contract.low_notional_multiplier, contract.high_notional_multiplier)
        return min(max(raw, lower), upper)

