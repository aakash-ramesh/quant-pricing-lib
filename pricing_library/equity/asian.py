from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Literal

from pricing_library.equity.options import OptionType, _normal_cdf

AverageType = Literal["arithmetic"]


@dataclass(frozen=True)
class ArithmeticAsianOption:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: OptionType
    dividend_yield: float = 0.0
    observations: int = 12
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.observations <= 0:
            raise ValueError("observations must be positive")


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    paths: int
    seed: int | None
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)
    variance_reduction: tuple[str, ...] = ()


class AsianMonteCarloPricer:
    """Risk-neutral Monte Carlo pricer for fixed-strike arithmetic Asian options."""

    def __init__(
        self,
        paths: int = 20_000,
        seed: int | None = 7,
        antithetic: bool = True,
        *,
        moment_matching: bool = False,
        control_variate: bool = False,
    ):
        if paths <= 0:
            raise ValueError("paths must be positive")
        self.paths = paths
        self.seed = seed
        self.antithetic = antithetic
        self.moment_matching = moment_matching
        self.control_variate = control_variate

    def price(self, option: ArithmeticAsianOption) -> MonteCarloResult:
        shock_paths = self._shock_paths(option)
        if self.moment_matching:
            shock_paths = self._moment_match(shock_paths)

        discount = math.exp(-option.rate * option.maturity)
        arithmetic_values: list[float] = []
        geometric_values: list[float] = []
        for shocks in shock_paths:
            arithmetic_payoff, geometric_payoff = self._path_payoffs(option, shocks)
            arithmetic_values.append(arithmetic_payoff * discount)
            geometric_values.append(geometric_payoff * discount)

        variance_reduction: list[str] = []
        if self.antithetic:
            variance_reduction.append("antithetic")
        if self.moment_matching:
            variance_reduction.append("moment_matching")

        values = arithmetic_values
        if self.control_variate:
            variance_reduction.append("geometric_control_variate")
            values = self._control_variate_values(option, arithmetic_values, geometric_values)

        price = fmean(values) * option.quantity
        standard_error = pstdev(values) * option.quantity / math.sqrt(len(values)) if len(values) > 1 else 0.0
        ci = (price - 1.96 * standard_error, price + 1.96 * standard_error)
        return MonteCarloResult(
            price=price,
            standard_error=standard_error,
            paths=len(values),
            seed=self.seed,
            confidence_interval_95=ci,
            variance_reduction=tuple(variance_reduction),
        )

    def _shock_paths(self, option: ArithmeticAsianOption) -> list[list[float]]:
        rng = random.Random(self.seed)
        shock_paths: list[list[float]] = []
        while len(shock_paths) < self.paths:
            shocks = [rng.gauss(0.0, 1.0) for _ in range(option.observations)]
            shock_paths.append(shocks)
            if self.antithetic and len(shock_paths) < self.paths:
                shock_paths.append([-z for z in shocks])
        return shock_paths

    @staticmethod
    def _moment_match(shock_paths: list[list[float]]) -> list[list[float]]:
        flattened = [shock for path in shock_paths for shock in path]
        mean = fmean(flattened)
        stddev = pstdev(flattened)
        if stddev == 0:
            return shock_paths
        return [[(shock - mean) / stddev for shock in path] for path in shock_paths]

    @staticmethod
    def _path_payoffs(option: ArithmeticAsianOption, shocks: list[float]) -> tuple[float, float]:
        dt = option.maturity / option.observations
        drift = (option.rate - option.dividend_yield - 0.5 * option.volatility * option.volatility) * dt
        diffusion = option.volatility * math.sqrt(dt)
        spot = option.spot
        observations: list[float] = []
        for shock in shocks:
            spot *= math.exp(drift + diffusion * shock)
            observations.append(spot)
        average = fmean(observations)
        geometric_average = math.exp(fmean(math.log(observation) for observation in observations))
        if option.option_type == "call":
            return max(average - option.strike, 0.0), max(geometric_average - option.strike, 0.0)
        return max(option.strike - average, 0.0), max(option.strike - geometric_average, 0.0)

    @staticmethod
    def _control_variate_values(
        option: ArithmeticAsianOption,
        arithmetic_values: list[float],
        geometric_values: list[float],
    ) -> list[float]:
        geometric_price = _geometric_asian_price(option)
        geometric_mean = fmean(geometric_values)
        arithmetic_mean = fmean(arithmetic_values)
        covariance = fmean(
            (arithmetic - arithmetic_mean) * (geometric - geometric_mean)
            for arithmetic, geometric in zip(arithmetic_values, geometric_values)
        )
        geometric_variance = fmean((geometric - geometric_mean) ** 2 for geometric in geometric_values)
        if geometric_variance == 0:
            return arithmetic_values
        beta = covariance / geometric_variance
        return [
            arithmetic - beta * (geometric - geometric_price)
            for arithmetic, geometric in zip(arithmetic_values, geometric_values)
        ]


def _geometric_asian_price(option: ArithmeticAsianOption) -> float:
    observations = option.observations
    average_time = option.maturity * (observations + 1) / (2.0 * observations)
    variance = option.volatility * option.volatility * option.maturity
    variance *= (observations + 1) * (2 * observations + 1) / (6.0 * observations * observations)
    mean = math.log(option.spot) + (
        option.rate - option.dividend_yield - 0.5 * option.volatility * option.volatility
    ) * average_time
    discount = math.exp(-option.rate * option.maturity)

    if variance == 0:
        geometric_forward = math.exp(mean)
        if option.option_type == "call":
            return discount * max(geometric_forward - option.strike, 0.0)
        return discount * max(option.strike - geometric_forward, 0.0)

    stddev = math.sqrt(variance)
    d1 = (mean - math.log(option.strike) + variance) / stddev
    d2 = d1 - stddev
    expected_geometric = math.exp(mean + 0.5 * variance)
    if option.option_type == "call":
        return discount * (expected_geometric * _normal_cdf(d1) - option.strike * _normal_cdf(d2))
    return discount * (option.strike * _normal_cdf(-d2) - expected_geometric * _normal_cdf(-d1))
