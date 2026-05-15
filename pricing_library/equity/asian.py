from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Literal

from pricing_library.equity.options import OptionType

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


class AsianMonteCarloPricer:
    """Risk-neutral Monte Carlo pricer for fixed-strike arithmetic Asian options."""

    def __init__(self, paths: int = 20_000, seed: int | None = 7, antithetic: bool = True):
        if paths <= 0:
            raise ValueError("paths must be positive")
        self.paths = paths
        self.seed = seed
        self.antithetic = antithetic

    def price(self, option: ArithmeticAsianOption) -> MonteCarloResult:
        rng = random.Random(self.seed)
        payoffs: list[float] = []
        target_paths = self.paths
        while len(payoffs) < target_paths:
            shocks = [rng.gauss(0.0, 1.0) for _ in range(option.observations)]
            payoffs.append(self._path_payoff(option, shocks))
            if self.antithetic and len(payoffs) < target_paths:
                payoffs.append(self._path_payoff(option, [-z for z in shocks]))

        discounted = [p * math.exp(-option.rate * option.maturity) * option.quantity for p in payoffs]
        price = fmean(discounted)
        standard_error = pstdev(discounted) / math.sqrt(len(discounted)) if len(discounted) > 1 else 0.0
        return MonteCarloResult(price=price, standard_error=standard_error, paths=len(discounted), seed=self.seed)

    @staticmethod
    def _path_payoff(option: ArithmeticAsianOption, shocks: list[float]) -> float:
        dt = option.maturity / option.observations
        drift = (option.rate - option.dividend_yield - 0.5 * option.volatility * option.volatility) * dt
        diffusion = option.volatility * math.sqrt(dt)
        spot = option.spot
        observations: list[float] = []
        for shock in shocks:
            spot *= math.exp(drift + diffusion * shock)
            observations.append(spot)
        average = fmean(observations)
        if option.option_type == "call":
            return max(average - option.strike, 0.0)
        return max(option.strike - average, 0.0)

