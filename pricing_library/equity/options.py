from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True)
class EquityOption:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: OptionType
    dividend_yield: float = 0.0
    exercise_style: ExerciseStyle = "european"
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity < 0:
            raise ValueError("maturity cannot be negative")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.exercise_style not in {"european", "american"}:
            raise ValueError("exercise_style must be 'european' or 'american'")


@dataclass(frozen=True)
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class BinomialResult:
    price: float
    delta: float
    steps: int


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _payoff(option_type: OptionType, spot: float, strike: float) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


class BlackScholesPricer:
    """Closed-form Black-Scholes-Merton pricer for European equity options."""

    def price(self, option: EquityOption) -> BlackScholesResult:
        if option.exercise_style != "european":
            raise ValueError("Black-Scholes closed form only supports European exercise")

        s = option.spot
        k = option.strike
        t = option.maturity
        r = option.rate
        q = option.dividend_yield
        sigma = option.volatility
        qty = option.quantity

        if t == 0 or sigma == 0:
            intrinsic = _payoff(option.option_type, s, k) * qty
            if option.option_type == "call":
                delta = qty if s > k else 0.0
            else:
                delta = -qty if s < k else 0.0
            return BlackScholesResult(intrinsic, delta, 0.0, 0.0, 0.0, 0.0)

        sqrt_t = math.sqrt(t)
        d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        discount_rate = math.exp(-r * t)
        discount_dividend = math.exp(-q * t)

        if option.option_type == "call":
            price = s * discount_dividend * _normal_cdf(d1) - k * discount_rate * _normal_cdf(d2)
            delta = discount_dividend * _normal_cdf(d1)
            theta = (
                -s * discount_dividend * _normal_pdf(d1) * sigma / (2.0 * sqrt_t)
                - r * k * discount_rate * _normal_cdf(d2)
                + q * s * discount_dividend * _normal_cdf(d1)
            )
            rho = k * t * discount_rate * _normal_cdf(d2)
        else:
            price = k * discount_rate * _normal_cdf(-d2) - s * discount_dividend * _normal_cdf(-d1)
            delta = discount_dividend * (_normal_cdf(d1) - 1.0)
            theta = (
                -s * discount_dividend * _normal_pdf(d1) * sigma / (2.0 * sqrt_t)
                + r * k * discount_rate * _normal_cdf(-d2)
                - q * s * discount_dividend * _normal_cdf(-d1)
            )
            rho = -k * t * discount_rate * _normal_cdf(-d2)

        gamma = discount_dividend * _normal_pdf(d1) / (s * sigma * sqrt_t)
        vega = s * discount_dividend * _normal_pdf(d1) * sqrt_t
        return BlackScholesResult(
            price=price * qty,
            delta=delta * qty,
            gamma=gamma * qty,
            vega=vega * qty,
            theta=theta * qty,
            rho=rho * qty,
        )


class BinomialTreePricer:
    """Cox-Ross-Rubinstein tree pricer for European and American options."""

    def __init__(self, steps: int = 250):
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.steps = steps

    def price(self, option: EquityOption) -> BinomialResult:
        if option.maturity == 0 or option.volatility == 0:
            intrinsic = _payoff(option.option_type, option.spot, option.strike) * option.quantity
            return BinomialResult(price=intrinsic, delta=0.0, steps=self.steps)

        dt = option.maturity / self.steps
        up = math.exp(option.volatility * math.sqrt(dt))
        down = 1.0 / up
        growth = math.exp((option.rate - option.dividend_yield) * dt)
        probability = (growth - down) / (up - down)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("invalid CRR probability; increase steps or check inputs")

        discount = math.exp(-option.rate * dt)
        values = [
            _payoff(option.option_type, option.spot * (up**j) * (down ** (self.steps - j)), option.strike)
            for j in range(self.steps + 1)
        ]

        first_step_values: tuple[float, float] | None = None
        for step in range(self.steps - 1, -1, -1):
            next_values: list[float] = []
            for j in range(step + 1):
                continuation = discount * (probability * values[j + 1] + (1.0 - probability) * values[j])
                if option.exercise_style == "american":
                    node_spot = option.spot * (up**j) * (down ** (step - j))
                    continuation = max(continuation, _payoff(option.option_type, node_spot, option.strike))
                next_values.append(continuation)

            if step == 1:
                first_step_values = (next_values[0], next_values[1])
            values = next_values

        if first_step_values is None:
            delta = 0.0
        else:
            down_spot = option.spot * down
            up_spot = option.spot * up
            delta = (first_step_values[1] - first_step_values[0]) / (up_spot - down_spot)

        return BinomialResult(price=values[0] * option.quantity, delta=delta * option.quantity, steps=self.steps)

