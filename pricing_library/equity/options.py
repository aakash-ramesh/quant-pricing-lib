from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Callable, Literal

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]
BinomialModel = Literal["crr", "jarrow_rudd", "tian"]
LocalVolatility = Callable[[float, float], float]


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
    model: BinomialModel = "crr"


@dataclass(frozen=True)
class HestonMonteCarloResult:
    price: float
    standard_error: float
    paths: int
    seed: int | None


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
    """Binomial tree pricer for European and American options."""

    def __init__(
        self,
        steps: int = 250,
        *,
        model: BinomialModel = "crr",
        local_volatility: LocalVolatility | None = None,
        max_local_vol_steps: int = 20,
    ):
        if steps <= 0:
            raise ValueError("steps must be positive")
        if model not in {"crr", "jarrow_rudd", "tian"}:
            raise ValueError("model must be 'crr', 'jarrow_rudd', or 'tian'")
        if max_local_vol_steps <= 0:
            raise ValueError("max_local_vol_steps must be positive")
        self.steps = steps
        self.model: BinomialModel = model
        self.local_volatility = local_volatility
        self.max_local_vol_steps = max_local_vol_steps

    def price(self, option: EquityOption) -> BinomialResult:
        if option.maturity == 0 or (option.volatility == 0 and self.local_volatility is None):
            intrinsic = _payoff(option.option_type, option.spot, option.strike) * option.quantity
            return BinomialResult(price=intrinsic, delta=0.0, steps=self.steps, model=self.model)

        if self.local_volatility is not None:
            return self._price_local_vol(option)
        return self._price_recombining(option)

    def _price_recombining(self, option: EquityOption) -> BinomialResult:
        dt = option.maturity / self.steps
        up, down, probability = _tree_parameters(
            self.model,
            option.volatility,
            dt,
            option.rate,
            option.dividend_yield,
        )
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

        return BinomialResult(
            price=values[0] * option.quantity,
            delta=delta * option.quantity,
            steps=self.steps,
            model=self.model,
        )

    def _price_local_vol(self, option: EquityOption) -> BinomialResult:
        if self.steps > self.max_local_vol_steps:
            raise ValueError(
                "local-volatility trees are non-recombining; reduce steps or increase max_local_vol_steps"
            )

        dt = option.maturity / self.steps
        discount = math.exp(-option.rate * dt)
        spot_levels: list[list[float]] = [[option.spot]]

        for step in range(self.steps):
            time = step * dt
            next_spots: list[float] = []
            for spot in spot_levels[-1]:
                volatility = self._local_volatility(time, spot)
                up, down, _ = _tree_parameters(
                    self.model,
                    volatility,
                    dt,
                    option.rate,
                    option.dividend_yield,
                )
                next_spots.extend((spot * down, spot * up))
            spot_levels.append(next_spots)

        values = [_payoff(option.option_type, spot, option.strike) for spot in spot_levels[-1]]
        first_step_values: tuple[float, float] | None = None

        for step in range(self.steps - 1, -1, -1):
            time = step * dt
            next_values: list[float] = []
            for node_index, spot in enumerate(spot_levels[step]):
                volatility = self._local_volatility(time, spot)
                _, _, probability = _tree_parameters(
                    self.model,
                    volatility,
                    dt,
                    option.rate,
                    option.dividend_yield,
                )
                down_value = values[2 * node_index]
                up_value = values[2 * node_index + 1]
                continuation = discount * (probability * up_value + (1.0 - probability) * down_value)
                if option.exercise_style == "american":
                    continuation = max(continuation, _payoff(option.option_type, spot, option.strike))
                next_values.append(continuation)
            if step == 1:
                first_step_values = (next_values[0], next_values[1])
            values = next_values

        if first_step_values is None:
            delta = 0.0
        else:
            down_spot, up_spot = spot_levels[1][0], spot_levels[1][1]
            delta = (
                0.0
                if up_spot == down_spot
                else (first_step_values[1] - first_step_values[0]) / (up_spot - down_spot)
            )

        return BinomialResult(
            price=values[0] * option.quantity,
            delta=delta * option.quantity,
            steps=self.steps,
            model=self.model,
        )

    def _local_volatility(self, time: float, spot: float) -> float:
        assert self.local_volatility is not None
        volatility = self.local_volatility(time, spot)
        if volatility < 0:
            raise ValueError("local volatility cannot be negative")
        return volatility


def _tree_parameters(
    model: BinomialModel,
    volatility: float,
    dt: float,
    rate: float,
    dividend_yield: float,
) -> tuple[float, float, float]:
    if volatility == 0:
        growth = math.exp((rate - dividend_yield) * dt)
        return growth, growth, 1.0

    growth = math.exp((rate - dividend_yield) * dt)
    if model == "crr":
        up = math.exp(volatility * math.sqrt(dt))
        down = 1.0 / up
        probability = (growth - down) / (up - down)
    elif model == "jarrow_rudd":
        drift = (rate - dividend_yield - 0.5 * volatility * volatility) * dt
        up = math.exp(drift + volatility * math.sqrt(dt))
        down = math.exp(drift - volatility * math.sqrt(dt))
        probability = 0.5
    else:
        variance_growth = math.exp(volatility * volatility * dt)
        root = math.sqrt(variance_growth * variance_growth + 2.0 * variance_growth - 3.0)
        up = 0.5 * growth * variance_growth * (variance_growth + 1.0 + root)
        down = 0.5 * growth * variance_growth * (variance_growth + 1.0 - root)
        probability = (growth - down) / (up - down)

    if up <= down:
        raise ValueError("invalid tree parameters; up factor must exceed down factor")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("invalid tree probability; increase steps or check inputs")
    return up, down, probability


class HestonMonteCarloPricer:
    """Monte Carlo pricer for European options under stochastic variance."""

    def __init__(
        self,
        paths: int = 20_000,
        steps: int = 252,
        seed: int | None = 7,
        antithetic: bool = True,
    ):
        if paths <= 0:
            raise ValueError("paths must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.paths = paths
        self.steps = steps
        self.seed = seed
        self.antithetic = antithetic

    def price(
        self,
        option: EquityOption,
        *,
        kappa: float,
        theta: float,
        vol_of_vol: float,
        correlation: float,
        initial_variance: float | None = None,
    ) -> HestonMonteCarloResult:
        if option.exercise_style != "european":
            raise ValueError("Heston Monte Carlo supports European exercise")
        if kappa < 0:
            raise ValueError("kappa cannot be negative")
        if theta < 0:
            raise ValueError("theta cannot be negative")
        if vol_of_vol < 0:
            raise ValueError("vol_of_vol cannot be negative")
        if not -1.0 <= correlation <= 1.0:
            raise ValueError("correlation must be between -1 and 1")

        variance = option.volatility * option.volatility if initial_variance is None else initial_variance
        if variance < 0:
            raise ValueError("initial_variance cannot be negative")
        if option.maturity == 0:
            intrinsic = _payoff(option.option_type, option.spot, option.strike) * option.quantity
            return HestonMonteCarloResult(intrinsic, 0.0, self.paths, self.seed)

        rng = random.Random(self.seed)
        payoffs: list[float] = []
        while len(payoffs) < self.paths:
            spot_shocks = [rng.gauss(0.0, 1.0) for _ in range(self.steps)]
            variance_shocks = [rng.gauss(0.0, 1.0) for _ in range(self.steps)]
            payoffs.append(
                self._simulate_payoff(
                    option,
                    variance,
                    kappa,
                    theta,
                    vol_of_vol,
                    correlation,
                    spot_shocks,
                    variance_shocks,
                )
            )
            if self.antithetic and len(payoffs) < self.paths:
                payoffs.append(
                    self._simulate_payoff(
                        option,
                        variance,
                        kappa,
                        theta,
                        vol_of_vol,
                        correlation,
                        [-shock for shock in spot_shocks],
                        [-shock for shock in variance_shocks],
                    )
                )

        discount = math.exp(-option.rate * option.maturity)
        values = [payoff * discount * option.quantity for payoff in payoffs]
        price = fmean(values)
        standard_error = pstdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return HestonMonteCarloResult(price, standard_error, len(values), self.seed)

    def _simulate_payoff(
        self,
        option: EquityOption,
        initial_variance: float,
        kappa: float,
        theta: float,
        vol_of_vol: float,
        correlation: float,
        spot_shocks: list[float],
        variance_shocks: list[float],
    ) -> float:
        dt = option.maturity / self.steps
        sqrt_dt = math.sqrt(dt)
        independent_weight = math.sqrt(max(1.0 - correlation * correlation, 0.0))
        spot = option.spot
        variance = initial_variance
        for spot_shock, independent_variance_shock in zip(spot_shocks, variance_shocks):
            variance_positive = max(variance, 0.0)
            variance_shock = correlation * spot_shock + independent_weight * independent_variance_shock
            spot *= math.exp(
                (option.rate - option.dividend_yield - 0.5 * variance_positive) * dt
                + math.sqrt(variance_positive) * sqrt_dt * spot_shock
            )
            variance += (
                kappa * (theta - variance_positive) * dt
                + vol_of_vol * math.sqrt(variance_positive) * sqrt_dt * variance_shock
            )
            variance = max(variance, 0.0)
        return _payoff(option.option_type, spot, option.strike)
