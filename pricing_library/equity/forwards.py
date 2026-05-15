from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EquityForward:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float = 0.0
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity < 0:
            raise ValueError("maturity cannot be negative")


@dataclass(frozen=True)
class EquityForwardValuation:
    forward_price: float
    pv: float


class EquityForwardPricer:
    """No-arbitrage equity forward pricer with continuous dividend yield."""

    def price(self, contract: EquityForward) -> EquityForwardValuation:
        forward_price = contract.spot * math.exp((contract.rate - contract.dividend_yield) * contract.maturity)
        pv = contract.quantity * (
            contract.spot * math.exp(-contract.dividend_yield * contract.maturity)
            - contract.strike * math.exp(-contract.rate * contract.maturity)
        )
        return EquityForwardValuation(forward_price=forward_price, pv=pv)

