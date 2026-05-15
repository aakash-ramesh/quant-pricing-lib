from __future__ import annotations

import math
import unittest
from datetime import date

from pricing_library.core.curves import DiscountCurve
from pricing_library.equity import (
    ASRContract,
    ASRMonteCarloPricer,
    BinomialTreePricer,
    BlackScholesPricer,
    EquityForward,
    EquityForwardPricer,
    EquityOption,
)
from pricing_library.rates import InterestRateSwapPricer, build_vanilla_interest_rate_swap


class PricingLibraryTests(unittest.TestCase):
    def test_black_scholes_call_matches_reference_value(self) -> None:
        option = EquityOption(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            volatility=0.20,
            option_type="call",
        )
        result = BlackScholesPricer().price(option)
        self.assertAlmostEqual(result.price, 10.4506, places=4)

    def test_put_call_parity(self) -> None:
        call = EquityOption(100.0, 100.0, 1.0, 0.05, 0.20, "call")
        put = EquityOption(100.0, 100.0, 1.0, 0.05, 0.20, "put")
        bs = BlackScholesPricer()
        lhs = bs.price(call).price - bs.price(put).price
        rhs = 100.0 - 100.0 * math.exp(-0.05)
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_american_put_is_worth_at_least_european_put(self) -> None:
        european = EquityOption(45.0, 50.0, 1.0, 0.05, 0.25, "put", exercise_style="european")
        american = EquityOption(45.0, 50.0, 1.0, 0.05, 0.25, "put", exercise_style="american")
        tree = BinomialTreePricer(steps=300)
        self.assertGreaterEqual(tree.price(american).price, tree.price(european).price)

    def test_equity_forward_no_arbitrage_price(self) -> None:
        forward = EquityForward(spot=100.0, strike=101.0, maturity=1.0, rate=0.04, dividend_yield=0.01)
        value = EquityForwardPricer().price(forward)
        self.assertAlmostEqual(value.forward_price, 100.0 * math.exp(0.03), places=12)

    def test_par_swap_has_near_zero_pv(self) -> None:
        ref = date(2026, 5, 15)
        curve = DiscountCurve(ref, ((0.0, 0.045), (1.0, 0.043), (3.0, 0.041), (5.0, 0.040)))
        pricer = InterestRateSwapPricer()
        seed_swap = build_vanilla_interest_rate_swap(
            ref,
            ref,
            date(2031, 5, 15),
            notional=10_000_000,
            fixed_rate=0.04,
            direction="payer",
        )
        par_rate = pricer.price(seed_swap, curve).par_rate
        par_swap = build_vanilla_interest_rate_swap(
            ref,
            ref,
            date(2031, 5, 15),
            notional=10_000_000,
            fixed_rate=par_rate,
            direction="payer",
        )
        self.assertAlmostEqual(pricer.price(par_swap, curve).pv, 0.0, delta=1_000.0)

    def test_asr_zero_volatility_returns_finite_settlement(self) -> None:
        trade_date = date(2026, 5, 15)
        maturity_date = date(2026, 6, 15)
        contract = ASRContract.with_business_day_averaging(
            cash_notional=10_000_000,
            initial_spot=100.0,
            discount=1.0,
            trade_date=trade_date,
            maturity_date=maturity_date,
            upfront_fraction=0.8,
        )
        value = ASRMonteCarloPricer(
            spot=100.0,
            rate=0.03,
            volatility=0.0,
            paths=100,
            seed=1,
        ).price(contract)
        self.assertTrue(math.isfinite(value.pv))
        self.assertGreater(value.expected_total_shares, value.upfront_shares)


if __name__ == "__main__":
    unittest.main()

