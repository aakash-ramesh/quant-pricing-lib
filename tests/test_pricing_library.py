from __future__ import annotations

import math
import unittest
from datetime import date

from pricing_library.core.curves import DiscountCurve
from pricing_library.core.schedule import adjust_business_day, generate_schedule, is_business_day
from pricing_library.equity import (
    ASRContract,
    ASRMonteCarloPricer,
    ArithmeticAsianOption,
    AsianMonteCarloPricer,
    BinomialTreePricer,
    BlackScholesPricer,
    EquityForward,
    EquityForwardPricer,
    EquityOption,
    HestonMonteCarloPricer,
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

    def test_named_calendar_and_end_of_month_schedule(self) -> None:
        self.assertFalse(is_business_day(date(2026, 12, 25), calendar="target2"))
        self.assertEqual(
            adjust_business_day(date(2026, 7, 4), "following", calendar="us_federal"),
            date(2026, 7, 6),
        )
        schedule = generate_schedule(
            date(2026, 1, 31),
            date(2026, 7, 31),
            1,
            adjustment="modified_following",
            calendar="us_federal",
            end_of_month_rule=True,
        )
        self.assertEqual(schedule[1], date(2026, 2, 27))
        self.assertEqual(schedule[-1], date(2026, 7, 31))

    def test_curve_supports_discount_factor_input_and_log_linear_interpolation(self) -> None:
        curve = DiscountCurve.from_discount_factors(
            date(2026, 5, 15),
            ((0.0, 1.0), (1.0, math.exp(-0.04)), (2.0, math.exp(-0.10))),
        )
        self.assertAlmostEqual(curve.discount_factor(1.5), math.exp(-0.07), places=12)
        self.assertAlmostEqual(curve.zero_rate(1.5), 0.07 / 1.5, places=12)

    def test_american_put_is_worth_at_least_european_put(self) -> None:
        european = EquityOption(45.0, 50.0, 1.0, 0.05, 0.25, "put", exercise_style="european")
        american = EquityOption(45.0, 50.0, 1.0, 0.05, 0.25, "put", exercise_style="american")
        tree = BinomialTreePricer(steps=300)
        self.assertGreaterEqual(tree.price(american).price, tree.price(european).price)

    def test_alternate_binomial_models_track_black_scholes(self) -> None:
        option = EquityOption(100.0, 100.0, 1.0, 0.04, 0.22, "call")
        reference = BlackScholesPricer().price(option).price
        for model in ("jarrow_rudd", "tian"):
            value = BinomialTreePricer(steps=400, model=model).price(option)
            self.assertAlmostEqual(value.price, reference, delta=0.08)
            self.assertEqual(value.model, model)

    def test_local_volatility_tree_prices(self) -> None:
        option = EquityOption(100.0, 100.0, 1.0, 0.03, 0.20, "put", exercise_style="american")
        value = BinomialTreePricer(
            steps=5,
            local_volatility=lambda time, spot: 0.18 + 0.0005 * abs(spot - 100.0),
        ).price(option)
        self.assertGreater(value.price, 0.0)

    def test_heston_monte_carlo_returns_finite_european_value(self) -> None:
        option = EquityOption(100.0, 100.0, 1.0, 0.04, 0.20, "call")
        value = HestonMonteCarloPricer(paths=1_000, steps=24, seed=19).price(
            option,
            kappa=1.5,
            theta=0.04,
            vol_of_vol=0.30,
            correlation=-0.5,
        )
        self.assertTrue(math.isfinite(value.price))
        self.assertGreater(value.price, 0.0)

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

    def test_swap_can_use_separate_forecast_curve(self) -> None:
        ref = date(2026, 5, 15)
        discount_curve = DiscountCurve(ref, ((0.0, 0.03), (5.0, 0.03)))
        forecast_curve = DiscountCurve(ref, ((0.0, 0.05), (5.0, 0.05)))
        swap = build_vanilla_interest_rate_swap(
            ref,
            ref,
            date(2031, 5, 15),
            notional=10_000_000,
            fixed_rate=0.04,
            direction="payer",
            calendar="us_federal",
        )
        single_curve = InterestRateSwapPricer().price(swap, discount_curve)
        multi_curve = InterestRateSwapPricer().price(swap, discount_curve, forecast_curve)
        self.assertGreater(multi_curve.par_rate, single_curve.par_rate)
        self.assertNotEqual(multi_curve.forecast_dv01, 0.0)

    def test_asian_control_variate_reduces_standard_error(self) -> None:
        option = ArithmeticAsianOption(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.04,
            volatility=0.25,
            option_type="call",
            observations=12,
        )
        plain = AsianMonteCarloPricer(paths=2_000, seed=11, antithetic=False).price(option)
        controlled = AsianMonteCarloPricer(
            paths=2_000,
            seed=11,
            antithetic=False,
            control_variate=True,
            moment_matching=True,
        ).price(option)
        self.assertLess(controlled.standard_error, plain.standard_error)
        self.assertIn("geometric_control_variate", controlled.variance_reduction)

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
