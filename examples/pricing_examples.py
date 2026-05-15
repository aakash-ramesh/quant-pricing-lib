from __future__ import annotations

from datetime import date

from pricing_library.core.curves import DiscountCurve
from pricing_library.equity import (
    ASRContract,
    ASRMonteCarloPricer,
    ArithmeticAsianOption,
    AsianMonteCarloPricer,
    BinomialTreePricer,
    BlackScholesPricer,
    Dividend,
    EquityForward,
    EquityForwardPricer,
    EquityOption,
)
from pricing_library.rates import InterestRateSwapPricer, build_vanilla_interest_rate_swap


def synthetic_usd_curve(ref_date: date) -> DiscountCurve:
    return DiscountCurve(
        ref_date,
        (
            (0.0, 0.0490),
            (0.25, 0.0485),
            (0.5, 0.0475),
            (1.0, 0.0460),
            (2.0, 0.0435),
            (3.0, 0.0420),
            (5.0, 0.0405),
            (7.0, 0.0410),
            (10.0, 0.0425),
        ),
    )


def price_interest_rate_swap() -> None:
    valuation_date = date(2026, 5, 15)
    curve = synthetic_usd_curve(valuation_date)
    swap = build_vanilla_interest_rate_swap(
        valuation_date,
        valuation_date,
        date(2031, 5, 15),
        notional=50_000_000,
        fixed_rate=0.0415,
        direction="payer",
        floating_spread=0.0008,
    )
    value = InterestRateSwapPricer().price(swap, curve)
    print("Interest rate swap")
    print(f"  PV: ${value.pv:,.2f}")
    print(f"  Par fixed rate: {value.par_rate:.4%}")
    print(f"  DV01 for +1bp parallel bump: ${value.dv01:,.2f}")


def price_european_option() -> None:
    call = EquityOption(
        spot=102.50,
        strike=100.00,
        maturity=0.75,
        rate=0.045,
        dividend_yield=0.012,
        volatility=0.24,
        option_type="call",
        exercise_style="european",
        quantity=10_000,
    )
    analytic = BlackScholesPricer().price(call)
    tree = BinomialTreePricer(steps=500).price(call)
    print("\nEuropean equity call")
    print(f"  Black-Scholes PV: ${analytic.price:,.2f}")
    print(f"  Binomial PV:      ${tree.price:,.2f}")
    print(f"  Delta:            {analytic.delta:,.2f} shares")


def price_american_option() -> None:
    put = EquityOption(
        spot=48.00,
        strike=52.00,
        maturity=1.25,
        rate=0.043,
        dividend_yield=0.005,
        volatility=0.31,
        option_type="put",
        exercise_style="american",
        quantity=25_000,
    )
    value = BinomialTreePricer(steps=750).price(put)
    print("\nAmerican equity put")
    print(f"  Binomial PV: ${value.price:,.2f}")
    print(f"  Tree delta:  {value.delta:,.2f} shares")


def price_equity_forward() -> None:
    forward = EquityForward(
        spot=188.75,
        strike=190.00,
        maturity=0.5,
        rate=0.044,
        dividend_yield=0.006,
        quantity=100_000,
    )
    value = EquityForwardPricer().price(forward)
    print("\nEquity forward")
    print(f"  No-arbitrage forward price: ${value.forward_price:,.4f}")
    print(f"  Contract PV: ${value.pv:,.2f}")


def price_asian_option() -> None:
    asian = ArithmeticAsianOption(
        spot=76.20,
        strike=75.00,
        maturity=1.0,
        rate=0.042,
        dividend_yield=0.008,
        volatility=0.28,
        option_type="call",
        observations=12,
        quantity=50_000,
    )
    value = AsianMonteCarloPricer(paths=12_000, seed=23).price(asian)
    print("\nArithmetic Asian call")
    print(f"  Monte Carlo PV: ${value.price:,.2f}")
    print(f"  Standard error: ${value.standard_error:,.2f}")


def price_asr() -> None:
    trade_date = date(2026, 5, 15)
    maturity_date = date(2026, 11, 16)
    contract = ASRContract.with_business_day_averaging(
        cash_notional=250_000_000,
        initial_spot=92.30,
        discount=1.15,
        trade_date=trade_date,
        maturity_date=maturity_date,
        upfront_fraction=0.82,
        average_floor=75.00,
        average_cap=115.00,
        low_level=80.00,
        high_level=110.00,
        low_notional_multiplier=1.08,
        high_notional_multiplier=0.94,
        dividends=(Dividend(date(2026, 8, 14), 0.42),),
    )
    value = ASRMonteCarloPricer(
        spot=93.10,
        rate=0.044,
        volatility=0.27,
        borrow_or_dividend_yield=0.006,
        paths=8_000,
        seed=31,
    ).price(contract)
    print("\nAccelerated share repurchase")
    print(f"  PV of final settlement shares: ${value.pv:,.2f}")
    print(f"  Expected average price: ${value.expected_final_average:,.4f}")
    print(f"  Upfront shares: {value.upfront_shares:,.0f}")
    print(f"  Expected settlement shares: {value.expected_settlement_shares:,.0f}")


def main() -> None:
    price_interest_rate_swap()
    price_european_option()
    price_american_option()
    price_equity_forward()
    price_asian_option()
    price_asr()


if __name__ == "__main__":
    main()

