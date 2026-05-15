# Master Pricing Library

This repository is a Python pricing library for common rates and equity derivatives. It integrates the existing interest-rate swap work into a reusable rates module, adds an ASR module inspired by the existing ASR implementation, and extends the library to European options, American options, equity forwards, and arithmetic Asian options.

The code is intended for interview, learning, and model-prototyping use. It is not a production trading, risk, or valuation system.

## Repository Layout

```text
pricing_library/
  core/
    curves.py        # Discount curves and flat curves
    daycount.py      # ACT/360, ACT/365F, 30/360
    schedule.py      # Coupon schedules and business-day averaging dates
  rates/
    swaps.py         # Plain-vanilla interest-rate swaps
  equity/
    options.py       # European Black-Scholes and binomial option pricing
    forwards.py      # Equity forward pricing
    asian.py         # Arithmetic Asian option Monte Carlo
    asr.py           # Accelerated share repurchase Monte Carlo
examples/
  pricing_examples.py
tests/
  test_pricing_library.py
```

The original `irs_pricer/` package and `asr_pricer.py` script are left in place as reference implementations.

## Quick Start

The package has no required third-party runtime dependency.

```bash
python3 -m unittest discover -s tests
python3 -m examples.pricing_examples
```

For editable installation:

```bash
python3 -m pip install -e .
```

## Implemented Products

### Interest Rate Swap

A plain-vanilla fixed-vs-floating interest rate swap exchanges fixed coupons for floating coupons on a notional amount. No notional is exchanged.

Contractual features:

- Notional
- Fixed rate
- Payer or receiver direction
- Fixed and floating payment frequencies
- Fixed and floating day-count conventions
- Floating spread
- Coupon schedules with simple business-day adjustment

Pricing math:

For payment dates \(T_i\), accrual fractions \(\alpha_i\), notional \(N\), fixed rate \(K\), and discount factor \(P(0,T_i)\):

\[
PV_{\text{fixed}} = N K \sum_i \alpha_i P(0,T_i)
\]

The floating forward for period \([T_{i-1}, T_i]\) is:

\[
F_i = \frac{P(0,T_{i-1})/P(0,T_i)-1}{\alpha_i}
\]

With spread \(s\):

\[
PV_{\text{float}} = N \sum_i (F_i+s)\alpha_i P(0,T_i)
\]

For a fixed-rate payer:

\[
PV = PV_{\text{float}} - PV_{\text{fixed}}
\]

For a fixed-rate receiver, the sign is reversed. The par fixed rate is:

\[
K^* = \frac{PV_{\text{float}}}{N\sum_i \alpha_i P(0,T_i)}
\]

### European Equity Options

A European option can be exercised only at maturity. The library implements Black-Scholes-Merton closed-form valuation and a Cox-Ross-Rubinstein binomial tree.

Contractual features:

- Call or put
- Spot
- Strike
- Maturity in years
- Continuously compounded risk-free rate
- Continuous dividend yield
- Volatility
- Quantity

Black-Scholes math:

\[
d_1 = \frac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1-\sigma\sqrt{T}
\]

European call:

\[
C = S_0 e^{-qT}N(d_1)-K e^{-rT}N(d_2)
\]

European put:

\[
P = K e^{-rT}N(-d_2)-S_0 e^{-qT}N(-d_1)
\]

The implementation also returns delta, gamma, vega, theta, and rho.

### American Equity Options

An American option can be exercised on any tree step up to maturity. There is no simple general closed form for American puts, so the library uses a Cox-Ross-Rubinstein tree.

Tree math:

\[
u=e^{\sigma\sqrt{\Delta t}},
\qquad
d=\frac{1}{u},
\qquad
p=\frac{e^{(r-q)\Delta t}-d}{u-d}
\]

At each node:

\[
V = e^{-r\Delta t}\left(pV_u+(1-p)V_d\right)
\]

For American exercise:

\[
V = \max(V,\text{intrinsic value})
\]

### Equity Forward

An equity forward is an agreement to buy or sell stock at a fixed strike on a future date.

No-arbitrage forward price with continuous dividend yield:

\[
F_0 = S_0 e^{(r-q)T}
\]

Long forward PV:

\[
PV = S_0e^{-qT} - K e^{-rT}
\]

### Arithmetic Asian Option

An arithmetic Asian option pays based on the arithmetic average of observed prices rather than only the terminal price. The arithmetic-average payoff does not generally have a simple Black-Scholes closed form, so the library uses risk-neutral Monte Carlo.

Call payoff:

\[
\max(\bar{S}-K,0)
\]

Put payoff:

\[
\max(K-\bar{S},0)
\]

where:

\[
\bar{S}=\frac{1}{m}\sum_{j=1}^{m}S_{t_j}
\]

The simulated equity process is:

\[
S_{t+\Delta t}=S_t\exp\left((r-q-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}Z\right)
\]

### Accelerated Share Repurchase

An accelerated share repurchase is a structured equity transaction where a company pays a cash notional upfront, receives an initial delivery of shares, and later settles based on the average stock price over an averaging period.

Implemented contractual features:

- Cash notional
- Initial spot
- Current spot
- Upfront share delivery fraction
- Averaging dates
- Discount to average price
- Average cap and floor
- Variable notional multiplier between low and high average levels
- Discrete cash dividends
- Realized average and realized observation count for in-flight ASRs

Core settlement math:

Upfront shares:

\[
\text{UpfrontShares} = \frac{N \times f}{S_{\text{initial}}}
\]

Final net average:

\[
A_{\text{net}} = \min(\max(\bar{S}-d,\text{floor}),\text{cap})
\]

Variable notional multiplier \(w(\bar{S})\) is linearly interpolated between low and high levels and clamped to the configured multiplier bounds.

Total shares deliverable:

\[
\text{TotalShares} = \frac{N \times w(\bar{S})}{A_{\text{net}}}
\]

Final settlement shares:

\[
\text{SettlementShares} = \text{TotalShares} - \text{UpfrontShares}
\]

The ASR pricer reports the present value of final settlement shares from the corporate client's perspective:

\[
PV = e^{-rT}\mathbb{E}[\text{SettlementShares}\times S_T]
\]

The model uses risk-neutral Monte Carlo with optional antithetic paths. It is intentionally simpler than the legacy `asr_pricer.py` PDE script, which includes a Crank-Nicolson path-variable PDE and requires `numpy`/`scipy`.

## Example Usage

```python
from datetime import date

from pricing_library.core.curves import DiscountCurve
from pricing_library.rates import InterestRateSwapPricer, build_vanilla_interest_rate_swap

valuation_date = date(2026, 5, 15)
curve = DiscountCurve(
    valuation_date,
    ((0.0, 0.049), (1.0, 0.046), (3.0, 0.042), (5.0, 0.0405)),
)

swap = build_vanilla_interest_rate_swap(
    valuation_date,
    valuation_date,
    date(2031, 5, 15),
    notional=50_000_000,
    fixed_rate=0.0415,
    direction="payer",
)

value = InterestRateSwapPricer().price(swap, curve)
print(value.pv, value.par_rate, value.dv01)
```

Run all example products:

```bash
python3 -m examples.pricing_examples
```

## Engineering Practices Used

- Typed dataclasses for product contracts and valuation outputs
- Separate `core`, `rates`, and `equity` modules
- No hidden global market state
- Deterministic seeds for stochastic examples and tests
- Unit tests for reference option values, parity, American exercise, swaps, forwards, and ASR sanity
- Minimal runtime dependencies for easier GitHub reuse
- `.gitignore` and package metadata for clean repository publishing

## Known Limitations

- Calendars use weekends and optional static holiday lists only.
- Curves use continuously compounded zero rates with linear interpolation in maturity.
- Swaps use a single curve for forwarding and discounting.
- The binomial tree is a CRR tree, not a local-volatility or stochastic-volatility model.
- Monte Carlo pricers are educational and do not include variance-reduction beyond antithetic paths.
- The ASR pricer models scheduled final settlement. It does not optimize a counterparty early-termination right.

