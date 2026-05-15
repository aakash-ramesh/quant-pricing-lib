# Master Pricing Library

This repository is a Python pricing library for common rates and equity derivatives. It integrates the existing interest-rate swap work into a reusable rates module, adds an ASR module inspired by the existing ASR implementation, and extends the library to European options, American options, stochastic-volatility options, equity forwards, and arithmetic Asian options.

## Repository Layout

```text
pricing_library/
  core/
    curves.py        # Discount curves, flat curves, interpolation, compounding
    daycount.py      # ACT/360, ACT/365F, 30/360
    schedule.py      # Market calendars, coupon schedules, business-day dates
  rates/
    swaps.py         # Single-curve and multi-curve interest-rate swaps
  equity/
    options.py       # Black-Scholes, binomial, local-vol, Heston Monte Carlo
    forwards.py      # Equity forward pricing
    asian.py         # Arithmetic Asian option Monte Carlo with variance reduction
    asr.py           # Accelerated share repurchase Monte Carlo
examples/
  pricing_examples.py
tests/
  test_pricing_library.py
```

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

## Core Infrastructure

### Calendars and Schedules

The schedule module supports weekend-only calendars, static custom holidays, and named rule-based calendars:

- `weekend`
- `us_federal`
- `target2`
- `nyse`

Supported business-day adjustments are `following`, `modified_following`, `preceding`, and `modified_preceding`. Schedule generation also supports end-of-month rolling.

### Curves

`DiscountCurve` supports:

- Zero-rate or discount-factor input
- Continuous, simple, annual, semiannual, quarterly, and monthly compounding
- Linear zero-rate interpolation
- Linear discount-factor interpolation
- Log-linear discount-factor interpolation
- Parallel zero-rate bumps
- Simple money-market discount-factor bootstrapping

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
- Coupon schedules with following, modified-following, preceding, and modified-preceding adjustment
- Weekend, custom-holiday, US Federal, TARGET2, and NYSE calendars
- End-of-month schedule generation

Pricing features:

- Single-curve pricing by default
- Optional separate discounting and floating-rate forecast curves
- Discount, forecast, and total DV01

Pricing math:

For payment dates $T_i$, accrual fractions $\alpha_i$, notional $N$, fixed rate $K$, discount factor $P^d(0,T_i)$, and forecast discount factor $P^f(0,T_i)$:

$$
PV_{\text{fixed}} = N K \sum_i \alpha_i P^d(0,T_i)
$$

The floating forward for period $[T_{i-1}, T_i]$ is:

$$
F_i = \frac{P^f(0,T_{i-1})/P^f(0,T_i)-1}{\alpha_i}
$$

With spread $s$:

$$
PV_{\text{float}} = N \sum_i (F_i+s)\alpha_i P^d(0,T_i)
$$

For a fixed-rate payer:

$$
PV = PV_{\text{float}} - PV_{\text{fixed}}
$$

For a fixed-rate receiver, the sign is reversed. The par fixed rate is:

$$
K^* = \frac{PV_{\text{float}}}{N\sum_i \alpha_i P^d(0,T_i)}
$$

### European Equity Options

A European option can be exercised only at maturity. The library implements Black-Scholes-Merton closed-form valuation, CRR/Jarrow-Rudd/Tian binomial trees, optional local-volatility trees, and Heston stochastic-volatility Monte Carlo.

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

$$
d_1 = \frac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1-\sigma\sqrt{T}
$$

European call:

$$
C = S_0 e^{-qT}N(d_1)-K e^{-rT}N(d_2)
$$

European put:

$$
P = K e^{-rT}N(-d_2)-S_0 e^{-qT}N(-d_1)
$$

The implementation also returns delta, gamma, vega, theta, and rho.

Heston Monte Carlo uses full-truncation Euler variance steps:

$$
dS_t=(r-q)S_tdt+\sqrt{v_t}S_tdW^S_t
$$

$$
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW^v_t,
\qquad
dW^SdW^v=\rho dt
$$

### American Equity Options

An American option can be exercised on any tree step up to maturity. There is no simple general closed form for American puts, so the library uses binomial trees. Supported tree models are `crr`, `jarrow_rudd`, and `tian`; a spot/time local-volatility callback is also available for smaller non-recombining trees.

Tree math:

$$
u=e^{\sigma\sqrt{\Delta t}},
\qquad
d=\frac{1}{u},
\qquad
p=\frac{e^{(r-q)\Delta t}-d}{u-d}
$$

At each node:

$$
V = e^{-r\Delta t}\left(pV_u+(1-p)V_d\right)
$$

For American exercise:

$$
V = \max(V,\text{intrinsic value})
$$

### Equity Forward

An equity forward is an agreement to buy or sell stock at a fixed strike on a future date.

No-arbitrage forward price with continuous dividend yield:

$$
F_0 = S_0 e^{(r-q)T}
$$

Long forward PV:

$$
PV = S_0e^{-qT} - K e^{-rT}
$$

### Arithmetic Asian Option

An arithmetic Asian option pays based on the arithmetic average of observed prices rather than only the terminal price. The arithmetic-average payoff does not generally have a simple Black-Scholes closed form, so the library uses risk-neutral Monte Carlo.

Call payoff:

$$
\max(\bar{S}-K,0)
$$

Put payoff:

$$
\max(K-\bar{S},0)
$$

where:

$$
\bar{S}=\frac{1}{m}\sum_{j=1}^{m}S_{t_j}
$$

The simulated equity process is:

$$
S_{t+\Delta t}=S_t\exp\left((r-q-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}Z\right)
$$

Variance-reduction options include:

- Antithetic paths
- Moment matching
- Geometric-average Asian control variate

The Monte Carlo result includes a standard error and 95% confidence interval.

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

$$
\text{UpfrontShares} = \frac{N \times f}{S_{\text{initial}}}
$$

Final net average:

$$
A_{\text{net}} = \min(\max(\bar{S}-d,\text{floor}),\text{cap})
$$

Variable notional multiplier $w(\bar{S})$ is linearly interpolated between low and high levels and clamped to the configured multiplier bounds.

Total shares deliverable:

$$
\text{TotalShares} = \frac{N \times w(\bar{S})}{A_{\text{net}}}
$$

Final settlement shares:

$$
\text{SettlementShares} = \text{TotalShares} - \text{UpfrontShares}
$$

The ASR pricer reports the present value of final settlement shares from the corporate client's perspective:

$$
PV = e^{-rT}\mathbb{E}[\text{SettlementShares}\times S_T]
$$

The model uses risk-neutral Monte Carlo with optional antithetic paths. It is intentionally simpler than the legacy `asr_pricer.py` PDE script, which includes a Crank-Nicolson path-variable PDE and requires `numpy`/`scipy`.

## Example Usage

```python
from datetime import date

from pricing_library.core.curves import DiscountCurve
from pricing_library.rates import InterestRateSwapPricer, build_vanilla_interest_rate_swap

valuation_date = date(2026, 5, 15)
discount_curve = DiscountCurve(
    valuation_date,
    ((0.0, 0.049), (1.0, 0.046), (3.0, 0.042), (5.0, 0.0405)),
    interpolation="log_linear_discount",
)
forecast_curve = DiscountCurve(
    valuation_date,
    ((0.0, 0.050), (1.0, 0.048), (3.0, 0.044), (5.0, 0.0420)),
    interpolation="log_linear_discount",
)

swap = build_vanilla_interest_rate_swap(
    valuation_date,
    valuation_date,
    date(2031, 5, 15),
    notional=50_000_000,
    fixed_rate=0.0415,
    direction="payer",
    calendar="us_federal",
)

value = InterestRateSwapPricer().price(swap, discount_curve, forecast_curve)
print(value.pv, value.par_rate, value.discount_dv01, value.forecast_dv01)
```

Run all example products:

```bash
python3 -m examples.pricing_examples
```
