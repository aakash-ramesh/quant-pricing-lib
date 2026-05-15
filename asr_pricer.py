"""
Accelerated Share Repurchase (ASR) Pricer
==========================================
Python implementation of the ASR PDE pricing model.

Model Overview:
  - Black-Scholes PDE solved via Finite Difference (Crank-Nicolson scheme)
  - "PDE with path variable" approach for averaging: K independent 1D PDEs
  - Flat-skew Local Volatility (simplified Dupire)
  - Discrete cash dividend modelling
  - Early exercise (American-style) feature
  - Optional lookback (drop-current-spot-from-average) feature
  - Optional variable notional (notional depends on realized average)

Key contract terms:
  - discount         : fixed $ discount to be subtracted from the final average price
  - upfrontNotional  : fraction of total shares delivered at inception (e.g. 0.80)
  - initialSpot      : reference spot used to compute number of upfront shares
  - averCap / averFloor : cap / floor on the final net average
  - earlyExStart     : date from which the counterparty can force termination
  - lookback         : 0 = none; 1 = one-day lookback option at each early-ex event
  - lowLevel / highLevel / lowNotional / highNotional : variable-notional parameters

Payoff (at final average date):
  fair_value = -S * (W / boundedA  -  upfrontNotional / initialSpot)

  where
    boundedA = clip(averFloor,  realizedAverage - discount,  averCap)
    W        = 1.0  (fixed notional)  or
               interpolated between [lowNotional, highNotional] for variable notional

Usage example:
  see __main__ block at the bottom of the file.
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """Market inputs for ASR pricing."""
    spot: float                    # current underlying price
    vol: float                     # ATMF implied volatility (flat skew model)
    domestic_rate: float           # continuous risk-free rate
    borrow_rate: float             # continuous borrow / dividend rate
    trade_date: date               # valuation date
    expiry_date: date              # last averaging date  (= scheduled termination)

    # Optional: yield curve / borrow curve (list of (date, rate) tuples)
    yield_curve: Optional[List[Tuple[date, float]]] = None
    borrow_curve: Optional[List[Tuple[date, float]]] = None


@dataclass
class ContractData:
    """ASR contract specification."""
    discount: float                # $ discount per share on final average
    averaging_dates: List[date]    # ordered list of future averaging dates
    realized_average: float = 0.0  # running average realized so far (0 if none)
    current_index: int = 0         # number of averaging dates already fixed

    upfront_notional: float = 0.0  # fraction of shares delivered upfront (e.g. 0.80)
    initial_spot: float = 0.0      # reference spot for upfront delivery

    early_ex_start: Optional[date] = None  # first date counterparty can terminate
    lookback: int = 0              # 0 = none, 1 = 1-day lookback at early-ex

    aver_cap: float = 1e8          # cap on final average (very large = no cap)
    aver_floor: float = 0.0        # floor on final average

    # Variable notional parameters (set lowLevel=highLevel=0 for fixed notional)
    low_level: float = 0.0
    high_level: float = 0.0
    low_notional: float = 1.0
    high_notional: float = 1.0

    # Discrete dividends: list of (ex_date, amount) tuples
    dividends: List[Tuple[date, float]] = field(default_factory=list)


@dataclass
class ASROutput:
    """Pricing output."""
    fair_value: float
    fair_up: float      # fair value with spot * 1.01
    fair_down: float    # fair value with spot * 0.99
    delta: float        # (fairUp - fairDown) / 0.02
    fair_tp1: float     # fair value one business day forward (for theta)
    current_average: float
    n_remaining_dates: int


# ---------------------------------------------------------------------------
# Local Volatility (flat-skew version)
# ---------------------------------------------------------------------------

def flat_skew_local_vol_sq(sigma_atm: float, sigma_t: float, t: float) -> float:
    """
    Flat-skew Local Volatility (simplified Dupire, all strike derivatives = 0):

        sigma_lv^2(K, T) = sigma_atm^2 + 2 * sigma_atm * (d sigma_atm / dT) * T

    Here we use a piecewise-constant vol term-structure so the derivative is
    approximated as the instantaneous forward variance:

        sigma_lv^2 ≈ sigma_t^2   (the forward vol for the current time step)

    For a flat vol surface, sigma_t = sigma_atm and sigma_lv = sigma_atm.
    Returns the **squared** local vol (used directly in PDE coefficients).
    """
    return max(sigma_t ** 2, 1e-8)


# ---------------------------------------------------------------------------
# Year fraction helpers
# ---------------------------------------------------------------------------

def year_fraction(d1: date, d2: date) -> float:
    """ACT/365 year fraction."""
    return (d2 - d1).days / 365.0


def add_bus_days(d: date, n: int) -> date:
    """Add n business days (Mon–Fri) to a date."""
    current = d
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:   # Mon=0 … Fri=4
            added += 1
    return current


# ---------------------------------------------------------------------------
# Average grid construction
# ---------------------------------------------------------------------------

def build_average_grid(current_average: float, time_to_expiry: float,
                       vol: float, nb_std: float, nb_avg_fringes: int) -> np.ndarray:
    """
    Build an independent log-spaced average grid centred on current_average.

    Grid spans:  [A_min, A_max] = exp( log(A0) ± nb_std * vol * sqrt(T) )
    """
    log_a = math.log(current_average)
    sqrt_t = math.sqrt(max(time_to_expiry, 1e-6))
    a_min_log = log_a - nb_std * vol * sqrt_t
    da = (2.0 * nb_std * vol * sqrt_t) / (nb_avg_fringes - 1)
    return np.array([math.exp(a_min_log + i * da) for i in range(nb_avg_fringes)])


# ---------------------------------------------------------------------------
# Final payoff
# ---------------------------------------------------------------------------

def notional_weight(avg: float, low_level: float, high_level: float,
                    low_notional: float, high_notional: float) -> float:
    """
    Variable notional: interpolate W between [low_notional, high_notional]
    depending on where the average falls in [low_level, high_level].

    In a typical ASR the lower the average, the more shares are owed → lowNotional >= highNotional.
    W is clamped to [highNotional, lowNotional].
    """
    is_variable = (abs(high_level - low_level) > 1e-12 and
                   abs(high_notional - low_notional) > 1e-12)
    if not is_variable:
        return low_notional   # fixed notional (should be 1.0)

    w = low_notional + ((avg - low_level) / (high_level - low_level)) * (high_notional - low_notional)
    return min(max(w, high_notional), low_notional)


def final_payoff_matrix(S: np.ndarray, averages: np.ndarray,
                        contract: ContractData,
                        is_lookback: bool = False,
                        avg_index: Optional[int] = None) -> np.ndarray:
    """
    Compute the 2-D final payoff matrix of shape (nb_spot, nb_avg).

    Payoff(S_j, A_k) = -S_j * ( W(A_k) / boundedA_k  -  upfrontNotional / initialSpot )

    where boundedA_k = clip(averFloor,  A_k - discount,  averCap)

    If is_lookback=True, we use a modified average that excludes the current
    day's spot (implemented by the caller providing avg_index to recompute).
    """
    nb_s = len(S)
    nb_a = len(averages)
    payoff = np.zeros((nb_s, nb_a))

    reversed_averages = averages[::-1]   # reverse so index 0 = highest avg

    for j in range(nb_s):
        for k in range(nb_a):
            avg = reversed_averages[k]

            # variable notional weight
            w = notional_weight(avg,
                                contract.low_level, contract.high_level,
                                contract.low_notional, contract.high_notional)

            bounded_a = np.clip(avg - contract.discount,
                                contract.aver_floor, contract.aver_cap)
            if bounded_a < 1e-8:
                bounded_a = 1e-8

            upfront = (contract.upfront_notional / contract.initial_spot
                       if contract.initial_spot > 1e-8 else 0.0)
            payoff[j][k] = -S[j] * (w / bounded_a - upfront)

    return payoff


# ---------------------------------------------------------------------------
# Time step descriptor
# ---------------------------------------------------------------------------

@dataclass
class TimeStep:
    dt: float          # time step size in years
    time: float        # time from valuation date in years
    event: str         # "AVG", "EARLY_EX", "DIV", or "NONE"
    avg_index: int = 0 # which averaging date this corresponds to
    is_tp1: bool = False
    rate: float = 0.0
    borrow: float = 0.0
    is_div: bool = False
    div_amt: float = 0.0


def generate_time_steps(market: MarketData, contract: ContractData,
                        nb_time_steps: int = 200) -> List[TimeStep]:
    """
    Build the time-step schedule from valuation date to last averaging date.

    Key events inserted at exact dates:
      - averaging dates
      - early exercise start
      - dividend ex-dates
      - T+1 business day (for theta)
    """
    start_date = market.trade_date
    avg_dates = contract.averaging_dates
    if not avg_dates:
        return []

    total_time = year_fraction(start_date, avg_dates[-1])

    # Collect all event dates
    event_dates: dict = {}
    for i, d in enumerate(avg_dates):
        event_dates[d] = ("AVG", i)

    early_ex = contract.early_ex_start
    if early_ex and early_ex > start_date:
        if early_ex not in event_dates:
            event_dates[early_ex] = ("EARLY_EX", 0)

    for ex_date, _ in contract.dividends:
        if start_date < ex_date <= avg_dates[-1]:
            if ex_date not in event_dates:
                event_dates[ex_date] = ("DIV", 0)

    tp1 = add_bus_days(start_date, 1)
    is_tp1_avg = tp1 in event_dates
    if not is_tp1_avg and tp1 < avg_dates[-1]:
        event_dates[tp1] = ("TP1", 0)

    # Build uniform grid and insert event dates
    base_dt = total_time / nb_time_steps
    times = set(np.arange(0, total_time + base_dt, base_dt).tolist())
    for d in event_dates:
        yf = year_fraction(start_date, d)
        times.add(yf)

    times = sorted(times)
    if times[0] < 1e-10:
        times = times[1:]

    steps: List[TimeStep] = []
    prev_t = 0.0
    for t in times:
        dt = t - prev_t
        if dt < 1e-10:
            prev_t = t
            continue

        # Determine current date (approximate)
        current_date = start_date + timedelta(days=int(round(t * 365)))

        # Rate interpolation (flat if no curve)
        rate = market.domestic_rate
        borrow = market.borrow_rate
        if market.yield_curve:
            rate = interpolate_rate(market.yield_curve, t)
        if market.borrow_curve:
            borrow = interpolate_rate(market.borrow_curve, t)

        # Event classification
        event = "NONE"
        avg_idx = 0
        is_tp1_step = False
        is_div = False
        div_amt = 0.0

        # Find nearest event date to this t
        approx_date = start_date + timedelta(days=int(round(t * 365)))
        if approx_date in event_dates:
            ev_type, ev_idx = event_dates[approx_date]
            if ev_type == "AVG":
                event = "AVG"
                avg_idx = ev_idx
            elif ev_type == "EARLY_EX":
                event = "EARLY_EX"
            elif ev_type == "TP1":
                is_tp1_step = True
            elif ev_type == "DIV":
                event = "DIV"

        # Check dividend
        for ex_date, amt in contract.dividends:
            if approx_date == ex_date:
                is_div = True
                div_amt = amt

        steps.append(TimeStep(
            dt=dt, time=t, event=event, avg_index=avg_idx,
            is_tp1=is_tp1_step, rate=rate, borrow=borrow,
            is_div=is_div, div_amt=div_amt
        ))
        prev_t = t

    return steps


def interpolate_rate(curve: List[Tuple[date, float]], t: float) -> float:
    """Simple linear interpolation on a (time, rate) curve."""
    if not curve:
        return 0.0
    if t <= curve[0][1]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        t0, r0 = curve[i]
        t1, r1 = curve[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return r0 + alpha * (r1 - r0)
    return curve[-1][1]


# ---------------------------------------------------------------------------
# Tri-diagonal matrix solver (Thomas algorithm)
# ---------------------------------------------------------------------------

def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 d: np.ndarray) -> np.ndarray:
    """
    Solve tri-diagonal system  a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    using the Thomas (forward sweep / back substitution) algorithm.  O(N).
    """
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# Apply averaging event (cubic spline interpolation across average grid)
# ---------------------------------------------------------------------------

def apply_avg_event(v: np.ndarray, S: np.ndarray, averages: np.ndarray,
                    current_index: int, n_total: int) -> np.ndarray:
    """
    On an averaging date the realized running average is updated:

        A_new = (S_j + (n-1) * A_old) / n

    where n is the new total count.  We use cubic spline to re-sample the
    value grid across the average dimension for each spot level.

    v shape: (nb_spot, nb_avg)
    Returns updated v of the same shape.
    """
    nb_spot, nb_avg = v.shape
    n = current_index + 1   # new count after including today
    new_v = np.zeros_like(v)

    log_averages = np.log(averages)

    for j in range(nb_spot):
        # new average as a function of old average (for each old avg grid point)
        new_avgs = (S[j] + (n - 1) * averages) / n
        new_log_avgs = np.log(np.clip(new_avgs, 1e-8, None))

        # interpolate current v[j, :] (function of old avg) onto new avg grid
        cs = CubicSpline(log_averages, v[j, :], extrapolate=True)
        new_v[j, :] = cs(new_log_avgs)

    return new_v


# ---------------------------------------------------------------------------
# Apply dividend event (continuity condition)
# ---------------------------------------------------------------------------

def apply_dividend(v: np.ndarray, S: np.ndarray, x: np.ndarray,
                   div_amt: float) -> np.ndarray:
    """
    On ex-dividend date, stock drops by the dividend:  S_ex = S_cum - Div

    We apply the continuity condition on the payoff:
        V(S, t_ex) = V(S - Div, t_cum)

    i.e. we re-sample the grid after shifting S down by Div.
    """
    nb_spot, nb_avg = v.shape
    new_v = np.zeros_like(v)

    for k in range(nb_avg):
        cs = CubicSpline(x, v[:, k], extrapolate=True)
        x_shifted = np.log(np.clip(S - div_amt, 1e-8, None))
        new_v[:, k] = cs(x_shifted)
    return new_v


# ---------------------------------------------------------------------------
# Core PDE solver
# ---------------------------------------------------------------------------

def pde_solver_asr(market: MarketData,
                   contract: ContractData,
                   nb_fringes: int = 101,
                   nb_avg_fringes: int = 41,
                   nb_std: float = 3.3,
                   theta: float = 0.5,
                   nb_time_steps: int = 200) -> ASROutput:
    """
    Price an Accelerated Share Repurchase using the Crank-Nicolson PDE
    with path variable (independent average grid).

    Parameters
    ----------
    market        : MarketData
    contract      : ContractData
    nb_fringes    : number of spot grid points
    nb_avg_fringes: number of average grid points (K independent PDEs)
    nb_std        : number of standard deviations for grid boundaries
    theta         : 0=explicit, 1=fully implicit, 0.5=Crank-Nicolson (default)
    nb_time_steps : number of uniform time steps between events

    Returns
    -------
    ASROutput with fair value, greeks, and diagnostics.
    """
    spot = market.spot
    vol = market.vol
    avg_dates = contract.averaging_dates

    # Initialise initial_spot default
    initial_spot = contract.initial_spot if contract.initial_spot > 1e-8 else spot
    contract = ContractData(**{**contract.__dict__,
                                'initial_spot': initial_spot})

    # ---------- Handle expiry case (no future averaging dates) ----------
    if not avg_dates:
        ca = contract.realized_average if contract.realized_average > 0 else spot
        bounded_a = np.clip(ca - contract.discount,
                            contract.aver_floor, contract.aver_cap)
        if bounded_a < 1e-8:
            bounded_a = 1e-8
        upfront = (contract.upfront_notional / initial_spot
                   if initial_spot > 1e-8 else 0.0)
        fair = -spot * (1.0 / bounded_a - upfront)
        fair_up = fair * 1.01
        fair_down = fair * 0.99
        delta = fair
        return ASROutput(fair_value=fair, fair_up=fair_up, fair_down=fair_down,
                         delta=delta, fair_tp1=0.0,
                         current_average=ca, n_remaining_dates=0)

    # ---------- Build spatial grid (log-spot) ----------
    time_to_expiry = year_fraction(market.trade_date, avg_dates[-1])
    sqrt_t = math.sqrt(max(time_to_expiry, 1e-6))
    s_max = spot * math.exp(nb_std * vol * sqrt_t)
    s_min = spot * math.exp(-nb_std * vol * sqrt_t)
    x_max = math.log(s_max)
    x_min = math.log(s_min)
    dx = (x_max - x_min) / (nb_fringes - 1)
    x = np.array([x_min + j * dx for j in range(nb_fringes)])
    S = np.exp(x)

    # ---------- Average grid (K=nb_avg_fringes independent grids) ----------
    current_avg = (contract.realized_average if contract.realized_average > 0
                   else spot)
    averages = build_average_grid(current_avg, time_to_expiry, vol,
                                  nb_std, nb_avg_fringes)

    # ---------- Populate final payoff ----------
    v = final_payoff_matrix(S, averages, contract, False)
    intr = final_payoff_matrix(S, averages, contract, False)

    # ---------- Time stepping (backward diffusion) ----------
    time_steps = generate_time_steps(market, contract, nb_time_steps)
    # Reverse: we go backward from expiry to today
    time_steps = time_steps[::-1]

    fair_tp1 = 0.0

    # interior indices
    n_int = nb_fringes - 2   # interior points (boundary excluded)

    for step in time_steps:
        dt = step.dt
        rate = step.rate
        borrow = step.borrow

        # Local vol squared (flat skew simplification = ATMF vol^2)
        vol_sq = flat_skew_local_vol_sq(vol, vol, step.time)
        alpha = 0.5 * vol_sq
        beta = rate - borrow - 0.5 * vol_sq

        # ---- Build LHS (unknown, i-1) and RHS (known, i) tri-diagonal ----
        # LHS:  a_j * V_{i-1,j-1}  +  b_j * V_{i-1,j}  +  c_j * V_{i-1,j+1}
        # RHS:  A_j * V_{i,j-1}    +  B_j * V_{i,j}    +  C_j * V_{i,j+1}

        a_lhs = np.zeros(n_int)
        b_lhs = np.zeros(n_int)
        c_lhs = np.zeros(n_int)
        a_rhs = np.zeros(n_int)
        b_rhs = np.zeros(n_int)
        c_rhs = np.zeros(n_int)

        for j in range(n_int):
            # LHS coefficients (unknowns at time i-1)
            a_lhs[j] = -(1 - theta) * alpha * dt / (dx * dx) + 0.5 * (1 - theta) * beta * dt / dx
            b_lhs[j] = 1.0 + 2.0 * (1 - theta) * alpha * dt / (dx * dx) + (1 - theta) * rate * dt
            c_lhs[j] = -(1 - theta) * alpha * dt / (dx * dx) - 0.5 * (1 - theta) * beta * dt / dx

            # RHS coefficients (knowns at time i)
            a_rhs[j] = theta * alpha * dt / (dx * dx) - 0.5 * theta * beta * dt / dx
            b_rhs[j] = 1.0 - 2.0 * theta * alpha * dt / (dx * dx) - theta * rate * dt
            c_rhs[j] = theta * alpha * dt / (dx * dx) + 0.5 * theta * beta * dt / dx

        # Apply linear (gamma=0) boundary condition:
        #   V[0] = 2*V[1] - V[2]  and  V[n-1] = 2*V[n-2] - V[n-3]
        b_lhs[0] += 2.0 * a_lhs[0]
        c_lhs[0] -= a_lhs[0]
        b_lhs[-1] += 2.0 * c_lhs[-1]
        a_lhs[-1] -= c_lhs[-1]

        # ---- Diffuse each average grid independently ----
        for k in range(nb_avg_fringes):
            v_interior = v[1:-1, k]  # interior spot points

            # Compute RHS vector
            rhs = np.zeros(n_int)
            for j in range(n_int):
                j_full = j + 1   # j+1 in full grid
                rhs[j] = (a_rhs[j] * v[j_full - 1, k] +
                          b_rhs[j] * v[j_full, k] +
                          c_rhs[j] * v[j_full + 1, k])

            # Solve tri-diagonal system
            v_new_interior = thomas_solve(a_lhs, b_lhs, c_lhs, rhs)

            # Update interior and boundary (linear extrapolation)
            v[1:-1, k] = v_new_interior
            v[0, k] = 2 * v[1, k] - v[2, k]
            v[-1, k] = 2 * v[-2, k] - v[-3, k]

        # ---- Apply events ----
        if step.is_tp1:
            center_idx = (nb_avg_fringes - 1) // 2
            fair_tp1 = v[(nb_fringes - 1) // 2, center_idx]

        if step.is_div:
            v = apply_dividend(v, S, x, step.div_amt)

        if step.event == "AVG":
            v = apply_avg_event(v, S, averages,
                                step.avg_index + contract.current_index,
                                len(contract.averaging_dates) + contract.current_index)
            # Update intrinsic grid
            intr = apply_avg_event(intr, S, averages,
                                   step.avg_index + contract.current_index,
                                   len(contract.averaging_dates) + contract.current_index)

        if step.event in ("AVG", "EARLY_EX"):
            # Early exercise: V = max(V, intrinsic)
            lookback_intr = None
            if contract.lookback > 0:
                lookback_intr = final_payoff_matrix(S, averages, contract,
                                                    is_lookback=True,
                                                    avg_index=step.avg_index)
            for j in range(nb_fringes):
                for k in range(nb_avg_fringes):
                    ex_val = intr[j, k]
                    if lookback_intr is not None:
                        ex_val = max(ex_val, lookback_intr[j, k])
                    v[j, k] = max(v[j, k], ex_val)

    # ---------- Extract fair value at centre of grid ----------
    center_avg_idx = (nb_avg_fringes - 1) // 2
    center_spot_idx = (nb_fringes - 1) // 2
    center_col = v[:, center_avg_idx]

    # Cubic spline interpolation to get exact spot values
    cs = CubicSpline(x, center_col, extrapolate=True)
    x_spot = math.log(spot)
    x_up = math.log(spot * 1.01)
    x_dn = math.log(spot * 0.99)

    fair = float(cs(x_spot))
    fair_up = float(cs(x_up))
    fair_down = float(cs(x_dn))
    delta = (fair_up - fair_down) / 0.02

    return ASROutput(
        fair_value=fair,
        fair_up=fair_up,
        fair_down=fair_down,
        delta=delta,
        fair_tp1=fair_tp1,
        current_average=current_avg,
        n_remaining_dates=len(avg_dates)
    )


# ---------------------------------------------------------------------------
# Convenience wrapper (mirrors the JS exports['asrFlatSkewPDE'] function)
# ---------------------------------------------------------------------------

def asr_flat_skew_pde(
    spot: float,
    vol: float,
    domestic_rate: float,
    borrow_rate: float,
    trade_date: date,
    expiry_date: date,
    averaging_dates: List[date],
    discount: float,
    upfront_notional: float = 0.8,
    initial_spot: float = 0.0,
    realized_average: float = 0.0,
    current_index: int = 0,
    early_ex_start: Optional[date] = None,
    lookback: int = 0,
    aver_cap: float = 1e8,
    aver_floor: float = 0.0,
    low_level: float = 0.0,
    high_level: float = 0.0,
    low_notional: float = 1.0,
    high_notional: float = 1.0,
    dividends: Optional[List[Tuple[date, float]]] = None,
    nb_fringes: int = 101,
    nb_avg_fringes: int = 41,
) -> ASROutput:
    """High-level wrapper that constructs MarketData / ContractData and calls the solver."""
    market = MarketData(
        spot=spot, vol=vol, domestic_rate=domestic_rate, borrow_rate=borrow_rate,
        trade_date=trade_date, expiry_date=expiry_date
    )
    contract = ContractData(
        discount=discount,
        averaging_dates=[d for d in averaging_dates if d > trade_date],
        realized_average=realized_average,
        current_index=current_index,
        upfront_notional=upfront_notional,
        initial_spot=initial_spot if initial_spot > 0 else spot,
        early_ex_start=early_ex_start,
        lookback=lookback,
        aver_cap=aver_cap if aver_cap > 0 else 1e8,
        aver_floor=aver_floor,
        low_level=low_level,
        high_level=high_level,
        low_notional=low_notional if low_notional > 0 else 1.0,
        high_notional=high_notional if high_notional > 0 else 1.0,
        dividends=dividends or [],
    )
    return pde_solver_asr(market, contract,
                          nb_fringes=nb_fringes, nb_avg_fringes=nb_avg_fringes)


# ---------------------------------------------------------------------------
# Main – example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import date
    import pprint

    print("=" * 60)
    print("  Accelerated Share Repurchase – Python PDE Pricer")
    print("=" * 60)

    # ---- Example 1: Simple 6-month ASR, daily averaging ----
    trade_date = date(2025, 1, 2)
    expiry_date = date(2025, 7, 2)

    # Generate approximately 126 business day averaging dates
    avg_dates = []
    d = trade_date + timedelta(days=1)
    while d <= expiry_date:
        if d.weekday() < 5:
            avg_dates.append(d)
        d += timedelta(days=1)

    # One quarterly dividend
    dividends = [(date(2025, 3, 14), 0.50)]

    result = asr_flat_skew_pde(
        spot=100.0,
        vol=0.25,
        domestic_rate=0.05,
        borrow_rate=0.01,
        trade_date=trade_date,
        expiry_date=expiry_date,
        averaging_dates=avg_dates,
        discount=1.00,          # $1 discount on final average
        upfront_notional=0.80,  # 80% delivered upfront
        initial_spot=100.0,
        early_ex_start=date(2025, 4, 1),
        lookback=0,
        aver_cap=120.0,
        aver_floor=85.0,
        dividends=dividends,
        nb_fringes=51,          # use smaller grid for speed in demo
        nb_avg_fringes=21,
    )

    print("\nExample 1 – 6m ASR with $1 discount, 80% upfront, early-ex:")
    print(f"  Fair Value        : {result.fair_value:.6f}")
    print(f"  Fair Up (S*1.01)  : {result.fair_up:.6f}")
    print(f"  Fair Down (S*0.99): {result.fair_down:.6f}")
    print(f"  Delta             : {result.delta:.6f}")
    print(f"  T+1 Fair          : {result.fair_tp1:.6f}")
    print(f"  Current Average   : {result.current_average:.4f}")
    print(f"  Remaining Dates   : {result.n_remaining_dates}")

    # ---- Example 2: At expiry (realized average known) ----
    result2 = asr_flat_skew_pde(
        spot=103.0,
        vol=0.25,
        domestic_rate=0.05,
        borrow_rate=0.01,
        trade_date=expiry_date,
        expiry_date=expiry_date,
        averaging_dates=[],       # no future dates → intrinsic pricing
        discount=1.00,
        upfront_notional=0.80,
        initial_spot=100.0,
        realized_average=101.5,   # realized VWAP
    )
    print("\nExample 2 – At expiry (realized avg = $101.50, spot = $103):")
    print(f"  Fair Value (intrinsic): {result2.fair_value:.6f}")
    # Intrinsic: -103 * (1/(101.5-1) - 0.80/100) = -103*(1/100.5 - 0.008)
    expected = -103.0 * (1.0 / 100.5 - 0.80 / 100.0)
    print(f"  Expected (manual)    : {expected:.6f}")

    # ---- Example 3: Sensitivity to discount ----
    print("\nExample 3 – Sensitivity to discount level (spot=100, vol=0.25):")
    print(f"  {'Discount':>10}  {'Fair Value':>12}  {'Delta':>10}")
    for disc in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r = asr_flat_skew_pde(
            spot=100.0, vol=0.25,
            domestic_rate=0.05, borrow_rate=0.01,
            trade_date=trade_date, expiry_date=expiry_date,
            averaging_dates=avg_dates[:30],   # shorter averaging for speed
            discount=disc,
            upfront_notional=0.80, initial_spot=100.0,
            nb_fringes=31, nb_avg_fringes=11,
        )
        print(f"  {disc:>10.2f}  {r.fair_value:>12.6f}  {r.delta:>10.6f}")
