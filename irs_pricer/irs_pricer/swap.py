from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal
from datetime import date
from .daycount import DAYCOUNT_MAP
from .schedule import generate_schedule

LegType = Literal["fixed", "float"]
PayerType = Literal["payer", "receiver"]  # payer pays fixed, receiver receives fixed (convention)

@dataclass
class FixedLeg:
    notional: float
    rate: float            # fixed rate (as decimal)
    pay_freq_months: int
    daycount: str          # e.g. "30/360"
    schedule: List[date]

@dataclass
class FloatLeg:
    notional: float
    pay_freq_months: int
    daycount: str          # e.g. "ACT/360"
    schedule: List[date]
    index_spread: float = 0.0  # in decimal

@dataclass
class Swap:
    trade_date: date
    start_date: date
    end_date: date
    payer: PayerType
    fixed_leg: FixedLeg
    float_leg: FloatLeg

def build_plain_vanilla_swap(trade_date: date, start: date, end: date,
                             notional: float, fixed_rate: float,
                             fixed_dc: str = "30/360", float_dc: str = "ACT/360",
                             fixed_freq_m: int = 6, float_freq_m: int = 3,
                             payer: PayerType = "payer",
                             index_spread: float = 0.0) -> Swap:
    fixed_sched = generate_schedule(start, end, fixed_freq_m)
    float_sched = generate_schedule(start, end, float_freq_m)
    fixed_leg = FixedLeg(notional, fixed_rate, fixed_freq_m, fixed_dc, fixed_sched)
    float_leg = FloatLeg(notional, float_freq_m, float_dc, float_sched, index_spread)
    return Swap(trade_date, start, end, payer, fixed_leg, float_leg)
