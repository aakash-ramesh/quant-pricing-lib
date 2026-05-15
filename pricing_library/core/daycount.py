from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Callable


class DayCount(str, Enum):
    """Supported day-count conventions."""

    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360 = "30/360"


def _validate_dates(start: date, end: date) -> None:
    if end < start:
        raise ValueError(f"end date {end!s} is before start date {start!s}")


def year_fraction_act_360(start: date, end: date) -> float:
    _validate_dates(start, end)
    return (end - start).days / 360.0


def year_fraction_act_365f(start: date, end: date) -> float:
    _validate_dates(start, end)
    return (end - start).days / 365.0


def year_fraction_30_360_us(start: date, end: date) -> float:
    _validate_dates(start, end)
    d1 = min(start.day, 30)
    d2 = end.day if start.day < 30 else min(end.day, 30)
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + d2 - d1) / 360.0


DAY_COUNT_FUNCTIONS: dict[str, Callable[[date, date], float]] = {
    DayCount.ACT_360.value: year_fraction_act_360,
    DayCount.ACT_365F.value: year_fraction_act_365f,
    DayCount.THIRTY_360.value: year_fraction_30_360_us,
}


def normalize_day_count(convention: str | DayCount) -> str:
    value = convention.value if isinstance(convention, DayCount) else convention
    if value not in DAY_COUNT_FUNCTIONS:
        supported = ", ".join(sorted(DAY_COUNT_FUNCTIONS))
        raise ValueError(f"unsupported day-count convention {value!r}; supported: {supported}")
    return value


def year_fraction(start: date, end: date, convention: str | DayCount = DayCount.ACT_365F) -> float:
    """Return the accrual fraction between two dates under a supported convention."""

    return DAY_COUNT_FUNCTIONS[normalize_day_count(convention)](start, end)

