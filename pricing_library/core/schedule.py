from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from typing import Iterable, Literal

BusinessDayAdjustment = Literal["none", "following", "modified_following"]


def is_business_day(day: date, holidays: Iterable[date] = ()) -> bool:
    return day.weekday() < 5 and day not in set(holidays)


def add_months(day: date, months: int) -> date:
    if months <= 0:
        raise ValueError("months must be positive")
    year = day.year + (day.month - 1 + months) // 12
    month = (day.month - 1 + months) % 12 + 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


def adjust_business_day(
    day: date,
    adjustment: BusinessDayAdjustment = "none",
    holidays: Iterable[date] = (),
) -> date:
    if adjustment == "none" or is_business_day(day, holidays):
        return day
    if adjustment not in {"following", "modified_following"}:
        raise ValueError(f"unsupported business-day adjustment {adjustment!r}")

    adjusted = day
    holiday_set = set(holidays)
    while not is_business_day(adjusted, holiday_set):
        adjusted += timedelta(days=1)

    if adjustment == "modified_following" and adjusted.month != day.month:
        adjusted = day
        while not is_business_day(adjusted, holiday_set):
            adjusted -= timedelta(days=1)
    return adjusted


def generate_schedule(
    start: date,
    end: date,
    frequency_months: int,
    *,
    adjustment: BusinessDayAdjustment = "none",
    holidays: Iterable[date] = (),
) -> list[date]:
    """Generate an unadjusted or business-day-adjusted coupon schedule."""

    if end <= start:
        raise ValueError("end date must be after start date")

    dates = [start]
    current = start
    while True:
        current = add_months(current, frequency_months)
        if current >= end:
            dates.append(end)
            break
        dates.append(current)

    return [adjust_business_day(d, adjustment, holidays) for d in dates]


def business_days(
    start: date,
    end: date,
    *,
    include_start: bool = False,
    include_end: bool = True,
    holidays: Iterable[date] = (),
) -> list[date]:
    if end < start:
        raise ValueError("end date must be on or after start date")

    holiday_set = set(holidays)
    current = start if include_start else start + timedelta(days=1)
    days: list[date] = []
    while current <= end:
        if (current < end or include_end) and is_business_day(current, holiday_set):
            days.append(current)
        current += timedelta(days=1)
    return days

