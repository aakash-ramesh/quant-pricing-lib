from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable, Literal, Union

BusinessDayAdjustment = Literal[
    "none",
    "following",
    "modified_following",
    "preceding",
    "modified_preceding",
]
NamedCalendar = Literal["weekend", "us_federal", "target2", "nyse"]
HolidayRule = Callable[[int], set[date]]


@dataclass(frozen=True)
class MarketCalendar:
    """Weekend and holiday calendar used for schedule generation."""

    name: str = "custom"
    holidays: frozenset[date] = frozenset()
    holiday_rule: HolidayRule | None = None
    weekend: tuple[int, ...] = (5, 6)

    def holidays_for_year(self, year: int) -> set[date]:
        rule_holidays = self.holiday_rule(year) if self.holiday_rule else set()
        return {day for day in self.holidays if day.year == year} | rule_holidays

    def is_business_day(self, day: date) -> bool:
        return day.weekday() not in self.weekend and day not in self.holidays_for_year(day.year)

    def adjust(self, day: date, adjustment: BusinessDayAdjustment = "none") -> date:
        return adjust_business_day(day, adjustment, calendar=self)

    def combine(self, *others: "MarketCalendar") -> "MarketCalendar":
        calendars = (self, *others)

        def combined_rule(year: int) -> set[date]:
            holidays: set[date] = set()
            for calendar in calendars:
                holidays.update(calendar.holidays_for_year(year))
            return holidays

        weekends = tuple(sorted({weekday for calendar in calendars for weekday in calendar.weekend}))
        name = "+".join(calendar.name for calendar in calendars)
        return MarketCalendar(name=name, holiday_rule=combined_rule, weekend=weekends)


CalendarInput = Union[MarketCalendar, NamedCalendar, str, None]


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    days_until_weekday = (weekday - current.weekday()) % 7
    return current + timedelta(days=days_until_weekday + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month, monthrange(year, month)[1])
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _us_federal_holidays(year: int) -> set[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _observed_fixed_holiday(year, 11, 11),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }
    # Observed New Year's Day can fall on the prior calendar year.
    next_new_year = date(year + 1, 1, 1)
    if next_new_year.weekday() == 5:
        holidays.add(date(year, 12, 31))
    return holidays


def _target2_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def _nyse_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }


_NAMED_CALENDARS: dict[str, MarketCalendar] = {
    "weekend": MarketCalendar(name="weekend"),
    "us_federal": MarketCalendar(name="us_federal", holiday_rule=_us_federal_holidays),
    "target2": MarketCalendar(name="target2", holiday_rule=_target2_holidays),
    "nyse": MarketCalendar(name="nyse", holiday_rule=_nyse_holidays),
}


def get_calendar(calendar: CalendarInput = None, holidays: Iterable[date] = ()) -> MarketCalendar:
    if isinstance(calendar, MarketCalendar):
        base = calendar
    elif calendar is None:
        base = _NAMED_CALENDARS["weekend"]
    else:
        key = str(calendar).lower()
        if key not in _NAMED_CALENDARS:
            supported = ", ".join(sorted(_NAMED_CALENDARS))
            raise ValueError(f"unsupported calendar {calendar!r}; supported: {supported}")
        base = _NAMED_CALENDARS[key]

    extra_holidays = frozenset(holidays)
    if not extra_holidays:
        return base

    def rule(year: int) -> set[date]:
        return base.holidays_for_year(year) | {day for day in extra_holidays if day.year == year}

    return MarketCalendar(name=base.name, holiday_rule=rule, weekend=base.weekend)


def is_business_day(
    day: date,
    holidays: Iterable[date] = (),
    *,
    calendar: CalendarInput = None,
) -> bool:
    return get_calendar(calendar, holidays).is_business_day(day)


def is_end_of_month(day: date) -> bool:
    return day.day == monthrange(day.year, day.month)[1]


def end_of_month(day: date) -> date:
    return date(day.year, day.month, monthrange(day.year, day.month)[1])


def add_months(day: date, months: int, *, end_of_month_rule: bool = False) -> date:
    if months <= 0:
        raise ValueError("months must be positive")
    year = day.year + (day.month - 1 + months) // 12
    month = (day.month - 1 + months) % 12 + 1
    if end_of_month_rule and is_end_of_month(day):
        return date(year, month, monthrange(year, month)[1])
    return date(year, month, min(day.day, monthrange(year, month)[1]))


def adjust_business_day(
    day: date,
    adjustment: BusinessDayAdjustment = "none",
    holidays: Iterable[date] = (),
    *,
    calendar: CalendarInput = None,
) -> date:
    market_calendar = get_calendar(calendar, holidays)
    if adjustment == "none" or market_calendar.is_business_day(day):
        return day
    if adjustment not in {"following", "modified_following", "preceding", "modified_preceding"}:
        raise ValueError(f"unsupported business-day adjustment {adjustment!r}")

    adjusted = day
    if adjustment in {"following", "modified_following"}:
        while not market_calendar.is_business_day(adjusted):
            adjusted += timedelta(days=1)
        if adjustment == "modified_following" and adjusted.month != day.month:
            adjusted = day
            while not market_calendar.is_business_day(adjusted):
                adjusted -= timedelta(days=1)
        return adjusted

    while not market_calendar.is_business_day(adjusted):
        adjusted -= timedelta(days=1)
    if adjustment == "modified_preceding" and adjusted.month != day.month:
        adjusted = day
        while not market_calendar.is_business_day(adjusted):
            adjusted += timedelta(days=1)
    return adjusted


def generate_schedule(
    start: date,
    end: date,
    frequency_months: int,
    *,
    adjustment: BusinessDayAdjustment = "none",
    holidays: Iterable[date] = (),
    calendar: CalendarInput = None,
    end_of_month_rule: bool = False,
) -> list[date]:
    """Generate an unadjusted or business-day-adjusted coupon schedule."""

    if end <= start:
        raise ValueError("end date must be after start date")

    dates = [start]
    current = start
    while True:
        current = add_months(current, frequency_months, end_of_month_rule=end_of_month_rule)
        if current >= end:
            dates.append(end)
            break
        dates.append(current)

    return [adjust_business_day(d, adjustment, holidays, calendar=calendar) for d in dates]


def business_days(
    start: date,
    end: date,
    *,
    include_start: bool = False,
    include_end: bool = True,
    holidays: Iterable[date] = (),
    calendar: CalendarInput = None,
) -> list[date]:
    if end < start:
        raise ValueError("end date must be on or after start date")

    market_calendar = get_calendar(calendar, holidays)
    current = start if include_start else start + timedelta(days=1)
    days: list[date] = []
    while current <= end:
        if (current < end or include_end) and market_calendar.is_business_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days
