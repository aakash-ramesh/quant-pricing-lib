from datetime import date

def is_leap(y: int) -> bool:
    return (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)

def year_fraction_ACT_360(d1: date, d2: date) -> float:
    return (d2 - d1).days / 360.0

def year_fraction_ACT_365F(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.0

def year_fraction_30_360_US(d1: date, d2: date) -> float:
    # US 30/360
    d1d = min(d1.day, 30)
    d2d = d2.day if (d1.day < 30) else min(d2.day, 30)
    return ((d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2d - d1d)) / 360.0

DAYCOUNT_MAP = {
    "ACT/360": year_fraction_ACT_360,
    "ACT/365F": year_fraction_ACT_365F,
    "30/360": year_fraction_30_360_US,
}
