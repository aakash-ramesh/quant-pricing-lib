from datetime import date, timedelta
from typing import List, Tuple

def add_months(d: date, n: int) -> date:
    # Simple month add (no business-day adjustment)
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    # clamp day
    day = min(d.day, [31,29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return date(y, m, day)

def generate_schedule(start: date, end: date, freq_months: int) -> List[date]:
    dates = [start]
    d = start
    while True:
        d = add_months(d, freq_months)
        if d >= end:
            dates.append(end)
            break
        dates.append(d)
    return dates
