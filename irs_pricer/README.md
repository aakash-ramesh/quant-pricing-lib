# IRS Pricer (Python)

A minimal, self-contained interest rate swap pricer with:
- Log-linear zero curve (continuously compounded zeros)
- Fixed vs. floating legs (vanilla IRS)
- Support for ACT/360, ACT/365F, 30/360
- PV, par rate, DV01
- Simple schedule generation (no business day adj. for brevity)
- Unit tests and demo

## Install & Run

```bash
python -m pip install -r requirements.txt  # (empty requirements; stdlib only)
python demo.py
pytest -q
```

## Files

- `irs_pricer/daycount.py` – day count conventions
- `irs_pricer/schedule.py` – simple monthly schedule generator
- `irs_pricer/curve.py` – discount curve with log-linear zero interpolation
- `irs_pricer/swap.py` – swap data structures and builder
- `irs_pricer/pricer.py` – PV, par rate, DV01
- `demo.py` – builds a synthetic curve, prices a few swaps and prints results
- `tests/test_swap.py` – basic unit tests (par swap close to zero PV, payer/receiver sign, DV01) 

## Notes
- Floating leg uses forward rates implied by discount factors and adds the standard `N*(DF(start)-DF(end))` term.
- For real production use, add business-day calendars, accrual cutoffs, stubs, holidays, compounding, OIS discounting, and index curves.
