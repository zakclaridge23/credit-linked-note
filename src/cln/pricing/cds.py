"""Single-name CDS: legs on a fine time grid, par spreads, upfronts, bootstrap.

For a contract on [t0, T] with quarterly premium dates t_1 < ... < t_n = T and
recovery R, discount factors P and (conditional) survival S:

    premium leg (per unit spread, "RPV01")
        = sum_i Delta_i P(t_i) S(t_i)                                   (coupons)
        + sum_i int_{t_{i-1}}^{t_i} (s - t_{i-1}) P(s) (-dS(s))          (accrual on default)
    protection leg
        = (1 - R) int_{t0}^T P(s) (-dS(s)).

Integrals are computed on a fine grid (90 steps per quarter) as
sum_k P(mid_k) (S_k - S_{k+1}), which is the approach of the ISDA standard model
(piecewise-flat hazard and forward rates, numerical integration between knots).
Accrual fractions Delta_i = t_i - t_{i-1} (a 0.25 year-fraction per quarter;
real contracts use ACT/360 and IMM dates, see docs/DESIGN_DECISIONS.md).
The protection buyer pays the premium; a positive value is good for the buyer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from cln.pricing.curves import DiscountCurve, HazardCurve, PiecewiseFlat

STEPS_PER_QUARTER = 90
FREQ = 4


def time_grid(t0: float, T: float, steps: int = STEPS_PER_QUARTER) -> np.ndarray:
    """Fine grid on [t0, T] containing every quarterly date after t0 and T itself."""
    pay = premium_dates(t0, T)
    pts = [np.array([t0])]
    prev = t0
    for d in pay:
        n = max(round((d - prev) * FREQ * steps), 1)
        pts.append(np.linspace(prev, d, n + 1)[1:])
        prev = d
    return np.concatenate(pts)


def premium_dates(t0: float, T: float) -> np.ndarray:
    """Quarterly dates counted back from maturity, strictly after t0 (short first period)."""
    n = int(np.ceil((T - t0) * FREQ - 1e-9))
    d = T - np.arange(n)[::-1] / FREQ
    return d[d > t0 + 1e-12]


@dataclass(frozen=True)
class Legs:
    rpv01: float
    protection: float  # (1 - R) times the default leg

    def par_spread(self) -> float:
        return self.protection / self.rpv01


def legs_from_survival(
    grid: np.ndarray, S: np.ndarray, P: np.ndarray, t0: float, T: float, recovery: float
) -> Legs:
    """Legs given survival S and discount P on ``grid`` (both relative to t0)."""
    dS = S[:-1] - S[1:]
    Pm = 0.5 * (P[:-1] + P[1:])
    mid = 0.5 * (grid[:-1] + grid[1:])
    prot = (1 - recovery) * float(np.sum(Pm * dS))
    pay = premium_dates(t0, T)
    starts = np.concatenate([[pay[0] - 1 / FREQ], pay[:-1]])
    idx = np.searchsorted(grid, pay - 1e-12)
    coupons = float(np.sum((pay - starts) * P[idx] * S[idx]))
    period = np.searchsorted(pay, mid)  # which premium period each grid cell is in
    accr_start = np.maximum(starts[np.minimum(period, len(pay) - 1)], t0)
    aod = float(np.sum((mid - accr_start) * Pm * dS))
    return Legs(coupons + aod, prot)


def cds_legs(
    hc: HazardCurve,
    dc: DiscountCurve,
    T: float,
    recovery: float,
    t0: float = 0.0,
    steps: int = STEPS_PER_QUARTER,
) -> Legs:
    g = time_grid(t0, T, steps)
    S = hc.survival(g) / hc.survival(t0)
    P = dc.df(g) / dc.df(t0)
    return legs_from_survival(g, S, P, t0, T, recovery)


def par_spread(
    hc: HazardCurve, dc: DiscountCurve, T: float, recovery: float, t0: float = 0.0
) -> float:
    return cds_legs(hc, dc, T, recovery, t0).par_spread()


def upfront(
    hc: HazardCurve, dc: DiscountCurve, T: float, recovery: float, coupon: float, t0: float = 0.0
) -> float:
    """Upfront (fraction of notional) paid by the protection buyer for a standard coupon
    (e.g. 100bp or 500bp): protection - coupon * RPV01."""
    lg = cds_legs(hc, dc, T, recovery, t0)
    return lg.protection - coupon * lg.rpv01


def value_to_buyer(
    hc: HazardCurve,
    dc: DiscountCurve,
    T: float,
    recovery: float,
    contract_spread: float,
    t0: float = 0.0,
) -> float:
    lg = cds_legs(hc, dc, T, recovery, t0)
    return lg.protection - contract_spread * lg.rpv01


def bootstrap(
    tenors: np.ndarray, spreads: np.ndarray, dc: DiscountCurve, recovery: float
) -> HazardCurve:
    """Piecewise-constant hazards reproducing each par spread, solved tenor by tenor."""
    tenors, spreads = np.asarray(tenors, float), np.asarray(spreads, float)
    rates: list[float] = []
    for k, (T, s) in enumerate(zip(tenors, spreads, strict=True)):

        def err(lam: float, k: int = k, T: float = T, s: float = s) -> float:
            hc = HazardCurve(PiecewiseFlat(tenors[: k + 1], np.array([*rates, lam])))
            return par_spread(hc, dc, T, recovery) - s

        lo, hi = 1e-10, 50.0
        if err(lo) > 0 or err(hi) < 0:
            raise ValueError(
                f"{T:g}y spread {s * 1e4:.1f}bp cannot be matched with a non-negative hazard "
                "given the shorter tenors (the quotes are inconsistent)"
            )
        rates.append(brentq(err, lo, hi, xtol=1e-14))
    return HazardCurve(PiecewiseFlat(tenors, np.array(rates)))
