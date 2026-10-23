"""Valuation of the funded first-to-default credit-linked note.

Investor pays N = 1 at issue. Until the earlier of maturity T and the first
credit event tau_1 in the basket, the investor receives quarterly coupons of
floating (OIS) + s. If tau_1 <= T the note terminates and the investor receives
the recovery R_i of the defaulted name at tau_1 (auction settlement, paid
immediately; no accrued coupon); otherwise par at T. Issuer/collateral risk is
ignored (the SPV's collateral is risk-free). With deterministic rates
independent of defaults:

    PV = sum_i [P(t_{i-1}) - P(t_i)] S_1(t_i)          floating coupons
       + s sum_i Delta_i P(t_i) S_1(t_i)                spread coupons
       + int_0^T P(t) sum_i R_i dF_{1,i}(t)             recovery at the first default
       + P(T) S_1(T)                                    principal

(the floating coupon for (t_{i-1}, t_i] is worth P(t_{i-1}) - P(t_i) at a fixed
date, so a default-free floating-rate note is worth exactly par).
The fair spread s* makes PV = 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cln.pricing.cds import STEPS_PER_QUARTER, premium_dates, time_grid
from cln.pricing.copula import ftd_distribution
from cln.pricing.curves import DiscountCurve, HazardCurve


@dataclass(frozen=True)
class NoteValue:
    floating: float
    spread_annuity: float  # value of 1 unit of spread
    recovery: float
    principal: float

    def pv(self, s: float) -> float:
        return self.floating + s * self.spread_annuity + self.recovery + self.principal

    def fair_spread(self) -> float:
        return (1.0 - self.floating - self.recovery - self.principal) / self.spread_annuity


def note_value(
    curves: Sequence[HazardCurve],
    dc: DiscountCurve,
    T: float,
    recoveries: Sequence[float],
    rho: float,
    t0: float = 0.0,
    n_nodes: int = 64,
    steps: int = STEPS_PER_QUARTER,
) -> NoteValue:
    g = time_grid(t0, T, steps)
    Q = np.array([c.survival(g) / c.survival(t0) for c in curves])
    S1, share = ftd_distribution(Q, rho, n_nodes)
    P = dc.df(g) / dc.df(t0)
    pay = premium_dates(t0, T)
    idx = np.searchsorted(g, pay - 1e-12)
    starts = np.concatenate([[max(pay[0] - 0.25, t0)], pay[:-1]])
    P_start = dc.df(starts) / dc.df(t0)
    floating = float(np.sum((P_start - P[idx]) * S1[idx]))
    annuity = float(np.sum((pay - starts) * P[idx] * S1[idx]))
    Pm = 0.5 * (P[:-1] + P[1:])
    rec = float(np.sum(Pm * (np.asarray(recoveries)[:, None] * share).sum(axis=0)))
    return NoteValue(floating, annuity, rec, float(P[-1] * S1[-1]))
