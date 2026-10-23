"""Risk measures of the CLN, by bump and reprice.

Sign conventions
----------------
* ``investor`` numbers are changes in the value of the note to the investor
  (long the note = short FtD protection): widening spreads lose money.
* The dealer/issuer is the mirror image: it sold the note, holds risk-free
  collateral, and is economically *long* FtD protection. Its hedge is to
  *sell* single-name protection.

CS01: change in value for a +1bp parallel shift of one name's par spread
curve (re-bootstrapped), or of a single tenor (bucketed). Correl01: +1
percentage point of copula correlation. Rec01: recovery assumption -1
percentage point, spreads held fixed (curves re-bootstrapped). JTD: the
instantaneous P&L if a name defaults now.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cln.market import TENORS, Market, build_market
from cln.pricing.cds import bootstrap, cds_legs
from cln.pricing.curves import HazardCurve
from cln.pricing.note import note_value

BP = 1e-4


@dataclass(frozen=True)
class NoteSpec:
    T: float = 5.0
    rho: float = 0.3
    spread: float | None = None  # contractual spread; None = fair spread at inception


def _rebuild(mk: Market, spreads: np.ndarray) -> Market:
    curves = tuple(
        bootstrap(TENORS, row, mk.dc, r) for row, r in zip(spreads, mk.recoveries, strict=True)
    )
    return replace(mk, curves=curves, spreads=spreads)


def note_pv(mk: Market, spec: NoteSpec, s: float, rho: float | None = None) -> float:
    nv = note_value(mk.curves, mk.dc, spec.T, mk.recoveries, spec.rho if rho is None else rho)
    return nv.pv(s)


def fair_spread(mk: Market, spec: NoteSpec) -> float:
    return note_value(mk.curves, mk.dc, spec.T, mk.recoveries, spec.rho).fair_spread()


@dataclass(frozen=True)
class RiskReport:
    fair_spread: float
    cs01: np.ndarray  # investor, per name, per 1bp parallel
    cs01_bucketed: np.ndarray  # investor, (names, tenors)
    correl01: float  # investor, per +1% correlation
    rec01: float  # investor, per -1% recovery (spreads fixed)
    jtd: np.ndarray  # investor, per name
    cds_cs01: np.ndarray  # per unit notional of 5y protection *bought*, per 1bp
    cds_rpv01: np.ndarray
    hedge_notional: np.ndarray  # dealer sells this much 5y protection per name (CS01-neutral)


def risk_report(mk: Market, spec: NoteSpec, bucketed: bool = True) -> RiskReport:
    s = fair_spread(mk, spec) if spec.spread is None else spec.spread
    base = note_pv(mk, spec, s)
    n, k = mk.spreads.shape
    cs01 = np.zeros(n)
    buck = np.zeros((n, k))
    cds_cs01 = np.zeros(n)
    rpv = np.zeros(n)
    for i in range(n):
        sp = mk.spreads.copy()
        sp[i] += BP
        bumped = _rebuild(mk, sp)
        cs01[i] = note_pv(bumped, spec, s) - base
        c5 = mk.spreads[i, list(TENORS).index(spec.T)] if spec.T in TENORS else None
        lg0 = cds_legs(mk.curves[i], mk.dc, spec.T, mk.recoveries[i])
        lg1 = cds_legs(bumped.curves[i], mk.dc, spec.T, mk.recoveries[i])
        strike = lg0.par_spread() if c5 is None else c5
        cds_cs01[i] = (lg1.protection - strike * lg1.rpv01) - (lg0.protection - strike * lg0.rpv01)
        rpv[i] = lg0.rpv01
        if bucketed:
            for j in range(k):
                sp = mk.spreads.copy()
                sp[i, j] += BP
                buck[i, j] = note_pv(_rebuild(mk, sp), spec, s) - base
    correl01 = note_pv(mk, spec, s, spec.rho + 0.01) - base
    lower_r = build_market(mk.spreads * 1e4, recovery=mk.recoveries[0] - 0.01)
    rec01 = note_pv(replace(lower_r, dc=mk.dc), spec, s) - base
    jtd = np.array(mk.recoveries) - base
    # dealer CS01 = -cs01 (long FtD protection); selling h_i of protection has CS01 -h_i*cds_cs01
    hedge = -cs01 / cds_cs01
    return RiskReport(s, cs01, buck, correl01, rec01, jtd, cds_cs01, rpv, hedge)


def hazard_shift_cs01(curves: tuple[HazardCurve, ...], recovery: float) -> float:
    """Hazard shift equivalent to ~1bp of spread via the credit triangle s ~ (1 - R) lambda."""
    del curves
    return BP / (1 - recovery)
