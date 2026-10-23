"""Discount and hazard-rate curves, both piecewise-constant in their instantaneous rate.

Conventions
-----------
* Time t in years from today (ACT/365F in spirit; no calendar is modelled).
* Discount curve: P(0, t) = exp(-int_0^t f(s) ds) with piecewise-constant
  forward rate f ("flat-forward" interpolation of zero rates).
* Hazard curve: survival Q(t) = exp(-int_0^t lambda(s) ds), lambda piecewise
  constant between knots and flat beyond the last knot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PiecewiseFlat:
    """Piecewise-constant rate: rates[k] on (knots[k-1], knots[k]]; flat after the last knot."""

    knots: np.ndarray  # increasing, > 0
    rates: np.ndarray

    def integral(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        edges = np.concatenate([[0.0], self.knots])
        out = np.zeros_like(t)
        for k, r in enumerate(self.rates):
            lo = edges[k]
            hi = np.inf if k == len(self.rates) - 1 else edges[k + 1]
            out += r * np.clip(np.minimum(t, hi) - lo, 0.0, None)
        return out

    def rate(self, t: np.ndarray | float) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.knots, t, side="left")
        return np.asarray(self.rates[np.minimum(idx, len(self.rates) - 1)])

    def scaled(self, factor: float) -> PiecewiseFlat:
        return PiecewiseFlat(self.knots, self.rates * factor)

    def shifted(self, shift: float) -> PiecewiseFlat:
        return PiecewiseFlat(self.knots, self.rates + shift)


@dataclass(frozen=True)
class DiscountCurve:
    fwd: PiecewiseFlat

    @classmethod
    def from_zero_rates(cls, tenors: np.ndarray, zeros: np.ndarray) -> DiscountCurve:
        """Continuously compounded zero rates at tenors -> flat forwards between them."""
        tenors, zeros = np.asarray(tenors, float), np.asarray(zeros, float)
        integ = zeros * tenors
        fwd = np.diff(np.concatenate([[0.0], integ])) / np.diff(np.concatenate([[0.0], tenors]))
        return cls(PiecewiseFlat(tenors, fwd))

    @classmethod
    def flat(cls, r: float) -> DiscountCurve:
        return cls(PiecewiseFlat(np.array([1.0]), np.array([r])))

    def df(self, t: np.ndarray | float) -> np.ndarray:
        return np.exp(-self.fwd.integral(t))


@dataclass(frozen=True)
class HazardCurve:
    hz: PiecewiseFlat

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        return np.exp(-self.hz.integral(t))

    def cumulative(self, t: np.ndarray | float) -> np.ndarray:
        return self.hz.integral(t)

    def inverse_cumulative(self, L: np.ndarray) -> np.ndarray:
        """t with Lambda(t) = L (piecewise-linear inversion); inf where L is infinite."""
        L = np.asarray(L, dtype=float)
        edges = np.concatenate([[0.0], self.hz.knots])
        cum = self.hz.integral(edges)
        out = np.full(L.shape, np.inf)
        for k, r in enumerate(self.hz.rates):
            lo_t = edges[k]
            lo_L = cum[k]
            hi_L = cum[k + 1] if k + 1 < len(edges) and k < len(self.hz.rates) - 1 else np.inf
            m = (L >= lo_L) & (L < hi_L)
            out[m] = lo_t + (L[m] - lo_L) / r
        return out
