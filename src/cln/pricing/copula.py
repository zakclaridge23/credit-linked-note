"""First-to-default pricing in the one-factor Gaussian copula.

Default times tau_i = Q_i^{-1}(Phi(X_i)), X_i = sqrt(rho) M + sqrt(1 - rho) Z_i, with
M, Z_i i.i.d. N(0, 1). Conditional on M the defaults are independent with

    p_i(t | M) = P(tau_i <= t | M) = Phi( (Phi^{-1}(1 - Q_i(t)) - sqrt(rho) M) / sqrt(1 - rho) ),

so the first-to-default survival is

    S_1(t) = E[ prod_i (1 - p_i(t | M)) ],

computed by Gauss–Hermite quadrature in M (Hull & White 2004 style semi-analytic
pricing; a fine trapezoid rule replaces it for rho > 0.9). For name-specific
recoveries the protection leg needs which name defaults first: conditional on M,
P(tau_1 in dt, first is i) =
dp_i(t|M) prod_{j != i} (1 - p_j(t|M)), discretised on the fine grid and then
rescaled so that the shares add up to the exact -dS_1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from cln.pricing.cds import Legs, legs_from_survival, premium_dates, time_grid
from cln.pricing.curves import DiscountCurve, HazardCurve

RHO_MAX = 1 - 1e-10


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for E[f(M)], M ~ N(0, 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / np.sqrt(2 * np.pi)


def conditional_default(Q: np.ndarray, rho: float, M: np.ndarray) -> np.ndarray:
    """p_i(t|M): Q has shape (names, times); returns (nodes, names, times)."""
    rho = min(max(rho, 0.0), RHO_MAX)
    c = norm.ppf(np.clip(1 - Q, 1e-300, 1 - 1e-16))
    return np.asarray(norm.cdf((c[None] - np.sqrt(rho) * M[:, None, None]) / np.sqrt(1 - rho)))


def factor_quadrature(rho: float, n_nodes: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature for E[f(M)]. Gauss–Hermite is very accurate for moderate rho, but as
    rho -> 1 the conditional default probabilities become step functions of M and a
    fixed set of nodes cannot resolve them; above rho = 0.9 we switch to a fine
    trapezoid rule on [-8, 8]."""
    if rho <= 0.9:
        return gauss_hermite(n_nodes)
    M = np.linspace(-8.0, 8.0, 401)
    w = norm.pdf(M) * (M[1] - M[0])
    w[[0, -1]] *= 0.5
    return M, w / w.sum()


def ftd_distribution(Q: np.ndarray, rho: float, n_nodes: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """S_1 on the grid and the first-default shares dF_{1,i} on each grid cell."""
    M, w = factor_quadrature(rho, n_nodes)
    p = conditional_default(Q, rho, M)  # (m, n, k)
    surv = 1 - p
    S1 = np.einsum("m,mk->k", w, np.prod(surv, axis=1))
    dp = p[:, :, 1:] - p[:, :, :-1]
    sm = 0.5 * (surv[:, :, 1:] + surv[:, :, :-1])
    prod_all = np.prod(sm, axis=1, keepdims=True)
    others = prod_all / np.maximum(sm, 1e-300)
    share = np.einsum("m,mnk->nk", w, dp * others)
    tot = share.sum(axis=0)
    dS1 = S1[:-1] - S1[1:]
    share = share * np.where(tot > 0, dS1 / np.where(tot > 0, tot, 1.0), 0.0)
    return S1, share


def ftd_legs(
    curves: Sequence[HazardCurve],
    dc: DiscountCurve,
    T: float,
    recoveries: Sequence[float],
    rho: float,
    t0: float = 0.0,
    n_nodes: int = 64,
) -> tuple[Legs, dict[str, np.ndarray]]:
    """Premium (RPV01) and protection legs of an FtD swap on [t0, T], conditional on no
    default before t0 (marginals Q_i(t)/Q_i(t0))."""
    g = time_grid(t0, T)
    Q = np.array([c.survival(g) / c.survival(t0) for c in curves])
    S1, share = ftd_distribution(Q, rho, n_nodes)
    P = dc.df(g) / dc.df(t0)
    base = legs_from_survival(g, S1, P, t0, T, 0.0)  # protection here = int P dF_1
    Pm = 0.5 * (P[:-1] + P[1:])
    lgd = 1 - np.asarray(recoveries, dtype=float)
    prot = float(np.sum(Pm * (lgd[:, None] * share).sum(axis=0)))
    return Legs(base.rpv01, prot), {"grid": g, "S1": S1, "share": share, "P": P}


def ftd_spread(
    curves: Sequence[HazardCurve],
    dc: DiscountCurve,
    T: float,
    recoveries: Sequence[float],
    rho: float,
    n_nodes: int = 64,
) -> float:
    return ftd_legs(curves, dc, T, recoveries, rho, n_nodes=n_nodes)[0].par_spread()


__all__ = ["ftd_distribution", "ftd_legs", "ftd_spread", "gauss_hermite", "premium_dates"]
