"""Monte Carlo pricing of the FtD swap by simulating default times.

tau_i = Lambda_i^{-1}(-log U_i), U_i = F(X_i), with X_i = sqrt(rho) M + sqrt(1-rho) Z_i
(Gaussian copula, F = Phi) or the Student-t copula X_i = (sqrt(rho) M + sqrt(1-rho) Z_i)
/ sqrt(W/nu), W ~ chi^2_nu, F = t_nu CDF (tail dependence: joint extreme defaults are
more likely for the same rho). Antithetic variates flip (M, Z).

Per path the FtD legs are computed exactly for the realised tau_1:
premium = sum of coupons paid before tau_1 plus accrued premium at tau_1;
protection = (1 - R_first) P(tau_1) 1{tau_1 <= T}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from cln.pricing.cds import premium_dates
from cln.pricing.curves import DiscountCurve, HazardCurve


@dataclass(frozen=True)
class MCResult:
    spread: float
    spread_se: float
    rpv01: float
    protection: float
    n_paths: int


def default_times(
    curves: Sequence[HazardCurve],
    rho: float,
    n: int,
    rng: np.random.Generator,
    copula: str = "gaussian",
    nu: float = 4.0,
    antithetic: bool = True,
) -> np.ndarray:
    k = len(curves)
    half = n // 2 if antithetic else n
    M = rng.standard_normal((half, 1))
    Z = rng.standard_normal((half, k))
    if antithetic:
        M, Z = np.vstack([M, -M]), np.vstack([Z, -Z])
    X = np.sqrt(rho) * M + np.sqrt(1 - rho) * Z
    if copula == "gaussian":
        U = norm.cdf(X)
    elif copula == "student":
        W = rng.chisquare(nu, (half, 1))
        if antithetic:
            W = np.vstack([W, W])
        U = student_t.cdf(X / np.sqrt(W / nu), nu)
    else:
        raise ValueError(copula)
    L = -np.log(np.clip(U, 1e-300, 1.0))
    return np.column_stack([c.inverse_cumulative(L[:, i]) for i, c in enumerate(curves)])


def ftd_mc(
    curves: Sequence[HazardCurve],
    dc: DiscountCurve,
    T: float,
    recoveries: Sequence[float],
    rho: float,
    n: int,
    rng: np.random.Generator,
    copula: str = "gaussian",
    nu: float = 4.0,
) -> MCResult:
    tau = default_times(curves, rho, n, rng, copula, nu)
    first = np.argmin(tau, axis=1)
    t1 = tau[np.arange(len(tau)), first]
    R = np.asarray(recoveries)[first]
    pay = premium_dates(0.0, T)
    starts = np.concatenate([[0.0], pay[:-1]])
    dfp = dc.df(pay)
    alive = t1[:, None] > pay[None, :]
    prem = (alive * (pay - starts) * dfp).sum(axis=1)
    hit = t1 <= T
    k = np.searchsorted(pay, t1[hit])  # period of default
    accr = t1[hit] - starts[np.minimum(k, len(pay) - 1)]
    prem[hit] += accr * dc.df(t1[hit])
    prot = np.zeros(len(t1))
    prot[hit] = (1 - R[hit]) * dc.df(t1[hit])
    A, B = prem.mean(), prot.mean()
    # delta method for the ratio, using antithetic pairs as independent units
    half = len(t1) // 2
    pa, pb = 0.5 * (prem[:half] + prem[half:]), 0.5 * (prot[:half] + prot[half:])
    cov = np.cov(np.vstack([pb, pa]))
    var = (cov[0, 0] - 2 * (B / A) * cov[0, 1] + (B / A) ** 2 * cov[1, 1]) / (A**2 * half)
    return MCResult(B / A, float(np.sqrt(var)), float(A), float(B), len(t1))
