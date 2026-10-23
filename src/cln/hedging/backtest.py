"""Hedge backtest of the dealer's position over a horizon (default one year).

Dealer position
---------------
The dealer issued the CLN at its fair spread s* and holds risk-free collateral:
economically long first-to-default protection. Marked clean, its value is
V_t = 1 - PV_note(t); it pays carry s* per year while no name has defaulted; at
the first default it pays the investor R_i, sells the collateral for 1 and
gains 1 - R_i - V_{t-}. The CS01 hedge sells 5y-to-original-maturity
protection on each name, notionals chosen so that the book is CS01-neutral
per name; hedges are rebalanced monthly (unwound at mark-to-market and
re-struck at the current par spread), they earn their coupon as carry, and
the defaulted name's hedge pays 1 - R_i per unit notional at default. After
the first default every hedge is unwound.

Market dynamics (synthetic)
---------------------------
Hazard curves move by a common lognormal factor per name,
lambda_i(t, u) = Z_i(t) lambda^0_i(u), Z_i(t) = exp(s W_i(t) - s^2 t / 2), with
correlated Brownian motions (correlation ``spread_corr``). Default of name i
occurs when the realised cumulative hazard int_0^t Z_i(v) lambda^0_i(v) dv first
exceeds E_i = -log U_i, with U drawn from the Gaussian copula with the pricing
correlation. Spreads therefore widen before defaults, and defaults are
correlated. Survivors' spreads do not jump at a default (no contagion), and
the pricing model is re-marked to the current curves at every date. Cash is
not discounted over the horizon.

P&L is split into spread (mark-to-market changes without default), carry
(coupons), and default (cash at default minus the pre-default mark).
For speed the CS01 used for hedging bumps hazards by 1bp / (1 - R) (about
1bp of spread), and pricing uses a coarser time grid (15 steps per quarter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from cln.market import Market
from cln.pricing.cds import cds_legs
from cln.pricing.curves import HazardCurve
from cln.pricing.note import note_value

STEPS = 15
NODES = 32


@dataclass(frozen=True)
class BacktestConfig:
    T: float = 5.0
    rho: float = 0.3
    horizon: float = 1.0
    rebalance_per_year: int = 12
    substeps: int = 8  # default monitoring steps per rebalance period
    spread_vol: float = 0.5
    spread_corr: float = 0.5
    hedge: bool = True


@dataclass(frozen=True)
class PathResult:
    spread: float
    carry: float
    default: float
    defaulted: int  # index of first defaulted name, or -1

    @property
    def total(self) -> float:
        return self.spread + self.carry + self.default


def _curves(mk: Market, Z: np.ndarray) -> tuple[HazardCurve, ...]:
    return tuple(HazardCurve(c.hz.scaled(float(z))) for c, z in zip(mk.curves, Z, strict=True))


def _dealer_value(curves, mk: Market, cfg: BacktestConfig, s: float, t: float) -> float:
    nv = note_value(curves, mk.dc, cfg.T, mk.recoveries, cfg.rho, t0=t, n_nodes=NODES, steps=STEPS)
    return 1.0 - nv.pv(s)


def _cds_seller_value(
    c: HazardCurve, mk: Market, i: int, cfg: BacktestConfig, strike: float, t: float
) -> float:
    lg = cds_legs(c, mk.dc, cfg.T, mk.recoveries[i], t0=t, steps=STEPS)
    return strike * lg.rpv01 - lg.protection


def _hedge_ratios(
    curves, mk: Market, cfg: BacktestConfig, s: float, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """Notionals of protection to sell (CS01-neutral) and current par spreads."""
    base = _dealer_value(curves, mk, cfg, s, t)
    h, par = np.zeros(len(curves)), np.zeros(len(curves))
    for i, c in enumerate(curves):
        d = 1e-4 / (1 - mk.recoveries[i])
        bumped = list(curves)
        bumped[i] = HazardCurve(c.hz.shifted(d))
        dv = _dealer_value(tuple(bumped), mk, cfg, s, t) - base
        lg0 = cds_legs(c, mk.dc, cfg.T, mk.recoveries[i], t0=t, steps=STEPS)
        lg1 = cds_legs(bumped[i], mk.dc, cfg.T, mk.recoveries[i], t0=t, steps=STEPS)
        par[i] = lg0.par_spread()
        dcds = (par[i] * lg1.rpv01 - lg1.protection) - (par[i] * lg0.rpv01 - lg0.protection)
        h[i] = -dv / dcds  # seller's CS01 dcds < 0, dealer's dv > 0
    return h, par


def run_path(mk: Market, cfg: BacktestConfig, s: float, rng: np.random.Generator) -> PathResult:
    n = len(mk.curves)
    nreb = round(cfg.horizon * cfg.rebalance_per_year)
    dt = cfg.horizon / nreb
    h_dt = dt / cfg.substeps
    # copula-correlated default thresholds
    X = np.sqrt(cfg.rho) * rng.standard_normal() + np.sqrt(1 - cfg.rho) * rng.standard_normal(n)
    E = -np.log(norm.cdf(X))
    L = np.linalg.cholesky(cfg.spread_corr * np.ones((n, n)) + (1 - cfg.spread_corr) * np.eye(n))
    logZ = np.zeros(n)
    cum = np.zeros(n)
    t = 0.0
    curves = mk.curves
    V = _dealer_value(curves, mk, cfg, s, 0.0)
    if cfg.hedge:
        h, strikes = _hedge_ratios(curves, mk, cfg, s, 0.0)
    else:
        h, strikes = np.zeros(n), np.zeros(n)
    H = np.zeros(n)  # hedge marks (zero at strike)
    pnl_spread = pnl_carry = pnl_default = 0.0
    for _ in range(nreb):
        default_name = -1
        for _ in range(cfg.substeps):
            z = np.exp(logZ)
            lam = np.array([c.hz.rate(t)[0] for c in mk.curves]) * z
            cum += lam * h_dt
            pnl_carry += (-s + float(np.sum(h * strikes))) * h_dt
            logZ += (
                cfg.spread_vol * np.sqrt(h_dt) * (L @ rng.standard_normal(n))
                - 0.5 * cfg.spread_vol**2 * h_dt
            )
            t += h_dt
            hit = np.where(cum >= E)[0]
            if hit.size:
                default_name = (
                    int(hit[np.argmin(E[hit] - cum[hit])]) if hit.size > 1 else int(hit[0])
                )
                break
        curves = _curves(mk, np.exp(logZ))
        if default_name >= 0:
            R = mk.recoveries[default_name]
            # survivors' hedges are unwound at their current marks (a spread P&L)
            for j in range(n):
                if j != default_name and h[j] != 0:
                    Hj = h[j] * _cds_seller_value(curves[j], mk, j, cfg, strikes[j], t)
                    pnl_spread += Hj - H[j]
            pnl_default += (1 - R - V) + (-h[default_name] * (1 - R) - H[default_name])
            return PathResult(pnl_spread, pnl_carry, pnl_default, default_name)
        V_new = _dealer_value(curves, mk, cfg, s, t)
        H_new = np.array(
            [h[j] * _cds_seller_value(curves[j], mk, j, cfg, strikes[j], t) for j in range(n)]
        )
        pnl_spread += (V_new - V) + float(np.sum(H_new - H))
        V, H = V_new, H_new
        if cfg.hedge and t < cfg.horizon - 1e-9:
            h, strikes = _hedge_ratios(curves, mk, cfg, s, t)  # unwind at mark, re-strike at par
            H = np.zeros(n)
    return PathResult(pnl_spread, pnl_carry, pnl_default, -1)
