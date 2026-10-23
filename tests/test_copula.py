"""FtD pricing: limits, bounds, monotonicity, and Monte Carlo agreement."""

import numpy as np
import pytest

from cln.market import build_market
from cln.pricing.cds import par_spread, time_grid
from cln.pricing.copula import ftd_distribution, ftd_spread
from cln.pricing.curves import HazardCurve, PiecewiseFlat
from cln.pricing.montecarlo import ftd_mc

MK = build_market()
REC = MK.recoveries


def test_independence_gives_product_of_survivals():
    g = time_grid(0, 5)
    Q = np.array([c.survival(g) for c in MK.curves])
    S1, _ = ftd_distribution(Q, 0.0)
    np.testing.assert_allclose(S1, Q.prod(axis=0), rtol=1e-12)


def test_flat_identical_names_limits():
    hc = HazardCurve(PiecewiseFlat(np.array([1.0]), np.array([0.015])))
    single = par_spread(hc, MK.dc, 5.0, 0.4)
    curves = [hc] * 5
    # rho -> 0: FtD hazard is the sum of hazards
    five = HazardCurve(PiecewiseFlat(np.array([1.0]), np.array([0.075])))
    assert ftd_spread(curves, MK.dc, 5.0, [0.4] * 5, 0.0) == pytest.approx(
        par_spread(five, MK.dc, 5.0, 0.4), rel=1e-6
    )
    # rho -> 1: all default together; FtD -> single name (slowly, like sqrt(1 - rho))
    gaps = [abs(ftd_spread(curves, MK.dc, 5.0, [0.4] * 5, r) - single) for r in (0.99, 0.999, 1.0)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 5e-3 * single


def test_bounds_and_monotone_in_correlation():
    rhos = np.linspace(0, 0.99, 12)
    s = np.array([ftd_spread(MK.curves, MK.dc, 5.0, REC, r) for r in rhos])
    widest = max(par_spread(c, MK.dc, 5.0, 0.4) for c in MK.curves)
    total = sum(par_spread(c, MK.dc, 5.0, 0.4) for c in MK.curves)
    assert np.all(np.diff(s) < 0)
    assert np.all(s <= total + 1e-12) and np.all(s >= widest - 1e-6)


def test_quadrature_converged():
    a = ftd_spread(MK.curves, MK.dc, 5.0, REC, 0.5, n_nodes=32)
    b = ftd_spread(MK.curves, MK.dc, 5.0, REC, 0.5, n_nodes=128)
    assert abs(a - b) < 1e-8


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.95])
def test_monte_carlo_agrees_with_semi_analytic(rho):
    semi = ftd_spread(MK.curves, MK.dc, 5.0, REC, rho)
    mc = ftd_mc(MK.curves, MK.dc, 5.0, REC, rho, 400_000, np.random.default_rng(int(rho * 100)))
    assert abs(mc.spread - semi) < 3.5 * mc.spread_se


def test_student_t_copula_has_lower_ftd_spread_than_gaussian():
    """Tail dependence concentrates defaults: fewer scenarios with exactly one early default."""
    g = ftd_mc(MK.curves, MK.dc, 5.0, REC, 0.3, 200_000, np.random.default_rng(1))
    t = ftd_mc(MK.curves, MK.dc, 5.0, REC, 0.3, 200_000, np.random.default_rng(1), "student")
    assert t.spread < g.spread - 5 * (g.spread_se + t.spread_se)
