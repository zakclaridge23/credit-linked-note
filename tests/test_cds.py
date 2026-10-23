import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cln.market import TENORS, build_market
from cln.pricing.cds import bootstrap, cds_legs, par_spread, premium_dates, upfront
from cln.pricing.curves import DiscountCurve, HazardCurve, PiecewiseFlat

MK = build_market()


def test_bootstrap_reprices_every_quote_and_hazards_positive():
    for curve, row in zip(MK.curves, MK.spreads, strict=True):
        for T, s in zip(TENORS, row, strict=True):
            assert abs(par_spread(curve, MK.dc, T, 0.4) - s) < 0.1e-4 * 1e-3  # << 0.1bp
        assert np.all(curve.hz.rates > 0)


@settings(max_examples=25, deadline=None)
@given(st.floats(10, 500), st.lists(st.floats(0.95, 1.5), min_size=4, max_size=4))
def test_bootstrap_random_curves(s1, steps):
    s = s1 * np.cumprod([1.0, *steps]) * 1e-4
    hc = bootstrap(TENORS, s, MK.dc, 0.4)
    reprice = [par_spread(hc, MK.dc, T, 0.4) for T in TENORS]
    np.testing.assert_allclose(reprice, s, atol=1e-9)


def test_inconsistent_quotes_are_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        bootstrap(TENORS, np.array([10, 10, 10, 10, 800.0]) * 1e-4, MK.dc, 0.4)


def test_flat_hazard_closed_forms():
    lam, r, R, T = 0.02, 0.03, 0.4, 5.0
    hc = HazardCurve(PiecewiseFlat(np.array([1.0]), np.array([lam])))
    dc = DiscountCurve.flat(r)
    lg = cds_legs(hc, dc, T, R)
    exact_prot = (1 - R) * lam / (lam + r) * (1 - np.exp(-(lam + r) * T))
    assert lg.protection == pytest.approx(exact_prot, rel=1e-9)
    # credit triangle: s ~ (1 - R) lambda, within a fraction of a bp
    assert abs(lg.par_spread() - (1 - R) * lam) < 1e-4


def test_upfront_sign_and_par_consistency():
    hc, s5 = MK.curves[3], MK.spreads[3, 2]  # 125bp name
    assert upfront(hc, MK.dc, 5.0, 0.4, s5) == pytest.approx(0.0, abs=1e-12)
    assert upfront(hc, MK.dc, 5.0, 0.4, 0.01) > 0  # coupon 100bp < 125bp: buyer pays upfront
    assert upfront(hc, MK.dc, 5.0, 0.4, 0.05) < 0  # coupon 500bp: buyer receives


def test_schedule_and_inverse_cumulative():
    np.testing.assert_allclose(premium_dates(0.0, 1.0), [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(premium_dates(0.1, 1.0), [0.25, 0.5, 0.75, 1.0])
    t = np.array([0.3, 2.0, 4.9, 8.0, 15.0])
    hc = MK.curves[0]
    np.testing.assert_allclose(hc.inverse_cumulative(hc.cumulative(t)), t, rtol=1e-12)
