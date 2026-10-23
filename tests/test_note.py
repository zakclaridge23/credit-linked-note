import numpy as np
import pytest

from cln.hedging.sensitivities import NoteSpec, risk_report
from cln.market import build_market
from cln.pricing.copula import ftd_spread
from cln.pricing.curves import HazardCurve, PiecewiseFlat
from cln.pricing.note import note_value

MK = build_market()


def test_default_free_note_is_a_par_floater():
    z = [HazardCurve(PiecewiseFlat(np.array([1.0]), np.array([1e-14])))] * 5
    nv = note_value(z, MK.dc, 5.0, [0.4] * 5, 0.3)
    assert nv.pv(0.0) == pytest.approx(1.0, abs=1e-10)


def test_fair_spread_prices_at_par_and_exceeds_ftd_spread():
    nv = note_value(MK.curves, MK.dc, 5.0, MK.recoveries, 0.3)
    s = nv.fair_spread()
    assert nv.pv(s) == pytest.approx(1.0, abs=1e-12)
    ftd = ftd_spread(MK.curves, MK.dc, 5.0, MK.recoveries, 0.3)
    # the investor gets no accrued coupon at default, so needs slightly more than the FtD spread
    assert 0 < s - ftd < 15e-4


@pytest.fixture(scope="module")
def report():
    return risk_report(MK, NoteSpec(), bucketed=True)


def test_risk_signs_and_structure(report):
    assert np.all(report.cs01 < 0)  # investor is short protection
    assert report.correl01 > 0  # higher correlation, lower FtD risk
    np.testing.assert_allclose(report.jtd, 0.4 - 1.0, atol=1e-9)  # issued at par
    np.testing.assert_allclose(report.cs01_bucketed[:, 3:], 0.0, atol=1e-12)  # beyond 5y
    assert np.allclose(report.cs01_bucketed.sum(axis=1), report.cs01, rtol=0.02)
    assert np.all((0 < report.hedge_notional) & (report.hedge_notional < 1))


def test_cs01_neutral_is_not_jtd_neutral_and_correlation_is_unhedgeable(report):
    h = report.hedge_notional
    dealer_jtd = -report.jtd - h * (1 - 0.4)  # long FtD JTD, short h of single-name protection
    assert np.all(dealer_jtd > 0.1)
    # single-name CDS do not depend on the copula correlation at all
    from cln.pricing.cds import cds_legs

    lg = cds_legs(MK.curves[0], MK.dc, 5.0, 0.4)
    assert lg.par_spread() == pytest.approx(MK.spreads[0, 2], abs=1e-12)
