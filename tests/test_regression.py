"""Regression on headline numbers. Update together with README/notes/results.json."""

import pytest

from cln.hedging.sensitivities import NoteSpec, risk_report
from cln.market import build_market
from cln.pricing.copula import ftd_spread


def test_headline_numbers():
    mk = build_market()
    assert ftd_spread(mk.curves, mk.dc, 5.0, mk.recoveries, 0.0) * 1e4 == pytest.approx(
        443.22, abs=0.01
    )
    assert ftd_spread(mk.curves, mk.dc, 5.0, mk.recoveries, 0.3) * 1e4 == pytest.approx(
        366.58, abs=0.01
    )
    rep = risk_report(mk, NoteSpec(), bucketed=False)
    assert rep.fair_spread * 1e4 == pytest.approx(372.13, abs=0.01)
    assert rep.cs01.sum() * 1e4 == pytest.approx(-15.13, abs=0.01)
    assert rep.correl01 * 1e4 == pytest.approx(11.08, abs=0.01)
    assert rep.hedge_notional.sum() == pytest.approx(3.431, abs=0.002)
