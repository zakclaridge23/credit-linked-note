import numpy as np

from cln.hedging.backtest import BacktestConfig, run_path
from cln.market import build_market
from cln.pricing.note import note_value

MK = build_market()
S = note_value(MK.curves, MK.dc, 5.0, MK.recoveries, 0.3).fair_spread()


def test_hedge_reduces_spread_pnl_dispersion():
    cfg_u = BacktestConfig(hedge=False, horizon=0.5, rebalance_per_year=12)
    cfg_h = BacktestConfig(hedge=True, horizon=0.5, rebalance_per_year=12)
    u = [run_path(MK, cfg_u, S, np.random.default_rng(i)) for i in range(30)]
    h = [run_path(MK, cfg_h, S, np.random.default_rng(i)) for i in range(30)]
    su = np.std([r.spread for r in u if r.defaulted < 0])
    sh = np.std([r.spread for r in h if r.defaulted < 0])
    assert sh < 0.5 * su


def test_zero_volatility_no_default_path_is_carry_plus_time_decay():
    """With no spread moves a surviving path is deterministic: carry -s*horizon and a spread
    P&L equal to the change in the dealer's mark from pure passage of time."""
    from cln.hedging.backtest import _dealer_value

    cfg = BacktestConfig(hedge=False, spread_vol=0.0, horizon=0.25, rebalance_per_year=12)
    decay = _dealer_value(MK.curves, MK, cfg, S, 0.25) - _dealer_value(MK.curves, MK, cfg, S, 0.0)
    for i in range(10):
        r = run_path(MK, cfg, S, np.random.default_rng(i))
        if r.defaulted < 0:
            assert abs(r.carry + S * 0.25) < 1e-12
            assert abs(r.spread - decay) < 1e-12
