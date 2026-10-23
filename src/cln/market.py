"""Synthetic, illustrative market data (not real quotes).

Five investment-grade reference entities with CDS par spread curves in a
realistic IG range, 40% recovery, and an OIS zero curve with flat-forward
interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cln.pricing.cds import bootstrap
from cln.pricing.curves import DiscountCurve, HazardCurve

TENORS = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
NAMES = ["Name A", "Name B", "Name C", "Name D", "Name E"]
SPREADS_BP = np.array(
    [
        [40, 60, 80, 95, 110],
        [55, 80, 100, 115, 125],
        [30, 45, 60, 70, 80],
        [70, 100, 125, 140, 150],
        [45, 70, 90, 105, 115],
    ],
    dtype=float,
)
RECOVERY = 0.40
OIS_TENORS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
OIS_ZEROS = np.array([0.040, 0.039, 0.037, 0.036, 0.036, 0.037, 0.038])


@dataclass(frozen=True)
class Market:
    dc: DiscountCurve
    curves: tuple[HazardCurve, ...]
    spreads: np.ndarray  # decimal, (names, tenors)
    recoveries: tuple[float, ...]
    names: tuple[str, ...]


def build_market(
    spreads_bp: np.ndarray = SPREADS_BP, recovery: float = RECOVERY, zeros: np.ndarray = OIS_ZEROS
) -> Market:
    dc = DiscountCurve.from_zero_rates(OIS_TENORS, zeros)
    s = np.asarray(spreads_bp, dtype=float) * 1e-4
    curves = tuple(bootstrap(TENORS, row, dc, recovery) for row in s)
    return Market(dc, curves, s, tuple([recovery] * len(curves)), tuple(NAMES[: len(curves)]))
