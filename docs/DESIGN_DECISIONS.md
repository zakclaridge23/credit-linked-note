# Design decisions

Material modelling choices, the alternatives considered, and why. Items marked **(needs sign-off)**
are open for review.

## Product

| Decision | Alternatives | Why |
|---|---|---|
| 5-year funded FtD CLN on 5 IG names, floating (OIS) + spread, quarterly | Fixed coupon; unfunded FtD swap; nth-to-default | The brief's default design; the floating coupon makes the default-free part a par floater, isolating the credit spread. |
| At the first credit event: investor receives N·R of the defaulted name, coupons stop, **no accrued coupon** | Pay accrued coupon to the event date | Common in CLN documentation and simple; it is why the CLN fair spread (372bp) exceeds the FtD swap spread (367bp). |
| Recovery paid immediately at τ₁ | Auction delay of weeks | Immaterial for valuation; noted. |
| Issuer/SPV and collateral risk ignored | Model collateral default (extension) | Base case per the brief. |

## Market data (synthetic)

| Decision | Alternatives | Why |
|---|---|---|
| Five names with upward-sloping par spread curves, 5y spreads 60–125bp, 40% recovery | Real quotes | No licensed data; values are in a realistic IG range and labelled illustrative. |
| OIS zero curve 3.6–4.0% with flat-forward interpolation | Flat curve; full OIS swap bootstrap | Non-flat enough to exercise the code; a swap bootstrap adds nothing to the credit questions. |

## Single-name CDS

| Decision | Alternatives | Why |
|---|---|---|
| Piecewise-constant hazards, sequential bootstrap, flat forwards | Smooth hazard curves | The ISDA standard model approach; local, positive hazards; exact repricing. |
| Quarterly accrual Δ = 0.25, dates counted back from maturity, no calendar | ACT/360, IMM dates, business-day adjustment | Same convention in bootstrap and repricing, so the internal results are consistent; real-world quotes would need the full ISDA conventions. **(needs sign-off: worth implementing ACT/360 and IMM dates?)** |
| Legs integrated on a fine grid (90 steps per quarter) with accrual on default | Closed-form integrals between knots | Simple and accurate (closed forms checked to 1e-11); the same integrator serves the FtD curve, which has no closed form. |
| Bootstrap rejects quotes needing negative hazards | Allow negative hazards; floor at zero | Inconsistent quotes are a data problem to surface, not to hide. |

## Default correlation

| Decision | Alternatives | Why |
|---|---|---|
| One-factor Gaussian copula with a single flat correlation | Base correlation; multi-factor; stochastic correlation | Market-standard starting point for bespoke baskets; one parameter to study. The Student-t copula is implemented in Monte Carlo as an extension. |
| Gauss–Hermite quadrature (64 nodes) for ρ ≤ 0.9, a 401-point trapezoid in M above | Gauss–Hermite everywhere | Near ρ = 1 the integrand is a step in M and Gauss–Hermite was off by 12bp at ρ = 0.99 (found by comparing with Monte Carlo). |
| First-to-default shares by name via conditional densities, rescaled to the exact −dS₁ | Assume equal recoveries | Supports name-specific recoveries at negligible cost. |
| Monte Carlo with antithetic variates and delta-method standard errors on the ratio | Control variates | Enough for validation at 1–2bp precision with 400k paths. |

## Hedging

| Decision | Alternatives | Why |
|---|---|---|
| CS01 = +1bp parallel par-spread bump of one name, re-bootstrapped; bucketed by tenor | Hazard bumps | Matches how desks quote risk. |
| Backtest hedges use a hazard bump of 1bp/(1−R) and a coarser grid (15 steps/quarter, 32 nodes) | Full par-spread bump | About 10× faster; equivalent to within a few percent for hedge ratios. |
| Hedge instruments: 5y (to note maturity) single-name CDS struck at par at each rebalance | Standard-coupon CDS with upfront; multiple maturities | One instrument per name gives exact CS01 neutrality; bucketed risk is concentrated at 5y anyway. |
| Backtest dynamics: lognormal hazard multipliers per name (vol 50%, correlation 0.5); default when realised cumulative hazard crosses a copula-correlated threshold | Independent default draws; CIR intensities | Spreads widen before defaults and defaults are correlated, with little code. Not derived from the pricing model; no contagion at default. **(needs sign-off: add contagion jumps?)** |
| Monthly rebalancing, 1-year horizon, undiscounted cash | Daily; full 5 years | Keeps 3,000 paths × 2 strategies to ~7 minutes on 4 cores. |
| Report the hedged book's +29bp mean as a model mismatch | Tune the dynamics until it is zero | The mismatch (no contagion in the simulated world vs. contagion implied by the copula) is itself the lesson about hedging correlation. |

## Open questions **(need sign-off)**

1. Implement full ISDA conventions (ACT/360, IMM dates, standard coupons with upfront quoting) for real-world quotes?
2. Add contagion (survivors' spreads jump at a default) to the backtest dynamics?
3. Extensions to prioritise: collateral default, nth-to-default, base-correlation mapping to an index.
