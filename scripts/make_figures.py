"""Regenerate every figure, table and headline number in the README.

    python scripts/make_figures.py --out figures/ [--paths 3000]

Writes PNGs, results.json and risk_table.md. Deterministic (fixed seeds).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from cln.hedging.backtest import BacktestConfig, run_path
from cln.hedging.sensitivities import NoteSpec, risk_report
from cln.market import TENORS, build_market
from cln.plotting import AQUA, BLUE, INK_2, MAGENTA, MUTED, ORANGE, YELLOW, apply_style, plt, save
from cln.pricing.cds import par_spread, time_grid
from cln.pricing.copula import ftd_distribution, ftd_spread
from cln.pricing.montecarlo import ftd_mc
from cln.pricing.note import note_value

MK = build_market()
T = 5.0
RHO = 0.3
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
BP = 1e4


def fig_survival(out: Path) -> None:
    g = time_grid(0, 10)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for c, name, col, row in zip(MK.curves, MK.names, SERIES, MK.spreads, strict=True):
        ax[0].plot(g, c.survival(g), color=col, label=f"{name} (5y {row[2] * BP:.0f}bp)")
    ax[0].set_xlabel("years")
    ax[0].set_ylabel("survival probability Q(t)")
    ax[0].set_title("Bootstrapped single-name survival curves", fontsize=10)
    ax[0].legend()
    g5 = time_grid(0, T)
    Q = np.array([c.survival(g5) for c in MK.curves])
    for rho, col in zip([0.0, 0.3, 0.9], [BLUE, ORANGE, AQUA], strict=True):
        S1, _ = ftd_distribution(Q, rho)
        ax[1].plot(g5, S1, color=col, label=f"no default in basket, ρ = {rho}")
    ax[1].plot(g5, Q[3], color=INK_2, ls="--", label="riskiest single name (D)")
    ax[1].set_xlabel("years")
    ax[1].set_ylabel("P(no default by t)")
    ax[1].set_title("First-to-default survival: correlation keeps the basket alive", fontsize=10)
    ax[1].legend()
    save(fig, out / "survival_curves.png")


def _mc(args: tuple[float, str, int]) -> dict[str, float]:
    rho, cop, seed = args
    r = ftd_mc(MK.curves, MK.dc, T, MK.recoveries, rho, 400_000, np.random.default_rng(seed), cop)
    return {"rho": rho, "copula": cop, "spread": r.spread * BP, "se": r.spread_se * BP}


def fig_correlation(pool: ProcessPoolExecutor, out: Path, res: dict[str, Any]) -> None:
    rhos = np.linspace(0, 0.99, 34)
    semi = np.array([ftd_spread(MK.curves, MK.dc, T, MK.recoveries, r) * BP for r in rhos])
    cln = np.array(
        [note_value(MK.curves, MK.dc, T, MK.recoveries, r).fair_spread() * BP for r in rhos]
    )
    pts = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    mc = list(
        pool.map(_mc, [(r, c, 10 + i) for i, r in enumerate(pts) for c in ("gaussian", "student")])
    )
    singles = [par_spread(c, MK.dc, T, 0.4) * BP for c in MK.curves]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axhline(sum(singles), color=MUTED, ls="--", lw=1.2)
    ax.text(
        0.98,
        sum(singles) + 6,
        f"sum of 5y spreads {sum(singles):.0f}bp (independence bound)",
        ha="right",
        fontsize=8,
        color=INK_2,
    )
    ax.axhline(max(singles), color=MUTED, ls="--", lw=1.2)
    ax.text(
        0.02,
        max(singles) - 16,
        f"widest single name {max(singles):.0f}bp (comonotone bound)",
        fontsize=8,
        color=INK_2,
    )
    ax.plot(rhos, semi, color=BLUE, lw=2, label="FtD par spread: semi-analytic (Gauss–Hermite)")
    g = [m for m in mc if m["copula"] == "gaussian"]
    t = [m for m in mc if m["copula"] == "student"]
    ax.errorbar(
        [m["rho"] for m in g],
        [m["spread"] for m in g],
        yerr=[1.96 * m["se"] for m in g],
        fmt="o",
        color=BLUE,
        ms=6,
        label="Monte Carlo, Gaussian copula (400k paths)",
    )
    ax.errorbar(
        [m["rho"] for m in t],
        [m["spread"] for m in t],
        yerr=[1.96 * m["se"] for m in t],
        fmt="s",
        color=MAGENTA,
        ms=6,
        label="Monte Carlo, Student-t copula (ν = 4)",
    )
    ax.plot(rhos, cln, color=ORANGE, lw=1.6, ls="-.", label="CLN fair spread over OIS")
    ax.set_xlabel("default correlation ρ (one-factor copula)")
    ax.set_ylabel("spread (bp per year)")
    ax.set_title("First-to-default spread falls with correlation, between the two bounds")
    ax.legend(loc="center left", fontsize=8)
    ax.set_ylim(0, sum(singles) * 1.12)
    save(fig, out / "ftd_vs_correlation.png")
    i3 = int(np.argmin(np.abs(rhos - 0.3)))
    res["correlation"] = {
        "sum_singles_bp": round(sum(singles), 2),
        "widest_bp": round(max(singles), 2),
        "ftd_rho0_bp": round(float(semi[0]), 2),
        "ftd_rho099_bp": round(float(semi[-1]), 2),
        "mc": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()} for m in mc],
    }
    del i3


def fig_risk(out: Path, res: dict[str, Any]) -> None:
    rep = risk_report(MK, NoteSpec(T=T, rho=RHO))
    s = rep.fair_spread
    res["note"] = {
        "rho": RHO,
        "fair_spread_bp": round(s * BP, 2),
        "ftd_spread_bp": round(ftd_spread(MK.curves, MK.dc, T, MK.recoveries, RHO) * BP, 2),
        "cs01_bp_per_bp": [round(x * BP, 3) for x in rep.cs01],
        "cs01_total": round(float(rep.cs01.sum() * BP), 3),
        "hedge_notional": [round(x, 3) for x in rep.hedge_notional],
        "correl01_bp": round(rep.correl01 * BP, 2),
        "rec01_bp": round(rep.rec01 * BP, 2),
        "jtd_investor": [round(x, 4) for x in rep.jtd],
    }
    dealer_jtd_unhedged = -rep.jtd
    dealer_jtd_hedged = -rep.jtd - rep.hedge_notional * (1 - np.array(MK.recoveries))
    res["note"]["dealer_jtd_hedged"] = [round(x, 4) for x in dealer_jtd_hedged]
    lines = [
        "| Name | 5y spread (bp) | CS01, investor (bp of notional per bp) | 5y CDS hedge "
        "notional (dealer sells) | Dealer JTD unhedged | Dealer JTD CS01-hedged |",
        "|---|---|---|---|---|---|",
    ]
    for i, n in enumerate(MK.names):
        lines.append(
            f"| {n} | {MK.spreads[i, 2] * BP:.0f} | {rep.cs01[i] * BP:.2f} | "
            f"{rep.hedge_notional[i]:.3f} | {dealer_jtd_unhedged[i]:+.3f} | "
            f"{dealer_jtd_hedged[i]:+.3f} |"
        )
    lines.append(
        f"| **Total** | | {rep.cs01.sum() * BP:.2f} | {rep.hedge_notional.sum():.3f} | | |"
    )
    lines += [
        "",
        f"Correl01 (investor, +1% correlation): {rep.correl01 * BP:+.2f}bp of notional. "
        f"Rec01 (investor, recovery −1%, spreads fixed): {rep.rec01 * BP:+.2f}bp. "
        "JTD per unit notional; the dealer is long FtD protection.",
    ]
    (out / "risk_table.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(MK.names))
    ax[0].bar(x, -rep.cs01 * BP, color=BLUE, width=0.6)
    ax[0].set_xticks(x, MK.names, fontsize=8)
    ax[0].set_ylabel("loss per +1bp (bp of notional)")
    ax[0].set_title("Investor CS01 per name (5y FtD CLN, ρ = 0.3)", fontsize=10)
    w = 0.16
    for j, (tn, col) in enumerate(zip(TENORS, SERIES, strict=True)):
        ax[1].bar(
            x + (j - 2) * w, -rep.cs01_bucketed[:, j] * BP, width=w, color=col, label=f"{tn:g}y"
        )
    ax[1].set_xticks(x, MK.names, fontsize=8)
    ax[1].set_ylabel("loss per +1bp at one tenor (bp)")
    ax[1].set_title("Bucketed CS01: almost all at the 5y node", fontsize=10)
    ax[1].legend(ncol=5, fontsize=7)
    ax[2].bar(x - 0.17, rep.hedge_notional, width=0.32, color=ORANGE, label="CS01-neutral hedge")
    ax[2].bar(x + 0.17, dealer_jtd_hedged, width=0.32, color=AQUA, label="dealer JTD after hedging")
    ax[2].set_xticks(x, MK.names, fontsize=8)
    ax[2].set_ylabel("fraction of note notional")
    ax[2].set_title("Hedge notionals and residual jump-to-default", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle(
        "Spread risk is hedgeable name by name; jump-to-default and correlation are not",
        fontsize=12,
        fontweight="bold",
    )
    save(fig, out / "cs01.png")


def _bt(args: tuple[bool, int, float]) -> tuple[float, float, float, int]:
    hedge, seed, s = args
    r = run_path(MK, BacktestConfig(T=T, rho=RHO, hedge=hedge), s, np.random.default_rng(seed))
    return r.spread, r.carry, r.default, r.defaulted


def fig_backtest(pool: ProcessPoolExecutor, paths: int, out: Path, res: dict[str, Any]) -> None:
    s = note_value(MK.curves, MK.dc, T, MK.recoveries, RHO).fair_spread()
    out_rows: dict[str, np.ndarray] = {}
    for hedge in (False, True):
        rows = list(pool.map(_bt, [(hedge, i, s) for i in range(paths)], chunksize=20))
        out_rows["hedged" if hedge else "unhedged"] = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(-1500, 7000, 86)
    summary = {}
    for k, (key, col) in enumerate([("unhedged", BLUE), ("hedged", ORANGE)]):
        a = out_rows[key]
        tot = a[:, :3].sum(axis=1) * BP
        dflt = a[:, 3] >= 0
        ax[k].hist(tot[~dflt], bins=bins, color=col, alpha=0.8, label="no default in the year")
        ax[k].hist(
            tot[dflt],
            bins=bins,
            color=MAGENTA,
            alpha=0.9,
            label=f"first default occurred ({dflt.sum()} of {len(tot)})",
        )
        ax[k].set_yscale("log")
        ax[k].set_xlabel("one-year P&L (bp of note notional)")
        ax[k].set_ylabel("number of paths (log)")
        ax[k].set_title(
            f"Dealer {key}"
            + (
                " (CS01-neutral, monthly rebalance)"
                if key == "hedged"
                else " (long FtD protection)"
            ),
            fontsize=10,
        )
        ax[k].legend(fontsize=8)
        nd = ~dflt
        summary[key] = {
            "paths": len(tot),
            "defaults": int(dflt.sum()),
            "mean_total_bp": round(float(tot.mean()), 1),
            "sd_total_bp": round(float(tot.std(ddof=1)), 1),
            "sd_no_default_bp": round(float(tot[nd].std(ddof=1)), 1),
            "mean_spread_bp": round(float(a[nd, 0].mean() * BP), 1),
            "sd_spread_no_default_bp": round(float(a[nd, 0].std(ddof=1) * BP), 1),
            "mean_carry_bp": round(float(a[:, 1].mean() * BP), 1),
            "mean_default_pnl_given_default_bp": round(float(a[dflt, 2].mean() * BP), 1)
            if dflt.any()
            else None,
            "p5_bp": round(float(np.percentile(tot, 5)), 1),
            "p95_bp": round(float(np.percentile(tot, 95)), 1),
        }
    fig.suptitle(
        "One-year hedge backtest: CS01 hedging removes spread risk but leaves a long "
        "default exposure",
        fontsize=12,
        fontweight="bold",
    )
    save(fig, out / "hedge_pnl.png")
    res["backtest"] = summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--paths", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    res: dict[str, Any] = {}
    fig_survival(out)
    fig_risk(out, res)
    with ProcessPoolExecutor(a.workers) as pool:
        fig_correlation(pool, out, res)
        fig_backtest(pool, a.paths, out, res)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
