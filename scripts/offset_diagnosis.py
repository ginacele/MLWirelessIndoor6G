import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled_all.csv")

ANCHOR_DB = 33.3
SEED = 0
RNG = np.random.default_rng(SEED)
N_BOOT = 2000

FEATURES = ["tenlog10d", "N_total", "config_C2"]
BUILDINGS = ["Comms", "Library", "SSE"]


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["ref30"] = (df["Comments"].astype(str)
                   .str.contains("-30 dBm", na=False).astype(int))
    df["delta"] = df["PL"] - (ANCHOR_DB + 20.0 * df["log10d"])
    return df


def boot_median(x, n_boot=N_BOOT):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan
    key = int(abs(np.sum(x) * 1000)) + len(x) * 7919 + SEED
    rng = np.random.default_rng(key % (2**32))
    draws = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fit_floating(df, extra=()):
    cols = list(FEATURES) + list(extra)
    X = sm.add_constant(df[cols].astype(float))
    return sm.OLS(df["PL_excess"].astype(float), X).fit()


def fit_anchored(df):
    return sm.OLS(df["PL_excess"].astype(float), df[FEATURES].astype(float)).fit()


def test1_reference_level(df):
    print("\nT1a  zero-obstruction points split by reference level, unmatched")
    los = df[(df.N_total == 0) & (df.near_noise_floor == 0)]
    for b in BUILDINGS:
        g = los[los.building == b]
        n50, n30 = int((g.ref30 == 0).sum()), int((g.ref30 == 1).sum())
        if n50 < 3 or n30 < 3:
            print(f"  {b:8s} n50={n50:3d} n30={n30:3d}  one group only")
            continue
        d50, d30 = g.loc[g.ref30 == 0, "delta"], g.loc[g.ref30 == 1, "delta"]
        cfg50 = sorted(g.loc[g.ref30 == 0, "config"].unique())
        cfg30 = sorted(g.loc[g.ref30 == 1, "config"].unique())
        print(f"  {b:8s} -50 n={n50:3d} median {d50.median():6.2f} dB  configs {cfg50}")
        print(f"  {b:8s} -30 n={n30:3d} median {d30.median():6.2f} dB  configs {cfg30}")
        print(f"  {b:8s} difference {d30.median() - d50.median():+.2f} dB")

    print("\nT1b  matched: same building, same config, overlapping distance range")
    print(f"  {'building':9s} {'cfg':4s} {'n30':>4s} {'n50':>4s} {'ref30':>7s}"
          f" {'ci_lo':>7s} {'ci_hi':>7s} {'p':>8s}")
    for b in BUILDINGS:
        g = df[df.building == b]
        r30 = g[g.ref30 == 1]
        if len(r30) == 0:
            continue
        lo, hi = r30.Distance.min(), r30.Distance.max()
        for cfg in ("C1", "C2"):
            s = g[(g.config == cfg) & (g.Distance.between(lo, hi))]
            n30, n50 = int(s.ref30.sum()), int((s.ref30 == 0).sum())
            if n30 < 3 or n50 < 3:
                print(f"  {b:9s} {cfg:4s} {n30:4d} {n50:4d}  not identified")
                continue
            X = sm.add_constant(
                s[["tenlog10d", "N_total", "ref30"]].astype(float))
            m = sm.OLS(s["PL_excess"].astype(float), X).fit()
            cl, ch = m.conf_int().loc["ref30"]
            print(f"  {b:9s} {cfg:4s} {n30:4d} {n50:4d} {m.params['ref30']:+7.2f}"
                  f" {cl:+7.2f} {ch:+7.2f} {m.pvalues['ref30']:8.3g}")


def test2_reference_level_in_fit(df):
    print("\nT2  ref30 dummy, unmatched")
    print(f"  {'scope':9s} {'n30':>5s} {'n':>5s} {'pct':>6s} {'coef':>7s}"
          f" {'ci_lo':>7s} {'ci_hi':>7s} {'p':>8s} {'offset':>16s} {'n_hat':>14s}")
    for b in BUILDINGS + ["POOLED"]:
        g = df if b == "POOLED" else df[df.building == b]
        if g.ref30.std() < 1e-12:
            print(f"  {b:9s} no variation in flag")
            continue
        m = fit_floating(g, extra=["ref30"])
        lo, hi = m.conf_int().loc["ref30"]
        base = fit_floating(g)
        print(f"  {b:9s} {int(g.ref30.sum()):5d} {len(g):5d}"
              f" {100*g.ref30.mean():5.1f}% {m.params['ref30']:+7.2f}"
              f" {lo:+7.2f} {hi:+7.2f} {m.pvalues['ref30']:8.3g}"
              f"  {base.params['const']:+6.2f} -> {m.params['const']:+6.2f}"
              f"  {base.params['tenlog10d']:.3f} -> {m.params['tenlog10d']:.3f}")


def test3_los_delta(df):
    los = df[(df.N_total == 0) & (df.near_noise_floor == 0)]
    print(f"\nT3  short-range LOS delta above anchor   "
          f"{len(los)} points, d <= {los.Distance.max():.2f} m")
    for label, sub in (("C1 only", los[los.config == "C1"]),
                       ("both configs", los)):
        print(f"  {label}")
        for b in BUILDINGS:
            g = sub[sub.building == b]
            if len(g) < 3:
                print(f"    {b:8s} n={len(g):3d}  too few")
                continue
            lo, hi = boot_median(g["delta"])
            print(f"    {b:8s} n={len(g):3d} median {g['delta'].median():6.2f} dB"
                  f"  CI [{lo:6.2f}, {hi:6.2f}]"
                  f"  d {g.Distance.min():.2f}-{g.Distance.max():.2f} m")

    print("  split by reference level")
    for b in BUILDINGS:
        for r, name in ((0, "-50 dBm"), (1, "-30 dBm")):
            g = los[(los.building == b) & (los.ref30 == r)]
            if len(g) < 3:
                continue
            lo, hi = boot_median(g["delta"])
            print(f"    {b:8s} {name:8s} n={len(g):3d}"
                  f" median {g['delta'].median():6.2f} dB"
                  f"  CI [{lo:6.2f}, {hi:6.2f}]")


def test4_censoring(df):
    print("\nT4  lowest P_rx values and row counts")
    vc = df["P_rx"].value_counts().sort_index().head(14)
    for v, n in vc.items():
        print(f"  {v:7.0f} dBm {n:4d}  {'#' * min(n, 40)}")
    print(f"  min P_rx per building: {dict(df.groupby('building')['P_rx'].min())}")

    print("  rows flagged near_noise_floor")
    for b in BUILDINGS:
        g = df[df.building == b]
        print(f"    {b:8s} {int(g.near_noise_floor.sum()):4d} / {len(g):4d}"
              f"  {100*g.near_noise_floor.mean():5.1f}%")

    print("  refit without flagged rows (distance-range truncation, not a "
          "censoring test)")
    print(f"    {'scope':9s} {'n_hat':>16s} {'delta':>8s} {'offset':>18s}"
          f" {'delta':>8s}")
    for b in BUILDINGS + ["POOLED"]:
        g = df if b == "POOLED" else df[df.building == b]
        if g.near_noise_floor.sum() == 0:
            print(f"    {b:9s} none flagged")
            continue
        full, clean = fit_floating(g), fit_floating(g[g.near_noise_floor == 0])
        print(f"    {b:9s} {full.params['tenlog10d']:6.3f} ->"
              f" {clean.params['tenlog10d']:6.3f}"
              f" {clean.params['tenlog10d'] - full.params['tenlog10d']:+8.3f}"
              f"  {full.params['const']:+7.2f} -> {clean.params['const']:+7.2f}"
              f" {clean.params['const'] - full.params['const']:+8.2f}")


def test5_qfunction(df, n_points=3):
    print("\nT5  Q-function at 50% exceedance, floating vs anchored reference")
    for b in BUILDINGS:
        g = df[df.building == b]
        print(f"  {b}")
        for name, m, has_const in (("floating", fit_floating(g), True),
                                   ("anchored", fit_anchored(g), False)):
            sigma = float(np.std(m.resid, ddof=1))
            X = sm.add_constant(g[FEATURES].astype(float)) if has_const \
                else g[FEATURES].astype(float)
            pred_prx = 10.0 - (m.predict(X) + ANCHOR_DB)

            idx = np.linspace(0, len(g) - 1, n_points).astype(int)
            sel = g.sort_values("Distance").index[idx]

            print(f"    {name:8s} sigma {sigma:.2f} dB")
            print(f"      {'d (m)':>7} {'walls':>5} {'meas':>8} {'pred':>8}"
                  f" {'err':>7} {'err/sig':>8} {'Q':>7}")
            for i in sel:
                err = g.loc[i, "P_rx"] - pred_prx.loc[i]
                z = err / sigma
                print(f"      {g.loc[i,'Distance']:7.2f} "
                      f"{int(g.loc[i,'N_total']):5d} "
                      f"{g.loc[i,'P_rx']:8.1f} {pred_prx.loc[i]:8.1f} "
                      f"{err:+7.1f} {z:+8.2f} {norm.sf(abs(z)):7.3f}")


def test6_config_paired(df):
    print("\nT6  C2 - C1 paired at identical coordinates, by distance band")
    for b in BUILDINGS:
        g = df[df.building == b]
        piv = g.pivot_table(index="Coord.", columns="config",
                            values=["PL", "Distance"], aggfunc="first").dropna()
        if ("PL", "C1") not in piv or ("PL", "C2") not in piv or len(piv) < 5:
            print(f"  {b:8s} too few paired coordinates")
            continue
        diff = piv[("PL", "C2")] - piv[("PL", "C1")]
        dist = piv[("Distance", "C1")]
        print(f"  {b:8s} {len(diff)} paired coordinates")
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40)):
            s = diff[(dist >= lo) & (dist < hi)]
            if len(s) < 3:
                continue
            cl, ch = boot_median(s)
            print(f"    d {lo:2d}-{hi:2d} m n={len(s):4d}"
                  f" median {s.median():+6.2f} dB  CI [{cl:+6.2f}, {ch:+6.2f}]")

    print("  refit with a config x log-distance interaction")
    print(f"    {'scope':9s} {'offset':>18s} {'n_hat':>16s} {'sigma':>16s}"
          f" {'interaction':>12s} {'p':>9s}")
    d = df.copy()
    d["c2_x_log"] = d["config_C2"] * d["tenlog10d"]
    for b in BUILDINGS:
        g = d[d.building == b]
        base = fit_floating(g)
        inter = fit_floating(g, extra=["c2_x_log"])
        print(f"    {b:9s} {base.params['const']:+7.2f} ->"
              f" {inter.params['const']:+7.2f}"
              f"  {base.params['tenlog10d']:.3f} ->"
              f" {inter.params['tenlog10d']:.3f}"
              f"  {np.std(base.resid, ddof=1):.2f} ->"
              f" {np.std(inter.resid, ddof=1):.2f}"
              f" {inter.params['c2_x_log']:+12.3f}"
              f" {inter.pvalues['c2_x_log']:9.3g}")


def main():
    df = load()
    print(f"n={len(df)}  anchor={ANCHOR_DB} dB  seed={SEED}")
    test1_reference_level(df)
    test2_reference_level_in_fit(df)
    test3_los_delta(df)
    test4_censoring(df)
    test5_qfunction(df)
    test6_config_paired(df)


if __name__ == "__main__":
    main()
