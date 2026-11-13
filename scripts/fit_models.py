import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "primary_results.csv")

ANCHOR_DB = 33.3
SEED = 0
N_BOOT = 500
NEAR_M = 5.0

BUILDINGS = ["Comms", "Library", "SSE"]

SPEC_A = ["tenlog10d", "N_total", "config_C2"]
SPEC_E = ["tenlog10d", "N_total", "config_C2", "c2_near"]

MATERIALS = ["Num_brick_wall", "Num_wood_wall", "Num_glass_wall",
             "Num_drywall", "Num_column"]


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["near"] = (df["Distance"] < NEAR_M).astype(int)
    df["c2_near"] = df["config_C2"] * df["near"]
    return df


def fit(g, feats):
    X = sm.add_constant(g[feats].astype(float))
    return sm.OLS(g["PL_excess"].astype(float), X).fit()


def block_bootstrap_n(g, feats, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    blocks = [b for _, b in g.groupby(["config", "grid_letter"], dropna=False)]
    if len(blocks) < 3:
        return np.nan, np.nan
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(blocks), len(blocks))
        s = pd.concat([blocks[i] for i in idx], ignore_index=True)
        if s[feats].std().min() <= 1e-12:
            continue
        try:
            draws.append(fit(s, feats).params["tenlog10d"])
        except Exception:
            continue
    if len(draws) < 20:
        return np.nan, np.nan
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def formal_subgroup_test(values, group, label=""):
    a = np.asarray(values)[np.asarray(group) == 0]
    b = np.asarray(values)[np.asarray(group) == 1]
    if len(a) < 3 or len(b) < 3:
        return None
    t, p, _ = sm.stats.ttest_ind(a, b, usevar="unequal")
    return {"label": label, "n0": len(a), "n1": len(b),
            "mean0": float(a.mean()), "mean1": float(b.mean()),
            "t": float(t), "p": float(p)}


def row(spec_name, scope, g, feats):
    m = fit(g, feats)
    resid = m.resid.values
    lo, hi = block_bootstrap_n(g, feats)
    trend = sm.OLS(resid, sm.add_constant(g["log10d"].astype(float))).fit()
    tl, th = trend.conf_int().iloc[1]
    span = np.log10(g["Distance"].max() / g["Distance"].min())
    return {
        "spec": spec_name,
        "scope": scope,
        "n_points": len(g),
        "n_hat": round(float(m.params["tenlog10d"]), 3),
        "n_hat_se": round(float(m.bse["tenlog10d"]), 3),
        "n_ci_lo": round(lo, 3) if lo == lo else np.nan,
        "n_ci_hi": round(hi, 3) if hi == hi else np.nan,
        "offset_dB": round(float(m.params["const"]), 2),
        "L_total_dB": round(float(m.params["N_total"]), 2),
        "config_C2_dB": round(float(m.params["config_C2"]), 2),
        "c2_near_dB": (round(float(m.params["c2_near"]), 2)
                       if "c2_near" in feats else np.nan),
        "c2_near_p": (round(float(m.pvalues["c2_near"]), 4)
                      if "c2_near" in feats else np.nan),
        "sigma_dB": round(float(np.std(resid, ddof=1)), 2),
        "bias_dB": round(float(np.mean(resid)), 2),
        "trend_slope": round(float(trend.params.iloc[1]), 2),
        "trend_lo": round(float(tl), 2),
        "trend_hi": round(float(th), 2),
        "d_span_decades": round(float(span), 2),
    }


def main():
    df = load()
    print(f"n={len(df)}  anchor={ANCHOR_DB} dB  near_cut={NEAR_M} m  seed={SEED}")

    print(f"\npaired C2 - C1 at identical coordinates, split at {NEAR_M} m")
    print(f"{'building':9s} {'mean d>=5':>10s} {'n':>5s} {'mean d<5':>9s}"
          f" {'n':>4s} {'Welch p':>9s} {'sd pair':>8s}")
    for b in BUILDINGS:
        g = df[df.building == b]
        piv = g.pivot_table(index="Coord.", columns="config",
                            values=["PL", "Distance"], aggfunc="first").dropna()
        if len(piv) < 10:
            continue
        d = (piv[("PL", "C2")] - piv[("PL", "C1")]).values
        near = (piv[("Distance", "C1")] < NEAR_M).astype(int).values
        r = formal_subgroup_test(d, near, b)
        if r:
            print(f"{b:9s} {r['mean0']:+10.2f} {r['n0']:5d} {r['mean1']:+9.2f}"
                  f" {r['n1']:4d} {r['p']:9.4g} {d.std(ddof=1):8.1f}")

    rows = []
    for spec_name, feats in (("A", SPEC_A), ("E", SPEC_E)):
        print(f"\nspec {spec_name}: PL_excess ~ const + " + " + ".join(feats))
        print(f"{'scope':9s} {'n':>5s} {'n_hat':>7s} {'se':>6s} {'ci_lo':>7s}"
              f" {'ci_hi':>7s} {'offset':>7s} {'L_tot':>6s} {'cfg':>6s}"
              f" {'c2near':>7s} {'p':>7s} {'sigma':>6s} {'decades':>8s}")
        for b in BUILDINGS:
            r = row(spec_name, b, df[df.building == b], feats)
            rows.append(r)
            print(f"{b:9s} {r['n_points']:5d} {r['n_hat']:7.3f}"
                  f" {r['n_hat_se']:6.3f} {r['n_ci_lo']:7.2f} {r['n_ci_hi']:7.2f}"
                  f" {r['offset_dB']:+7.2f} {r['L_total_dB']:+6.2f}"
                  f" {r['config_C2_dB']:+6.2f}"
                  + (f" {r['c2_near_dB']:+7.2f} {r['c2_near_p']:7.4g}"
                     if spec_name == "E" else f" {'-':>7s} {'-':>7s}")
                  + f" {r['sigma_dB']:6.2f} {r['d_span_decades']:8.2f}")

        r = row(spec_name, "POOLED_shared_offset", df, feats)
        rows.append(r)
        print(f"{'POOLED':9s} {r['n_points']:5d} {r['n_hat']:7.3f}"
              f" {r['n_hat_se']:6.3f} {r['n_ci_lo']:7.2f} {r['n_ci_hi']:7.2f}"
              f" {r['offset_dB']:+7.2f} {r['L_total_dB']:+6.2f}"
              f" {r['config_C2_dB']:+6.2f}"
              + (f" {r['c2_near_dB']:+7.2f} {r['c2_near_p']:7.4g}"
                 if spec_name == "E" else f" {'-':>7s} {'-':>7s}")
              + f" {r['sigma_dB']:6.2f} {r['d_span_decades']:8.2f}")

        sub = [x for x in rows if x["spec"] == spec_name
               and x["scope"] in BUILDINGS]
        v = [x["n_hat"] for x in sub]
        o = [x["offset_dB"] for x in sub]
        cl = [x["n_hat"] for x in sub if x["scope"] != "SSE"]
        print(f"exponent spread {max(v)-min(v):.3f}   offset spread "
              f"{max(o)-min(o):.2f} dB   Comms vs Library {max(cl)-min(cl):.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)

    print("\nblock-bootstrap interval overlap")
    for spec_name in ("A", "E"):
        sub = [x for x in rows if x["spec"] == spec_name
               and x["scope"] in BUILDINGS]
        los = [x["n_ci_lo"] for x in sub]
        his = [x["n_ci_hi"] for x in sub]
        if any(v != v for v in los + his):
            print(f"  spec {spec_name}: an interval failed, cannot compare")
            continue
        lo, hi = max(los), min(his)
        spread = max(x["n_hat"] for x in sub) - min(x["n_hat"] for x in sub)
        print(f"  spec {spec_name}")
        for x in sub:
            print(f"    {x['scope']:9s} n_hat {x['n_hat']:6.3f}"
                  f"  OLS +/-{x['n_hat_se']:.3f}"
                  f"  boot [{x['n_ci_lo']:5.2f}, {x['n_ci_hi']:5.2f}]"
                  f"  {x['n_ci_hi']-x['n_ci_lo']:.2f} wide"
                  f"  {(x['n_ci_hi']-x['n_ci_lo'])/(2*1.96*x['n_hat_se']):.1f}x OLS")
        if lo <= hi:
            print(f"    overlap [{lo:.2f}, {hi:.2f}]   point spread {spread:.3f}")
        else:
            print(f"    no common overlap   point spread {spread:.3f}")

    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
