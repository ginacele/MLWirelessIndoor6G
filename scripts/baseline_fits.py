import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")

WALL_COLS = [
    "Num_brick_wall",
    "Num_wood_wall",
    "Num_glass_wall",
    "Num_drywall",
    "Num_column",
    "Num_elevator",
]

FEATURES_FULL = ["tenlog10d"] + WALL_COLS + ["config_C2"]
FEATURES_TOTAL = ["tenlog10d", "N_total", "config_C2"]

N_BOOT = 500
RNG = np.random.default_rng(0)


def drop_degenerate(df, feats):
    keep, dropped = [], []
    for f in feats:
        if df[f].std() > 1e-12:
            keep.append(f)
        else:
            dropped.append(f)
    return keep, dropped


def fit_anchored(df, feats, anchored=True):
    X = df[feats].astype(float)
    if not anchored:
        X = sm.add_constant(X, has_constant="add")
    y = df["PL_excess"].astype(float)
    return sm.OLS(y, X).fit()


def vif_table(df, feats):
    X = df[feats].astype(float).values
    out = {}
    for i, f in enumerate(feats):
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            out[f] = 1.0
            continue
        r2 = sm.OLS(X[:, i], sm.add_constant(others)).fit().rsquared
        out[f] = np.inf if r2 > 1 - 1e-12 else 1.0 / (1.0 - r2)
    return pd.Series(out)


def block_bootstrap_n(df, feats, anchored=True, n_boot=N_BOOT):
    blocks = [g for _, g in df.groupby(["config", "grid_letter"], dropna=False)]
    if len(blocks) < 3:
        return np.array([])
    draws = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(blocks), len(blocks))
        samp = pd.concat([blocks[i] for i in idx], ignore_index=True)
        if samp[feats].std().min() <= 1e-12:
            continue
        try:
            draws.append(fit_anchored(samp, feats, anchored).params["tenlog10d"])
        except Exception:
            continue
    return np.array(draws)


def residual_diagnostics(df, resid):
    x = df["log10d"].astype(float)
    trend = sm.OLS(resid, sm.add_constant(x)).fit()
    slope = trend.params.iloc[1]
    lo, hi = trend.conf_int().iloc[1]
    return {
        "sigma_dB": float(np.std(resid, ddof=1)),
        "bias_dB": float(np.mean(resid)),
        "trend_slope": float(slope),
        "trend_ci": (float(lo), float(hi)),
        "trend_ok": bool(lo < 0 < hi),
        "shapiro_p": float(stats.shapiro(resid[:5000])[1]) if len(resid) >= 3 else np.nan,
    }


def report(df, label, feats, name, anchored=True):
    keep, dropped = drop_degenerate(df, feats)
    if "tenlog10d" not in keep:
        print(f"    [{name}] no distance variance, skipped")
        return None

    m = fit_anchored(df, keep, anchored)
    n_hat = m.params["tenlog10d"]
    resid = m.resid.values
    d = residual_diagnostics(df, resid)
    boot = block_bootstrap_n(df, keep, anchored)

    print(f"    [{name}] n={len(df)}", end="")
    if not anchored:
        c = m.params["const"]
        print(f"  offset {c:+.2f} dB  PL(1m) {33.3 + c:.1f} dB", end="")
    print(f"  n_hat {n_hat:.3f}", end="")
    if boot.size:
        print(f"  boot [{np.percentile(boot, 2.5):.3f}, "
              f"{np.percentile(boot, 97.5):.3f}]  sd {boot.std():.3f}")
    else:
        print()
    if dropped:
        print(f"      dropped, no variance: {', '.join(dropped)}")

    for f in keep:
        if f == "tenlog10d":
            continue
        print(f"      {f:18s} {m.params[f]:7.2f}  p={m.pvalues[f]:.3g}")

    print(f"      sigma {d['sigma_dB']:.2f} dB  bias {d['bias_dB']:.2f} dB"
          f"  resid trend {d['trend_slope']:.2f}"
          f" CI[{d['trend_ci'][0]:.2f},{d['trend_ci'][1]:.2f}]"
          f" {'ok' if d['trend_ok'] else 'STRUCTURE LEFT IN RESIDUALS'}"
          f"  shapiro p={d['shapiro_p']:.3g}")

    v = vif_table(df, keep)
    worst = v.drop("tenlog10d", errors="ignore")
    print(f"      VIF(tenlog10d) {v['tenlog10d']:.1f}"
          + (f"  max other {worst.max():.1f} ({worst.idxmax()})"
             if len(worst) else ""))

    return {
        "scope": label, "model": name, "anchored": anchored,
        "n": len(df), "n_hat": n_hat,
        "offset_dB": (m.params["const"] if not anchored else 0.0),
        "n_hat_lo": np.percentile(boot, 2.5) if boot.size else np.nan,
        "n_hat_hi": np.percentile(boot, 97.5) if boot.size else np.nan,
        "sigma_dB": d["sigma_dB"], "bias_dB": d["bias_dB"],
        "trend_ok": d["trend_ok"], "vif_dist": v["tenlog10d"],
        "dropped": ";".join(dropped),
    }


def main():
    df = pd.read_csv(POOLED)
    if not (df["audit_is_LOS_geom"] == 1).any():
        print("WARNING: pooled.csv has no zero-obstruction rows, "
              "so LOS is empty and ALL == NLOS\n")
    df["tenlog10d"] = 10.0 * df["log10d"]
    for c in WALL_COLS:
        if c not in df.columns:
            df[c] = 0.0

    rows = []
    variants = (("FULL", FEATURES_FULL), ("TOTAL", FEATURES_TOTAL))

    subsets = {
        "ALL":  df,
        "NLOS": df[df["audit_is_LOS_geom"] == 0],
        "LOS":  df[df["audit_is_LOS_geom"] == 1],
    }

    for sub_name, sub in subsets.items():
        print(f"\nsubset {sub_name}   n={len(sub)}")

        conventions = (True, False) if sub_name == "ALL" else (True,)

        for anchored in conventions:
            print(f"\n  intercept {'anchored at 33.3 dB' if anchored else 'floating'}")

            use = variants if sub_name != "LOS" else (("TOTAL", FEATURES_TOTAL),)

            for b, g in sub.groupby("building"):
                if len(g) < 10:
                    print(f"  {b}  n={len(g)}, too few, skipped")
                    continue
                print(f"  {b}")
                for name, feats in use:
                    r = report(g, b, feats, name, anchored)
                    if r:
                        r["subset"] = sub_name
                        rows.append(r)

            print("  POOLED")
            for name, feats in use:
                r = report(sub, "POOLED", feats, name, anchored)
                if r:
                    r["subset"] = sub_name
                    rows.append(r)

    res = pd.DataFrame(rows)

    print("\nALL vs NLOS exponent")
    a = res[(res.subset == "ALL") & (res.anchored)]
    nl = res[res.subset == "NLOS"]
    for name in ("FULL", "TOTAL"):
        for scope in ("Comms", "Library", "SSE", "POOLED"):
            va = a[(a.model == name) & (a.scope == scope)]["n_hat"]
            vn = nl[(nl.model == name) & (nl.scope == scope)]["n_hat"]
            if len(va) and len(vn):
                print(f"  {name:6s} {scope:8s} ALL {va.iloc[0]:.3f}  "
                      f"NLOS {vn.iloc[0]:.3f}  delta {vn.iloc[0]-va.iloc[0]:+.3f}")

    print("\nexponent spread across buildings")
    for sub_name in ("ALL", "NLOS", "LOS"):
        for anchored in (True, False):
            for name in ("FULL", "TOTAL"):
                sub = res[(res.subset == sub_name) & (res.model == name)
                          & (res.anchored == anchored) & (res.scope != "POOLED")]
                if len(sub) < 2:
                    continue
                spread = sub.n_hat.max() - sub.n_hat.min()
                lab = "anchored" if anchored else "floating"
                vals = "  ".join(f"{s}={v:.2f}" for s, v in zip(sub.scope, sub.n_hat))
                print(f"  {sub_name:5s} {lab:9s} {name:6s} {spread:.2f}   {vals}")

    out = os.path.join(DATA_DIR, "baseline_results.csv")
    res.to_csv(out, index=False)
    print(f"\nWrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
