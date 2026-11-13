import os
import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.stats import norm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "qfunction_results.csv")

THRESHOLD_DBM = -70.0
N_POINTS = 3
PTX_DBM = 10.0
ANCHOR_DB = 33.3
NEAR_M = 5.0

FEATS = ["tenlog10d", "N_total", "config_C2", "c2_near"]
BUILDINGS = ["Comms", "Library", "SSE"]


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["c2_near"] = df["config_C2"] * (df["Distance"] < NEAR_M).astype(int)
    return df


def design(g):
    return np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float)
                                                for f in FEATS])


def fit_building(g):
    X = design(g)
    y = g["PL_excess"].to_numpy(float)
    beta = la.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    mu = PTX_DBM - ANCHOR_DB - (X @ beta)
    return mu, float(np.std(resid, ddof=1)), beta


def main():
    df = load()
    T = THRESHOLD_DBM
    print(f"n={len(df)}  T={T:.0f} dBm")

    rows = []
    for b in BUILDINGS:
        g = df[df.building == b].reset_index(drop=True)
        mu, sigma, _ = fit_building(g)
        meas = g["P_rx"].to_numpy(float)

        p_soft = norm.sf((T - mu) / sigma)
        p_hard = (mu > T).astype(float)
        obs = (meas > T).astype(float)

        print(f"\n{b}  n={len(g)}  sigma={sigma:.2f} dB")
        print(f"  coverage measured   {obs.mean():.3f}")
        print(f"  coverage Q-function {p_soft.mean():.3f}"
              f"   error {p_soft.mean()-obs.mean():+.3f}")
        print(f"  coverage mean only  {p_hard.mean():.3f}"
              f"   error {p_hard.mean()-obs.mean():+.3f}")

        order = np.argsort(g["Distance"].to_numpy())
        pick = order[np.linspace(2, len(g) - 3, N_POINTS).astype(int)]
        print(f"  {'d (m)':>7} {'walls':>5} {'mu':>8} {'z':>7} {'Q(z)':>7}"
              f" {'measured':>9} {'> T?':>6} {'correct':>8}")
        for i in pick:
            z = (T - mu[i]) / sigma
            q = norm.sf(z)
            above = meas[i] > T
            ok = (q > 0.5) == above
            print(f"  {g['Distance'][i]:7.2f} {int(g['N_total'][i]):5d}"
                  f" {mu[i]:8.1f} {z:7.2f} {q:7.3f} {meas[i]:9.1f}"
                  f" {'yes' if above else 'no':>6} {'yes' if ok else 'NO':>8}")
            rows.append({
                "building": b, "threshold_dBm": T,
                "distance_m": float(g["Distance"][i]),
                "n_obstructions": int(g["N_total"][i]),
                "mu_dBm": round(float(mu[i]), 2), "sigma_dB": round(sigma, 2),
                "z": round(float(z), 3), "Q_z": round(float(q), 4),
                "measured_dBm": float(meas[i]), "above_T": bool(above),
                "classified_correctly": bool(ok),
            })

        rows.append({
            "building": b, "threshold_dBm": T, "distance_m": np.nan,
            "n_obstructions": np.nan, "mu_dBm": np.nan,
            "sigma_dB": round(sigma, 2), "z": np.nan,
            "Q_z": round(float(p_soft.mean()), 4),
            "measured_dBm": np.nan, "above_T": np.nan,
            "classified_correctly": np.nan,
            "summary": "aggregate",
            "coverage_q": round(float(p_soft.mean()), 4),
            "coverage_mean_only": round(float(p_hard.mean()), 4),
            "coverage_measured": round(float(obs.mean()), 4),
        })

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
