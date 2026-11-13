import os
import numpy as np
import numpy.linalg as la
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "a_vs_e.csv")

NEAR_M = 5.0
ANCHOR_DB = 33.3
FEATS_A = ["tenlog10d", "N_total", "config_C2"]
FEATS_E = FEATS_A + ["c2_near"]
BUILDINGS = ["Comms", "Library", "SSE"]
NAMES = {"const": "offset", "tenlog10d": "n", "N_total": "L_total",
         "config_C2": "C2", "c2_near": "C2_near"}

_WARNED = False


def warn_if_nlos_only(df):
    global _WARNED
    if _WARNED or (df["N_total"] == 0).any():
        return
    _WARNED = True
    print("WARNING: pooled.csv has no zero-obstruction rows, "
          "so ALL == NLOS below\n")


def load(subset="ALL"):
    df = pd.read_csv(POOLED)
    warn_if_nlos_only(df)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["c2_near"] = df["config_C2"] * (df["Distance"] < NEAR_M).astype(int)
    if subset == "NLOS":
        df = df[df["N_total"] > 0]
    elif subset == "LOS":
        df = df[df["N_total"] == 0]
    return df.reset_index(drop=True)


def fit(g, feats):
    X = np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float) for f in feats])
    y = g["PL_excess"].to_numpy(float)
    b, *_ = la.lstsq(X, y, rcond=None)
    r = y - X @ b
    dof = len(g) - X.shape[1]
    s2 = float(r @ r) / dof
    se = np.sqrt(np.diag(s2 * la.inv(X.T @ X)))
    return b, se, float(np.std(r, ddof=1)), dof


def main():
    rows = []

    for subset in ("ALL", "NLOS"):
        df = load(subset)
        print(f"\nsubset {subset}   n={len(df)}")
        print(f"  {'building':9s} {'model':5s} {'n':>6s} {'offset':>8s}"
              f" {'L_total':>8s} {'C2':>7s} {'C2_near':>8s} {'sigma':>7s}"
              f" {'N':>5s}")
        for b in BUILDINGS:
            g = df[df.building == b]
            for label, feats in (("A", FEATS_A), ("E", FEATS_E)):
                beta, se, sd, dof = fit(g, feats)
                cols = ["const"] + feats
                v = dict(zip(cols, beta))
                line = (f"  {b:9s} {label:5s} {v['tenlog10d']:6.3f}"
                        f" {v['const']:8.2f} {v['N_total']:8.2f}"
                        f" {v['config_C2']:7.2f}")
                line += (f" {v['c2_near']:8.2f}" if "c2_near" in v
                         else f" {'-':>8s}")
                line += f" {sd:7.2f} {len(g):5d}"
                print(line)
                for c, bb, ss in zip(cols, beta, se):
                    rows.append({"subset": subset, "building": b,
                                 "model": label, "term": NAMES[c],
                                 "coef": round(float(bb), 4),
                                 "se": round(float(ss), 4),
                                 "t": round(float(bb / ss), 3),
                                 "sigma_dB": round(sd, 3), "n": len(g)})

    print(f"\nt on C2_near in model E")
    print(f"  {'building':9s} {'subset':8s} {'C2_near':>9s} {'t':>7s}"
          f" {'sigma A':>9s} {'sigma E':>9s} {'gain':>7s}")
    for b in BUILDINGS:
        for subset in ("ALL", "NLOS"):
            g = load(subset)
            g = g[g.building == b]
            _, _, sa, _ = fit(g, FEATS_A)
            be, see, se_, _ = fit(g, FEATS_E)
            t = be[-1] / see[-1]
            print(f"  {b:9s} {subset:8s} {be[-1]:9.2f} {t:7.2f}"
                  f" {sa:9.2f} {se_:9.2f} {sa - se_:7.2f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
