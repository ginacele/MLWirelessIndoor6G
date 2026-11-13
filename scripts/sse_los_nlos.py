import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "sse_los_nlos.csv")

BUILDING = "SSE"
MATCH_TOL_M = 0.25
NEAR_M = 4.0
FADING_SD = 5.53


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    s = df[df.building == BUILDING].reset_index(drop=True)
    if not (s["N_total"] == 0).any():
        print(f"WARNING: no zero-obstruction rows for {BUILDING}, "
              f"matched sections will be empty\n")
    return s


def matched_pairs(s, tol=MATCH_TOL_M):
    out = []
    for cfg in sorted(s.config.unique()):
        g = s[s.config == cfg]
        los, nlos = g[g.N_total == 0], g[g.N_total > 0]
        for _, l in los.iterrows():
            near = nlos[(nlos.Distance - l.Distance).abs() <= tol]
            for _, n in near.iterrows():
                out.append({
                    "config": cfg,
                    "los_coord": l["Coord."], "nlos_coord": n["Coord."],
                    "los_d": l.Distance, "nlos_d": n.Distance,
                    "los_prx": l.P_rx, "nlos_prx": n.P_rx,
                    "los_minus_nlos": l.P_rx - n.P_rx,
                })
    return pd.DataFrame(out)


def main():
    s = load()
    rows = []

    mp = matched_pairs(s)
    n_los = int((s.N_total == 0).sum())
    print(f"{BUILDING}: {n_los} LOS points, {len(mp)} with an NLOS point "
          f"within {MATCH_TOL_M} m")
    if len(mp):
        print(f"  {'cfg':4s} {'LOS':6s} {'NLOS':6s} {'d_LOS':>7s} {'d_NLOS':>7s}"
              f" {'P_LOS':>7s} {'P_NLOS':>7s} {'LOS-NLOS':>9s}")
        for _, r in mp.iterrows():
            print(f"  {r.config:4s} {r.los_coord:6s} {r.nlos_coord:6s}"
                  f" {r.los_d:7.3f} {r.nlos_d:7.3f} {r.los_prx:7.0f}"
                  f" {r.nlos_prx:7.0f} {r.los_minus_nlos:+9.0f}")
        rows += mp.assign(section="matched_pairs").to_dict("records")

    los_c, nlos_c = sorted(set(mp.los_coord)), sorted(set(mp.nlos_coord))
    print(f"  distinct coordinates: LOS {los_c}, NLOS {nlos_c}")

    print(f"\nC2 - C1 at identical coordinates, d < {NEAR_M} m")
    piv = s.pivot_table(index="Coord.", columns="config",
                        values=["P_rx", "N_total", "Distance"],
                        aggfunc="first").dropna()
    d = pd.DataFrame({
        "diff": piv[("P_rx", "C2")] - piv[("P_rx", "C1")],
        "N": piv[("N_total", "C1")],
        "dist": piv[("Distance", "C1")],
    })
    near = d[d.dist < NEAR_M].sort_values("diff")
    for c, r in near.iterrows():
        mark = ""
        if len(los_c) and c == los_c[0]:
            mark = "   <- LOS point in the comparison"
        if len(nlos_c) and c == nlos_c[0]:
            mark = "   <- NLOS point in the comparison"
        print(f"  {c:6s} N={int(r.N)}  d={r.dist:5.2f} m  "
              f"C2-C1 = {r['diff']:+5.0f} dB{mark}")
    print(f"  range {near['diff'].min():+.0f} to {near['diff'].max():+.0f} dB,"
          f" sd {near['diff'].std(ddof=1):.1f} dB")
    rows += [{"section": "c2_minus_c1", "coord": c, "n_total": int(r.N),
              "distance_m": r.dist, "c2_minus_c1_dB": r["diff"]}
             for c, r in near.iterrows()]

    print(f"\nobstruction count vs distance in that window")
    print(f"  {'N':>3s} {'n':>4s} {'distance range':>18s} {'mean C2-C1':>12s}")
    for n in sorted(near.N.unique()):
        g = near[near.N == n]
        print(f"  {int(n):3d} {len(g):4d} {g.dist.min():7.2f}-{g.dist.max():<9.2f}"
              f" {g['diff'].mean():+12.1f}")

    print("\npaired differences, location cancels")
    for label, data in (("all SSE coordinates", d),
                        (f"d < {NEAR_M} m only", near)):
        print(f"  {label}  n={len(data)}")
        for cols in (["dist"], ["dist", "N"]):
            X = sm.add_constant(data[cols].astype(float))
            m = sm.OLS(data["diff"].astype(float), X).fit()
            line = f"    {str(cols):14s} R2={m.rsquared:.3f}"
            if "N" in cols:
                lo, hi = m.conf_int().loc["N"]
                line += (f"   N coef {m.params['N']:+6.2f} dB"
                         f"  CI [{lo:+.2f}, {hi:+.2f}]  p={m.pvalues['N']:.3f}")
                rows.append({"section": "paired_regression", "scope": label,
                             "N_coef_dB": round(float(m.params["N"]), 3),
                             "ci_lo": round(float(lo), 3),
                             "ci_hi": round(float(hi), 3),
                             "p": round(float(m.pvalues["N"]), 4),
                             "n": len(data)})
            print(line)

    g = s[s.N_total <= 1]
    X = sm.add_constant(g[["tenlog10d", "N_total", "config_C2"]].astype(float))
    m = sm.OLS(g.PL_excess, X).fit()
    lo, hi = m.conf_int().loc["N_total"]
    print(f"\nfirst obstruction cost, all {len(g)} links with N <= 1")
    print(f"  {m.params['N_total']:+.2f} dB   CI [{lo:+.2f}, {hi:+.2f}]"
          f"   p={m.pvalues['N_total']:.3f}")
    rows.append({"section": "powered_estimate", "scope": "N<=1, all SSE",
                 "N_coef_dB": round(float(m.params["N_total"]), 3),
                 "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                 "p": round(float(m.pvalues["N_total"]), 4), "n": len(g)})

    sd_pair = FADING_SD * np.sqrt(2)
    print(f"\nscale for a single pair: fading {FADING_SD:.2f} dB per measurement,"
          f" sd of a difference {sd_pair:.2f} dB")
    print(f"  +6 dB is {6/sd_pair:.2f} sd, -4 dB is {4/sd_pair:.2f} sd from zero")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
