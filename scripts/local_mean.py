import os
import itertools
import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.spatial import cKDTree

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "local_mean_results.csv")

SEED = 0
BUILDINGS = ["Comms", "Library", "SSE"]
FEATS = ["tenlog10d", "N_total", "config_C2", "c2_near"]
FEATS_NOCFG = ["tenlog10d", "N_total"]
SPACING = {"Comms": 1.0, "Library": 1.355, "SSE": 1.0}
WINDOW_R = [1.5, 2.5, 4.0]
N_FOLDS = 5
DIST_TOL = 0.25


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["c2_near"] = df["config_C2"] * (df["Distance"] < 5.0).astype(int)
    c = df["Coord."].astype(str)
    letters = c.str.extract(r"^([A-Za-z]+)")[0].str.upper()
    df["gx"] = [sum((ord(ch) - 64) * 26 ** i
                    for i, ch in enumerate(reversed(str(s))))
                for s in letters]
    df["gy"] = c.str.extract(r"(\d+)$")[0].astype(float)
    df["x_m"] = df["gx"] * df["building"].map(SPACING)
    df["y_m"] = df["gy"] * df["building"].map(SPACING)
    return df


def Xc(g, feats):
    return np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float)
                                                for f in feats])


def fit(g, feats, target):
    return la.lstsq(Xc(g, feats), g[target].to_numpy(float), rcond=None)[0]


def rmse(r):
    return float(np.sqrt(np.mean(r ** 2)))


def cell_floor(g, target, tol=DIST_TOL):
    h = g.copy()
    h["dbin"] = (h["Distance"] / tol).round().astype(int)
    v = (h.groupby(["dbin", "N_total", "config_C2", "c2_near"])[target]
         .agg(["size", "std"]).dropna())
    v = v[v["size"] >= 2]
    if v.empty:
        return np.nan
    return float(np.sqrt((v["std"] ** 2 * (v["size"] - 1)).sum()
                         / (v["size"] - 1).sum()))


def fading_estimate(df):
    out = {}
    for b in BUILDINGS:
        g = df[df.building == b]
        piv = g.pivot_table(index="Coord.", columns="config",
                            values=["PL", "Distance", "N_total"],
                            aggfunc="first").dropna()
        if ("PL", "C1") not in piv or len(piv) < 20:
            continue
        same = piv[("N_total", "C1")] == piv[("N_total", "C2")]
        piv = piv[same]
        d = (piv[("PL", "C2")] - piv[("PL", "C1")]).to_numpy(float)
        d = d - np.median(d)
        sd_diff = float(np.std(d, ddof=1))
        out[b] = {"n_pairs": len(d), "sd_diff": sd_diff,
                  "sigma_single": sd_diff / np.sqrt(2.0),
                  "sigma_mean2": sd_diff / 2.0}
    return out


def pair_mean_frame(df):
    frames = []
    for b in BUILDINGS:
        g = df[df.building == b]
        piv = g.pivot_table(
            index="Coord.", columns="config",
            values=["PL", "PL_excess", "Distance", "N_total", "tenlog10d",
                    "log10d", "x_m", "y_m", "gx", "gy"],
            aggfunc="first").dropna()
        if ("PL", "C1") not in piv:
            continue
        piv = piv[piv[("N_total", "C1")] == piv[("N_total", "C2")]]
        if len(piv) < 20:
            continue
        off = float(np.median(piv[("PL", "C2")] - piv[("PL", "C1")]))
        h = pd.DataFrame({
            "building": b,
            "Coord.": piv.index,
            "PL": (piv[("PL", "C1")] + (piv[("PL", "C2")] - off)) / 2.0,
            "PL_excess": (piv[("PL_excess", "C1")]
                          + (piv[("PL_excess", "C2")] - off)) / 2.0,
            "Distance": piv[("Distance", "C1")],
            "N_total": piv[("N_total", "C1")],
            "tenlog10d": piv[("tenlog10d", "C1")],
            "log10d": piv[("log10d", "C1")],
            "x_m": piv[("x_m", "C1")], "y_m": piv[("y_m", "C1")],
            "gx": piv[("gx", "C1")], "gy": piv[("gy", "C1")],
            "config_C2": 0.0, "c2_near": 0.0,
            "config": "PAIRMEAN",
            "grid_letter": [str(i).split("-")[0].upper() for i in piv.index],
            "config_offset_removed": off,
        })
        frames.append(h)
    return pd.concat(frames, ignore_index=True) if frames else None


def window_mean(g, radius, src_mask=None):
    pts = g[["x_m", "y_m"]].to_numpy(float)
    y = g["PL_excess"].to_numpy(float)
    key = (g["N_total"].astype(int).astype(str) + "|"
           + g["config_C2"].astype(int).astype(str)).to_numpy()
    src = np.ones(len(g), bool) if src_mask is None else np.asarray(src_mask)

    tree = cKDTree(pts[src])
    src_y, src_key = y[src], key[src]
    idx_lists = tree.query_ball_point(pts, r=radius)

    out = np.full(len(g), np.nan)
    cnt = np.zeros(len(g), int)
    for i, nb in enumerate(idx_lists):
        if not nb:
            continue
        nb = np.asarray(nb)
        nb = nb[src_key[nb] == key[i]]
        if len(nb) == 0:
            continue
        out[i] = src_y[nb].mean()
        cnt[i] = len(nb)
    return out, cnt


def letter_folds(g, n_folds=N_FOLDS):
    letters = sorted(pd.Series(g["grid_letter"].dropna().unique()))
    rng = np.random.default_rng(SEED)
    letters = list(rng.permutation(letters))
    return [set(letters[i::n_folds]) for i in range(n_folds)]


def blocked_rmse(g, feats, target):
    parts = []
    for fold in letter_folds(g):
        te = g[g["grid_letter"].isin(fold)]
        tr = g[~g["grid_letter"].isin(fold)]
        if len(te) < 5 or len(tr) < 40:
            continue
        if min(tr[f].std() for f in feats) <= 1e-12:
            continue
        beta = fit(tr, feats, target)
        parts.append(te[target].to_numpy(float) - Xc(te, feats) @ beta)
    return rmse(np.concatenate(parts)) if parts else np.nan


def blocked_rmse_window(g, feats, radius):
    parts = []
    for fold in letter_folds(g):
        temask = g["grid_letter"].isin(fold).to_numpy()
        tgt, cnt = window_mean(g, radius, src_mask=~temask)
        h = g.copy()
        h["_win"] = tgt
        ok = h["_win"].notna().to_numpy()
        te = h[temask & ok]
        tr = h[(~temask) & ok]
        if len(te) < 5 or len(tr) < 40:
            continue
        if min(tr[f].std() for f in feats) <= 1e-12:
            continue
        beta = fit(tr, feats, "_win")
        parts.append(te["_win"].to_numpy(float) - Xc(te, feats) @ beta)
    return rmse(np.concatenate(parts)) if parts else np.nan


def main():
    df = load()
    print(f"n={len(df)}  seed={SEED}")
    rows = []

    print("\nfading from the C1/C2 pairing, identical obstruction counts")
    fe = fading_estimate(df)
    print(f"  {'building':10s} {'pairs':>7s} {'sd(C2-C1)':>10s}"
          f" {'sigma single':>13s} {'sigma mean-of-2':>16s}")
    for b, v in fe.items():
        print(f"  {b:10s} {v['n_pairs']:7d} {v['sd_diff']:10.2f}"
              f" {v['sigma_single']:13.2f} {v['sigma_mean2']:16.2f}")
        rows.append({"section": "fading", "scope": b, "target": "-",
                     "metric": "sigma_single",
                     "value": round(v["sigma_single"], 3)})

    print("\nwithin-feature-cell sd by target")
    pm = pair_mean_frame(df)
    print(f"  {'building':10s} {'raw PL':>9s} {'pair mean':>11s}"
          + "".join(f"{'win R=' + str(r):>11s}" for r in WINDOW_R))
    floors = {}
    for b in BUILDINGS:
        g = df[df.building == b]
        f_raw = cell_floor(g, "PL_excess")
        line = f"  {b:10s} {f_raw:9.2f}"
        floors[(b, "raw")] = f_raw
        if pm is not None and (pm.building == b).any():
            h = pm[pm.building == b]
            f_pm = cell_floor(h, "PL_excess")
            floors[(b, "pair")] = f_pm
            line += f" {f_pm:11.2f}"
        else:
            line += f" {'-':>11s}"
        for r in WINDOW_R:
            tgt, cnt = window_mean(g, r)
            h = g.copy()
            h["_win"] = tgt
            h = h[h["_win"].notna()]
            f_w = cell_floor(h, "_win") if len(h) > 20 else np.nan
            floors[(b, f"win{r}")] = f_w
            line += f" {f_w:11.2f}"
        print(line)
        for k, v in list(floors.items()):
            if k[0] == b:
                rows.append({"section": "floor", "scope": b, "target": k[1],
                             "metric": "cell_sd",
                             "value": round(v, 3) if v == v else np.nan})

    print("\nmean window membership")
    for b in BUILDINGS:
        g = df[df.building == b]
        s = []
        for r in WINDOW_R:
            _, cnt = window_mean(g, r)
            s.append(f"R={r}: {cnt[cnt > 0].mean():.1f} pts, "
                     f"{100*(cnt > 1).mean():.0f}% have >1")
        print(f"  {b:10s} " + " | ".join(s))
    if pm is not None:
        print("  pair-mean rows: "
              + ", ".join(f"{b} {int((pm.building==b).sum())}"
                          for b in BUILDINGS))

    print("\nmodel vs its own floor, spatially blocked CV   RMSE dB")
    print(f"  {'building':10s} {'target':11s} {'RMSE':>7s} {'floor':>7s}"
          f" {'ratio':>7s}")
    for b in BUILDINGS:
        g = df[df.building == b]
        r_raw = blocked_rmse(g, FEATS, "PL_excess")
        f_raw = floors[(b, "raw")]
        print(f"  {b:10s} {'raw PL':11s} {r_raw:7.2f} {f_raw:7.2f}"
              f" {r_raw/f_raw:7.2f}")
        rows.append({"section": "model", "scope": b, "target": "raw",
                     "metric": "blocked_rmse", "value": round(r_raw, 3)})

        if pm is not None and (pm.building == b).any():
            h = pm[pm.building == b]
            r_pm = blocked_rmse(h, FEATS_NOCFG, "PL_excess")
            f_pm = floors.get((b, "pair"), np.nan)
            print(f"  {b:10s} {'pair mean':11s} {r_pm:7.2f} {f_pm:7.2f}"
                  f" {r_pm/f_pm:7.2f}")
            rows.append({"section": "model", "scope": b, "target": "pair",
                         "metric": "blocked_rmse", "value": round(r_pm, 3)})

        for rad in WINDOW_R:
            r_w = blocked_rmse_window(g, FEATS, rad)
            f_w = floors.get((b, f"win{rad}"), np.nan)
            print(f"  {b:10s} {'win R=' + str(rad):11s} {r_w:7.2f}"
                  f" {f_w:7.2f} {r_w/f_w:7.2f}")
            rows.append({"section": "model", "scope": b,
                         "target": f"win{rad}", "metric": "blocked_rmse",
                         "value": round(r_w, 3)})

    print("\ncross-building transfer by target   blind RMSE dB")
    print(f"  {'train':10s} {'test':10s} {'raw PL':>9s} {'pair mean':>11s}"
          f" {'win R=2.5':>11s}")
    win = {}
    for b in BUILDINGS:
        g = df[df.building == b].copy()
        tgt, _ = window_mean(g, 2.5)
        g["_win"] = tgt
        win[b] = g[g["_win"].notna()]
    for a, b in itertools.permutations(BUILDINGS, 2):
        r_raw = rmse(df[df.building == b]["PL_excess"].to_numpy(float)
                     - Xc(df[df.building == b], FEATS)
                     @ fit(df[df.building == a], FEATS, "PL_excess"))
        line = f"  {a:10s} {b:10s} {r_raw:9.2f}"
        if pm is not None and (pm.building == a).any() \
                and (pm.building == b).any():
            ta, tb = pm[pm.building == a], pm[pm.building == b]
            r_pm = rmse(tb["PL_excess"].to_numpy(float)
                        - Xc(tb, FEATS_NOCFG)
                        @ fit(ta, FEATS_NOCFG, "PL_excess"))
            line += f" {r_pm:11.2f}"
        else:
            line += f" {'-':>11s}"
        wa, wb = win[a], win[b]
        r_w = rmse(wb["_win"].to_numpy(float)
                   - Xc(wb, FEATS) @ fit(wa, FEATS, "_win"))
        line += f" {r_w:11.2f}"
        print(line)
        rows.append({"section": "transfer", "scope": f"{a}->{b}",
                     "target": "raw", "metric": "blind_rmse",
                     "value": round(r_raw, 2)})
        rows.append({"section": "transfer", "scope": f"{a}->{b}",
                     "target": "win2.5", "metric": "blind_rmse",
                     "value": round(r_w, 2)})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
