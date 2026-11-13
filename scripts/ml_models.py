import os
import itertools
import numpy as np
import pandas as pd
import numpy.linalg as la
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "ml_results.csv")

SEED = 0
BUILDINGS = ["Comms", "Library", "SSE"]
FEATS = ["tenlog10d", "N_total", "config_C2", "c2_near"]

K_VALUES = [3, 10, 20]
N_REPEATS = 50
MIN_SPACING_M = 3.0

WITHIN_SIGMA = 6.58


def gbm():
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=SEED)


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["c2_near"] = df["config_C2"] * (df["Distance"] < 5.0).astype(int)
    return df


def X(g):
    return g[FEATS].to_numpy(float)


def Xc(g):
    return np.column_stack([np.ones(len(g)), X(g)])


def fit_param(g):
    return la.lstsq(Xc(g), g["PL_excess"].to_numpy(float), rcond=None)[0]


def train_all(tr):
    y = tr["PL_excess"].to_numpy(float)
    beta = fit_param(tr)

    resid = y - Xc(tr) @ beta
    gh = gbm().fit(X(tr), resid)

    gp = gbm().fit(X(tr), y)

    return {
        "PARAM": lambda g: Xc(g) @ beta,
        "HYBRID": lambda g: Xc(g) @ beta + gh.predict(X(g)),
        "PURE": lambda g: gp.predict(X(g)),
    }


def stats(resid):
    return (float(np.sqrt(np.mean(resid ** 2))),
            float(np.mean(resid)),
            float(np.std(resid, ddof=1)))


def spaced_idx(dist, k, rng):
    order = rng.permutation(len(dist))
    picked = []
    for i in order:
        if all(abs(dist[i] - dist[j]) >= MIN_SPACING_M for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return np.array(picked, dtype=int)


def row(model, evaluation, train, test, cond, k, resid, repeat=0):
    r, b, s = stats(resid)
    return (model, evaluation, train, test, cond, k, repeat,
            r, b, s, int(len(resid)))


def run_within(df, rows):
    for b in BUILDINGS:
        g = df[df.building == b]
        letters = sorted(g["grid_letter"].dropna().unique())
        if len(letters) < 3:
            continue
        acc = {m: [] for m in ("PARAM", "HYBRID", "PURE")}
        for L in letters:
            te = g[g.grid_letter == L]
            tr = g[g.grid_letter != L]
            if len(te) < 5 or len(tr) < 50:
                continue
            if min(tr[f].std() for f in FEATS) <= 1e-12:
                continue
            models = train_all(tr)
            yte = te["PL_excess"].to_numpy(float)
            for m, pred in models.items():
                acc[m].append(yte - pred(te))
        for m, parts in acc.items():
            if parts:
                rows.append(row(m, "WITHIN", b, b, "blocked_cv", 0,
                                np.concatenate(parts)))


def run_transfer(df, rows):
    folds = [("LOBO", [x for x in BUILDINGS if x != t], t) for t in BUILDINGS]
    folds += [("PAIR", [a], b) for a, b in itertools.permutations(BUILDINGS, 2)]

    for dsg, train_b, test_b in folds:
        tr = df[df.building.isin(train_b)]
        te = df[df.building == test_b]
        if min(tr[f].std() for f in FEATS) <= 1e-12:
            continue
        models = train_all(tr)
        tname = "+".join(train_b)
        yte = te["PL_excess"].to_numpy(float)
        dist = te["Distance"].to_numpy(float)
        n = len(te)

        for m, pred in models.items():
            base = yte - pred(te)
            rows.append(row(m, dsg, tname, test_b, "blind", 0, base))

            rng = np.random.default_rng(SEED)
            for k in K_VALUES:
                for repn in range(N_REPEATS):
                    cal = spaced_idx(dist, k, rng)
                    if len(cal) < min(k, 3):
                        continue
                    mask = np.ones(n, dtype=bool)
                    mask[cal] = False
                    if mask.sum() < 10:
                        continue
                    rows.append(row(m, dsg, tname, test_b, "kshot_intercept",
                                    k, base[mask] - base[cal].mean(), repn))


def main():
    df = load()
    print(f"n={len(df)}  seed={SEED}  repeats={N_REPEATS}")
    print(f"features: {', '.join(FEATS)}")
    print(f"within_sigma={WITHIN_SIGMA} dB")

    rows = []
    run_within(df, rows)
    run_transfer(df, rows)

    res = pd.DataFrame(rows, columns=["model", "evaluation", "train", "test",
                                      "condition", "k", "repeat", "rmse_dB",
                                      "bias_dB", "sigma_dB", "n_test"])
    res.to_csv(OUT, index=False)

    def m_of(g, cond, k):
        s = g[(g.condition == cond) & (g.k == k)]
        return s.rmse_dB.median() if len(s) else np.nan

    print("\nwithin building, spatial-block CV   RMSE dB")
    w = res[res.evaluation == "WITHIN"]
    print(f"  {'building':10s} {'PARAM':>9s} {'HYBRID':>9s} {'PURE':>9s}"
          f"   {'best':>8s}")
    for b in BUILDINGS:
        g = w[w.test == b]
        if g.empty:
            continue
        v = {m: g[g.model == m].rmse_dB.iloc[0] for m in g.model.unique()}
        best = min(v, key=v.get)
        print(f"  {b:10s} " + " ".join(f"{v.get(m, np.nan):9.2f}"
                                       for m in ("PARAM", "HYBRID", "PURE"))
              + f"   {best:>8s}")

    for dsg in ("LOBO", "PAIR"):
        print(f"\n{dsg} transfer   RMSE dB, median over repeats")
        sub = res[res.evaluation == dsg]
        print(f"  {'train':16s} {'test':9s} {'model':7s}"
              f" {'blind':>8s} {'bias':>8s} {'k=3':>7s} {'k=10':>7s} {'k=20':>7s}")
        for (tr, te), gg in sub.groupby(["train", "test"], sort=False):
            for m in ("PARAM", "HYBRID", "PURE"):
                g = gg[gg.model == m]
                if g.empty:
                    continue
                bl = g[g.condition == "blind"].iloc[0]
                print(f"  {tr:16s} {te:9s} {m:7s}"
                      f" {bl.rmse_dB:8.2f} {bl.bias_dB:+8.2f}"
                      f" {m_of(g,'kshot_intercept',3):7.2f}"
                      f" {m_of(g,'kshot_intercept',10):7.2f}"
                      f" {m_of(g,'kshot_intercept',20):7.2f}")

    wv = {b: {m: w[(w.test == b) & (w.model == m)].rmse_dB.iloc[0]
              for m in ("PARAM", "HYBRID", "PURE")}
          for b in BUILDINGS if not w[w.test == b].empty}

    print("\nM1  PURE vs PARAM within building")
    for b, v in wv.items():
        print(f"  {b:10s} PARAM {v['PARAM']:6.2f}  PURE {v['PURE']:6.2f}"
              f"  {v['PURE']-v['PARAM']:+6.2f} dB")
    t = res[res.evaluation == "PAIR"]
    for m in ("PARAM", "HYBRID", "PURE"):
        s = t[(t.model == m) & (t.condition == "kshot_intercept") & (t.k == 20)]
        print(f"  transfer k=20 median, {m:7s} {s.rmse_dB.median():.2f} dB")

    print("\nM2  HYBRID never worse than PARAM")
    bad = []
    for dsg in ("WITHIN", "LOBO", "PAIR"):
        s = res[res.evaluation == dsg]
        for (tr, te), g in s.groupby(["train", "test"]):
            p = g[(g.model == "PARAM") & (g.condition != "kshot_intercept")]
            h = g[(g.model == "HYBRID") & (g.condition != "kshot_intercept")]
            if len(p) and len(h) and h.rmse_dB.iloc[0] > p.rmse_dB.iloc[0] + 0.05:
                bad.append(f"{dsg} {tr}->{te}: "
                           f"{p.rmse_dB.iloc[0]:.2f} -> {h.rmse_dB.iloc[0]:.2f}")
    print("  " + ("holds everywhere" if not bad else "fails in: " + "; ".join(bad)))

    print("\nM3  blind PAIR transfer, PARAM vs PURE")
    s = res[(res.evaluation == "PAIR") & (res.condition == "blind")]
    for (tr, te), g in s.groupby(["train", "test"]):
        v = {m: g[g.model == m].rmse_dB.iloc[0] for m in g.model.unique()}
        gap = v.get("PURE", np.nan) - v.get("PARAM", np.nan)
        print(f"  {tr:16s} {te:9s} PARAM {v.get('PARAM',np.nan):6.2f}"
              f"  PURE {v.get('PURE',np.nan):6.2f}  gap {gap:+6.2f} dB")

    print("\nM4  HYBRID vs PARAM within building")
    for b, v in wv.items():
        print(f"  {b:10s} PARAM {v['PARAM']:6.2f}  HYBRID {v['HYBRID']:6.2f}"
              f"  {v['HYBRID']-v['PARAM']:+6.2f} dB")

    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
