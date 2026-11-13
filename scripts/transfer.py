import os
import itertools
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "transfer_results.csv")

SEED = 0
BUILDINGS = ["Comms", "Library", "SSE"]

SPECS = {
    "A": ["tenlog10d", "N_total", "config_C2"],
    "E": ["tenlog10d", "N_total", "config_C2", "c2_near"],
}

K_VALUES = [1, 3, 5, 10, 20]
N_REPEATS = 200
MIN_SPACING_M = 3.0

WITHIN_SIGMA = 6.58
POOLED_SIGMA = 7.74


def load():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["c2_near"] = df["config_C2"] * (df["Distance"] < 5.0).astype(int)
    return df


def design(g, feats):
    return np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float)
                                                for f in feats])


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def spaced_idx(dist, k, rng):
    order = rng.permutation(len(dist))
    picked = []
    for i in order:
        d = dist[i]
        if all(abs(d - dist[j]) >= MIN_SPACING_M for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return np.array(picked, dtype=int)


def rec(spec, dsg, train, test, cond, k, resid, repeat=0):
    return (spec, dsg, train, test, cond, k, repeat,
            float(np.sqrt(np.mean(resid ** 2))),
            float(np.mean(resid)),
            float(np.std(resid, ddof=1)),
            int(len(resid)))


def run_fold(df, spec, feats, dsg, train_b, test_b, rows):
    tr = df[df.building.isin(train_b)]
    te = df[df.building == test_b]
    if min(tr[f].std() for f in feats) <= 1e-12:
        return
    beta = ols(design(tr, feats), tr["PL_excess"].to_numpy(float))

    Xte, yte = design(te, feats), te["PL_excess"].to_numpy(float)
    base = yte - Xte @ beta
    tname = "+".join(train_b)
    rows.append(rec(spec, dsg, tname, test_b, "blind", 0, base))

    dist = te["Distance"].to_numpy(float)
    tlog = te["tenlog10d"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    n = len(te)

    for k in K_VALUES:
        for repn in range(N_REPEATS):
            cal = spaced_idx(dist, k, rng)
            if len(cal) < min(k, 3):
                continue
            mask = np.ones(n, dtype=bool)
            mask[cal] = False
            if mask.sum() < 10:
                continue

            shift = base[cal].mean()
            rows.append(rec(spec, dsg, tname, test_b, "kshot_intercept", k,
                            base[mask] - shift, repn))

            if len(cal) >= 4 and tlog[cal].std() > 1e-9:
                Xc = np.column_stack([np.ones(len(cal)), tlog[cal]])
                b2 = ols(Xc, yte[cal])
                pred = b2[0] + b2[1] * tlog[mask]
                rows.append(rec(spec, dsg, tname, test_b, "kshot_both", k,
                                yte[mask] - pred, repn))


def main():
    df = load()
    print(f"n={len(df)}  seed={SEED}  repeats={N_REPEATS}  "
          f"min_spacing={MIN_SPACING_M} m")
    print(f"within_sigma={WITHIN_SIGMA} dB  pooled_sigma={POOLED_SIGMA} dB")

    rows = []
    for spec, feats in SPECS.items():
        for test_b in BUILDINGS:
            run_fold(df, spec, feats, "LOBO",
                     [b for b in BUILDINGS if b != test_b], test_b, rows)
        for a, b in itertools.permutations(BUILDINGS, 2):
            run_fold(df, spec, feats, "PAIR", [a], b, rows)

    res = pd.DataFrame(rows, columns=["spec", "design", "train", "test",
                                      "condition", "k", "repeat", "rmse_dB",
                                      "bias_dB", "sigma_dB", "n_test"])
    res.to_csv(OUT, index=False)

    def med(g, cond, k):
        s = g[(g.condition == cond) & (g.k == k)]
        return s.rmse_dB.median() if len(s) else np.nan

    for spec in SPECS:
        for dsg in ("LOBO", "PAIR"):
            sub = res[(res.spec == spec) & (res.design == dsg)]
            if sub.empty:
                continue
            print(f"\nspec {spec}  {dsg}  RMSE dB, median over repeats")
            print(f"  {'train':16s} {'test':9s} {'blind':>8s} {'bias':>8s}"
                  f" {'k=1':>7s} {'k=5':>7s} {'k=10':>7s} {'k=20':>7s}"
                  f" {'k=20 both':>10s}")
            for (tr, te), g in sub.groupby(["train", "test"], sort=False):
                bl = g[g.condition == "blind"].iloc[0]
                print(f"  {tr:16s} {te:9s} {bl.rmse_dB:8.2f} "
                      f"{bl.bias_dB:+8.2f}"
                      f" {med(g,'kshot_intercept',1):7.2f}"
                      f" {med(g,'kshot_intercept',5):7.2f}"
                      f" {med(g,'kshot_intercept',10):7.2f}"
                      f" {med(g,'kshot_intercept',20):7.2f}"
                      f" {med(g,'kshot_both',20):10.2f}")

    e = res[res.spec == "E"]
    bl = e[(e.condition == "blind") & (e.design == "PAIR")]
    print(f"\nspec E  blind |bias| over the 6 pairs: "
          f"min {bl.bias_dB.abs().min():.2f}  "
          f"median {bl.bias_dB.abs().median():.2f}  "
          f"max {bl.bias_dB.abs().max():.2f} dB")
    for _, r in bl.iterrows():
        print(f"  {r.train:16s} {r.test:9s} {r.bias_dB:+7.2f}")

    print(f"\nspec E  k-shot progression, median RMSE dB")
    for dsg in ("LOBO", "PAIR"):
        s = e[e.design == dsg]
        b = s[s.condition == "blind"].rmse_dB.median()
        line = "  ".join(f"k={k} {med(s,'kshot_intercept',k):.2f}"
                         for k in K_VALUES)
        print(f"  {dsg:6s} blind {b:.2f}  {line}")

    print(f"\nspec E  PAIR k=20 intercept-corrected RMSE dB")
    s = e[(e.design == "PAIR") & (e.condition == "kshot_intercept")
          & (e.k == 20)]
    for (tr, te), g in s.groupby(["train", "test"]):
        print(f"  {tr:16s} {te:9s} {g.rmse_dB.median():7.2f}")

    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
