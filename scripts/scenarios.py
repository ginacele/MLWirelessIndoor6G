import os
import numpy as np
import numpy.linalg as la
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")

OUT_METRICS = os.path.join(DATA_DIR, "scenario_metrics.csv")
OUT_COEFS = os.path.join(DATA_DIR, "scenario_coefficients.csv")
OUT_RESID = os.path.join(DATA_DIR, "scenario_residuals.csv")

SEED = 0
K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30]
N_REPEATS = 200
MIN_SPACING_M = 3.0
SPACING_MODE = "2d"
WRITE_RESIDUALS = True

SPACING_M = {"Comms": 1.0, "Library": 1.355, "SSE": 1.0}

NEAR_M = 5.0
ANCHOR_DB = 33.3

REFERENCE_LEVELS = {
    "within_building_floor": 6.58,
    "pooled_shared_offset": 7.74,
    "fading_per_measurement": 5.2,
    "floor_Comms": 6.15,
    "floor_Library": 5.72,
    "floor_SSE": 4.88,
}

SPECS = {
    "A": ["tenlog10d", "N_total", "config_C2"],
    "E": ["tenlog10d", "N_total", "config_C2", "c2_near"],
}

SCENARIOS = [
    {"name": "S1", "train": "Comms",   "validate": "SSE",     "test": "Library"},
    {"name": "S2", "train": "SSE",     "validate": "Library", "test": "Comms"},
    {"name": "S3", "train": "Library", "validate": "Comms",   "test": "SSE"},
]


def load(path=POOLED):
    df = pd.read_csv(path)
    df["tenlog10d"] = 10.0 * df["log10d"]
    df["c2_near"] = df["config_C2"] * (df["Distance"] < NEAR_M).astype(int)

    c = df["Coord."].astype(str)
    letters = c.str.extract(r"^([A-Za-z]+)")[0].str.upper()
    df["gx"] = [sum((ord(ch) - 64) * 26 ** i
                    for i, ch in enumerate(reversed(str(s))))
                for s in letters]
    df["gy"] = c.str.extract(r"(\d+)$")[0].astype(float)
    df["x_m"] = df["gx"] * df["building"].map(SPACING_M)
    df["y_m"] = df["gy"] * df["building"].map(SPACING_M)
    return df


def design(g, feats):
    return np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float)
                                                for f in feats])


def fit(g, feats):
    return la.lstsq(design(g, feats), g["PL_excess"].to_numpy(float),
                    rcond=None)[0]


def spaced_idx(pos, k, rng, min_spacing=MIN_SPACING_M):
    picked = []
    for i in rng.permutation(len(pos)):
        if all(np.linalg.norm(pos[i] - pos[j]) >= min_spacing for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return np.array(picked, dtype=int)


def positions(te):
    if SPACING_MODE == "2d":
        return te[["x_m", "y_m"]].to_numpy(float)
    return te[["Distance"]].to_numpy(float)


def metrics(resid):
    r = np.asarray(resid, dtype=float)
    return {
        "mse": float(np.mean(r ** 2)),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "bias": float(np.mean(r)),
        "sd": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
        "mae": float(np.mean(np.abs(r))),
        "p90": float(np.percentile(np.abs(r), 90)),
    }


def evaluate_fold(beta, te, feats, spec, scenario, train, role,
                  k_values=K_VALUES, n_repeats=N_REPEATS, seed=SEED):
    y = te["PL_excess"].to_numpy(float)
    base = y - design(te, feats) @ beta
    pos = positions(te)
    n = len(te)

    tag = {"spec": spec, "scenario": scenario, "train": train,
           "role": role, "building": te["building"].iloc[0]}

    rows = [{**tag, "condition": "blind", "k": 0, "repeat": 0,
             "n_test": n, "n_cal": 0, **metrics(base)}]

    rng = np.random.default_rng(seed)
    for k in k_values:
        for rep in range(n_repeats):
            cal = spaced_idx(pos, k, rng)
            if len(cal) == 0:
                continue
            mask = np.ones(n, dtype=bool)
            mask[cal] = False
            if mask.sum() < 10:
                continue
            r = base[mask] - base[cal].mean()
            rows.append({**tag, "condition": "kshot_intercept", "k": k,
                         "repeat": rep, "n_test": int(mask.sum()),
                         "n_cal": len(cal), **metrics(r)})
    return rows


def residual_rows(beta, te, feats, spec, scenario, role):
    pred = design(te, feats) @ beta
    obs = te["PL_excess"].to_numpy(float)
    return pd.DataFrame({
        "spec": spec, "scenario": scenario, "role": role,
        "building": te["building"].to_numpy(),
        "coord": te["Coord."].to_numpy(),
        "distance": te["Distance"].to_numpy(),
        "n_total": te["N_total"].to_numpy(),
        "config": te["config"].to_numpy(),
        "observed": obs + ANCHOR_DB,
        "predicted": pred + ANCHOR_DB,
        "residual": obs - pred,
    })


def run_all(df, specs=SPECS, scenarios=SCENARIOS, **kw):
    met, coefs, resid = [], [], []
    for spec, feats in specs.items():
        for sc in scenarios:
            tr = df[df.building == sc["train"]]
            beta = fit(tr, feats)

            for term, value in zip(["offset"] + feats, beta):
                coefs.append({
                    "spec": spec, "scenario": sc["name"], "train": sc["train"],
                    "n_train": len(tr),
                    "d_min": float(tr.Distance.min()),
                    "d_max": float(tr.Distance.max()),
                    "term": term, "value": float(value),
                })

            for role in ("validate", "test"):
                te = df[df.building == sc[role]]
                met += evaluate_fold(beta, te, feats, spec, sc["name"],
                                     sc["train"], role, **kw)
                if WRITE_RESIDUALS:
                    resid.append(residual_rows(beta, te, feats, spec,
                                               sc["name"], role))

    return (pd.DataFrame(met),
            pd.DataFrame(coefs),
            pd.concat(resid, ignore_index=True) if resid else pd.DataFrame())


def print_summary(met, coefs, spec="E"):
    fl = REFERENCE_LEVELS["within_building_floor"]
    e = met[met.spec == spec]

    print(f"\nspec {spec}  median over {N_REPEATS} draws")
    ks = [k for k in (1, 3, 5, 10, 20) if k in K_VALUES]
    for sc in SCENARIOS:
        c = coefs[(coefs.spec == spec) & (coefs.scenario == sc["name"])]
        g = dict(zip(c.term, c.value))
        n_tr = int(c.n_train.iloc[0])
        print(f"\n  {sc['name']}  train {sc['train']}  n={n_tr}  "
              f"d {c.d_min.iloc[0]:.1f}-{c.d_max.iloc[0]:.1f} m  "
              f"n_hat {g.get('tenlog10d', np.nan):.3f}  "
              f"offset {g.get('offset', np.nan):+.2f} dB  "
              f"L_total {g.get('N_total', np.nan):+.2f} dB")
        print(f"       {'role':9s} {'building':9s} {'n':>5s} {'blind':>7s}"
              f" {'bias':>8s} {'sd':>7s}"
              + "".join(f"{'k=' + str(k):>7s}" for k in ks))
        for role in ("validate", "test"):
            s = e[(e.scenario == sc["name"]) & (e.role == role)]
            b = s[s.condition == "blind"].iloc[0]
            cells = "".join(
                f"{s[(s.condition=='kshot_intercept') & (s.k==k)].rmse.median():7.2f}"
                for k in ks)
            print(f"       {role:9s} {b.building:9s} {b.n_test:5d}"
                  f" {b.rmse:7.2f} {b.bias:+8.2f} {b.sd:7.2f}" + cells)

    print(f"\nmedian n_cal placed against requested k  "
          f"(mode {SPACING_MODE!r}, min {MIN_SPACING_M} m)")
    s = met[met.condition == "kshot_intercept"]
    piv = s.pivot_table(index="building", columns="k", values="n_cal",
                        aggfunc="median")
    print("   " + piv.to_string().replace("\n", "\n   "))
    capped = s[s.n_cal < s.k]
    if len(capped):
        worst = capped.groupby("building")["n_cal"].max()
        print("   spacing binds above: "
              + ", ".join(f"{b} {int(v)}" for b, v in worst.items()))
    else:
        print("   spacing never binds, n_cal == k everywhere")

    kmax = max(K_VALUES)
    print(f"\ntraining building ranked by transfer, rms over both held-out "
          f"folds at k={kmax}   (floor {fl})")
    s = e[(e.condition == "kshot_intercept") & (e.k == kmax)]
    agg = (s.groupby("train")["mse"].mean() ** 0.5).sort_values()
    for b, v in agg.items():
        print(f"   {b:9s} {v:.2f} dB")

    print(f"\nspec comparison, rms of mse over all six held-out folds at k={kmax}")
    for sp in SPECS:
        s = met[(met.spec == sp) & (met.condition == "kshot_intercept")
                & (met.k == kmax)]
        print(f"   {sp}: {s.mse.mean() ** 0.5:.2f} dB")


def main():
    df = load()
    print(f"n={len(df)}  seed={SEED}  repeats={N_REPEATS}  k={K_VALUES}  "
          f"min_spacing={MIN_SPACING_M} m")

    met, coefs, resid = run_all(df)

    met.to_csv(OUT_METRICS, index=False)
    coefs.to_csv(OUT_COEFS, index=False)
    if WRITE_RESIDUALS and len(resid):
        resid.to_csv(OUT_RESID, index=False)

    print_summary(met, coefs)

    print(f"\nWrote {os.path.normpath(OUT_METRICS)}  {len(met)} rows")
    print(f"Wrote {os.path.normpath(OUT_COEFS)}  {len(coefs)} rows")
    if WRITE_RESIDUALS and len(resid):
        print(f"Wrote {os.path.normpath(OUT_RESID)}  {len(resid)} rows")


if __name__ == "__main__":
    main()
