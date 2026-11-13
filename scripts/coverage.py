import os
import itertools
import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.stats import norm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
POOLED = os.path.join(DATA_DIR, "pooled.csv")
OUT = os.path.join(DATA_DIR, "coverage_results.csv")

THRESHOLDS_DBM = [-60.0, -65.0, -70.0]
RELIABILITY = [0.90, 0.95, 0.99]
N_CAL_POINTS = 10
N_REPEATS = 200
MIN_SPACING_M = 3.0
SEED = 0

PTX_DBM = 10.0
ANCHOR_DB = 33.3
NEAR_M = 5.0
SPACING_M = {"Comms": 1.0, "Library": 1.355, "SSE": 1.0}

FEATS = ["tenlog10d", "N_total", "config_C2", "c2_near"]
BUILDINGS = ["Comms", "Library", "SSE"]


def load():
    df = pd.read_csv(POOLED)
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


def X(g):
    return np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float)
                                                for f in FEATS])


def fit(g):
    return la.lstsq(X(g), g["PL_excess"].to_numpy(float), rcond=None)[0]


def predicted_prx(beta, g):
    return PTX_DBM - ANCHOR_DB - (X(g) @ beta)


def spaced_idx(pos, k, rng, min_spacing=MIN_SPACING_M):
    picked = []
    for i in rng.permutation(len(pos)):
        if all(np.linalg.norm(pos[i] - pos[j]) >= min_spacing for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return np.array(picked, dtype=int)


def coverage_metrics(mu, sigma, measured, T):
    p_hat = norm.sf((T - mu) / sigma)
    obs = measured > T
    pred_hard = mu > T

    tp = int((pred_hard & obs).sum())
    fp = int((pred_hard & ~obs).sum())
    fn = int((~pred_hard & obs).sum())
    tn = int((~pred_hard & ~obs).sum())

    return {
        "threshold_dBm": T,
        "n": len(mu),
        "coverage_measured": float(obs.mean()),
        "coverage_predicted_soft": float(p_hat.mean()),
        "coverage_predicted_hard": float(pred_hard.mean()),
        "coverage_error": float(p_hat.mean() - obs.mean()),
        "accuracy": float((pred_hard == obs).mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "brier": float(np.mean((p_hat - obs.astype(float)) ** 2)),
        "sigma_used": float(sigma),
    }


def calibration_table(mu, sigma, measured, T, nbins=5):
    p_hat = norm.sf((T - mu) / sigma)
    obs = (measured > T).astype(float)
    edges = np.linspace(0, 1, nbins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_hat >= lo) & (p_hat < hi if hi < 1 else p_hat <= 1)
        if m.sum() < 5:
            continue
        out.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()),
                    float(p_hat[m].mean()), float(obs[m].mean())))
    return out


def fade_margins(resid, sigma, reliability):
    out = {}
    for p in reliability:
        out[f"gauss_{int(p*100)}"] = float(sigma * norm.ppf(p))
        out[f"emp_{int(p*100)}"] = float(np.percentile(resid, p * 100))
    return out


def main():
    df = load()
    print(f"n={len(df)}  thresholds {THRESHOLDS_DBM} dBm  seed={SEED}")
    rows = []

    print("\nwithin building")
    print(f"  {'building':9s} {'T':>6s} {'measured':>9s} {'Q-function':>11s}"
          f" {'hard':>7s} {'error':>8s} {'accuracy':>9s} {'Brier':>7s}")
    for b in BUILDINGS:
        g = df[df.building == b]
        beta = fit(g)
        mu = predicted_prx(beta, g)
        resid = g["PL_excess"].to_numpy(float) - X(g) @ beta
        sigma = float(np.std(resid, ddof=1))
        meas = g["P_rx"].to_numpy(float)
        for T in THRESHOLDS_DBM:
            m = coverage_metrics(mu, sigma, meas, T)
            rows.append({"section": "within", "building": b,
                         "condition": "fitted", **m})
            print(f"  {b:9s} {T:6.0f} {m['coverage_measured']:9.3f}"
                  f" {m['coverage_predicted_soft']:11.3f}"
                  f" {m['coverage_predicted_hard']:7.3f}"
                  f" {m['coverage_error']:+8.3f}"
                  f" {m['accuracy']:9.3f} {m['brier']:7.3f}")

    print("\ncalibration at -70 dBm")
    for b in BUILDINGS:
        g = df[df.building == b]
        beta = fit(g)
        mu = predicted_prx(beta, g)
        resid = g["PL_excess"].to_numpy(float) - X(g) @ beta
        sigma = float(np.std(resid, ddof=1))
        tab = calibration_table(mu, sigma, g["P_rx"].to_numpy(float), -70.0)
        print(f"  {b}")
        print(f"    {'bin':10s} {'n':>5s} {'predicted':>11s} {'observed':>10s}"
              f" {'gap':>7s}")
        for lab, n, pm, om in tab:
            print(f"    {lab:10s} {n:5d} {pm:11.3f} {om:10.3f} {pm - om:+7.3f}")
            rows.append({"section": "calibration", "building": b,
                         "condition": lab, "threshold_dBm": -70.0, "n": n,
                         "coverage_predicted_soft": pm,
                         "coverage_measured": om,
                         "coverage_error": pm - om})

    print("\nfade margin dB above threshold, G gaussian / E empirical")
    print(f"  {'building':9s} {'sigma':>7s}" + "".join(
        f"{'G' + str(int(p*100)):>7s}{'E' + str(int(p*100)):>7s}"
        for p in RELIABILITY))
    for b in BUILDINGS:
        g = df[df.building == b]
        beta = fit(g)
        resid = g["PL_excess"].to_numpy(float) - X(g) @ beta
        sigma = float(np.std(resid, ddof=1))
        fm = fade_margins(resid, sigma, RELIABILITY)
        print(f"  {b:9s} {sigma:7.2f}" + "".join(
            f"{fm[f'gauss_{int(p*100)}']:7.2f}{fm[f'emp_{int(p*100)}']:7.2f}"
            for p in RELIABILITY))
        rows.append({"section": "fade_margin", "building": b,
                     "condition": "fitted", "sigma_used": sigma, **fm})

    print(f"\ntransfer at -70 dBm, blind and after {N_CAL_POINTS} spaced points,"
          f" median over {N_REPEATS} draws")
    print(f"  {'train':9s} {'test':9s} {'measured':>9s} {'blind Q':>9s}"
          f" {'err':>7s} {'k-shot Q':>9s} {'err':>7s} {'blind acc':>10s}"
          f" {'k-shot acc':>11s}")
    T = -70.0
    rng = np.random.default_rng(SEED)
    for a, b in itertools.permutations(BUILDINGS, 2):
        tr, te = df[df.building == a], df[df.building == b]
        beta = fit(tr)
        resid_tr = tr["PL_excess"].to_numpy(float) - X(tr) @ beta
        sigma = float(np.std(resid_tr, ddof=1))
        mu = predicted_prx(beta, te)
        meas = te["P_rx"].to_numpy(float)
        pos = te[["x_m", "y_m"]].to_numpy(float)

        blind = coverage_metrics(mu, sigma, meas, T)
        rows.append({"section": "transfer", "building": f"{a}->{b}",
                     "condition": "blind", **blind})

        shifts, accs, covs = [], [], []
        for _ in range(N_REPEATS):
            cal = spaced_idx(pos, N_CAL_POINTS, rng)
            if len(cal) < 3:
                continue
            mask = np.ones(len(te), dtype=bool)
            mask[cal] = False
            shift = float(np.mean(meas[cal] - mu[cal]))
            m = coverage_metrics(mu[mask] + shift, sigma, meas[mask], T)
            covs.append(m["coverage_predicted_soft"] - m["coverage_measured"])
            accs.append(m["accuracy"])
            shifts.append(shift)
        k_err = float(np.median(covs))
        k_acc = float(np.median(accs))
        rows.append({"section": "transfer", "building": f"{a}->{b}",
                     "condition": f"kshot{N_CAL_POINTS}",
                     "threshold_dBm": T, "coverage_error": k_err,
                     "accuracy": k_acc, "sigma_used": sigma,
                     "coverage_measured": blind["coverage_measured"]})
        print(f"  {a:9s} {b:9s} {blind['coverage_measured']:9.3f}"
              f" {blind['coverage_predicted_soft']:9.3f}"
              f" {blind['coverage_error']:+7.3f}"
              f" {blind['coverage_measured'] + k_err:9.3f} {k_err:+7.3f}"
              f" {blind['accuracy']:10.3f} {k_acc:11.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
