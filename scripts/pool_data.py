import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

NLOS_ONLY = False

PTX_DBM = 10.0
D0 = 1.0
FSPL_1M_SYSTEM = 33.3
NOISE_FLOOR_DBM = -115.0
NOISE_FLOOR_MARGIN = 10.0

WALL_COLS = [
    "Num_brick_wall",
    "Num_wood_wall",
    "Num_glass_wall",
    "Num_drywall",
    "Num_column",
]

PAPER_COUNTS = {
    ("Comms", "C1"): 721,
    ("Comms", "C2"): 677,
    ("SSE", "C1"): 107,
    ("SSE", "C2"): 107,
    ("Library", "C1"): 343,
    ("Library", "C2"): 344,
}


def load_one(path):
    name = os.path.basename(path)
    parts = name.replace(".csv", "").split("_")
    scenario, config = parts[1], parts[2]

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).split("(")[0].strip() for c in df.columns]

    if "LOS/NLOS Classification" in df.columns:
        df["audit_label"] = df["LOS/NLOS Classification"]

    df["building"] = scenario
    df["config"] = config
    df["source_file"] = name
    return df


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "Prx_*_classified.csv")))
    if not paths:
        raise FileNotFoundError(f"No Prx_*_classified.csv found in {DATA_DIR}")

    frames = [load_one(p) for p in paths]

    print(f"NLOS_ONLY = {NLOS_ONLY}")
    print(f"\n{'building':9s} {'cfg':4s} {'rows':>5s} {'paper':>6s}")
    for f in frames:
        b, c, n = f["building"].iloc[0], f["config"].iloc[0], len(f)
        exp = PAPER_COUNTS.get((b, c))
        flag = "" if exp == n else "  <--"
        print(f"{b:9s} {c:4s} {n:5d} {exp if exp else '-':>6}{flag}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"{'pooled':9s} {'':4s} {len(df):5d}")

    if "Elevator" in df.columns:
        df["has_elevator_feature"] = df["Elevator"].notna().astype(int)
        df["Num_elevator"] = pd.to_numeric(df["Elevator"], errors="coerce").fillna(0)
        df = df.drop(columns=["Elevator"])
    else:
        df["Num_elevator"] = 0
        df["has_elevator_feature"] = 0

    for c in WALL_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    df["N_total"] = df[WALL_COLS].sum(axis=1) + df["Num_elevator"]
    df["P_rx"] = pd.to_numeric(df["P_rx"], errors="coerce")

    impossible = df["P_rx"] >= PTX_DBM
    print(f"\ndropped, P_rx >= {PTX_DBM:.0f} dBm: {int(impossible.sum())}")
    if impossible.any():
        print(df.loc[impossible, ["source_file", "Coord.", "Distance", "P_rx"]]
              .to_string(index=False))
        df = df[~impossible].copy()

    floor_cut = NOISE_FLOOR_DBM + NOISE_FLOOR_MARGIN
    df["near_noise_floor"] = (df["P_rx"] <= floor_cut).astype(int)
    n_floor = int(df["near_noise_floor"].sum())
    print(f"flagged near_noise_floor, P_rx <= {floor_cut:.0f} dBm: {n_floor}")
    if n_floor:
        print(df[df["near_noise_floor"] == 1].groupby("building").size()
              .to_string())

    df["PL"] = PTX_DBM - df["P_rx"]
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    if "audit_label" not in df.columns:
        df["audit_label"] = np.nan
    df["audit_label"] = df["audit_label"].astype(str).str.strip().str.upper()
    df["audit_is_LOS_geom"] = (df["N_total"] == 0).astype(int)

    n0 = len(df)
    df = df[df["PL"].notna() & df["Distance"].notna()]
    print(f"dropped, PL or Distance missing: {n0 - len(df)}")

    n1 = len(df)
    df = df[df["Distance"] > D0]
    print(f"dropped, d <= {D0} m: {n1 - len(df)}")

    df["log10d"] = np.log10(df["Distance"])
    df["PL_excess"] = df["PL"] - FSPL_1M_SYSTEM
    df["config_C2"] = (df["config"] == "C2").astype(int)

    if "Comments" in df.columns:
        df["ref_level_changed"] = (
            df["Comments"].astype(str).str.contains("30", na=False).astype(int)
        )
    else:
        df["ref_level_changed"] = 0

    df["grid_letter"] = (
        df["Coord."].astype(str).str.extract(r"^([A-Za-z]+)")[0].str.upper()
    )

    df_all = df.copy()
    if NLOS_ONLY:
        los = df[df["audit_is_LOS_geom"] == 1]
        print(f"\ndropped, LOS (N_total == 0): {len(los)}")
        print(los.groupby(["building", "config"]).size().to_string())
        print(f"stated labels on those rows: "
              f"{dict(los['audit_label'].value_counts())}")
        df = df[df["audit_is_LOS_geom"] == 0].copy()

    print("\nper building x config")
    print(
        df.groupby(["building", "config"]).agg(
            n=("PL", "size"),
            d_min=("Distance", "min"),
            d_max=("Distance", "max"),
            PL_mean=("PL", "mean"),
            PL_sd=("PL", "std"),
            frac_LOS_geom=("audit_is_LOS_geom", "mean"),
        ).round(2)
    )

    print("\nstated label vs N_total == 0")
    print(pd.crosstab(df["audit_label"], df["audit_is_LOS_geom"]))

    print("\nobstruction sd by building, 0.00 = unidentifiable in that fold")
    print(df.groupby("building")[WALL_COLS + ["Num_elevator"]].std().round(2))

    print("\nnaive anchored exponent, no obstruction terms")
    for b, g in df.groupby("building"):
        n_hat = np.polyfit(10 * g["log10d"], g["PL_excess"], 1)[0]
        print(f"  {b:10s} {n_hat:6.2f}   n={len(g)}")
    n_all = np.polyfit(10 * df["log10d"], df["PL_excess"], 1)[0]
    print(f"  {'POOLED':10s} {n_all:6.2f}   n={len(df)}")

    out = os.path.join(DATA_DIR, "pooled.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {os.path.normpath(out)}  {len(df)} rows"
          + (", NLOS only" if NLOS_ONLY else ", unfiltered"))

    out_all = os.path.join(DATA_DIR, "pooled_all.csv")
    df_all.to_csv(out_all, index=False)
    print(f"Wrote {os.path.normpath(out_all)}  {len(df_all)} rows, unfiltered")

    return df


if __name__ == "__main__":
    main()
